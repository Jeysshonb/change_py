import os
import re
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# =========================
# CONFIG
# =========================
MODEL_ID = "jonathandinu/face-parsing"
HAIR_ID = 13  # clase "hair" en este modelo

# Presets de color (puedes editar)
COLOR_PRESETS = {
    "Personalizado (picker)": None,
    "Negro": "#121212",
    "Castaño": "#4b2e1f",
    "Rubio": "#d8c27a",
    "Platinado": "#d9d9d9",
    "Rojo": "#c1121f",
    "Azul": "#0077b6",
    "Verde": "#2a9d8f",
    "Morado": "#7209b7",
    "Rosa": "#ff4d8d",
}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

# Recomendado en Spaces para evitar timeouts raros al bajar modelos
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "30")

# Limita threads en CPU (opcional, mejora estabilidad)
try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
except Exception:
    pass

# =========================
# LOAD MODEL (una sola vez)
# =========================
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# =========================
# UTIL: color parsing robusto
# =========================
def parse_color_to_rgb(color):
    """
    Acepta:
      - "#RRGGBB"
      - "#RRGGBBAA" (ignora AA)
      - "#RGB"
      - "rgb(r,g,b)" / "rgba(r,g,b,a)"
      - (r,g,b) o [r,g,b]
      - dict con {"hex": "..."} (por si acaso)
    Devuelve (r,g,b) en 0..255
    """
    if color is None:
        return (255, 0, 0)

    if isinstance(color, dict):
        color = color.get("hex") or color.get("value") or color.get("color")

    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))

    if not isinstance(color, str):
        raise ValueError(f"Formato de color no soportado: {type(color)} -> {color}")

    s = color.strip()

    # rgb/rgba(...)
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s.lower())
    if m:
        r, g, b = map(int, m.groups())
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return (r, g, b)

    # hex
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:      # #RGB -> #RRGGBB
            h = "".join([c * 2 for c in h])
        if len(h) == 8:      # #RRGGBBAA -> ignora AA
            h = h[:6]
        if len(h) != 6:
            raise ValueError(f"HEX inválido: {s} (usa #RRGGBB)")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (r, g, b)

    raise ValueError(f"Color inválido: {color}")

# =========================
# IMAGE UTILS
# =========================
def resize_keep_aspect(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return pil.resize((nw, nh), Image.BILINEAR)

@torch.inference_mode()
def get_hair_mask(image: Image.Image, max_side: int = 640) -> Image.Image:
    """
    Devuelve una máscara L (0..255) del cabello, al tamaño original.
    """
    image = image.convert("RGB")
    ow, oh = image.size

    infer_img = resize_keep_aspect(image, max_side=max_side)

    inputs = processor(images=infer_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # (B,C,h,w)

    up = F.interpolate(
        logits,
        size=infer_img.size[::-1],  # (H,W)
        mode="bilinear",
        align_corners=False,
    )

    labels = up.argmax(dim=1)[0]  # (H,W)
    hair = (labels == HAIR_ID).to(torch.uint8).cpu().numpy() * 255

    mask = Image.fromarray(hair, mode="L")

    if mask.size != (ow, oh):
        mask = mask.resize((ow, oh), Image.NEAREST)

    return mask

def refine_mask(mask: Image.Image, close_kernel: int = 9, feather: int = 9) -> Image.Image:
    m = np.array(mask.convert("L"))

    # binariza
    _, mb = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # close
    k = max(3, int(close_kernel) | 1)  # impar
    kernel = np.ones((k, k), np.uint8)
    mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, kernel, iterations=1)

    # feather (blur)
    f = max(1, int(feather))
    if f % 2 == 0:
        f += 1
    mb = cv2.GaussianBlur(mb, (f, f), 0)

    return Image.fromarray(mb, mode="L")

def recolor_hair_lab(
    image: Image.Image,
    mask: Image.Image,
    color_input,
    strength: float = 0.85,
    brighten: float = 0.0,
) -> Image.Image:
    """
    Recolor en LAB para mantener sombras/luces.
    strength: 0..1 confirmando cuánto entra el color
    brighten: -0.3..0.3 (opcional, solo en cabello)
    """
    image_rgb = np.array(image.convert("RGB"))
    mask_f = np.array(mask.convert("L")).astype(np.float32) / 255.0
    alpha = np.clip(mask_f * float(strength), 0.0, 1.0)[..., None]  # (H,W,1)

    # RGB -> BGR -> LAB
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # color objetivo -> LAB
    r, g, b = parse_color_to_rgb(color_input)
    target_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    # Mezcla a/b hacia el objetivo
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - alpha[:, :, 0]) + target_lab[1] * alpha[:, :, 0]
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - alpha[:, :, 0]) + target_lab[2] * alpha[:, :, 0]

    # Ajuste de brillo en cabello
    if abs(brighten) > 1e-6:
        lab[:, :, 0] = np.clip(lab[:, :, 0] + (brighten * 255.0) * alpha[:, :, 0], 0, 255)

    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(out_rgb)

# =========================
# GRADIO RUN
# =========================
def run(image, preset, picked_color, strength, brighten, max_side, close_kernel, feather):
    try:
        if image is None:
            return None, None, "Sube una imagen primero."

        # color final
        preset_hex = COLOR_PRESETS.get(preset)
        final_color = (picked_color or "#ff0000") if preset_hex is None else preset_hex

        # máscara
        raw_mask = get_hair_mask(image, max_side=int(max_side))
        mask = refine_mask(raw_mask, close_kernel=int(close_kernel), feather=int(feather))

        # si la máscara salió vacía
        if np.mean(np.array(mask)) < 2.0:
            return image, mask, "No detecté cabello en esta foto. Prueba otra (mejor luz/frente)."

        # recolor
        result = recolor_hair_lab(
            image=image,
            mask=mask,
            color_input=final_color,
            strength=float(strength),
            brighten=float(brighten),
        )

        return result, mask, f"OK ✅ Color aplicado: {final_color}"

    except Exception as e:
        # devuelve el error visible en la app
        return None, None, f"ERROR: {type(e).__name__}: {e}"

DESCRIPTION = """
Sube una foto y cambia el color del cabello.
- Segmentación de cabello (hair mask)
- Recolor en LAB para conservar sombras/luces
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Cambiar color de cabello")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        inp = gr.Image(label="Tu foto", type="pil")
        out = gr.Image(label="Resultado", type="pil")

    with gr.Accordion("Controles", open=True):
        preset = gr.Dropdown(
            label="Preset",
            choices=list(COLOR_PRESETS.keys()),
            value="Personalizado (picker)",
        )
        picked_color = gr.ColorPicker(label="Color personalizado", value="#ff0000")
        strength = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Intensidad")
        brighten = gr.Slider(-0.3, 0.3, value=0.0, step=0.05, label="Brillo cabello (opcional)")
        max_side = gr.Slider(384, 1024, value=640, step=64, label="Resolución segmentación")
        close_kernel = gr.Slider(3, 21, value=9, step=2, label="Cerrar huecos (máscara)")
        feather = gr.Slider(1, 31, value=9, step=2, label="Suavizado bordes (máscara)")

    btn = gr.Button("Aplicar")
    mask_out = gr.Image(label="Máscara (debug)", type="pil")
    status = gr.Textbox(label="Estado", value="Listo.")

    btn.click(
        fn=run,
        inputs=[inp, preset, picked_color, strength, brighten, max_side, close_kernel, feather],
        outputs=[out, mask_out, status],
    )

if __name__ == "__main__":
    demo.launch()
