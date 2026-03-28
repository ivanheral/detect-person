import gradio as gr
from src.core import get_mod, predict
from PIL import Image

REPO = "ivanheral/detect-person"
m = get_mod(f"https://huggingface.co/{REPO}/resolve/main/weights/best.pt") or get_mod()

def det(img):
    if not m: return "Sin modelo", None
    r = predict(img, m)
    return {r.names[i].replace("_"," ").upper(): float(r.probs.data[i]) for i in range(len(r.names))}, Image.fromarray(r.plot()[:,:,::-1])

with gr.Blocks() as demo:
    gr.Markdown("# 🌟 Celebrity Detector v4.0")
    with gr.Row():
        with gr.Column(): i, b = gr.Image(type="pil"), gr.Button("Analizar", variant="primary")
        with gr.Column(): l, o = gr.Label(num_top_classes=3), gr.Image()
    b.click(det, i, [l, o])

if __name__ == "__main__": demo.launch()
