import gradio as gr
from src.core import get_predictor, predict
from PIL import Image

REPO = "ivanheral/detect-person"
model = get_predictor(f"https://huggingface.co/{REPO}/resolve/main/weights/best.pt") or get_predictor()

def detect(img):
    if not model: return "Sin modelo", None
    res = predict(img, model)
    return {res.names[i].replace("_"," ").upper(): float(res.probs.data[i]) for i in range(len(res.names))}, Image.fromarray(res.plot()[:,:,::-1])

with gr.Blocks() as demo:
    gr.Markdown("# 🌟 Celebrity Detector v3.0")
    with gr.Row():
        with gr.Column():
            i, b = gr.Image(type="pil"), gr.Button("Analizar", variant="primary")
        with gr.Column():
            l, o = gr.Label(num_top_classes=3), gr.Image()
    b.click(detect, i, [l, o])

if __name__ == "__main__": demo.launch()
