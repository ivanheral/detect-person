import gradio as gr
from ultralytics import YOLO
import os
from PIL import Image

# 1. CARGAMOS EL MODELO (Se descarga automáticamente de tu repo)
# Asegúrate de que el repo sea PÚBLICO para que Gradio pueda acceder a él.
REPO_ID = "ivanheral/detect-person"
MODEL_PATH = "weights/best.pt"

try:
    print(f"🔄 Cargando modelo desde {REPO_ID}...")
    model = YOLO(f"https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_PATH}")
    print("✅ Modelo cargado con éxito.")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    # Backup: Intentar cargar local si existe
    if os.path.exists("weights/best.pt"):
        model = YOLO("weights/best.pt")
    else:
        model = None

def detect_celebrity(image):
    if model is None:
        return "Error: Modelo no cargado. Verifica el REPO_ID.", None
    
    # Realizar predicción
    results = model(image)[0]
    
    # Obtener el resultado con mayor confianza
    probs = results.probs
    if probs is not None:
        top1_idx = probs.top1
        name = results.names[top1_idx].replace("_", " ").upper()
        conf = probs.top1conf.item()
        
        # Formatear salida para Gradio (diccionario de etiquetas)
        # Esto genera la barra de confianza visual
        label_output = {results.names[i].replace("_", " ").upper(): float(probs.data[i]) for i in range(len(results.names))}
        
        # Imagen con anotación (opcional en clasificación)
        # results.plot() devuelve un array numpy
        annotated_img = Image.fromarray(results.plot()[:, :, ::-1]) # RGB conversion
        
        return label_output, annotated_img
    
    return "No se detectó nada", None

# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(f"# 🌟 Celebrity AI Detector")
    gr.Markdown("Sube la foto de un famoso para que la Inteligencia Artificial lo identifique.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Foto del Famoso")
            btn = gr.Button("🔍 Identificar Famoso", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(num_top_classes=3, label="Predicción")
            output_img = gr.Image(label="Detección Visual")

    btn.click(fn=detect_celebrity, inputs=input_img, outputs=[output_labels, output_img])
    
    gr.Markdown("---")
    gr.Markdown(f"🚀 Modelo alojado en: [huggingface.co/{REPO_ID}](https://huggingface.co/{REPO_ID})")

if __name__ == "__main__":
    demo.launch()
