import os
import json
import shutil
from ultralytics import YOLO

# Rutas
model_pt = os.path.join("runs", "classify", "celebrity_detector", "run", "weights", "best.pt")
labels_json = os.path.join("WebGPU", "labels.json")
output_onnx = os.path.join("weights", "best.onnx")

def finalize():
    if not os.path.exists(model_pt):
        print(f"❌ Error: No se encontró {model_pt}")
        return

    print("--- 📑 GENERANDO ETIQUETAS ---")
    model = YOLO(model_pt)
    
    # Extraer y formatear nombres
    labels = {k: v.replace("_", " ").upper() for k, v in model.names.items()}
    
    # Guardar en WebGPU/labels.json
    os.makedirs(os.path.dirname(labels_json), exist_ok=True)
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)
    print(f"✅ Labels generados en {labels_json}")

    print("\n--- ⚙️ EXPORTANDO A ONNX (WEB) ---")
    # Exportar (esto crea best.onnx en la misma carpeta que best.pt)
    onnx_path = model.export(format="onnx", imgsz=224, simplify=True)
    
    # Mover a la carpeta weights/ central
    os.makedirs("weights", exist_ok=True)
    shutil.copy(onnx_path, output_onnx)
    print(f"✅ Modelo exportado a {output_onnx}")

if __name__ == "__main__":
    finalize()
