import torch
import os
import json
import shutil
from src.config import PATHS

def get_device():
    """Detecta GPU/CPU."""
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        return 0
    return "cpu"

def validate_dataset(path="dataset"):
    """Verifica si el dataset es válido."""
    if not os.path.exists(path): return False, [], "No existe 'dataset'."
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(classes) < 2: return False, classes, "Faltan clases (mínimo 2)."
    return True, classes, "Dataset OK."

def export_model_assets(model):
    """Genera labels.json y exporta ONNX centralizadamente."""
    print("\n--- 📦 EXPORTANDO MODELO Y ETIQUETAS ---")
    
    # 1. Labels
    labels = {k: v.replace("_", " ").upper() for k, v in model.names.items()}
    os.makedirs(os.path.dirname(PATHS["labels"]), exist_ok=True)
    with open(PATHS["labels"], "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)
    print(f"✅ Labels: {PATHS['labels']}")

    # 2. ONNX
    onnx_path = model.export(format="onnx", imgsz=224, simplify=True)
    os.makedirs(PATHS["weights"], exist_ok=True)
    shutil.copy(onnx_path, PATHS["onnx_export"])
    print(f"✅ ONNX: {PATHS['onnx_export']}")
