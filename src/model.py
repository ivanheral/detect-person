import os
from ultralytics import YOLO
from src.utils import get_device, validate_dataset, export_model_assets
from src.config import PATHS

def train(dataset=PATHS["dataset"], epochs=30):
    valido, clases, msg = validate_dataset(dataset)
    if not valido: return print(f"❌ {msg}")

    # Limpieza automática
    for f in os.listdir(dataset):
        if f.endswith(".cache"): os.remove(os.path.join(dataset, f))
    
    model = YOLO(os.path.join(PATHS["models"], "yolo11n-cls.pt"))
    print(f"🚀 Entrenando con {len(clases)} clases...")
    
    model.train(
        data=dataset, epochs=epochs, imgsz=224, device=get_device(),
        project="celebrity_detector", name="run", exist_ok=True,
        batch=16, workers=8, amp=True
    )

    # Exportación automática unificada
    export_model_assets(model)
    return model

def predict(img_path):
    model_path = os.path.join(PATHS["runs"], "classify", "celebrity_detector", "run", "weights", "best.pt")
    if not os.path.exists(model_path): return print("❌ Primero entrena (Opción 3).")

    res = YOLO(model_path)(img_path, verbose=False)[0]
    name = res.names[res.probs.top1].replace("_", " ").upper()
    conf = res.probs.top1conf.item() * 100
    print(f"\n🌟 DETECCIÓN: {name} ({conf:.1f}%)")
