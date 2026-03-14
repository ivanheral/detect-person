import os
from ultralytics import YOLO
from src.utils import get_device, validate_dataset
import shutil

def train(dataset="dataset", epochs=30):
    """Entrena el modelo YOLO11."""
    # 1. Validar dataset
    valido, clases, msg = validate_dataset(dataset)
    if not valido:
        print(f"❌ Error en el dataset: {msg}")
        print("💡 Asegúrate de haber descargado imágenes suficientes para al menos 2 personas.")
        return None

    # 2. Limpieza preventiva
    # Borramos splits anteriores y cachés que causan el error de 'found 1 classes'
    for path in ["dataset_split", os.path.join(dataset, "train"), os.path.join(dataset, "val")]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except:
                pass
    
    for f in os.listdir(dataset):
        if f.endswith(".cache"):
            os.remove(os.path.join(dataset, f))

    device = get_device()
    # Cargamos el modelo desde la carpeta 'models'
    model_path = os.path.join("models", "yolo11n-cls.pt")
    model = YOLO(model_path)
    
    print(f"🚀 Iniciando entrenamiento con {len(clases)} clases: {', '.join(clases)}")
    
    return model.train(
        data=dataset,
        epochs=epochs,
        imgsz=224,
        device=device,
        project="celebrity_detector",
        name="run",
        exist_ok=True, # Sobrescribe 'run' si ya existe para no llenar el disco de carpetas run1, run2...
        batch=16, # Ajustado para 16GB VRAM (puedes subirlo a 32 o 64 si quieres ir más rápido)
        workers=8,
        amp=True  # Automatic Mixed Precision para GPU CUDA modernas
    )

def predict(img_path):
    """Realiza una predicción con el modelo más reciente."""
    model_path = os.path.join("runs", "classify", "celebrity_detector", "run", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("Asegúrate de haber completado el entrenamiento (Opción 3).")
        return

    model = YOLO(model_path)
    res = model(img_path)[0]
    
    name = res.names[res.probs.top1].replace("_", " ").upper()
    conf = res.probs.top1conf.item() * 100
    print(f"\n🌟 DETECCIÓN: {name} ({conf:.1f}%)")
