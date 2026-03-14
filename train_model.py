import os
from ultralytics import YOLO
import torch

def train_celebrity_model(dataset_path="dataset", model_name="yolo11n-cls.pt"):
    """
    Entrena un modelo de clasificación con las imágenes descargadas.
    La GPU CUDA será detectada automáticamente.
    """
    # Intentar forzar el uso de CUDA si está disponible
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"¡GPU detectada!: {gpu_name}")
    else:
        device = "cpu"
        print("AVISO: No se detectó GPU. El entrenamiento será LENTO en CPU.")
        print("Asegúrate de tener instalado PyTorch con soporte CUDA (cu128).")

    # Cargar modelo pre-entrenado para clasificación (YOLO11 Nano - Cls)
    model = YOLO(model_name)

    print("Iniciando entrenamiento...")
    # Entrenamiento con los parámetros optimizados
    results = model.train(
        data=dataset_path, 
        epochs=30, 
        imgsz=224, 
        device=device,
        project="celebrity_detector",
        name="run1"
    )

    print(f"\nEntrenamiento completado. El modelo está guardado en: {results.save_dir}")
    return model

if __name__ == "__main__":
    if not os.path.exists("dataset") or not os.listdir("dataset"):
        print("Error: La carpeta 'dataset' no existe o está vacía.")
        print("Por favor, ejecuta 'python setup_full_dataset.py' para descargar imágenes primero.")
    else:
        # Iniciamos el entrenamiento
        try:
            train_celebrity_model()
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
