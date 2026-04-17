import torch
from ultralytics import YOLO
from pathlib import Path
import os
import json
from typing import Optional, Dict, Any
from src.config import P, load_state, save_state

def get_model(model_path: Path) -> Optional[YOLO]:
    """Inicializa y retorna un modelo YOLO si la ruta es válida."""
    if model_path.exists() or "http" in str(model_path):
        print(f"🔄 Cargando modelo de YOLO desde: {model_path.absolute()}")
        try:
            return YOLO(str(model_path))
        except Exception as e:
            print(f"❌ Error al cargar el modelo YOLO: {e}")
            return None
    else:
        print(f"❌ El modelo no se encuentra en: {model_path.absolute()}")
        return None

def train_model(dataset_path: Path, epochs: int = 30) -> Optional[YOLO]:
    """
    Entrena el modelo YOLO con el dataset proporcionado y ejecuta la exportación
    automáticamente si el entrenamiento es exitoso.

    Args:
        dataset_path: Ruta al dataset (directorio con las imágenes).
        epochs: Número de épocas de entrenamiento.

    Returns:
        El objeto YOLO del modelo entrenado, o None si falla.
    """
    print(f"\n--- Iniciando Entrenamiento YOLO con {dataset_path.absolute()} ({epochs} épocas) ---")

    if not dataset_path.exists() or not list(dataset_path.iterdir()):
        print("❌ No se encontraron datos en la ruta especificada. El entrenamiento no puede iniciarse.")
        return None

    # 1. Inicializar y entrenar
    # Asumiendo que el modelo base siempre es 'yolo11n-cls.pt' como en core.py
    model_base_path = Path("models/yolo11n-cls.pt")
    if not model_base_path.exists():
        print(f"⚠️ Advertencia: Modelo base {model_base_path} no encontrado. Asegúrate de descargarlo o inicializarlo.")
        # Intentamos obtenerlo desde la configuración si es un enlace HTTP, aunque es poco probable aquí.
        # Para el desarrollo, se debe asegurar que este modelo exista.

    m = YOLO(str(model_base_path))

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"▶️ Usando dispositivo: {device}")

    try:
        # El entrenamiento usa el parámetro 'project' y 'name' para generar la estructura 'runs/det/w/'
        results = m.train(data=str(dataset_path.absolute()),
                          epochs=epochs,
                          imgsz=224,
                          device=device,
                          project="det",
                          name="w",
                          exist_ok=True)

        print("✅ Entrenamiento completado con éxito.")
        return m
    except Exception as e:
        print(f"❌ ¡FALLO FATAL durante el entrenamiento! Detalle: {e}")
        return None

def train(dataset_path: Path, epochs: int = 30):
    """Wrapper público para iniciar el flujo de entrenamiento."""
    trained_model = train_model(dataset_path, epochs)

    if trained_model:
        # El éxito del entrenamiento debe actualizar el estado
        state = load_state()
        state["last_trained_weights"] = trained_model.model.path
        state["last_run_timestamp"] = datetime.now().isoformat()
        save_state(state)

        # 2. Exportación automática tras el entrenamiento
        from src.exporter import export_model_and_label
        export_model_and_label(trained_model)
    else:
        print("🚫 El flujo de entrenamiento se detuvo debido a un error o falta de datos.")

if __name__ == "__main__":
    from datetime import datetime
    # Este bloque de prueba requiere que el dataset exista y que el modelo base esté presente.
    # print("--- Prueba de módulo trainer ---")
    # train(dataset_path=P["d"], epochs=1)
