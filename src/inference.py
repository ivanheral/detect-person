import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Optional
from src.config import P

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

def run_prediction(image_path: str, model: Optional[YOLO]) -> Optional[str]:
    """
    Realiza la inferencia de una imagen dada.

    Args:
        image_path: Ruta a la imagen de entrada.
        model: Objeto YOLO ya cargado o None.

    Returns:
        Un string con el resultado formateado, o None si falla.
    """
    print("\n--- Ejecutando Predicción ---")
    if not Path(image_path).exists():
        print(f"❌ Ruta de imagen no encontrada: {image_path}")
        return None

    # 1. Obtener o cargar el modelo
    if model is None:
        # Intenta cargar el modelo exportado, que es lo que el frontend suele usar
        model_path = P["o"]
        model = get_model(model_path)

    if model is None:
        return "❌ No se pudo cargar ni el modelo ni la imagen para predecir."

    # 2. Ejecutar inferencia
    try:
        # Ejecutamos en modo silencioso para el CLI, pero el return captura resultados
        r = model(image_path, verbose=False)[0]

        # Formateo del resultado (imitando la lógica del core.py)
        top1_class_name = r.names[r.probs.top1].replace('_', ' ').upper()
        confidence = r.probs.top1conf.item() * 100

        resultado = f"🌟 {top1_class_name} ({confidence:.1f}%)"
        print(f"🌟 {resultado}")
        return resultado
    except Exception as e:
        print(f"❌ Error durante la inferencia: {e}")
        return None

def predict(image_path: str, model: Optional[YOLO] = None) -> Optional[str]:
    """Wrapper público para la predicción."""
    return run_prediction(image_path, model)

if __name__ == "__main__":
    # Prueba requerirá un modelo y una imagen de prueba.
    print("--- Prueba de módulo inference ---")
    # Simulación:
    # 1. Asegurarse de que exista un modelo para probar
    # 2. Asegurarse de que exista una imagen en test/
    # predict(image_path="ruta/a/imagen_de_prueba.jpg", model=model)
    pass
