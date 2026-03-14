import torch
import os

def get_device():
    """Detecta la mejor unidad de procesamiento disponible (GPU/CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detectada: {gpu_name}")
        return 0
    print("⚠️ AVISO: No se detectó GPU. Usando CPU (Lento).")
    return "cpu"

def ensure_dir(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def validate_dataset(dataset_path="dataset"):
    """
    Verifica que el dataset sea válido para el entrenamiento.
    Retorna (es_valido, lista_de_clases, mensaje_error)
    """
    if not os.path.exists(dataset_path):
        return False, [], "La carpeta 'dataset' no existe."
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(classes) < 2:
        return False, classes, f"Se necesitan al menos 2 clases (carpetas) para entrenar, pero se encontraron {len(classes)}."
    
    classes_with_data = []
    errors = []
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 10:
            errors.append(f"La clase '{cls}' solo tiene {len(images)} imágenes (mínimo 10 recomendado).")
        else:
            classes_with_data.append(cls)
            
    if len(classes_with_data) < 2:
        return False, classes_with_data, "No hay suficientes clases con imágenes válidas. " + " ".join(errors)
    
    return True, classes_with_data, "Dataset válido."
