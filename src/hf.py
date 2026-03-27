import os
import json
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
from ultralytics import YOLO

def export_labels(model, output_path):
    """Extrae las etiquetas del modelo y las guarda en un JSON."""
    print("📝 Generando labels.json con los nombres de los famosos...")
    # model.names devuelve {0: 'Leonardo_DiCaprio', 1: 'Brad_Pitt', ...}
    labels = {k: v.replace("_", " ").upper() for k, v in model.names.items()}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)
    return labels

def upload_model(repo_name=None):
    """
    Exporta el modelo actual a ONNX, genera etiquetas y lo sube a Hugging Face.
    """
    load_dotenv()
    
    weights_dir = os.path.join("runs", "classify", "celebrity_detector", "run", "weights")
    model_path = os.path.join(weights_dir, "best.pt")
    labels_local_path = os.path.join("WebGPU", "labels.json")

    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo PyTorch (.pt) en: {model_path}")
        print("💡 Asegúrate de haber completado el entrenamiento antes de subirlo.")
        return

    # 1. CARGAR MODELO Y EXPORTAR
    print("\n--- 📦 PREPARANDO MODELO PARA LA WEB ---")
    try:
        model = YOLO(model_path)
        
        # Generar labels.json
        export_labels(model, labels_local_path)
        print(f"✅ Etiquetas guardadas en: {labels_local_path}")

        # Exportar a ONNX
        print("⚙️ Convirtiendo a formato ONNX (esto puede tardar un poco)...")
        onnx_file = model.export(format="onnx", imgsz=224, simplify=True)
        print(f"✅ Exportación completada: {onnx_file}")
    except Exception as e:
        print(f"❌ Error durante la preparación: {e}")
        return

    # 2. AUTENTICACIÓN HUGGING FACE
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN no encontrado en el archivo .env.")
        return

    if not repo_name:
        repo_name = input("\nIntroduce el nombre para el repo en HF (ej: detector-famosos): ").strip()
    
    if not repo_name:
        print("❌ El nombre no puede estar vacío.")
        return

    print("\n--- 🚀 SUBIR A HUGGING FACE ---")
    try:
        login(token=token)
        api = HfApi()
        user = api.whoami()["name"]
        repo_id = f"{user}/{repo_name}"

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        
        # Archivos a subir: Pesos y Etiquetas
        files_to_upload = [
            (os.path.join(weights_dir, "best.pt"), "weights/best.pt"),
            (os.path.join(weights_dir, "best.onnx"), "weights/best.onnx"),
            (labels_local_path, "labels.json") # Subimos labels al raíz del repo
        ]
        
        for local_file, path_in_repo in files_to_upload:
            if os.path.exists(local_file):
                print(f"⬆️ Subiendo {os.path.basename(local_file)}...")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                )

        print("\n" + "✅"*20)
        print("¡Misión cumplida! Todo está en la nube.")
        print(f"🔗 Enlace: https://huggingface.co/{repo_id}")
        print("✅"*20)
        
    except Exception as e:
        print(f"\n❌ Error durante la subida: {e}")
