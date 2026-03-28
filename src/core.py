import os, json, shutil, torch
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from huggingface_hub import HfApi, login

# --- CONFIG & PATHS ---
ACTORES_BASE = ["Jordi Sánchez Antonio Recio", "Nathalie Seseña Berta Escobar", "Pablo Chiapella Amador Rivas", "Eva Isanta Maite Figueroa", "Nacho Guerreros Coque Calatrava", "Macarena Gómez Lola Trujillo", "José Luis Gil Enrique Pastor", "Ricardo Arroyo Vicente Maroto", "Loles León Menchu Carrascosa", "Petra Martínez Doña Fina", "Vanesa Romero Raquel Villanueva", "Fernando Tejero Fermín Trujillo", "Miren Ibarguren Yoli Morcillo", "Luis Merlo Bruno Quiroga", "Cristina Castaño Judith Becker", "Antonio Pagudo Javier Maroto", "Isabel Ordaz Araceli Madariaga", "Malena Alterio Cristina Aguilera", "Antonia San Juan Estela Reynolds", "Víctor Palmero Alba Recio", "Ernesto Sevilla Teodoro Rivas", "Eduardo Gómez Máximo Angulo", "Mariví Bilbao Izaskun Sagastume", "Laura Gómez-Lacueva Greta Garmendia"]

PATHS = {k: Path(v) for k, v in {"dataset": "dataset", "runs": "runs", "weights": "weights", "models": "models", "test": "test", "labels": "WebGPU/labels.json", "onnx": "weights/best.onnx", "pt": "runs/classify/celebrity_detector/run/weights/best.pt"}.items()}

# --- UTILS & MODEL ---
def get_device(): return 0 if torch.cuda.is_available() else "cpu"

def export_assets(model):
    PATHS["labels"].parent.mkdir(parents=True, exist_ok=True)
    labels = {k: v.replace("_", " ").upper() for k, v in model.names.items()}
    with open(PATHS["labels"], "w", encoding="utf-8") as f: json.dump(labels, f, indent=4, ensure_ascii=False)
    onnx_path = model.export(format="onnx", imgsz=224, simplify=True)
    PATHS["weights"].mkdir(exist_ok=True)
    shutil.copy(onnx_path, PATHS["onnx"])
    return labels

def get_predictor(path=None):
    p = Path(path) if path else PATHS["pt"]
    return YOLO(str(p)) if p.exists() or (path and "http" in str(path)) else None

def train(epochs=30):
    if not PATHS["dataset"].exists(): return print("❌ Sin dataset.")
    model = YOLO("models/yolo11n-cls.pt")
    model.train(data=str(PATHS["dataset"]), epochs=epochs, imgsz=224, device=get_device(), project="celebrity_detector", name="run", exist_ok=True, amp=True)
    export_assets(model)

def predict(img, model=None):
    mod = model or get_predictor()
    if not mod: return print("❌ Sin modelo.")
    res = mod(img, verbose=False)[0]
    print(f"🌟 {res.names[res.probs.top1].replace('_',' ').upper()} ({res.probs.top1conf.item()*100:.1f}%)")
    return res

# --- HUGGING FACE ---
def upload(repo=None):
    load_dotenv()
    if not PATHS["pt"].exists(): return print("❌ Entrena primero.")
    model = YOLO(str(PATHS["pt"]))
    export_assets(model)
    token = os.getenv("HF_TOKEN")
    if not token: return print("❌ Sin HF_TOKEN.")
    login(token=token)
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{repo or input('Repo: ')}"
    api.create_repo(repo_id=repo_id, exist_ok=True)
    for loc, rem in [(PATHS["pt"], "weights/best.pt"), (PATHS["onnx"], "weights/best.onnx"), (PATHS["labels"], "labels.json")]:
        api.upload_file(path_or_fileobj=str(loc), path_in_repo=rem, repo_id=repo_id)
    print(f"✅ https://huggingface.co/{repo_id}")
