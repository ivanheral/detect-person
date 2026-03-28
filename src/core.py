import os, json, shutil, torch, gradio as gr
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
from huggingface_hub import HfApi, login
from icrawler.builtin import BingImageCrawler

# --- CONFIG ---
ACTS = ["Adrià Collado Fernando Navarro", "Beatriz Carvajal María Jesús", "Daniel Guzmán Roberto Alonso", "Diego Martín Carlos de la Fuente", "Eduardo García Josemi Cuesta", "Eduardo Gómez Mariano Delgado", "Emma Penella Doña Concha", "Eva Isanta Bea Villarejo", "Fernando Tejero Emilio Delgado", "Gemma Cuervo Vicenta Benito", "Guillermo Ortega Paco", "Isabel Ordaz Isabel Hierbas", "José Luis Gil Juan Cuesta", "Juan Díaz Álex Guerra", "Laura Pamplona Alicia Sanz", "Loles León Paloma Cuesta", "Luis Merlo Mauri Hidalgo", "Malena Alterio Belén López", "María Adánez Lucía Álvarez", "Mariví Bilbao Marisa Benito", "Nacho Guerreros José María", "Santiago Ramos Andrés Guerra", "Sofía Nieto Natalia Cuesta", "Vanesa Romero Ana"]
P = {k: Path(v) for k, v in {"d": "dataset", "r": "runs", "w": "weights", "t": "test", "l": "WebGPU/labels.json", "o": "weights/best.onnx", "pt": "runs/classify/det/w/weights/best.pt"}.items()}

# --- DATA ---
def download(q, n=40):
    f, t = P["d"]/q.replace(" ","_").lower(), P["d"]/f"t_{q[:3]}"
    f.mkdir(parents=True, exist_ok=True)
    if t.exists(): shutil.rmtree(t)
    t.mkdir()
    BingImageCrawler(storage={'root_dir': str(t)}, log_level=40).crawl(keyword=f"{q} face", max_num=n*2)
    c = 0
    for file in sorted(t.iterdir()):
        if c >= n: break
        try:
            with Image.open(file) as im:
                im = im.convert("RGB")
                if min(im.size) < 512: continue
                s = min(im.size); l, tp = (im.size[0]-s)//2, (im.size[1]-s)//2
                im.crop((l, tp, l+s, tp+s)).save(f/f"{c}.jpg", "JPEG", quality=90)
                c += 1; print(f"[{c}/{n}] {q}")
        except: pass
    shutil.rmtree(t, ignore_errors=True)

# --- ENGINE ---
def get_mod(p=None):
    loc = Path(p) if p else P["pt"]
    return YOLO(str(loc)) if loc.exists() or (p and "http" in str(p)) else None

def export(m):
    P["l"].parent.mkdir(parents=True, exist_ok=True); P["w"].mkdir(exist_ok=True)
    with open(P["l"], "w", encoding="utf-8") as f: json.dump({k: v.replace("_"," ").upper() for k,v in m.names.items()}, f, indent=4, ensure_ascii=False)
    shutil.copy(m.export(format="onnx", imgsz=224, simplify=True), P["o"])

def train(e=30):
    if not P["d"].exists(): return print("❌ Sin datos.")
    m = YOLO("models/yolo11n-cls.pt")
    m.train(data=str(P["d"].absolute()), epochs=e, imgsz=224, device=0 if torch.cuda.is_available() else "cpu", project="det", name="w", exist_ok=True)
    export(m)

def predict(i, m=None):
    mod = m or get_mod()
    if not mod: return print("❌ Sin modelo.")
    r = mod(i, verbose=False)[0]
    print(f"🌟 {r.names[r.probs.top1].replace('_',' ').upper()} ({r.probs.top1conf.item()*100:.1f}%)")
    return r

def upload(repo=None):
    load_dotenv(); login(token=os.getenv("HF_TOKEN"))
    api = HfApi(); r_id = f"{api.whoami()['name']}/{repo or input('Repo: ')}"
    api.create_repo(repo_id=r_id, exist_ok=True)
    export(YOLO(str(P["pt"])))
    for l, rm in [(P["pt"], "weights/best.pt"), (P["o"], "weights/best.onnx"), (P["l"], "labels.json")]:
        api.upload_file(path_or_fileobj=str(l), path_in_repo=rm, repo_id=r_id)
    print(f"✅ https://huggingface.co/{r_id}")
