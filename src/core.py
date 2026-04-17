import os, json, shutil, torch, hashlib, gradio as gr
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
from huggingface_hub import HfApi, login
from icrawler.builtin import BingImageCrawler

# --- CONFIG ---
ACTS = ["Carmen Machi Aída García", "Paco León Luisma García", "Pepe Viyuela Chema Martínez", "Mariano Peña Mauricio Colmenero", "Melani Olivares Paz Bermejo", "David Castillo Jonathan García", "Ana Polvorosa Lorena García", "Miren Ibarguren Soraya García", "Eduardo Casanova Fidel Martínez", "Marisol Ayuso Eugenia García", "Secun de la Rosa Toni Colmenero", "Pepa Rus Macu", "Canco Rodríguez Barajas", "Dani Martínez Simón Bermejo", "Óscar Reyes Machupichu", "Sanseverina Lazar Aidita", "Bernabé Fernández Marcial", "Rafael Ramos Germán"]
P = {k: Path(v) for k, v in {"d": "dataset", "ds": "dataset_split", "r": "runs", "w": "weights", "t": "test", "l": "WebGPU/labels.json", "o": "weights/best.onnx", "pt": "runs/classify/det/w/weights/best.pt"}.items()}

# --- DATA ---
def download(q, n=40):
    f, t = P["d"]/q.replace(" ","_").lower(), P["d"]/f"t_{q[:3]}"
    f.mkdir(parents=True, exist_ok=True)
    def get_h(p):
        with open(p, "rb") as bf: return hashlib.md5(bf.read()).hexdigest()
    
    ex = [img for img in f.iterdir() if img.suffix.lower() == ".jpg"]
    hs = {get_h(img) for img in ex}
    
    if len(ex) < n:
        print(f"🔍 {q}: {len(ex)}/{n} fotos. Descargando...")
        if t.exists(): shutil.rmtree(t)
        t.mkdir()
        BingImageCrawler(storage={'root_dir': str(t)}, log_level=40).crawl(keyword=f"{q} actor portrait headshot closeup", max_num=n*4)
        c = len(ex)
        for file in sorted(t.iterdir()):
            if c >= n: break
            try:
                h = get_h(file)
                if h in hs: continue
                with Image.open(file) as im:
                    im = im.convert("RGB")
                    if min(im.size) < 512: continue
                    w, h_img = im.size
                    max_dim = max(w, h_img)
                    canvas = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                    canvas.paste(im, ((max_dim - w) // 2, (max_dim - h_img) // 2))
                    canvas.save(f/f"tmp_{c}.jpg", "JPEG", quality=90)
                    hs.add(h); c += 1; print(f"[{c}/{n}] {q}")
            except: pass
        shutil.rmtree(t, ignore_errors=True)
    
    print(f"♻️ Renombrando {q}...")
    imgs = sorted([img for img in f.iterdir() if img.suffix.lower() == ".jpg"])
    for i, img in enumerate(imgs): img.rename(f/f"sort_{i}.jpg")
    for i in range(len(imgs)): (f/f"sort_{i}.jpg").rename(f/f"{i}.jpg")
    print(f"✅ {q} listo: {len(imgs)} fotos.")

# --- ENGINE ---
def get_mod(p=None):
    loc = Path(p) if p else P["pt"]
    return YOLO(str(loc)) if loc.exists() or (p and "http" in str(p)) else None

def export(m):
    P["l"].parent.mkdir(parents=True, exist_ok=True); P["w"].mkdir(exist_ok=True)
    def fmt(v):
        p = [w.capitalize() for w in v.replace("_", " ").split()]
        return f"{' '.join(p[:2])} ({' '.join(p[2:])})" if len(p) > 2 else f"{' '.join(p[:2])} ({p[-1]})" if len(p) == 3 else " ".join(p)
    with open(P["l"], "w", encoding="utf-8") as f:
        json.dump({k: fmt(v) for k, v in m.names.items()}, f, indent=4, ensure_ascii=False)
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
