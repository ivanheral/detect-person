import shutil
from pathlib import Path
from icrawler.builtin import BingImageCrawler
from PIL import Image

def process_image(p, res=512):
    try:
        with Image.open(p) as img:
            img = img.convert("RGB")
            if min(img.size) < res: return None
            s = min(img.size)
            l, t = (img.size[0]-s)//2, (img.size[1]-s)//2
            return img.crop((l, t, l+s, t+s))
    except: return None

def download_celebrity(q, max_i=40):
    folder, temp = Path("dataset")/q.replace(" ","_").lower(), Path("dataset")/f"t_{q[:3]}"
    folder.mkdir(parents=True, exist_ok=True)
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir()
    BingImageCrawler(storage={'root_dir': str(temp)}, log_level=40).crawl(keyword=f"{q} face", max_num=max_i*2)
    c = 0
    for f in sorted(temp.iterdir()):
        if c >= max_i: break
        img = process_image(f)
        if img:
            img.save(folder/f"{c}.jpg", "JPEG", quality=90)
            c += 1
            print(f"[{c}/{max_i}] {q}")
    shutil.rmtree(temp, ignore_errors=True)
