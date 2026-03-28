import os, shutil
from icrawler.builtin import BingImageCrawler
from PIL import Image

def process_image(img_path, min_res=512):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < min_res or h < min_res: return None
        size = min(w, h)
        left, top = (w-size)//2, (h-size)//2
        return img.crop((left, top, left+size, top+size))
    except: return None

def download_celebrity(query, max_imgs=40, base_dir="dataset"):
    folder = os.path.join(base_dir, query.replace(" ", "_").lower())
    temp = os.path.join(base_dir, f"temp_{query[:5]}")
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(temp): shutil.rmtree(temp)
    os.makedirs(temp)

    print(f"🔎 Buscando rostros de '{query}'...")
    crawler = BingImageCrawler(storage={'root_dir': temp}, log_level=40)
    crawler.crawl(keyword=f"{query} face portrait actor", max_num=max_imgs*2)

    count = 0
    for f in sorted(os.listdir(temp)):
        if count >= max_imgs: break
        img = process_image(os.path.join(temp, f))
        if img:
            img.save(os.path.join(folder, f"{count}.jpg"), "JPEG", quality=90)
            count += 1
            print(f"[{count}/{max_imgs}] {query}")

    shutil.rmtree(temp, ignore_errors=True)
    return count
