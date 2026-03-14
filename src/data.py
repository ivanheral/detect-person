import os
import requests
import time
from ddgs import DDGS
from PIL import Image
from io import BytesIO

def process_image(img_content, min_res=512):
    """Procesa una imagen para que sea 1:1 y cumpla resolución mínima."""
    img = Image.open(BytesIO(img_content)).convert("RGB")
    w, h = img.size
    
    if w < min_res or h < min_res:
        return None
        
    # Recorte 1:1 centrado
    size = min(w, h)
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))

def download_celebrity(query, max_imgs=40, base_dir="dataset"):
    """Descarga imágenes de una celebridad con sistema de reintentos."""
    folder = os.path.join(base_dir, query.replace(" ", "_").lower())
    if not os.path.exists(folder): os.makedirs(folder)
    
    count = 0
    retries = 3
    
    for attempt in range(retries):
        print(f"Buscando '{query}' (Intento {attempt + 1}/{retries})...")
        try:
            with DDGS() as ddgs:
                # Pausa inicial respiratoria más larga en reintentos
                time.sleep(2 + attempt * 2)
                
                # Parámetros explícitos para ayudar a la búsqueda
                results = ddgs.images(
                    query, 
                    region="wt-wt", 
                    safesearch="off", 
                    max_results=max_imgs * 2
                )
                
                results_list = []
                # Capturamos los resultados inmediatamente
                for r in results:
                    results_list.append(r)
                    if len(results_list) >= max_imgs * 2: break
                
                if not results_list:
                    print(f"⚠️  No se encontraron resultados para '{query}' en este intento.")
                    if attempt < retries - 1:
                        print("Reintentando en unos segundos...")
                        continue
                    break

                for res in results_list:
                    if count >= max_imgs: break
                    try:
                        time.sleep(0.4) 
                        resp = requests.get(res['image'], timeout=10)
                        if resp.status_code == 200:
                            img = process_image(resp.content)
                            if img:
                                img.save(os.path.join(folder, f"{count}.jpg"), "JPEG", quality=90)
                                count += 1
                                print(f"[{count}/{max_imgs}] Guardada: {query}")
                    except: continue
                
                # Si hemos conseguido imágenes, salimos del bucle de reintentos
                if count > 0:
                    break
        
        except Exception as e:
            error_str = str(e).lower()
            if "ratelimit" in error_str or "403" in error_str:
                print(f"⚠️  Bloqueo detectado (Rate Limit).")
                if attempt < retries - 1:
                    wait_time = 15 + attempt * 10
                    print(f"Esperando {wait_time} segundos para reintentar...")
                    time.sleep(wait_time)
                else:
                    print("Se agotaron los reintentos debido al bloqueo.")
            else:
                print(f"Error en intento {attempt + 1}: {e}")
            
    return count
