import os
import requests
import time
from ddgs import DDGS
from PIL import Image
from io import BytesIO

def download_images(query, max_images=50, output_dir="dataset"):
    """
    Descarga imágenes de DuckDuckGo para un término de búsqueda.
    Incluye pausas para evitar el bloqueo por Rate Limit.
    """
    celebrity_name = query.replace(" ", "_").lower()
    path = os.path.join(output_dir, celebrity_name)
    count = 0 
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directorio creado: {path}")

    print(f"Buscando imágenes de '{query}'...")
    
    try:
        with DDGS() as ddgs:
            time.sleep(3) # Pausa inicial para resetear estado en servidor
            results = ddgs.images(query, max_results=max_images)
            
            for i, res in enumerate(results):
                if count >= max_images:
                    break
                    
                image_url = res['image']
                try:
                    time.sleep(0.5) # Pausa amigable entre descargas
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        img = img.convert("RGB")
                        
                        width, height = img.size
                        # Filtrar por resolución mínima
                        if width < 512 or height < 512:
                            continue
                            
                        # Recortar a formato cuadrado (1:1) centrado
                        min_side = min(width, height)
                        left = (width - min_side) / 2
                        top = (height - min_side) / 2
                        right = (width + min_side) / 2
                        bottom = (height + min_side) / 2
                        img = img.crop((left, top, right, bottom))
                        
                        filename = f"{celebrity_name}_{count}.jpg"
                        img.save(os.path.join(path, filename), "JPEG", quality=95)
                        print(f"[{count+1}/{max_images}] OK: {filename} ({min_side}px)")
                        count += 1
                except Exception:
                    pass

        if count > 0:
            print(f"\nFinalizado: Se han descargado {count} imágenes en '{path}'.")
        else:
            print("\nNo se encontraron imágenes válidas o hubo un problema en la búsqueda.")

    except Exception as e:
        if "Ratelimit" in str(e):
            print("\n[!] ERROR: DuckDuckGo ha bloqueado tu IP temporalmente (Rate Limit).")
            print("CONSEJO: Espera 5 minutos o usa una VPN para continuar.")
        else:
            print(f"\n[!] Error inesperado: {e}")

if __name__ == "__main__":
    try:
        nombre = input("Introduce el nombre de la celebridad: ")
        num = int(input("¿Cuántas imágenes quieres descargar? (Ej: 50): "))
        download_images(nombre, max_images=num)
    except KeyboardInterrupt:
        print("\nDescarga cancelada.")
    except Exception as e:
        print(f"\nError: {e}")
