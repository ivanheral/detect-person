from src.data import download_celebrity
import os

# Probamos con un actor que sabemos que falló
actor = "Antonio Banderas"
print(f"Probando descarga manual para: {actor}")
count = download_celebrity(actor, max_imgs=5)
print(f"\nResultado final: {count} imágenes descargadas.")

# Verificamos si la carpeta tiene algo
folder = os.path.join("dataset", actor.replace(" ", "_").lower())
if os.path.exists(folder):
    files = os.listdir(folder)
    print(f"Archivos en carpeta: {files}")
else:
    print("La carpeta no fue creada.")
