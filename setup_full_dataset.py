import time
from download_images import download_images

# Lista de actores para crear un detector robusto
ACTORES = [
    "Brad Pitt",
    "Antonio Banderas",
    "Leonardo DiCaprio",
    "Scarlett Johansson",
    "Tom Cruise",
    "Jennifer Lawrence",
    "Robert Downey Jr"
]

def setup():
    print(f"🚀 Iniciando creación de dataset para {len(ACTORES)} celebridades...")
    
    for actor in ACTORES:
        print(f"\n--- Preparando fotos de: {actor} ---")
        # Descargamos 40 fotos por actor (ajustado para no saturar DDG demasiado rápido)
        try:
            download_images(actor, max_images=40)
            # Pausa de seguridad entre actores para evitar bloqueos de IP
            print("Esperando 10 segundos antes del siguiente actor...")
            time.sleep(10)
        except Exception as e:
            print(f"Error con {actor}: {e}")

    print("\n✅ Dataset listo. Ahora puedes ejecutar: python train_model.py")

if __name__ == "__main__":
    setup()
