import os
import sys
from src.model import predict

def run_test_detection():
    test_dir = "test"
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"📁 Carpeta '{test_dir}' creada. Introduce ahí las imágenes que quieras probar.")
        return

    # Extensiones de imagen soportadas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]

    if not images:
        print(f"ℹ️ No hay imágenes en la carpeta '{test_dir}'.")
        print("Arrastra alguna imagen a esa carpeta para poder probar el modelo.")
        return

    print(f"\n--- Imágenes encontradas en '{test_dir}' ---")
    for i, img in enumerate(images):
        print(f"{i + 1}. {img}")
    
    try:
        choice = input("\nElige el número de la imagen (o 'a' para todas): ").strip().lower()
        
        if choice == 'a':
            for img in images:
                print(f"\nProcesando: {img}...")
                predict(os.path.join(test_dir, img))
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                predict(os.path.join(test_dir, images[idx]))
            else:
                print("❌ Selección inválida.")
    except ValueError:
        print("❌ Por favor, introduce un número válido.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_test_detection()
