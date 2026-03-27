import sys
import os
from src.data import download_celebrity
from src.model import train, predict
from src.hf import upload_model

# Actores por defecto para el dataset base
ACTORES_BASE = [
    "Leonardo DiCaprio", "Brad Pitt", "Tom Cruise",
    "Robert Downey Jr", "Scarlett Johansson",
    "Jennifer Lawrence", "Antonio Banderas"
]

def descargar_dataset_base():
    print(f"\n--- 📥 DESCARGANDO {len(ACTORES_BASE)} FAMOSOS (DATASET BASE) ---")
    img_por_actor = 40
    for actor in ACTORES_BASE:
        download_celebrity(actor, max_imgs=img_por_actor)
    print("\n✅ Dataset base descargado con éxito.")

def test_batch():
    if not os.path.exists("test"): os.makedirs("test")
    fotos = [f for f in os.listdir("test") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not fotos:
        print("\n⚠️ No hay fotos en la carpeta 'test/'.")
        path = input("Introduce la ruta completa de una imagen: ").strip('"')
        if os.path.exists(path):
            predict(path)
        else:
            print("❌ Archivo no encontrado.")
        return

    print("\nFotos encontradas en 'test/':")
    for i, f in enumerate(fotos): print(f"{i+1}. {f}")
    print(f"{len(fotos)+1}. 🚀 Evaluar TODAS a la vez")
    
    sel = input("\nElige el número de foto, 'todas', o introduce una ruta: ").strip()
    
    if sel.lower() == 'todas' or sel == str(len(fotos)+1):
        print("\n--- 🔍 EVALUACIÓN EN LOTE ---")
        for f in fotos:
            print(f"\nArchivo: {f}")
            predict(os.path.join("test", f))
    elif sel.isdigit() and 1 <= int(sel) <= len(fotos):
        path = os.path.join("test", fotos[int(sel)-1])
        predict(path)
    else:
        path = sel.strip('"')
        if os.path.exists(path):
            predict(path)
        else:
            print("❌ Archivo no encontrado.")


def menu():
    print("\n" + "="*45)
    print("      DETECTOR DE CELEBRIDADES (CUDA)       ")
    print("="*45)
    print("1. Descargar foto de famoso específico")
    print("2. Descargar dataset base completo (7 famosos)")
    print("3. Entrenar modelo (GPU Optimizada)")
    print("4. Realizar detecciones (Probar modelo)")
    print("5. Exportar a ONNX y subir a Hugging Face")
    print("6. Salir")
    
    op = input("\nElige una opción: ").strip()
    
    if op == "1":
        nombre = input("Nombre del actor: ")
        download_celebrity(nombre)
    elif op == "2":
        descargar_dataset_base()
    elif op == "3":
        train()
    elif op == "4":
        test_batch()
    elif op == "5":
        upload_model()
    elif op == "6":
        print("¡Hasta luego!")
        sys.exit()
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    for folder in ["dataset", "test", "models"]:
        if not os.path.exists(folder): os.makedirs(folder)
    while True:
        try:
            menu()
        except KeyboardInterrupt:
            print("\n\nSaliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
