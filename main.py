import sys, os, shutil
from src.data import download_celebrity
from src.model import train, predict
from src.hf import upload_model
from src.config import ACTORES_BASE, PATHS
from test import run_interactive_test

def reset_project():
    if input("\n⚠️ ¿Borrar TODO? (s/n): ").lower() != 's': return
    for f in ["dataset", "runs", "weights", "models"]:
        if os.path.exists(f): shutil.rmtree(f)
        os.makedirs(f)
    with open(PATHS["labels"], "w") as f: import json; json.dump({}, f)
    print("\n✅ Proyecto reseteado.")

def menu():
    opts = {
        "1": lambda: download_celebrity(input("Nombre: ")),
        "2": lambda: [download_celebrity(a) for a in ACTORES_BASE],
        "3": lambda: train(),
        "4": lambda: run_interactive_test(),
        "5": lambda: upload_model(),
        "6": lambda: reset_project(),
        "7": lambda: sys.exit()
    }
    print("\n" + "="*30 + "\n  LQSA DETECTOR (v2.0)\n" + "="*30)
    print("1. Descargar específico\n2. Descargar 24 vecinos\n3. Entrenar\n4. Probar modelo\n5. Subir a HF\n6. 🔥 Resetear\n7. Salir")
    
    sel = input("\nOpción: ").strip()
    if sel in opts: opts[sel]()
    else: print("Opción inválida.")

if __name__ == "__main__":
    for f in ["dataset", "test", "models", "weights"]: os.makedirs(f, exist_ok=True)
    while True:
        try: menu()
        except KeyboardInterrupt: break
        except Exception as e: print(f"❌ Error: {e}")
