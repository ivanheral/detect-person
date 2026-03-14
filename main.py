import sys
import os
from src.data import download_celebrity
from src.model import train, predict

def menu():
    print("\n" + "="*45)
    print("      DETECTOR DE CELEBRIDADES (CUDA)       ")
    print("="*45)
    print("1. Descargar un Actor Específico")
    print("2. Entrenar Modelo (GPU Optimizada)")
    print("3. Realizar Detección (Predecir)")
    print("4. Salir")
    
    op = input("\nElige una opción: ")
    
    if op == "1":
        nombre = input("Nombre del actor: ")
        download_celebrity(nombre)
    elif op == "2":
        train()
    elif op == "3":
        if not os.path.exists("test"): os.makedirs("test")
        fotos = [f for f in os.listdir("test") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not fotos:
            print("\n⚠️  No hay fotos en la carpeta 'test/'.")
            path = input("Introduce la ruta completa de una imagen: ").strip('"')
        else:
            print("\nFotos encontradas en 'test/':")
            for i, f in enumerate(fotos): print(f"{i+1}. {f}")
            sel = input("\nElige el número de foto o introduce una ruta: ")
            
            if sel.isdigit() and 1 <= int(sel) <= len(fotos):
                path = os.path.join("test", fotos[int(sel)-1])
            else:
                path = sel.strip('"')

        if os.path.exists(path):
            predict(path)
        else:
            print("❌ Archivo no encontrado.")
    elif op == "4":
        print("¡Hasta luego!")
        sys.exit()
    else:
        print("Opción inválida.")

if __name__ == "__main__":
    if not os.path.exists("dataset"): os.makedirs("dataset")
    if not os.path.exists("test"): os.makedirs("test")
    if not os.path.exists("models"): os.makedirs("models")
    while True:
        try:
            menu()
        except KeyboardInterrupt:
            print("\n\nSaliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
