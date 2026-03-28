import os
from ultralytics import YOLO
from src.config import PATHS

def run_interactive_test():
    model_path = os.path.join(PATHS["runs"], "classify", "celebrity_detector", "run", "weights", "best.pt")
    if not os.path.exists(model_path): return print("❌ Primero entrena (Opción 3).")
    
    print("\n--- 🔍 CARGANDO MODELO ---")
    model = YOLO(model_path)
    
    while True:
        fotos = [f for f in os.listdir(PATHS["test"]) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))] if os.path.exists(PATHS["test"]) else []
        
        print("\n📸 " + "-"*20 + f"\nFotos en {PATHS['test']}:")
        for i, f in enumerate(fotos): print(f"{i + 1}. {f}")
        print(f"{len(fotos) + 1}. 🚀 Evaluar TODAS\n{len(fotos) + 2}. 🔙 Volver")

        sel = input("\nElige opción: ").strip()
        if not sel or sel == str(len(fotos) + 2): break
        
        if sel == str(len(fotos) + 1):
            for f in fotos:
                res = model(os.path.join(PATHS["test"], f), verbose=False)[0]
                print(f"🖼️ {f} -> {res.names[res.probs.top1].replace('_', ' ').upper()} ({res.probs.top1conf.item()*100:.1f}%)")
        elif sel.isdigit() and 1 <= int(sel) <= len(fotos):
            res = model(os.path.join(PATHS["test"], fotos[int(sel)-1]))[0]
            print(f"\n✅ {res.names[res.probs.top1].upper()} ({res.probs.top1conf.item()*100:.1f}%)")
        else: print("❌ Selección no válida.")

if __name__ == "__main__":
    if not os.path.exists(PATHS["test"]): os.makedirs(PATHS["test"])
    run_interactive_test()
