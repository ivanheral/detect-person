from src.core import PATHS, ACTORES_BASE, train, upload, predict, get_predictor
from src.data import download_celebrity

def run_test():
    mod = get_predictor()
    if not mod: return print("❌ Sin modelo.")
    while True:
        fs = [f for f in os.listdir(PATHS["test"]) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if PATHS["test"].exists() else []
        [print(f"{i+1}. {f}") for i, f in enumerate(fs)]
        print(f"{len(fs)+1}. TODAS\n{len(fs)+2}. Salir")
        s = input("\n> ").strip()
        if not s or s == str(len(fs)+2): break
        if s == str(len(fs)+1): [predict(PATHS["test"]/f, mod) for f in fs]
        elif s.isdigit() and 1 <= int(s) <= len(fs): predict(PATHS["test"]/fs[int(s)-1], mod)

def reset():
    if input("⚠️ ¿Borrar todo? (s/n): ").lower() != 's': return
    for d in ["dataset", "runs", "weights", "models"]:
        if PATHS[d].exists(): shutil.rmtree(PATHS[d])
        PATHS[d].mkdir(exist_ok=True)
    print("✅ Borrado.")

def menu():
    print("\n" + "="*20 + "\n LQSA v3.0\n" + "="*20)
    print("1. Descargar 1\n2. Descargar 24\n3. Entrenar\n4. Probar\n5. Subir HF\n6. Reset\n7. Salir")
    s = input("\n> ").strip()
    ops = {"1": lambda: download_celebrity(input("Nombre: ")), "2": lambda: [download_celebrity(a) for a in ACTORES_BASE], "3": train, "4": run_test, "5": upload, "6": reset, "7": sys.exit}
    if s in ops: ops[s]()

if __name__ == "__main__":
    [PATHS[d].mkdir(exist_ok=True) for d in ["dataset", "test", "models", "weights"]]
    while True: menu()
