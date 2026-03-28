import sys, os, shutil
from src.core import P, ACTS, download, train, upload, predict, get_mod

def run_test():
    m = get_mod()
    fs = [f for f in os.listdir(P['t']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if P['t'].exists() else []
    [print(f"{i+1}. {f}") for i, f in enumerate(fs)]; print(f"{len(fs)+1}. TODAS\n{len(fs)+2}. Salir")
    s = input("\n> ").strip()
    if s and s != str(len(fs)+2):
        if s == str(len(fs)+1): [predict(P['t']/f, m) for f in fs]
        elif s.isdigit() and 1 <= int(s) <= len(fs): predict(P['t']/fs[int(s)-1], m)

def reset():
    if input("⚠️ ¿Borrar todo? (s/n): ").lower() == 's':
        [shutil.rmtree(P[d], ignore_errors=True) or P[d].mkdir(exist_ok=True) for d in ["d", "r", "w"]]
        print("✅ Borrado.")

if __name__ == "__main__":
    [P[d].mkdir(exist_ok=True) for d in ["d", "t", "w"]]
    while True:
        print("\n" + "="*15 + "\n ANHQV v4.0\n" + "="*15 + "\n1. Descargar 1\n2. Descargar 24\n3. Entrenar\n4. Probar\n5. Subir\n6. Reset\n7. Salir")
        s = input("\n> ").strip()
        ops = {"1": lambda: download(input("Nombre: ")), "2": lambda: [download(a) for a in ACTS], "3": train, "4": run_test, "5": upload, "6": reset, "7": sys.exit}
        if s in ops: ops[s]()
