try:
    from ddgs import DDGS
    print("Biblioteca 'ddgs' importada correctamente.")
    with DDGS() as ddgs:
        print("Buscando 'gato' para probar conectividad...")
        res = next(ddgs.images("gato", max_results=1))
        print(f"Éxito: {res['image']}")
except Exception as e:
    print(f"Fallo crítico: {e}")
