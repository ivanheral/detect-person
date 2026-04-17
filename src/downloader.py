import os
import json
import shutil
from pathlib import Path
from PIL import Image
import hashlib
from typing import List, Set
from ultralytics import YOLO
from src.config import P, ACTS, load_state, save_state

def _get_hash(path: Path) -> str:
    """Calcula el hash MD5 de un archivo de imagen."""
    try:
        with open(path, "rb") as bf:
            return hashlib.md5(bf.read()).hexdigest()
    except Exception:
        return ""

def download_dataset(query: str, n: int = 40) -> (int, Set[str]):
    """
    Descarga, normaliza y limpia el dataset para una consulta específica.

    Args:
        query: El nombre del actor o tema a descargar.
        n: El número máximo de imágenes a procesar/guardar.

    Returns:
        Una tupla (conteo_guardado, hashes_guardados)
    """
    print(f"--- Iniciando descarga para: {query} ---")

    # 1. Definición de rutas
    base_dir = P["d"]
    query_dir = base_dir / query.replace(" ", "_").lower()
    temp_dir = base_dir / f"t_{query[:3]}"

    query_dir.mkdir(parents=True, exist_ok=True)

    # 2. Revisión de estado
    state = load_state()
    hashes_guardados = state.get("last_download_hashes", set())

    # 3. Preparación y descarga
    ex_paths = [img for img in query_dir.iterdir() if img.suffix.lower() == ".jpg"]

    if len(ex_paths) < n:
        print(f"🔍 {query}: {len(ex_paths)}/{n} fotos detectadas. Iniciando descarga masiva...")

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Ejecuta el crawler (usando log_level=40 para reducir ruido)
        from icrawler.builtin import BingImageCrawler
        BingImageCrawler(storage={'root_dir': str(temp_dir)}, log_level=40).crawl(keyword=f"{query} actor portrait headshot closeup", max_num=n*4)

        # 4. Procesamiento y normalización de las imágenes descargadas
        hashes_actuales: Set[str] = set()
        count_saved = 0

        for file_path in sorted(temp_dir.iterdir()):
            if count_saved >= n: break
            try:
                h = _get_hash(file_path)
                if h in hashes_guardados: continue

                with Image.open(file_path) as im:
                    im = im.convert("RGB")
                    if min(im.size) < 512: continue
                    w, h_img = im.size
                    max_dim = max(w, h_img)
                    canvas = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                    canvas.paste(im, ((max_dim - w) // 2, (max_dim - h_img) // 2))

                    # Guardar temporalmente en el directorio de destino con nomenclatura única
                    output_path = query_dir / f"sort_{count_saved}.jpg"
                    canvas.save(output_path, "JPEG", quality=90)

                    hashes_actuales.add(h)
                    count_saved += 1
                    print(f"[{count_saved}/{n}] {query}")
            except Exception as e:
                # print(f"Error al procesar {file_path}: {e}") # Comentar para no saturar la salida
                pass

        # 5. Limpieza
        shutil.rmtree(temp_dir, ignore_errors=True)

        # 6. Renombrar para el formato final (0.jpg, 1.jpg, ...)
        imgs = sorted([img for img in query_dir.iterdir() if img.suffix.lower() == ".jpg"])
        new_imgs = []
        for i, img in enumerate(imgs):
            # Sólo necesitamos renombrar los que fueron procesados y guardados
            if i < n:
                # Reemplazamos la lógica de renombrado original por una más limpia
                img.rename(query_dir / f"{i}.jpg")
                new_imgs.append(query_dir / f"{i}.jpg")

        print(f"✅ {query} listos: {len(new_imgs)} fotos procesadas y guardadas.")
        return len(new_imgs), hashes_actuales

    else:
        # Si ya hay suficientes imágenes, solo comprobamos y reportamos
        hashes_encontrados = {_get_hash(img) for img in ex_paths}
        print(f"✅ {query} ya contiene {len(ex_paths)} fotos. No se requiere descarga.")
        return len(ex_paths), hashes_encontrados

def download(query: str, n: int = 40):
    """Wrapper público para la descarga de datasets."""
    count, hashes = download_dataset(query, n)

    # Actualiza el estado global
    state = load_state()
    state["last_download_hashes"] = hashes
    state["last_download_query"] = query
    state["last_download_count"] = count
    save_state(state)

    return count, hashes

if __name__ == "__main__":
    # Prueba rápida (necesita ejecutar el entrenamiento primero si se quiere una prueba real)
    print("--- Prueba de módulo downloader ---")
    # Esto fallará si no se ejecuta el entrenamiento primero, pero sirve para verificar la estructura.
    # count, hashes = download("Carmen Machi", n=5)
    # print(f"Prueba final: Contó {count} imágenes y se registraron {len(hashes)} hashes.")

