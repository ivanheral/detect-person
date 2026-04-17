# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos Comunes

### Configuración del Entorno
- Crear entorno virtual: `python -m venv .venv`
- Activar entorno (Windows): `.venv\Scripts\activate`
- Activar entorno (Linux/macOS): `source .venv/bin/activate`
- Instalar dependencias: `pip install -r requirements.txt`

### Desarrollo y Ejecución
- **Interfaz Principal (CLI)**: `python main.py`
    - Opción 1: Descargar dataset de un actor específico.
    - Opción 2: Descargar dataset completo (18 actores).
    - Opción 3: Entrenar modelo YOLO11-Cls (genera `weights/best.pt`, `weights/best.onnx` y `WebGPU/labels.json`).
    - Opción 4: Probar inferencia sobre imágenes en la carpeta `test/`.
    - Opción 5: Subir pesos y etiquetas a Hugging Face Hub (requiere `HF_TOKEN` en `.env`).
    - Opción 6: Resetear sistema (borra datasets y modelos).
- **Aplicación Web (Gradio)**: `python app.py` (Interfaz para Hugging Face Spaces).
- **Frontend WebGPU**: Servir la carpeta `WebGPU/` con cualquier servidor estático (ej. `python -m http.server`).

## Arquitectura del Código

### Estructura General
El proyecto sigue una arquitectura minimalista centralizada:

- **Motor Central (`src/core.py`)**: Contiene TODA la lógica de negocio, incluyendo:
    - `download()`: Descarga de imágenes mediante BingImageCrawler, con normalización (redimensionado, padding a cuadrado) y prevención de duplicados mediante hashing MD5.
    - `train()`: Entrenamiento de YOLO11-Cls usando Ultralytics.
    - `export()`: Exportación del modelo a ONNX optimizado y generación de `labels.json` con nombres formateados para el frontend.
    - `predict()`: Lógica de inferencia.
    - `upload()`: Integración con Hugging Face Hub para despliegue automatizado.
- **Puntos de Entrada**:
    - `main.py`: Orquestador de tareas de desarrollo.
    - `app.py`: Interfaz de usuario para despliegue en la nube.
- **Frontend (`WebGPU/`)**: Aplicación estática que consume el modelo ONNX directamente en el navegador usando ONNX Runtime Web con soporte WebGPU.

### Configuración
- Las rutas de archivos y nombres de actores están centralizados en `src/core.py` (diccionario `P` y lista `ACTS`).
- La configuración de Hugging Face se gestiona mediante variables de entorno en un archivo `.env`.

### Flujo de Datos
1. Imágenes -> `dataset/` (descargadas y normalizadas).
2. Entrenamiento -> `runs/` -> `weights/best.pt`.
3. Exportación -> `weights/best.onnx` + `WebGPU/labels.json`.
4. Inferencia -> `app.py` (Gradio) o `WebGPU/index.html` (Navegador).
