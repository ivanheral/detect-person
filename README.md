# 🌟 Celebrity AI Detector (YOLO11 + WebGPU)

Sistema integral de Inteligencia Artificial para la detección y clasificación de celebridades, optimizado para ejecutarse localmente (GPU CUDA) o directamente en el navegador (WebGPU/ONNX).

---

## 🚀 Características Principales

- **🔄 Pipeline Automatizado**: Descarga de dataset, entrenamiento optimizado para GPU (RTX 50 serie soportada) y despliegue.
- **🌐 Doble Despliegue**: 
    - **WebGPU**: Interfaz nativa en HTML/JS usando ONNX Runtime Web para máxima rapidez.
    - **Gradio Space**: Aplicación web lista para Hugging Face Spaces (`app.py`).
- **📦 Sincronización Directa**: Exportación automática de etiquetas (`labels.json`) y modelos al formato ONNX.

---

## 🛠️ Instalación y Uso

### 1. Clonar y Configurar Entorno
```bash
# Instalar dependencias necesarias
pip install -r requirements.txt
```

### 2. Ejecutar Menú Principal (CLI)
Todo el sistema se gestiona desde el panel principal:
```bash
python main.py
```

En el menú podrás:
1.  **Descargar imágenes** de cualquier famoso específico.
2.  **Preparar el dataset base** (7 celebridades populares).
3.  **Entrenar el modelo** YOLO11-Cls con optimización CUDA.
4.  **Probar detecciones** localmente.
5.  **Exportar y Subir**: Convertir a ONNX y subir a Hugging Face automáticamente.

---

## 🔬 Estructura del Proyecto

- `src/`: Lógica central (Modelos, Datos, Hugging Face).
- `WebGPU/`: Frontend HTML interactivo para ejecución en navegador.
- `app.py`: Aplicación Gradio para compartir en la nube.
- `models/`: Directorio de modelos base pre-entrenados.
- `requirements.txt`: Dependencias unificadas (Entorno local + Nube).

---

## 🎯 Cómo subir a Hugging Face Spaces

1. Crea un nuevo Space en **Hugging Face** (Select Gradio SDK).
2. Sube los archivos `app.py` y `requirements.txt`.
3. Asegúrate de configurar tu `HF_TOKEN` en el archivo `.env` para la subida automática de pesos.

---

## 🚀 Pruebas con WebGPU
Para ver el detector en acción en el navegador sin intermediarios, simplemente abre la carpeta `WebGPU/` y lanza un servidor local (o usa la extensión Live Server de VS Code). ¡Toda la computación ocurrirá en tu tarjeta gráfica de forma nativa!

---

💡 *Desarrollado con ❤️ para IA y Detección de Celebridades.*
