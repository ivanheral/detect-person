# 🌟 Celebrity AI Detector

Sistema de Inteligencia Artificial para la detección y clasificación de celebridades (específicamente actores de 'Aquí no hay quien viva'), optimizado para ejecución local (CUDA) y despliegue web (Hugging Face / WebGPU).

---

## 🚀 Características Principales

- **💎 Arquitectura Minimalista**: Toda la lógica central consolidada en un único motor (`src/core.py`).
- **🔄 Pipeline Todo-en-Uno**: Gestión completa desde un solo menú interactivo (`main.py`).
- **🌐 Inferencia Doble**:
    - **Gradio**: Aplicación web lista para Hugging Face Spaces (`app.py`).
    - **WebGPU**: Ejecución nativa en navegador usando ONNX Runtime.
- **📦 Sincronización Inteligente**: Generación automática de etiquetas y exportación ONNX simplificada.

---

## 🛠️ Instalación y Uso

### 1. Clonar y Configurar Entorno

Se recomienda usar un entorno virtual para mantener las dependencias aisladas:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Linux/macOS:
source .venv/bin/activate
# En Windows:
# .venv\Scripts\activate

# Instalar dependencias optimizadas
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Para habilitar la subida automática a Hugging Face, crea un archivo `.env` en la raíz del proyecto:

```bash
# Crear archivo .env (o créalo manualmente)
echo "HF_TOKEN=tu_token_de_hugging_face_aqui" > .env
```
> [!TIP]
> Puedes obtener tu token en [Hugging Face Settings](https://huggingface.co/settings/tokens).

### 3. Ejecutar Gestor Principal
```bash
python main.py
```
Desde este menú podrás:
1. **Descargar dataset**: Individual o el set base de 24 actores.
2. **Entrenar**: Crea tu modelo YOLO11-Cls optimizado.
3. **Probar**: Inferencia interactiva sobre la carpeta `test/`.
4. **Desplegar**: Sube automáticamente pesos y etiquetas a Hugging Face.

---

## 🔬 Estructura del Proyecto

El proyecto ha sido reducido a su expresión mínima funcional:
- `src/core.py`: Motor central (IA, Rutas, Hugging Face).
- `src/data.py`: Descarga y procesamiento de imágenes.
- `main.py`: Interfaz CLI y herramientas de prueba.
- `app.py`: Aplicación Gradio (Interfaz Web).
- `WebGPU/`: Frontend estático para ejecución nativa en cliente.
- `requirements.txt`: Dependencias limpias y organizadas.

---

## 🚀 Despliegue en Hugging Face
Crea un Space de tipo **Gradio SDK** y sube `app.py`, `requirements.txt` y tu carpeta `src/`. Configura tu `HF_TOKEN` en variables de entorno para que `main.py` pueda subir los modelos automáticamente.

---
