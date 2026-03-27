# 🌐 WebGPU Celebrity Detector

Este es el frontend web para tu detector de celebridades. Utiliza **Transformers.js** para ejecutar modelos de inteligencia artificial directamente en el navegador del usuario, aprovechando la aceleración por hardware (**WebGPU** / WebGL).

## 🚀 Cómo usar este sitio

1.  **Entrenar y subir**: Primero usa los scripts de Python para entrenar tu modelo y subirlo a Hugging Face usando `upload_to_hf.py`.
2.  **Configurar el modelo**: Abre `index.html` y cambia la constante `modelId` por la ruta de tu modelo en Hugging Face.
    *   Ejemplo: `const modelId = 'tu-usuario/tu-modelo';`
3.  **Ejecutar**: Para que las imágenes carguen correctamente por seguridad del navegador (CORS), debes servir la carpeta con un servidor web simple.
    *   Desde esta carpeta, ejecuta:
        ```bash
        python -m http.server 8000
        ```
    *   Abre tu navegador en: `http://localhost:8000`

## ✨ Características
*   **Privacidad**: El procesamiento ocurre 100% en el dispositivo del cliente. No se envían fotos a ningún servidor.
*   **Velocidad**: Utiliza ONNX Runtime Web para máxima potencia.
*   **Diseño Premium**: Interfaz moderna con animaciones fluidas y modo oscuro.

## 🛠️ Tecnologías
*   [Transformers.js](https://huggingface.co/docs/transformers.js)
*   ONNX Runtime Web
*   Vanilla HTML5 / CSS3 / JavaScript (ES Modules)
