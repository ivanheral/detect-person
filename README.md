# 🌟 Detector de Celebridades (CUDA) 🌟

¡Bienvenido! Este proyecto te permite crear tu propia IA capaz de reconocer celebridades usando el poder de tu tarjeta gráfica. 🚀

## 📋 ¿Qué hace este proyecto?
1. 📸 **Descarga** fotos de famosos de internet automáticamente.
2. 🧠 **Entrena** un cerebro digital (modelo YOLO) para aprender sus caras.
3. 🎯 **Detecta** quién es la persona en una foto nueva.

---

## 🛠️ Instalación Rápida
¡Sigue estos pasos y estarás listo en minutos! ⏱️

1. **Clona el repositorio** o descarga los archivos. 📂
2. **Crea un entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Instala las dependencias**: 📦
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Cómo se usa
Solo necesitas ejecutar un comando. La magia ocurre en `main.py`:

```bash
python main.py
```

### 📋 Opciones del Menú:
*   **1. Descargar un Actor Específico** 🔍: Escribe el nombre de alguien famoso y bajaremos fotos suyas.
*   **2. Entrenar Modelo** 🧠: Tu GPU (CUDA) empezará a aprender. ¡Es muy rápido!
*   **3. Realizar Detección** 🧐: Pon una foto en la carpeta `test/` y la IA te dirá quién es.
*   **4. Salir** 👋: Cierra el programa.

---

## 📂 Estructura del Proyecto
*   `main.py`: 🏠 El centro de mando.
*   `src/`: ⚙️ Los engranajes internos (lógica de descarga y entrenamiento).
*   `models/`: 🧠 Donde se guardan los modelos base descargados.
*   `dataset/`: 🖼️ Donde se guardan las fotos de entrenamiento.
*   `test/`: 🧪 Pon aquí las fotos que quieras que la IA adivine.
*   `runs/`: 💾 Donde se guarda el "cerebro" una vez entrenado.

---

## 💡 Requisitos del Sistema
*   Python 3.10+ 🐍
*   Tarjeta Gráfica NVIDIA (compatible con **CUDA**) para máxima velocidad. ⚡
*   Conexión a internet para bajar fotos. 🌐

---
*Hecho con ❤️ para amantes de la IA.*
