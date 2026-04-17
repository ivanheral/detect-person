import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from ultralytics import YOLO
from src.config import P, load_state, save_state

if TYPE_CHECKING:
    from ultralytics import YOLO

def export_model_and_label(model: YOLO):
    """
    Exporta el modelo YOLO entrenado a formato ONNX y genera el archivo de etiquetas JSON.
    Actualiza el estado del proyecto con la referencia al modelo exportado.
    """
    print("\n--- Iniciando Exportación de Modelo y Etiquetas ---")

    # 1. Exportar ONNX
    try:
        # Usamos la función de exportación del propio objeto YOLO para consistencia
        onnx_path = model.export(format="onnx", imgsz=224, simplify=True)
        P["o"].unlink(missing_ok=True) # Eliminar versión anterior
        shutil.copy(onnx_path, str(P["o"]))
        print(f"✅ Modelo ONNX exportado exitosamente a {P['o'].absolute()}")
    except Exception as e:
        print(f"❌ Fallo al exportar el modelo ONNX: {e}")
        return

    # 2. Generar labels.json
    def format_names(names: dict) -> dict:
        """Formatea el diccionario de nombres de clases para el JSON."""
        formatted = {}
        for key, v in names.items():
            # Lógica de formato compleja extraída del core.py
            p = [w.capitalize() for w in v.replace("_", " ").split()]
            if len(p) > 2:
                formatted[key] = f"{' '.join(p[:2])} ({' '.join(p[2:])})"
            elif len(p) == 3:
                formatted[key] = f"{' '.join(p[:2])} ({p[-1]})"
            else:
                formatted[key] = " ".join(p)
        return formatted

    try:
        # El modelo entrenado (model) ya contiene el diccionario names
        label_data = format_names(model.names)

        with open(P["l"], "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=4, ensure_ascii=False)
        print(f"✅ Etiquetas JSON generadas exitosamente en {P['l'].absolute()}")
    except Exception as e:
        print(f"❌ Fallo al generar labels.json: {e}")
        return

    # 3. Actualizar el estado global
    state = load_state()
    state["last_exported_model"] = f"ONNX ({P['o'].name})"
    state["last_run_timestamp"] = datetime.now().isoformat()
    save_state(state)

    print("✨ Exportación y actualización de estado completadas.")

if __name__ == "__main__":
    from ultralytics import YOLO
    from datetime import datetime
    print("--- Prueba de módulo exporter ---")
    # Este módulo ahora depende de que trainer.py ejecute primero el entrenamiento.
    pass
