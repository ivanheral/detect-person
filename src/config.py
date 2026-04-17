import os
from pathlib import Path
from typing import Dict, Any

# --- CONFIGURACIÓN GLOBAL DEL PROYECTO ---

# Lista de actores centralizada
ACTS = ["Carmen Machi Aída García", "Paco León Luisma García", "Pepe Viyuela Chema Martínez", "Mariano Peña Mauricio Colmenero", "Melani Olivares Paz Bermejo", "David Castillo Jonathan García", "Ana Polvorosa Lorena García", "Miren Ibarguren Soraya García", "Eduardo Casanova Fidel Martínez", "Marisol Ayuso Eugenia García", "Secun de la Rosa Toni Colmenero", "Pepa Rus Macu", "Canco Rodríguez Barajas", "Dani Martínez Simón Bermejo", "Óscar Reyes Machupichu", "Sanseverina Lazar Aidita", "Bernabé Fernández Marcial", "Rafael Ramos Germán"]

# Rutas clave centralizadas
P: Dict[str, Path] = {
    "d": Path("dataset"),
    "ds": Path("dataset_split"),
    "r": Path("runs"),
    "w": Path("weights"),
    "t": Path("test"),
    "l": Path("WebGPU/labels.json"),
    "o": Path("weights/best.onnx"),
    "pt": Path("runs/classify/det/w/weights/best.pt")
}

# --- GESTIÓN DE ESTADO ---

STATE_FILE = ".project_state.json"

def load_state() -> Dict[str, Any]:
    """Carga el estado del proyecto desde el archivo JSON."""
    if not Path(STATE_FILE).exists():
        print("⚠️ Archivo de estado no encontrado. Inicializando nuevo estado.")
        return {"last_download_hashes": set(), "last_trained_weights": None, "last_exported_model": None, "last_run_timestamp": None}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Advertencia: El archivo de estado es inválido. Inicializando nuevo estado.")
        return {"last_download_hashes": set(), "last_trained_weights": None, "last_exported_model": None, "last_run_timestamp": None}

def save_state(state: Dict[str, Any]):
    """Guarda el estado actual del proyecto al disco."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)
    print(f"💾 Estado del proyecto guardado exitosamente en {STATE_FILE}.")

def update_state(key: str, value: Any):
    """Actualiza una clave específica en el estado y guarda."""
    state = load_state()
    state[key] = value
    save_state(state)