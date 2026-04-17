import sys
import os
import shutil
from pathlib import Path
from src.config import P, ACTS, load_state, save_state
from src.downloader import download
from src.trainer import train
from src.exporter import export_model_and_label
from src.inference import predict
from ultralytics import YOLO # Necesario para inicializar en run_test si no se carga por la ruta

# --- ENTRADA MODIFICADA ---
def run_test():
    """Permite realizar inferencias sobre imágenes en el directorio test/."""
    print("\n--- Modo de Inferencia (Test) ---")
    test_dir = P["t"]
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print("⚠️ El directorio de prueba (test/) está vacío. Por favor, ejecute la descarga primero.")
        return

    # Listado de archivos
    fs = [f for f in test_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    if not fs:
        print("⚠️ No se encontraron imágenes JPG/JPEG/PNG en el directorio de prueba.")
        return

    print("\n--- Imágenes disponibles para prueba: ---")
    for i, f in enumerate(fs):
        print(f"{i+1}. {f.name}")
    print(f"{len(fs)+1}. Ver todas las imágenes")
    print(f"{len(fs)+2}. Salir")

    while True:
        try:
            s = input("\n> ").strip()
            if s == str(len(fs)+2):
                print("Saliendo del modo de prueba.")
                break

            if s == str(len(fs)+1):
                # Prueba con todas las imágenes
                print("\n--- Ejecutando predicción en TODAS las imágenes del directorio de prueba ---")
                for i, f in enumerate(fs):
                    predict(f"{f.absolute()}", model=None)
                print("--- Fin de la prueba en lote. ---")
            elif s.isdigit():
                index = int(s)
                if 1 <= index <= len(fs):
                    file_path = f"{fs[index-1].absolute()}"
                    predict(file_path, model=None)
                else:
                    print("Índice fuera de rango.")
            else:
                print("Opción no válida. Por favor, intente de nuevo.")
        except Exception as e:
            print(f"Ocurrió un error en el modo de prueba: {e}")
            break

def reset():
    """Resetea todo el estado del proyecto (dataset, modelos, resultados)."""
    if input("⚠️ ¿Borrar dataset, modelos y resultados? Esto es IRREVERSIBLE. (s/n): ").lower() == 's':
        dirs_to_clean = [P["d"], P["ds"], P["r"]]
        for d in dirs_to_clean:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(exist_ok=True)
                print(f"✅ Directorio {d.name} borrado y recreado.")

        # También limpiamos el estado persistente
        state_file_path = load_state.__globals__['STATE_FILE']
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
            print(f"✅ Estado persistente {state_file_path} borrado.")
        else:
            print(f"ℹ️ No se encontró el archivo de estado para borrar.")

        print("\n✨ Sistema reseteado. Por favor, vuelva a ejecutar el proceso.")

if __name__ == "__main__":
    # Asegurar la creación de directorios base
    [P[d].mkdir(exist_ok=True) for d in ["d", "t", "w"]]

    print("\n" + "="*15 + "\n SPOTLIGHT AI v2.0 - ORQUESTADOR\n" + "="*15 + "\n")
    while True:
        print("\n" + "="*15 + "\n MENÚ PRINCIPAL\n" + "="*15)
        print("1. Descargar Dataset (Por Actor)")
        print("2. Descargar Dataset (Todos los Actores)")
        print("3. Entrenar Modelo (YOLO) + Exportar")
        print("4. Probar Inferencia (Usando imágenes en test/)")
        print("5. Subir Modelo a Hugging Face")
        print("6. Resetear Sistema")
        print("7. Salir")

        s = input("\n> ").strip()

        if s == "1":
            query = input("Ingrese el nombre del actor para la descarga: ")
            download(query)
        elif s == "2":
            print("Se procederá a descargar todos los datasets. Esto puede tardar mucho tiempo. Por favor, confírmelo (s/n): ")
            if input("> ").lower() == 's':
                for act in ACTS:
                    download(act)
            else:
                print("Operación cancelada.")
        elif s == "3":
            # 3. Entrenar: Llama al proceso completo (Entrena Y Exporta)
            print("\n===============================")
            print("🚀 INICIANDO CICLO COMPLETO: ENTRENAMIENTO Y EXPORTACIÓN")
            print("===============================")
            try:
                # Intentamos forzar el entrenamiento, que a su vez llama a la exportación
                train(dataset_path=P["d"].resolve())
            except Exception as e:
                print(f"❌ Error crítico al ejecutar el entrenamiento/exportación: {e}")
        elif s == "4":
            run_test()
        elif s == "5":
            # 5. Subir: Ahora requiere que el modelo exista para funcionar
            if not Path(P["o"]).exists():
                print("🛑 ¡ADVERTENCIA! El archivo ONNX no existe. Por favor, ejecute primero la Opción 3 (Entrenar) para generar el modelo.")
            else:
                print("Procediendo con la subida a Hugging Face Hub...")
                try:
                    from src.downloader import download # Asegurar que el estado esté actualizado antes de subir
                    upload()
                except Exception as e:
                    print(f"❌ Fallo al subir el modelo: {e}")
        elif s == "6":
            reset()
        elif s == "7":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")