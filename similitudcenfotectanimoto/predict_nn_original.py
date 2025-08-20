
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Carga diferida de TensorFlow con mensaje claro si no esta instalado
def _safe_import_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return tf, load_model
    except Exception as e:
        raise ImportError(
            "TensorFlow no esta instalado o no se pudo importar.\n"
            "Instala con:\n"
            "  pip install tensorflow     (CPU)\n"
            "  pip install \"tensorflow[and-cuda]\"  (GPU NVIDIA, TF≥2.12)\n"
            f"Detalle: {e}"
        )

def _ensure_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV/DF: {missing}")
    return df[features]

def _load_artifacts(artifacts_dir: str):
    """Carga model.keras, x_scaler.pkl y meta.json desde artifacts_dir."""
    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(f"No existe el directorio de artefactos: {artifacts_dir}")

    # Cargar meta primero para validar contenido
    meta_path = os.path.join(artifacts_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Falta meta.json en {artifacts_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Cargar scaler
    scaler_path = os.path.join(artifacts_dir, "x_scaler.pkl")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Falta x_scaler.pkl en {artifacts_dir}")
    with open(scaler_path, "rb") as f:
        x_scaler = pickle.load(f)

    # Cargar modelo Keras (necesita TF)
    model_path = os.path.join(artifacts_dir, "model.keras")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Falta model.keras en {artifacts_dir}")
    tf, load_model = _safe_import_tf()
    model = load_model(model_path)

    return model, x_scaler, meta

def predict_from_dataframe(df: pd.DataFrame, artifacts_dir: str) -> np.ndarray:
    """
    Predice sobre un DataFrame.
    Devuelve un vector np.ndarray con las predicciones.
    """
    model, x_scaler, meta = _load_artifacts(artifacts_dir)
    features = meta.get("features", None)
    if not features:
        raise KeyError("meta.json no contiene la clave 'features' con la lista de columnas.")
    y_min = meta.get("y_min", None)
    y_max = meta.get("y_max", None)
    clip_output = bool(meta.get("clip_output", False))

    X = _ensure_features(df, features).to_numpy(dtype=float)
    Xs = x_scaler.transform(X)

    # Predicción
    preds = model.predict(Xs, verbose=0).ravel()
    if clip_output and (y_min is not None) and (y_max is not None):
        preds = np.clip(preds, y_min, y_max)
    return preds

def predict_from_csv(csv_path: str, artifacts_dir: str, save_csv: str | None = None) -> np.ndarray:
    """
    Predice sobre un CSV.
    Si save_csv esta definido, guarda un CSV con una columna extra 'pred'.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No existe el CSV de entrada: {csv_path}")
    df = pd.read_csv(csv_path)
    preds = predict_from_dataframe(df, artifacts_dir)
    if save_csv:
        out = df.copy()
        out["pred"] = preds
        out.to_csv(save_csv, index=False)
        print(f"✅ Predicciones guardadas en: {save_csv}")
    return preds

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Inferencia con NN original exportada (model.keras + x_scaler.pkl + meta.json)")
    parser.add_argument("--artifacts", type=str, default="artifacts_nn_original",
                        help="Carpeta con model.keras, x_scaler.pkl y meta.json")
    parser.add_argument("--csv", type=str, default="",
                        help="Ruta a CSV de entrada para predecir")
    parser.add_argument("--save_csv", type=str, default="",
                        help="Ruta de salida (CSV) con columna 'pred' (opcional)")
    args, _ = parser.parse_known_args(sys.argv[1:])

    if not args.csv:
        print("ℹ️ Uso en terminal:")
        print("   python predict_nn_original.py --artifacts artifacts_nn_original --csv datos.csv --save_csv salida.csv")
        print("\nℹ️ Uso en Jupyter:")
        print("   from predict_nn_original import predict_from_csv")
        print("   preds = predict_from_csv('datos.csv', 'artifacts_nn_original', 'salida.csv')")
        return

    preds = predict_from_csv(args.csv, args.artifacts, save_csv=(args.save_csv or None))
    print(f"Predicciones (n={len(preds)}):")
    n_show = min(10, len(preds))
    print(preds[:n_show] if len(preds) else "[]")

if __name__ == "__main__":
    _main()
