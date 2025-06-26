import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

SIMILARITY_RESULTS_DIR = Path("similarity_results")
RECOMMENDATIONS_DIR = Path("recommendations")
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_TOP = 0.8  # Umbral de decisión para considerar activo
THRESHOLD_BOTOM = 0.5  # Umbral de decisión para considerar inactivo

# Cargar modelo entrenado
print("Cargando modelo de Leishmania...")
clf = joblib.load("models/leishmania_model_v1.pkl")
print("Modelo cargado exitosamente.")

# 🧬 Función para obtener fingerprint desde SMILES
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def calculate_leishmania_probability(probability):
    if probability >= THRESHOLD_TOP:
        return "Alta"
    elif THRESHOLD_BOTOM <= probability < THRESHOLD_TOP:
        return "Media"
    else:
        return "Baja"

def add_leishmania_prediction_columns(file):
    # Leer archivo con los SMILES
    df = pd.read_csv(file)

    ref_smiles = file.stem.split("_")[0]

    print(f"Prediciendo actividad de leishmania para compuestos similares a: {ref_smiles}")

    # Generar fingerprints y filtrar inválidos
    fps = []
    valid_idx = []

    for idx, smiles in enumerate(df["smiles_canonical"]):
        fp = get_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)
        else:
            fps.append(None)  # Mantener alineación con el DataFrame

    # Predecir solo los válidos
    X_valid = np.array([fp for fp in fps if fp is not None])
    predictions = clf.predict(X_valid)
    probabilities = clf.predict_proba(X_valid)[:, 1]  # Solo prob. de clase 1 (Activo)

    # Insertar predicciones en el DataFrame
    df["actividad_predicha"] = None
    for idx, pred, prob in zip(valid_idx, predictions, probabilities):
        df.at[idx, "actividad_predicha"] = calculate_leishmania_probability(prob)

    # Guardar archivo con resultados
    output_csv = RECOMMENDATIONS_DIR / f"{ref_smiles}_recommendations.csv"
    output_excel = RECOMMENDATIONS_DIR / f"{ref_smiles}_recommendations.xlsx"
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)

if __name__ == "__main__":

        # Ejecutar el script principal
    print("Ejecutando predicciones de actividad para Leishmania...")

    for file in SIMILARITY_RESULTS_DIR.glob("*.csv"):
        print(f"Procesando archivo: {file.name}")
        add_leishmania_prediction_columns(file)

    print("Predicciones completadas.")