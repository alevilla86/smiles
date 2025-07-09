import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
from rdkit import RDLogger

from constants import SIMILARITY_RESULTS_DIR, RECOMMENDATIONS_DIR, LEISHMANIA_MODEL_PATH, LEISHMANIA_DONOVANI_MODEL_PATH

RDLogger.DisableLog('rdApp.*')

# SIMILARITY_RESULTS_DIR = Path("similarity_results")
# RECOMMENDATIONS_DIR = Path("recommendations")
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_TOP = 0.8  # Umbral de decisión para considerar activo
THRESHOLD_BOTOM = 0.5  # Umbral de decisión para considerar inactivo

# Cargar modelo entrenado
print("Cargando modelo de Leishmania...")
clf = joblib.load(LEISHMANIA_DONOVANI_MODEL_PATH)
print(f"Modelo cargado exitosamente: {LEISHMANIA_DONOVANI_MODEL_PATH}.")

# Función para obtener fingerprint desde SMILES
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
    
def reorder_columns(df, ref_smiles):
    # Columns to move
    priority_cols = ["actividad_predicha", "probabilidad_leishmania", "tanimoto_similarity", "cosine_similarity"]

    # Current columns list
    cols = list(df.columns)

    # Remove the priority columns from the current list (if they exist)
    for c in priority_cols:
        if c in cols:
            cols.remove(c)

    # Find index of 'lipinski_violations'
    lipinski_idx = cols.index("no_lipinski_violations")

    # Insert the priority columns right after 'lipinski_violations'
    new_cols = cols[:lipinski_idx + 1] + priority_cols + cols[lipinski_idx + 1:]

    # Reorder DataFrame columns
    df = df[new_cols]

    # Save to Excel
    output_csv = RECOMMENDATIONS_DIR / f"{ref_smiles}_recommendations.csv"
    df.to_csv(output_csv, index=False)

def add_leishmania_prediction_columns():

    for file in SIMILARITY_RESULTS_DIR.glob("*.csv"):

        print(f"Procesando archivo: {file.name}")
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
        df["probabilidad_leishmania"] = None
        for idx, pred, prob in zip(valid_idx, predictions, probabilities):
            df.at[idx, "actividad_predicha"] = calculate_leishmania_probability(prob)
            df.at[idx, "probabilidad_leishmania"] = prob

        # Ordenar por prioridades
        df_sorted = df.sort_values(by=["probabilidad_leishmania", "tanimoto_similarity", "cosine_similarity"], ascending=[False, False, False])

        # Guardar archivo con resultados
        #output_excel = RECOMMENDATIONS_DIR / f"{ref_smiles}_recommendations.xlsx"
        #df_sorted.to_excel(output_excel, index=False)
        reorder_columns(df_sorted, ref_smiles)

def calculate_leishmania_activity(smiles_list):
    """
    Calcula la actividad de Leishmania para una lista de SMILES.
    Retorna un DataFrame con las columnas:
    - SMILES
    - Probabilidad de ser activo
    - Actividad predicha (Alta, Media, Baja)
    """
    results = []
    
    for smiles in smiles_list:
        fp = get_fingerprint(smiles)
        if fp is not None:
            prob = clf.predict_proba([fp])[0][1]  # Probabilidad de ser activo
            results.append({
                "SMILES": smiles,
                "Probabilidad de ser activo": prob
            })
        else:
            results.append({
                "SMILES": smiles,
                "Probabilidad de ser activo": None
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":

        # Ejecutar el script principal
    print("Ejecutando predicciones de actividad para Leishmania...")

    add_leishmania_prediction_columns()

    print("Predicciones completadas.")