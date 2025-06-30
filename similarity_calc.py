import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from compound_properties import CompoundProperties
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

DATA = Path("data")
OUT_DIR  = Path("similarity_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columnas relevantes para similitud de coseno
FEAT_COLS = [
    "hydrogen_bond_acceptors",
    "hydrogen_bond_donors",
    "rotatable_bonds",
    "log_p",
    "molecular_weight",
    "heavy_atom_count",
    "polar_surface_area",
]

def do_tanimoto_similarity(df, ref_smiles):
    print(f"Calculando similitud de Tanimoto para SMILES: {ref_smiles}")

    # Se prepara el fingerprint del SMILES de referencia
    mol_ref = Chem.MolFromSmiles(ref_smiles)
    fp_ref  = AllChem.GetMorganFingerprintAsBitVect(mol_ref, radius=2, nBits=2048)

    # Se calcula la similitud de Tanimoto para cada SMILES en el DataFrame
    sims = []
    for s in df["smiles_canonical"]:
        mol  = Chem.MolFromSmiles(s)
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sim  = DataStructs.TanimotoSimilarity(fp_ref, fp)
        sims.append(sim)

    return sims

def do_cosine_similarity(df, query_vector, smiles):
    print(f"Calculando similitud de coseno para SMILES: {smiles}")
    
    # Limpiar nulos si los hubiera y tomar subset de columnas que nos interesan
    df_feat = df.dropna(subset=FEAT_COLS)

    query_vec = pd.DataFrame([query_vector])[FEAT_COLS]  # asegura el orden

    # Normalizar con MinMaxScaler ya que los valores no son normales (se utilizó Shapiro para verificar)
    mm_scaler = MinMaxScaler()
    # Ajustar el escalador a la base de datos
    mm_X_base  = mm_scaler.fit_transform(df_feat[FEAT_COLS])
    mm_X_query = mm_scaler.transform(query_vec)

    # Similitud de coseno para los registros vs. la consulta con el MinMaxScaler
    mm_sims = cosine_similarity(mm_X_query, mm_X_base)[0]
    return mm_sims

def do_combined_similarity(df):
    """
    Calcula la similitud combinada entre las columnas de similitud de Tanimoto y Coseno.
    Utiliza diferentes métodos de combinación: ponderada, geométrica y armónica.
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas de similitud de Tanimoto y Coseno.
    Returns:
        tuple: Una tupla que contiene listas de similitudes combinadas para diferentes ponderaciones y métodos.
    """
    combined_25 = []
    combined_50 = []
    combined_75 = []
    combined_geometric = []
    combined_harmonic = []
    for idx, row in df.iterrows():
        tanimoto_sim = row["tanimoto_similarity"]
        cosine_sim   = row["cosine_similarity"]
        combine_weighted_25 = combined_similarity(cosine_sim, tanimoto_sim, w=0.25, method='weighted')
        combine_weighted_50 = combined_similarity(cosine_sim, tanimoto_sim, w=0.50, method='weighted')
        combine_weighted_75 = combined_similarity(cosine_sim, tanimoto_sim, w=0.75, method='weighted')
        combine_geometric = combined_similarity(cosine_sim, tanimoto_sim, method='geometric')
        combine_harmonic = combined_similarity(cosine_sim, tanimoto_sim, method='harmonic')
        combined_25.append(combine_weighted_25)
        combined_50.append(combine_weighted_50)
        combined_75.append(combine_weighted_75)
        combined_geometric.append(combine_geometric)
        combined_harmonic.append(combine_harmonic)

    return (combined_25, combined_50, combined_75, combined_geometric, combined_harmonic)
    
def combined_similarity(cosine_sim, tanimoto_sim, w=0.5, method='weighted'):
    if method == 'weighted':
        return w * cosine_sim + (1 - w) * tanimoto_sim
    elif method == 'geometric':
        return (cosine_sim * tanimoto_sim) ** 0.5
    elif method == 'harmonic':
        if cosine_sim == 0 or tanimoto_sim == 0:
            return 0
        return 2 / (1 / cosine_sim + 1 / tanimoto_sim)
    else:
        raise ValueError("Method must be 'weighted', 'geometric', or 'harmonic'")

def calculate_similarity():
    for file in DATA.glob("*.csv"):

        ref_smiles = file.stem.split("_")[0]

        compound_properties = CompoundProperties(None, ref_smiles)
        if compound_properties is None:
            print(f"Skipping file {file} due to invalid SMILES: {ref_smiles}")
            continue

        # Cargar CSV de datos crudos
        df = pd.read_csv(file)

        cosine_sim = do_cosine_similarity(df, compound_properties.vectorize(), ref_smiles)
        tanimoto_sim = do_tanimoto_similarity(df, ref_smiles)
        
        df_sim = df.copy()
        df_sim["cosine_similarity"] = cosine_sim
        df_sim["tanimoto_similarity"] = tanimoto_sim

        combined_25, combined_50, combined_75, combined_geometric, combined_harmonic = do_combined_similarity(df_sim)

        #df_sim["combined_25"] = combined_25
        #df_sim["combined_50"] = combined_50
        df_sim["combined_75"] = combined_75
        #df_sim["combined_geometric"] = combined_geometric
        #df_sim["combined_harmonic"] = combined_harmonic

        # Output file para guardar los resultados
        result_file = OUT_DIR / f"{file.stem}_similarity_results.csv"
        df_sim.to_csv(result_file, index=False)

if __name__ == "__main__":
    calculate_similarity()