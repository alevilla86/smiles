# Notas importantes:
# Carga datos químicos de un archivo CSV (embeddings_dataset.csv) con compuestos de PubChem y añade seis benzimidazoles de referencia (Bz-1 a Bz-6) que ya tenemos con datos de actividad.
# Normaliza las estructuras químicas (SMILES) para que sean consistentes.
# Genera huellas moleculares ECFP4 (un tipo de "huella digital" química) para cada compuesto, que usamos para comparar similitudes.
# Calcula similitudes entre los compuestos usando tres métricas: Tanimoto, coseno, y un índice compuesto que mezcla ambas.
# Filtra benzimidazoles y selecciona los 10 compuestos más similares a Bz-1 a Bz-6.
# Añade información extra como IC50 (actividad), peso molecular, LogP (lipofilicidad), y una predicción de actividad para los compuestos desconocidos.
# Guarda los resultados en archivos CSV para que podamos analizarlos.
# El código está diseñado para ser robusto, pero a veces falla al buscar IC50 en ChEMBL (estoy trabajando en eso).
# Al final espero la idea es que, bueno se supone que a esta altura debe hacerlo, generar un archivo (ColumNewtop_n_benzimidazoles.csv) con los mejores candidatos, que son los que queremos probar en el laboratorio.

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Crippen import MolLogP
import base64
import csv
import os
from chembl_webresource_client.new_client import new_client
from sklearn.ensemble import RandomForestClassifier

# Qué hace: Define dónde está el archivo de entrada (embeddings_dataset.csv, con compuestos de PubChem) y dónde guardaremos los resultados.
# Por qué es importante: Los nombres de archivo nos aseguran que los datos se lean y guarden en el lugar correcto. 
# El archivo ColumNewtop_n_benzimidazoles.csv es el que contiene nuestros mejores candidatos.
# Archivos
input_file = "/cursoIA/ia_bioactivos/data/embeddings_dataset.csv"
output_file = "/cursoIA/ia_bioactivos/data/V2CalcularTanimotofingerprintsECFP4_dataset.csv"
embeddings_output_file = "/cursoIA/ia_bioactivos/data/V2CalcularTanimotoembEddingsECFP4_dataset.csv"
top_n_file = "/cursoIA/ia_bioactivos/data/V2Top_n_benzimidazoles.csv"
log_file = "/cursoIA/ia_bioactivos/data/invalid_smiles_log.txt"

# Qué hace: Lista las columnas de los CSVs de salida, incluyendo el nombre del compuesto, SMILES, CID, huella ECFP4, actividad, IC50, peso molecular, LogP, y los 881 bits de la huella (fp_0 a fp_880).
# Por qué es importante: Estas columnas nos dan toda la información que necesitamos para evaluar los compuestos.
# Esto debemos ir afinándolo.
headers = ["Compound_Name", "SMILES_or_CID", "CID", "FingerprintDisponible", "Fingerprint", "Activity", 
           "IC50", "Molecular_Weight", "LogP"] + [f'fp_{i}' for i in range(881)]

# Qué hace: Define los seis benzimidazoles de referencia que ya tenemos, con su nombre, SMILES, actividad (1 = activo, 0 = inactivo), e IC50 (en µM o "NC" si no es concluyente).
# Por qué es importante: Estos compuestos son nuestra base para comparar otros, ya que sabemos cuáles son activos contra la leishmaniasis (Bz-2, Bz-3, Bz-6).
bz_compounds = [
    {"name": "Bz-1", "smiles": "CC1=CC2=C(NC(=N2)C2=CC=C(O2)N(=O)=O)C=C1", "activity": 0, "ic50": "NC"},
    {"name": "Bz-2", "smiles": "CC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)=N2)C=C1", "activity": 1, "ic50": 1.3},
    {"name": "Bz-3", "smiles": "ClC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)N2)C=C1", "activity": 1, "ic50": 1.9},
    {"name": "Bz-4", "smiles": "CC1=CC=CN=C1C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2", "activity": 0, "ic50": "NC"},
    {"name": "Bz-5", "smiles": "ClC1=CC2=C(NC(NC(=O)C3=NC=CC4=C3C=CC=C4)=N2)C=C1", "activity": 0, "ic50": "NC"},
    {"name": "Bz-6", "smiles": "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2", "activity": 1, "ic50": 0.9}
]

# En este punto es donde se intenta buscar en ChEMBL el valor IC50 (actividad contra Leishmania) para un compuesto dado su SMILES.
# Usa una lista de especies de Leishmania para filtrar.
# Por qué es importante: Queremos saber si los nuevos compuestos tienen actividad conocida. Si ChEMBL no encuentra datos, retorna 'N/A'.
def get_ic50_from_chembl(smiles):
    try:
        activity = new_client.activity
        leishmania_species = ["Leishmania major", "Leishmania donovani", "Leishmania infantum", 
                              "Leishmania mexicana", "Leishmania braziliensis"]
        compounds = activity.filter(standard_type="IC50", target_organism__in=leishmania_species, 
                                   canonical_smiles=smiles)
        if compounds:
            return compounds[0].get('standard_value', 'N/A')
        return 'N/A'
    except Exception as e:
        print(f"Error fetching IC50 for SMILES {smiles}: {e}")
        return 'N/A'

# En este paso se calcula el peso molecular (mw) y la lipofilicidad (logp) de un compuesto usando RDKit.
# Ojo, estas propiedades nos ayudan a evaluar si los compuestos son viables como fármacos (por ejemplo, peso molecular < 500 Da, LogP < 5).
# No sé si le corresponde más a los doctores esta parte, pero bueno, aquí está el código.
def calculate_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mw = Descriptors.MolWt(mol)
        logp = MolLogP(mol)
        return mw, logp
    except:
        return None, None

# Aquí lee el CSV de entrada (embeddings_dataset.csv), extrae nombre, SMILES, y CID, y combina estos compuestos con Bz-1 a Bz-6.
# Elimina duplicados basándose en SMILES.
# Esto nos asegura que trabajamos con todos los compuestos relevantes sin repetir.
def load_compounds_from_csv(file_path):
    compounds = []
    invalid_smiles = []
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if {'Compound_Name', 'SMILES_or_CID'}.issubset(df.columns):
            for _, row in df.iterrows():
                if pd.isna(row['SMILES_or_CID']) or not isinstance(row['SMILES_or_CID'], str):
                    invalid_smiles.append((row.get('Compound_Name', 'Unknown'), row.get('SMILES_or_CID', 'N/A')))
                    print(f"Skipping row with invalid SMILES: {row.get('Compound_Name', 'Unknown')}")
                    continue
                try:
                    mol = Chem.MolFromSmiles(row['SMILES_or_CID'])
                    if mol is None:
                        invalid_smiles.append((row['Compound_Name'], row['SMILES_or_CID']))
                        print(f"Invalid SMILES: {row['SMILES_or_CID']} for compound {row['Compound_Name']}")
                        continue
                except:
                    invalid_smiles.append((row['Compound_Name'], row['SMILES_or_CID']))
                    print(f"Invalid SMILES: {row['SMILES_or_CID']} for compound {row['Compound_Name']}")
                    continue
                compounds.append({
                    "name": row['Compound_Name'],
                    "smiles": row['SMILES_or_CID'],
                    "cid": row.get('CID', 'N/A')
                })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    # Guardar SMILES inválidos en un log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Invalid SMILES Log\n")
        for name, smiles in invalid_smiles:
            f.write(f"Compound: {name}, SMILES: {smiles}\n")
    
    all_compounds = compounds + bz_compounds
    unique_compounds = list({c["smiles"]: c for c in all_compounds}.values())
    print(f"Processed {len(compounds)} valid compounds from CSV, {len(invalid_smiles)} invalid SMILES skipped")
    return unique_compounds

# Ojo, aquí hay que investigar un poco más, porque este código es para calcular huellas ECFP4.
# Convierte un SMILES en una huella ECFP4 (881 bits) usando RDKit, que es como una "huella digital" de la estructura química.
# Por qué es importante: Las huellas ECFP4 son esenciales para comparar la similitud entre compuestos.
def get_ecfp4_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, np.zeros(881)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
        return fp.ToBinary(), np.array(list(fp))
    except Exception as e:
        print(f"Error generating ECFP4 for SMILES {smiles}: {e}")
        return None, np.zeros(881)

# Aquí verificamos si un compuesto tiene el núcleo de benzimidazol usando un patrón SMARTS.
# Por qué es importante: Queremos enfocarnos en benzimidazoles, ya que son nuestro objetivo principal.
def has_benzimidazole(smiles):
    benzimidazole_smarts = Chem.MolFromSmarts('c1cc2c(c(c1))ncn2')
    mol = Chem.MolFromSmiles(smiles)
    return mol.HasSubstructMatch(benzimidazole_smarts) if mol else False

# Este fragmento lo que hace es comparar cada compuesto con Bz-1 a Bz-6, calcula Tanimoto (similitud estructural), coseno (similitud vectorial), y un índice compuesto (alpha = 0.25, 0.5, 0.75).
# También predice actividad para compuestos desconocidos.
# Por qué es importante: Estas métricas nos dicen qué compuestos son más similares a nuestros benzimidazoles activos, como Bz-2, Bz-3, y Bz-6.
def calculate_similarities(df, model):
    def to_rdkit_fp(row):
        fp = DataStructs.ExplicitBitVect(881)
        bits = np.where(row[[f"fp_{i}" for i in range(881)]] == 1.0)[0]
        for bit in bits:
            fp.SetBit(int(bit))
        return fp
    
    df['rdkit_fp'] = df.apply(to_rdkit_fp, axis=1)
    bz_fps = df[df['Compound_Name'].isin([c['name'] for c in bz_compounds])][['Compound_Name', 'rdkit_fp']]
    similarities = []
    for idx, row in df.iterrows():
        # Predecir actividad si no está disponible
        if row['Activity'] == 'N/A':
            fp_vector = row[[f'fp_{i}' for i in range(881)]].values.reshape(1, -1)
            predicted_activity = model.predict(fp_vector)[0] if model else 'N/A'
        else:
            predicted_activity = 'N/A'
        
        for _, bz_row in bz_fps.iterrows():
            bz_name = bz_row['Compound_Name']
            bz_fp = bz_row['rdkit_fp']
            tanimoto = DataStructs.TanimotoSimilarity(row['rdkit_fp'], bz_fp)
            fp_row = row[[f"fp_{i}" for i in range(881)]].values.astype(float)
            fp_bz = np.array(list(bz_fp)).astype(float)
            cosine = np.dot(fp_row, fp_bz) / (np.linalg.norm(fp_row) * np.linalg.norm(fp_bz)) if np.linalg.norm(fp_row) * np.linalg.norm(fp_bz) != 0 else 0
            similarities.append({
                'Compound': row['Compound_Name'],
                'Reference': bz_name,
                'SMILES': row['SMILES_or_CID'],
                'Tanimoto': tanimoto,
                'Cosine': cosine,
                'Score_alpha_0.25': 0.25 * tanimoto + 0.75 * cosine,
                'Score_alpha_0.5': 0.5 * tanimoto + 0.5 * cosine,
                'Score_alpha_0.75': 0.75 * tanimoto + 0.25 * cosine,
                'Is_Benzimidazole': has_benzimidazole(row['SMILES_or_CID']),
                'IC50': row['IC50'],
                'Molecular_Weight': row['Molecular_Weight'],
                'LogP': row['LogP'],
                'Predicted_Activity': predicted_activity
            })
    return pd.DataFrame(similarities)

# Aquí entrena un modelo Random Forest usando las huellas ECFP4 y actividad de Bz-1 a Bz-6 para predecir si otros compuestos son activos (1) o inactivos (0).
# Por qué es importante: Nos ayuda a estimar la actividad de compuestos sin datos de IC50.
def train_rf_model(df):
    df_train = df[df['Activity'] != 'N/A']
    if len(df_train) < 2:
        return None
    X = df_train[[f'fp_{i}' for i in range(881)]].values
    y = df_train['Activity'].astype(int)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Aquí está todo el flujo: carga datos, genera huellas, calcula propiedades, entrena el modelo, calcula similitudes, filtra benzimidazoles, y guarda los resultados.
# El main, el corazón del código.
def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Cargar compuestos
    compounds = load_compounds_from_csv(input_file)
    print(f"Loaded {len(compounds)} unique compounds")
    
    # Generar huellas ECFP4 y propiedades
    results = []
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for compound in compounds:
            name = compound["name"]
            smiles = compound["smiles"]
            cid = compound.get("cid", "N/A")
            activity = next((c["activity"] for c in bz_compounds if c["smiles"] == smiles), "N/A")
            ic50 = next((c["ic50"] for c in bz_compounds if c["smiles"] == smiles), get_ic50_from_chembl(smiles))
            mw, logp = calculate_properties(smiles)
            
            fingerprint_binary, fp_vector = get_ecfp4_fingerprint(smiles)
            if fingerprint_binary is not None:
                fingerprint = base64.b64encode(fingerprint_binary).decode('utf-8')
                fingerprint_available = "Yes"
            else:
                fingerprint = "N/A"
                fingerprint_available = "No"
                fp_vector = np.zeros(881)
            
            result = [name, smiles, cid, fingerprint_available, fingerprint, activity, ic50, mw, logp] + list(fp_vector)
            results.append(result)
            writer.writerow(result)
            print(f"Processed {name}")
    
    print(f"Fingerprints dataset saved to {output_file}")
    
    # Guardar embeddings
    results_df = pd.DataFrame(results, columns=headers)
    results_df.to_csv(embeddings_output_file, index=False, encoding='utf-8')
    print(f"Embeddings dataset saved to {embeddings_output_file}")
    
    # Entrenar modelo
    model = train_rf_model(results_df)
    if model is None:
        print("No se pudo entrenar el modelo, datos insuficientes")
    
    # Calcular similitudes y seleccionar top-N
    similarities_df = calculate_similarities(results_df, model)
    n = 10
    top_n = similarities_df.sort_values('Score_alpha_0.5', ascending=False).head(n)
    top_n_benzimidazoles = top_n[top_n['Is_Benzimidazole']]
    print("Top-N compounds (all):")
    print(top_n[['Compound', 'Reference', 'SMILES', 'Tanimoto', 'Cosine', 'Score_alpha_0.25', 'Score_alpha_0.5', 
                 'Score_alpha_0.75', 'Is_Benzimidazole', 'IC50', 'Molecular_Weight', 'LogP', 'Predicted_Activity']])
    print("\nTop-N benzimidazoles:")
    print(top_n_benzimidazoles[['Compound', 'Reference', 'SMILES', 'Tanimoto', 'Cosine', 'Score_alpha_0.25', 
                                'Score_alpha_0.5', 'Score_alpha_0.75', 'IC50', 'Molecular_Weight', 'LogP', 
                                'Predicted_Activity']])
    top_n_benzimidazoles.to_csv(top_n_file, index=False)
    print(f"Top-N benzimidazoles saved to {top_n_file}")

if __name__ == "__main__":
    main()