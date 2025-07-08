from chembl_webresource_client.new_client import new_client
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from itertools import islice
from pprint import pprint
import joblib
import os
from rdkit import RDLogger
from constants import LEISHMANIA_MODEL_PATH, LEISHMANIA_SPECIES, MAX_VALUE_UM_IC50

RDLogger.DisableLog('rdApp.*')

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

activity = new_client.activity
active_results = activity.filter(standard_type="IC50", 
                                 target_organism__in=LEISHMANIA_SPECIES,
                                 units="uM",
                                 value__lt=MAX_VALUE_UM_IC50,).only(['molecule_chembl_id', 'value', 'units', 'target_organism', 'canonical_smiles', 'assay_description', 'record_id', 'document_year', 'document_journal'])
print(f"Total de compuestos activos para alguna de las especies de Leishmania encontrados: {len(active_results)}")

print(f"Resultados de actividad para Leishmania: {len(active_results)} compuestos")

# Convertir los resultados a un DataFrame y extraer los IDs de compuestos activos
active_df = pd.DataFrame(active_results)
print("Compuestos activos convertidos a DataFrame")

active_df.to_csv("active_leishmania_compounds_CHEMBL_for_training.csv", index=False)

active_compound_smiles = set(active_df['canonical_smiles'].dropna())
total_unique_valid_compound_smiles = len(active_compound_smiles)
print(f"Total de compuestos activos únicos encontrados: {total_unique_valid_compound_smiles}")

molecule = new_client.molecule

# Para crear un dataset de moléculas que no son activas contra Leishmania
# Obtener todas las moléculas bioactivas
print("Obteniendo todas las moléculas bioactivas de ChEMBL...")
all_bioactive = molecule.filter(has_bioactivity="1").only(['molecule_chembl_id', 'molecule_structures'])
print(f"Obtenidas todas las moléculas bioactivas {len(all_bioactive)}")

all_bioactiv_top = list(islice(all_bioactive, total_unique_valid_compound_smiles * 2))
print(f"Reducimos la lista de compuestos bioactivos encontrados: {len(all_bioactiv_top)}")

# Filtrar las que no tienen actividad contra Leishmania
non_leishmania = [
    m for m in all_bioactiv_top
    if m.get('molecule_structures') and
       m['molecule_structures'].get('canonical_smiles') and
       m['molecule_structures']['canonical_smiles'] not in active_compound_smiles
]

non_leishmania_parsed = [
    {
        'molecule_chembl_id': m['molecule_chembl_id'],
        'canonical_smiles': m['molecule_structures']['canonical_smiles']
    }
    for m in non_leishmania
]

#non_leishmania = [m for m in all_bioactiv_top if m['molecule_structures']['canonical_smiles'] not in active_compound_smiles['canonical_smiles']]
print(f"Total de compuestos bioactivos NO Leishmania encontrados: {len(non_leishmania_parsed)}")

# Limitar el número de moléculas no Leishmania al mismo número que las activas
# Esto es para balancear el dataset
non_leishmania_smiles = non_leishmania_parsed[:len(active_compound_smiles)]
print(f"Total de compuestos no Leishmania seleccionados: {len(non_leishmania_smiles)}")

non_active_df = pd.DataFrame(non_leishmania_smiles)
print("Compuestos no Leishmania convertidos a DataFrame")
non_active_df.to_csv("active_NOT_leishmania_compounds_CHEMBL_for_training.csv", index=False)

# Positivos
positive_samples = []
for smiles in active_compound_smiles:
    fp = get_fingerprint(smiles)
    if fp is not None:
        positive_samples.append((fp, 1))
print(f"Procesados los smiles de los compuestos activos contra Leishmania {len(positive_samples)}")

# Negativos
negative_samples = []
for negative_smiles in non_leishmania_smiles:
    compound_smiles = negative_smiles['canonical_smiles']
    fp = get_fingerprint(compound_smiles)
    if fp is not None:
        negative_samples.append((fp, 0))
print(f"Procesados los smiles de los compuestos no activos contra Leishmania {len(negative_samples)}")

# Combinar los datos positivos y negativos
all_data = positive_samples + negative_samples
X = np.array([x[0] for x in all_data]) # todos los FPs
y = np.array([x[1] for x in all_data]) # etiquetas (1 para activo, 0 para no activo)

print("Datos combinados y preparados para el entrenamiento")

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=500, random_state=42)

print("Iniciando el entrenamiento del modelo...")
clf.fit(X_train, y_train)
print("Modelo entrenado exitosamente")

# Guardar el modelo entrenado
os.makedirs("models", exist_ok=True)
joblib.dump(clf, LEISHMANIA_MODEL_PATH)

# Evaluar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))