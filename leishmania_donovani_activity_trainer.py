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
import random
from rdkit import RDLogger
from constants import LEISHMANIA_MODEL_PATH, LEISHMANIA_DONOVANI_MODEL_PATH, LEISHMANIA_SPECIES_DONOVANI, LEISHMANIA_SPECIES_NOT_DONOVANI, MAX_VALUE_UM_IC50

RDLogger.DisableLog('rdApp.*')

print("Inicializando el cliente de ChEMBL...")
activity = new_client.activity
molecule = new_client.molecule
print("Cliente de ChEMBL inicializado correctamente")

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_active_compounds_unique_smiles(leshmania_species=LEISHMANIA_SPECIES_DONOVANI, max_value_um_ic50=MAX_VALUE_UM_IC50):
    """
    Obtiene los compuestos activos contra alguna especie desde ChEMBL.
    """
    active_results = activity.filter(standard_type="IC50", 
                                     target_organism__in=leshmania_species,
                                     units="uM",
                                     value__lt=max_value_um_ic50,).only(['molecule_chembl_id', 'value', 'units', 'target_organism', 'canonical_smiles', 'assay_description', 'record_id', 'document_year', 'document_journal'])
    # Convertir los resultados a un DataFrame y extraer los IDs de compuestos activos
    active_df = pd.DataFrame(active_results)

    active_compound_smiles = set(active_df['canonical_smiles'].dropna())

    return active_compound_smiles

leishmania_donovani_active_compound_smiles = get_active_compounds_unique_smiles(LEISHMANIA_SPECIES_DONOVANI, MAX_VALUE_UM_IC50)
total_unique_leishmania_donovani_compound_smiles = len(leishmania_donovani_active_compound_smiles)
print(f"Total de compuestos activos únicos encontrados contra L. donovani: {total_unique_leishmania_donovani_compound_smiles}")

# Paso 1: Cargar el SDF de ChEMBL
# Para crear un dataset de moléculas que no son activas contra Leishmania
# Obtener una muestra aleatoria de moléculas de ChemBL
sdf_file_path = "/Users/alejandro/Downloads/chembl_35.sdf"

supplier = Chem.SDMolSupplier(sdf_file_path)
all_mols = [mol for mol in supplier if mol is not None]

print(f"Total de moléculas válidas en el SDF: {len(all_mols)}")

# Paso 2: seleccionar al azar
sample_size = int(total_unique_leishmania_donovani_compound_smiles + (total_unique_leishmania_donovani_compound_smiles * 0.1))
sampled_mols = random.sample(all_mols, sample_size)

# Paso 3: obtener los SMILES
sampled_smiles = [Chem.MolToSmiles(mol) for mol in sampled_mols]

print(f"Total de SMILES aleatorios obtenidos de las moléculas muestreadas: {len(sampled_smiles)}")

# Positivos
donovani_positive_samples = []
for smiles in leishmania_donovani_active_compound_smiles:
    fp = get_fingerprint(smiles)
    if fp is not None:
        donovani_positive_samples.append((fp, 1))
print(f"Procesados los smiles de los compuestos activos contra Leishmania {len(donovani_positive_samples)}")

# Negativos
negative_samples = []
for negative_smiles in sampled_smiles:
    fp = get_fingerprint(negative_smiles)
    if fp is not None:
        negative_samples.append((fp, 0))
print(f"Procesados los smiles de los compuestos no activos contra Leishmania {len(negative_samples)}")

# Combinar los datos positivos y negativos
all_data = donovani_positive_samples + negative_samples
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
joblib.dump(clf, LEISHMANIA_DONOVANI_MODEL_PATH)

# Evaluar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))