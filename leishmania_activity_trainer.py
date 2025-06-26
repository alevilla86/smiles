from chembl_webresource_client.new_client import new_client
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from itertools import islice
import joblib
import os
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Obtener resultados de actividad para Leishmania
# Filtrar por tipo estándar "IC50" y organismos diana específicos
LEISHMANIA_SPECIES = ["Leishmania major", "Leishmania donovani", "Leishmania infantum", 
                      "Leishmania mexicana", "Leishmania braziliensis"]

MAX_VALUE_UM_IC50 = 20.0

activity = new_client.activity
active_results = activity.filter(standard_type="IC50", 
                                 target_organism__in=LEISHMANIA_SPECIES,
                                 units="uM",
                                 value__lt=MAX_VALUE_UM_IC50,).only(['molecule_chembl_id', 'molecule_structures', 'value', 'units', 'target_organism'])
print(f"Total de compuestos activos para alguna de las especies de Leishmania encontrados: {len(active_results)}")

print(f"Resultados de actividad para Leishmania: {len(active_results)} compuestos")

# Convertir los resultados a un DataFrame y extraer los IDs de compuestos activos
active_df = pd.DataFrame(active_results)
print("Compuestos activos convertidos a DataFrame")

active_compound_ids = set(active_df['molecule_chembl_id'].dropna())
print(f"Total de compuestos activos únicos encontrados: {len(active_compound_ids)}")

molecule = new_client.molecule

# Para crear un dataset de moléculas que no son activas contra Leishmania
# Obtener todas las moléculas bioactivas
all_bioactive = molecule.filter(has_bioactivity="1").only(['molecule_chembl_id', 'molecule_structures'])
all_bioactiv_top40k = list(islice(all_bioactive, 20000))
print(f"Total de compuestos bioactivos encontrados: {len(all_bioactiv_top40k)}")

# Filtrar las que no tienen actividad contra Leishmania
non_leishmania = [m for m in all_bioactiv_top40k if m['molecule_chembl_id'] not in active_compound_ids]
print(f"Total de compuestos no Leishmania encontrados: {len(non_leishmania)}")

# Limitar el número de moléculas no Leishmania al mismo número que las activas
# Esto es para balancear el dataset
non_leishmania = non_leishmania[:len(active_compound_ids)]
print(f"Total de compuestos no Leishmania seleccionados: {len(non_leishmania)}")

# Positivos
positive_samples = []
for mol_id in active_compound_ids:
    mol_data = molecule.get(mol_id)
    if 'molecule_structures' in mol_data and mol_data['molecule_structures']:
        smiles = mol_data['molecule_structures']['canonical_smiles']
        fp = get_fingerprint(smiles)
        if fp is not None:
            positive_samples.append((fp, 1))

print("Procesdos los smiles de los compuestos activos contra Leishmania")

# Negativos
negative_samples = []
for mol_data in non_leishmania:
    if 'molecule_structures' in mol_data and mol_data['molecule_structures']:
        smiles = mol_data['molecule_structures']['canonical_smiles']
        fp = get_fingerprint(smiles)
        if fp is not None:
            negative_samples.append((fp, 0))

print("Procesdos los smiles de los compuestos no activos contra Leishmania")

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
joblib.dump(clf, 'models/leishmania_model_v1.pkl')

# Cargar el modelo entrenado
# clf = joblib.load('models/leishmania_model_v1.pkl')

# Evaluar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))