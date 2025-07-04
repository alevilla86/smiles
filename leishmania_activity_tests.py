import joblib
from constants import LEISHMANIA_MODEL_PATH
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
import numpy as np

RDLogger.DisableLog('rdApp.*')

USAL_COMPOUNDS = [
    "CC1=CC2=C(NC(=N2)C2=CC=C(O2)N(=O)=O)C=C1",
    "ClC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)N2)C=C1",
    "CC1=CC=CN=C1C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2",
    "ClC1=CC2=C(NC(NC(=O)C3=NC=CC4=C3C=CC=C4)=N2)C=C1",
    "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2",
    "CC1=CC2=C(NC(NC(=O)C3=NC=CC=C3)=N2)C=C1",
    "COC(=O)C1=CC=C(NC(=O)CSC2=NC3=C(C=C(Cl)C=C3)N2CC2=CC(C)=CC(C)=C2)C(Cl)=C1",
    "COC(=O)C1=CC=C(NC(=O)CSC2=NC3=C(C=C(Cl)C=C3)N2S(=O)(=O)C2=CC(C)=CC(C)=C2)C=C1",
    "O=C(NC1=CC(=CC=C1)C1=NC2=C(N1)C=CC=C2)C1=CC=C(C=C1)C1=CC=CC=C1",
    "CN1C(NCC2=CC(Br)=CC=C2O)=NC2=C1C=CC=C2",
    "NC(=N)NC1=CC2=C(NC(=N2)C2=CC=C(C=C2)C2=CC=CC=C2)C=C1",
    "O=C(N\\N=C\\C1=CC=C(O1)N(=O)=O)C1=NC2=C(N1)C=CC=C2",
    "CNCC(C(C(C(CO)O)O)O)O.O[Sb](=O)=O",
    "COC1=CC2=C(NC(NCC3=CC=C(C)C=C3)=N2)C=C1",
    "O=C(N1CCCCCCC=C)N(CCCCCCC=C)C2=C1C=CC(C)=C2",
    "O=C1NC2=CC(C)=CC=C2N1CCCCCCCO",
    "CC(=O)Nc1ccc(O)cc1",
    "O=C1N(CCCCCCCOC(C2=NC=CC=C2)=O)C3=CC(C)=CC=C3N1CCCCCCCOC(C4=CC=CC=N4)=O",
    "O=C(NCC1=CC=C([N+]([O-])=O)C=C1)C(N2)=NC3=C2C=CC=C3",
    "OC(C1=NC2=CC=CC=C2N1)=O"]


# Cargar modelo entrenado
print("Cargando modelo de Leishmania...")
clf = joblib.load(LEISHMANIA_MODEL_PATH)
print("Modelo cargado exitosamente.")

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def leishmania_tests():
    for smiles in USAL_COMPOUNDS:
        fp = get_fingerprint(smiles)
        if fp is not None:
            prob = clf.predict_proba([fp])[0][1]  # Probabilidad de ser activo
            print(f"SMILES: {smiles}, Probabilidad: {prob:.4f}")
        else:
            print(f"SMILES inválido: {smiles}")

if __name__ == "__main__":

        # Ejecutar el script principal
    print("Ejecutando pruebas de actividad para Leishmania...")

    leishmania_tests()

    print("Pruebas de predicciones completadas.")