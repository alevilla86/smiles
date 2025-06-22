import pandas as pd
import os
from pathlib import Path

CSV_FILE_FOLDER = Path("data")
CSV_FILE_NAME = "{smiles}_similarity_{similarity}_data.csv"

# Define los nombres de las columnas que se van a utilizar en el DataFrame
# Asegúrate de que estos nombres coincidan con los que se usan en el DataFrame
# y en el archivo CSV que se va a crear.
columns = [
    "cid",
    "iupac_name",
    "smiles_canonical",
    "molecular_formula",
    "hydrogen_bond_acceptors",
    "hydrogen_bond_donors",
    "rotatable_bonds",
    "log_p",
    "molecular_weight",
    "heavy_atom_count",
    "polar_surface_area",
    "atom_surface_coefficient",
    "no_lipinski_violations"
]

def flatten_parsed_info(compound_properties):
    return {
        "cid": compound_properties.get("cid", ""),
        "iupac_name": compound_properties.get("iupac_name", {}),
        "smiles_canonical": compound_properties.get("smiles", {}),
        "molecular_formula": compound_properties.get("molecular_formula", ""),
        "hydrogen_bond_acceptors": compound_properties.get("hydrogen_bond_acceptors", ""),
        "hydrogen_bond_donors": compound_properties.get("hydrogen_bond_donors", ""),
        "rotatable_bonds": compound_properties.get("rotatable_bonds", ""),
        "log_p": compound_properties.get("log_p", ""),
        "molecular_weight": compound_properties.get("molecular_weight", ""),
        "heavy_atom_count": compound_properties.get("heavy_atom_count", ""),
        "polar_surface_area": compound_properties.get("polar_surface_area", ""),
        "atom_surface_coefficient": compound_properties.get("atom_surface_coefficient", ""),
        "no_lipinski_violations": compound_properties.get("no_lipinski_violations", "")
    }

def write_to_csv(smiles, compound_properties, similarity):
    CSV_FILE_FOLDER.mkdir(exist_ok=True)
    file_name = CSV_FILE_NAME.format(smiles=smiles, similarity=similarity)
    file = CSV_FILE_FOLDER / file_name
    
    if compound_properties:
        row = flatten_parsed_info(compound_properties)
        df = pd.DataFrame([row], columns=columns)

        if os.path.exists(file):
            df.to_csv(file, mode='a', header=False, index=False)
        else:
            df.to_csv(file, mode='w', header=True, index=False)
    else:
        print("Failed to parse the JSON data.")