from rdkit import Chem

# Ruta al archivo SDF
sdf_path = "/Users/alejandro/Downloads/chembl_35.sdf"  # Cambia esto por la ruta real

# Leer el archivo SDF
supplier = Chem.SDMolSupplier(sdf_path)

for i, mol in enumerate(supplier):
    if mol is not None:
        print(f"\nMolecule {i + 1}")
        for prop in mol.GetPropNames():
            print(f"{prop}: {mol.GetProp(prop)}")
    if i >= 4:  # Only show first 5 molecules
        break
