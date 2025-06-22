import json
from compound_properties import CompoundProperties

def count_lipinski_violations(compound_properties) -> int:
    """
    Cuenta el número de violaciones a las reglas de Lipinski en un compuesto dado.
    Args:
        compound_properties (CompoundProperties): Un objeto que contiene las propiedades del compuesto.
    Returns:
        int: El número de violaciones a las reglas de Lipinski.
    """
    violations = 0
    if compound_properties.hydrogen_bond_acceptors > 10:
        violations += 1
    if compound_properties.hydrogen_bond_donors > 5:
        violations += 1
    if compound_properties.molecular_weight > 500.0:
        violations += 1
    if compound_properties.log_p >5.0:
        violations += 1
    return violations

def parse_compound_json(json_string):
    """
    Parsea un JSON de PubChem para extraer las propiedades de un compuesto.
    Args:
        json_string (str): Cadena JSON que contiene la información del compuesto.
    Returns:
        dict: Un diccionario con las propiedades del compuesto, o None si hay un error.
    """
    try:
        data = json.loads(json_string)

        # Assuming 'PC_Compounds' is a list and we are interested in the first compound
        if not data or "PC_Compounds" not in data or not data["PC_Compounds"]:
            print("Error: 'PC_Compounds' key not found or is empty.")
            return None

        compound = data["PC_Compounds"][0]

        # Inicializamos un diccionario para almacenar las propiedades del compuesto.
        properties = {
            "cid": None,
            "iupac_name": {},
            "smiles": {},
            "molecular_formula": "",
            "hydrogen_bond_acceptors": 0,
            "hydrogen_bond_donors": 0,
            "rotatable_bonds": 0,
            "log_p": 0.0,
            "molecular_weight": 0.0,
            "heavy_atom_count": 0,
            "polar_surface_area": 0.0,
            "atom_surface_coefficient": 0.0,
            "no_lipinski_violations": False
        }

        cid = compound.get("id", {}).get("id", {}).get("cid")
        name = None
        smiles = None
        molecular_formula = None

        # Estas propiedades vienen de Pubchem. Solo tomamos el nombre IUPAC y SMILES.
        if "props" in compound and isinstance(compound["props"], list):
            for prop in compound["props"]:
                urn_label = prop.get("urn", {}).get("label")
                urn_name = prop.get("urn", {}).get("name")
                value = prop.get("value", {})
                if not value:
                    continue

                # Extraer el valor, independientemente de su tipo
                extracted_value = (
                    value.get("sval") or
                    value.get("fval") or
                    value.get("ival")
                )

                if urn_label == "IUPAC Name":
                    if urn_name == "Allowed":
                        name = extracted_value
                elif urn_label == "Molecular Formula":
                    molecular_formula = extracted_value
                elif urn_label == "SMILES":
                    if urn_name == "Canonical":
                        smiles = extracted_value

        #Calculamos las propiedades del compuesto usando CompoundProperties (rdkit) para esetandardizar las comparaciones.
        compound_properties = CompoundProperties(name, smiles)

        properties["cid"] = cid
        properties["iupac_name"] = compound_properties.name
        properties["smiles"] = compound_properties.smiles
        properties["molecular_formula"] = molecular_formula
        properties["hydrogen_bond_acceptors"] = compound_properties.hydrogen_bond_acceptors
        properties["hydrogen_bond_donors"] = compound_properties.hydrogen_bond_donors
        properties["rotatable_bonds"] = compound_properties.rotatable_bonds
        properties["log_p"] = compound_properties.log_p
        properties["molecular_weight"] = compound_properties.molecular_weight
        properties["heavy_atom_count"] = compound_properties.heavy_atom_count
        properties["polar_surface_area"] = compound_properties.polar_surface_area
        properties["atom_surface_coefficient"] = properties.get("heavy_atom_count", 0) / properties.get("polar_surface_area", 1) if properties.get("polar_surface_area") else 0
        properties["no_lipinski_violations"] = count_lipinski_violations(compound_properties)

        print(f"CID extraído: {cid}")

        return properties

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
