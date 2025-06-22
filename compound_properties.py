from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen

class CompoundProperties:
    """
    Clase que representa las propiedades que nos interesan de un compuesto químico.
    """

    def __init__(self, name, smiles):
        self.name = name
        self.smiles = smiles
        self.hydrogen_bond_acceptors = None
        self.hydrogen_bond_donors = None
        self.rotatable_bonds = None
        self.log_p = None
        self.molecular_weight = None
        self.heavy_atom_count = None
        self.polar_surface_area = None
        self.mol = Chem.MolFromSmiles(smiles)
        self.calculate_properties()

    def calculate_properties(self):
        """
        Calcula las propiedades del compuesto utilizando RDKit.
        """
        if self.mol is None:
            raise ValueError(f"Error al procesar SMILES: {self.smiles}")

        self.hydrogen_bond_acceptors = rdMolDescriptors.CalcNumHBA(self.mol)
        self.hydrogen_bond_donors = rdMolDescriptors.CalcNumHBD(self.mol)
        self.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(self.mol)
        self.log_p = Crippen.MolLogP(self.mol)
        self.molecular_weight = rdMolDescriptors.CalcExactMolWt(self.mol)
        self.heavy_atom_count = rdMolDescriptors.CalcNumHeavyAtoms(self.mol)
        self.polar_surface_area = rdMolDescriptors.CalcTPSA(self.mol)
        
    def vectorize(self):
        """
        Devuelve un diccionario con las propiedades calculadas.
        """
        if not hasattr(self, "mol") or self.mol is None:
            # Aseguramos que el objeto mol existe y es válido
            self.calculate_properties()

        return {
            "hydrogen_bond_acceptors": rdMolDescriptors.CalcNumHBA(self.mol),
            "hydrogen_bond_donors": rdMolDescriptors.CalcNumHBD(self.mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(self.mol),
            "log_p": Crippen.MolLogP(self.mol),
            "molecular_weight": rdMolDescriptors.CalcExactMolWt(self.mol),
            "heavy_atom_count": rdMolDescriptors.CalcNumHeavyAtoms(self.mol),
            "polar_surface_area": rdMolDescriptors.CalcTPSA(self.mol)
        }