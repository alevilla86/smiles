"""
Molecular fingerprint utilities for SMILES processing.

Provides standardized Morgan fingerprint generation and batch processing
for machine learning pipelines.
"""
from typing import Optional, Tuple, List

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# Suppress RDKit warnings globally when this module is imported
RDLogger.DisableLog('rdApp.*')

# Default fingerprint parameters (Morgan fingerprint)
DEFAULT_RADIUS = 2
DEFAULT_NBITS = 2048


def get_fingerprint(
    smiles: str,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_NBITS,
    dtype: type = np.float32,
) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a Morgan fingerprint vector.

    Args:
        smiles: SMILES string representation of a molecule
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Number of bits in the fingerprint (default: 2048)
        dtype: NumPy dtype for the output array (default: np.float32)

    Returns:
        NumPy array of shape (n_bits,) or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprints(
    smiles_list: List[str],
    label: int,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_NBITS,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """
    Convert a list of SMILES to fingerprints with labels.

    Args:
        smiles_list: List of SMILES strings
        label: Label to assign (1 for active, 0 for inactive)
        radius: Morgan fingerprint radius
        n_bits: Number of bits in the fingerprint
        verbose: Whether to print progress

    Returns:
        List of (fingerprint, label) tuples
    """
    samples = []
    for smiles in smiles_list:
        fp = get_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if fp is not None:
            samples.append((fp, label))

    if verbose:
        label_name = "active" if label == 1 else "inactive"
        print(f"  Processed {len(samples)} {label_name} compounds")

    return samples


def prepare_fingerprints(
    active_smiles: List[str],
    inactive_smiles: List[str],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SMILES lists to fingerprint arrays ready for ML training.

    Args:
        active_smiles: List of active compound SMILES
        inactive_smiles: List of inactive compound SMILES
        verbose: Whether to print progress

    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the label array
    """
    if verbose:
        print("\n=== PREPARING FINGERPRINTS ===\n")

    positive_samples = smiles_to_fingerprints(active_smiles, label=1, verbose=verbose)
    negative_samples = smiles_to_fingerprints(inactive_smiles, label=0, verbose=verbose)

    all_data = positive_samples + negative_samples
    X = np.array([x[0] for x in all_data])
    y = np.array([x[1] for x in all_data])

    if verbose:
        print(f"  Total samples: {len(X)}")

    return X, y
