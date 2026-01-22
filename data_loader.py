"""
Training data loading utilities for Leishmania activity prediction.

Handles loading and saving of SMILES datasets from text files.
"""
from pathlib import Path
from typing import Tuple, List

from constants import TRAINING_DATA_DIR

# Standard file paths for training data
ACTIVE_SMILES_FILE = TRAINING_DATA_DIR / "l_donovani_ACTIVE.txt"
INACTIVE_SMILES_FILE = TRAINING_DATA_DIR / "l_donovani_NOT_ACTIVE.txt"


def load_smiles_from_file(filepath: Path) -> List[str]:
    """
    Load SMILES strings from a text file (one per line).

    Args:
        filepath: Path to the text file

    Returns:
        List of SMILES strings (empty lines are filtered out)
    """
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def load_training_data(
    active_file: Path = ACTIVE_SMILES_FILE,
    inactive_file: Path = INACTIVE_SMILES_FILE,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load active and inactive SMILES from training data files.

    Args:
        active_file: Path to active compounds file
        inactive_file: Path to inactive compounds file
        verbose: Whether to print loading progress

    Returns:
        Tuple of (active_smiles, inactive_smiles) lists

    Raises:
        FileNotFoundError: If either data file doesn't exist
    """
    if verbose:
        print("\n=== LOADING EXISTING DATA ===\n")

    if not active_file.exists() or not inactive_file.exists():
        raise FileNotFoundError(
            f"Data files not found: {active_file}, {inactive_file}\n"
            "Run 'python leishmania_donovani_activity_trainer.py --from-scratch' first."
        )

    active_smiles = load_smiles_from_file(active_file)
    inactive_smiles = load_smiles_from_file(inactive_file)

    if verbose:
        print(f"  Loaded {len(active_smiles)} active compounds")
        print(f"  Loaded {len(inactive_smiles)} inactive compounds")

    return active_smiles, inactive_smiles


def save_smiles_to_file(smiles_list: List[str], filepath: Path) -> None:
    """
    Save SMILES strings to a text file (one per line).

    Args:
        smiles_list: List of SMILES strings
        filepath: Path to output file
    """
    with open(filepath, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")


def save_training_data(
    active_smiles: List[str],
    inactive_smiles: List[str],
    active_file: Path = ACTIVE_SMILES_FILE,
    inactive_file: Path = INACTIVE_SMILES_FILE,
    verbose: bool = True,
) -> None:
    """
    Save active and inactive SMILES to training data files.

    Args:
        active_smiles: List of active compound SMILES
        inactive_smiles: List of inactive compound SMILES
        active_file: Path to active compounds file
        inactive_file: Path to inactive compounds file
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\nSaving data to {active_file} and {inactive_file}...")

    save_smiles_to_file(active_smiles, active_file)
    save_smiles_to_file(inactive_smiles, inactive_file)
