"""
Random Forest trainer for Leishmania activity prediction.

Usage:
    # Train using existing data from txt files (fast)
    python leishmania_donovani_activity_trainer.py --use-existing

    # Prepare fresh data from ChEMBL and train (slow, requires chembl_35.sdf)
    python leishmania_donovani_activity_trainer.py --from-scratch
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import random
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from constants import (
    LEISHMANIA_SPECIES_DONOVANI,
    MAX_VALUE_UM_IC50,
    ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH,
    NON_ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH,
    TRAINING_DATA_DIR,
)
from model_utils import save_sklearn_model

RDLogger.DisableLog('rdApp.*')

# File paths for cached data
ACTIVE_SMILES_FILE = TRAINING_DATA_DIR / "l_donovani_ACTIVE.txt"
INACTIVE_SMILES_FILE = TRAINING_DATA_DIR / "l_donovani_NOT_ACTIVE.txt"
CHEMBL_SDF_PATH = Path("external_data") / "chembl_35.sdf"


def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fetch_active_compounds_from_chembl() -> set[str]:
    """Fetch active compounds from ChEMBL API."""
    from chembl_webresource_client.new_client import new_client

    print("Fetching active compounds from ChEMBL API...")
    activity = new_client.activity

    active_results = activity.filter(
        standard_type="IC50",
        target_organism__in=LEISHMANIA_SPECIES_DONOVANI,
        units="uM",
        value__lt=MAX_VALUE_UM_IC50,
    ).only([
        'molecule_chembl_id', 'value', 'units', 'target_organism',
        'canonical_smiles', 'assay_description', 'record_id',
        'document_year', 'document_journal'
    ])

    active_df = pd.DataFrame(active_results)
    active_smiles = set(active_df['canonical_smiles'].dropna())

    print(f"  Found {len(active_smiles)} unique active compounds from ChEMBL")
    return active_smiles


def sample_inactive_compounds_from_sdf(count: int) -> list[str]:
    """Sample random inactive compounds from ChEMBL SDF file."""
    print(f"Loading ChEMBL SDF from {CHEMBL_SDF_PATH}...")

    if not CHEMBL_SDF_PATH.exists():
        raise FileNotFoundError(
            f"ChEMBL SDF file not found at {CHEMBL_SDF_PATH}\n"
            "Download from: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
        )

    supplier = Chem.SDMolSupplier(str(CHEMBL_SDF_PATH))
    all_mols = [mol for mol in supplier if mol is not None]
    print(f"  Loaded {len(all_mols)} valid molecules")

    sampled_mols = random.sample(all_mols, count)
    sampled_smiles = [Chem.MolToSmiles(mol) for mol in sampled_mols]

    print(f"  Sampled {len(sampled_smiles)} random inactive compounds")
    return sampled_smiles


def prepare_data_from_scratch() -> tuple[list[str], list[str]]:
    """Prepare training data from ChEMBL (slow)."""
    print("\n=== PREPARING DATA FROM SCRATCH ===\n")

    # Get active compounds
    active_smiles = fetch_active_compounds_from_chembl()

    # Add manual benzimidazole compounds
    print(f"Adding {len(ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH)} manually curated active compounds")
    active_smiles.update(ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH)
    print(f"  Total active compounds: {len(active_smiles)}")

    # Sample inactive compounds (10% more than active for balance margin)
    sample_size = int(len(active_smiles) * 1.1)
    inactive_smiles = sample_inactive_compounds_from_sdf(sample_size)

    # Add manual inactive compounds
    print(f"Adding {len(NON_ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH)} manually curated inactive compounds")
    inactive_smiles.extend(NON_ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH)
    print(f"  Total inactive compounds: {len(inactive_smiles)}")

    # Save to files for future use
    print(f"\nSaving data to {ACTIVE_SMILES_FILE} and {INACTIVE_SMILES_FILE}...")
    with open(ACTIVE_SMILES_FILE, 'w') as f:
        for smiles in active_smiles:
            f.write(f"{smiles}\n")

    with open(INACTIVE_SMILES_FILE, 'w') as f:
        for smiles in inactive_smiles:
            f.write(f"{smiles}\n")

    return list(active_smiles), inactive_smiles


def load_existing_data() -> tuple[list[str], list[str]]:
    """Load training data from existing txt files (fast)."""
    print("\n=== LOADING EXISTING DATA ===\n")

    if not ACTIVE_SMILES_FILE.exists() or not INACTIVE_SMILES_FILE.exists():
        raise FileNotFoundError(
            f"Data files not found: {ACTIVE_SMILES_FILE}, {INACTIVE_SMILES_FILE}\n"
            "Run with --from-scratch first to generate the data files."
        )

    with open(ACTIVE_SMILES_FILE) as f:
        active_smiles = [line.strip() for line in f if line.strip()]

    with open(INACTIVE_SMILES_FILE) as f:
        inactive_smiles = [line.strip() for line in f if line.strip()]

    print(f"  Loaded {len(active_smiles)} active compounds")
    print(f"  Loaded {len(inactive_smiles)} inactive compounds")

    return active_smiles, inactive_smiles


def prepare_fingerprints(active_smiles: list[str], inactive_smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMILES to fingerprints and prepare training data."""
    print("\n=== PREPARING FINGERPRINTS ===\n")

    # Process active compounds
    positive_samples = []
    for smiles in active_smiles:
        fp = get_fingerprint(smiles)
        if fp is not None:
            positive_samples.append((fp, 1))
    print(f"  Processed {len(positive_samples)} active compounds")

    # Process inactive compounds
    negative_samples = []
    for smiles in inactive_smiles:
        fp = get_fingerprint(smiles)
        if fp is not None:
            negative_samples.append((fp, 0))
    print(f"  Processed {len(negative_samples)} inactive compounds")

    # Combine data
    all_data = positive_samples + negative_samples
    X = np.array([x[0] for x in all_data])
    y = np.array([x[1] for x in all_data])

    print(f"  Total samples: {len(X)}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, float]:
    """Train Random Forest model and return it with accuracy."""
    print("\n=== TRAINING MODEL ===\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)}, Test set: {len(X_test)}")

    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    print("  Training Random Forest (500 trees)...")
    clf.fit(X_train, y_train)
    print("  Training complete!")

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for Leishmania activity prediction"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing data from txt files (fast, no ChEMBL/SDF needed)"
    )
    group.add_argument(
        "--from-scratch",
        action="store_true",
        help="Prepare fresh data from ChEMBL API and SDF file (slow)"
    )

    args = parser.parse_args()

    # Load or prepare data
    if args.use_existing:
        active_smiles, inactive_smiles = load_existing_data()
    else:
        active_smiles, inactive_smiles = prepare_data_from_scratch()

    # Prepare fingerprints
    X, y = prepare_fingerprints(active_smiles, inactive_smiles)

    # Train model
    clf, accuracy = train_model(X, y)

    # Save model
    print("\n=== SAVING MODEL ===\n")
    save_sklearn_model(
        model=clf,
        model_name="leishmania_donovani_rf",
        accuracy=accuracy,
    )


if __name__ == "__main__":
    main()
