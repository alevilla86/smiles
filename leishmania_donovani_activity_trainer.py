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

import numpy as np
import pandas as pd
import random
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from constants import (
    LEISHMANIA_SPECIES_DONOVANI,
    MAX_VALUE_UM_IC50,
    ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH,
    NON_ACTIVE_BENZIMIDAZOLE_COMPOUNDS_MANUAL_SEARCH,
)
from data_loader import load_training_data, save_training_data
from fingerprint_utils import prepare_fingerprints
from model_utils import save_sklearn_model

CHEMBL_SDF_PATH = Path("external_data") / "chembl_35.sdf"


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
    save_training_data(list(active_smiles), inactive_smiles)

    return list(active_smiles), inactive_smiles


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
        active_smiles, inactive_smiles = load_training_data()
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
