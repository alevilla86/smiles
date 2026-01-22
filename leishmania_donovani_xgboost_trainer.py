"""
XGBoost trainer for Leishmania activity prediction.

Usage:
    # Train using existing data from txt files (fast)
    python leishmania_donovani_xgboost_trainer.py --use-existing

    # Prepare fresh data from ChEMBL and train (slow, requires chembl_35.sdf)
    python leishmania_donovani_xgboost_trainer.py --from-scratch
"""
import argparse

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_loader import load_training_data
from fingerprint_utils import prepare_fingerprints
from model_utils import save_sklearn_model


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[XGBClassifier, float]:
    """Train XGBoost model and return it with accuracy."""
    print("\n=== TRAINING MODEL ===\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Calculate scale_pos_weight for imbalanced data
    n_negative = sum(y_train == 0)
    n_positive = sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    print(f"  Class balance: {n_positive} positive, {n_negative} negative")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost configuration optimized for Morgan fingerprints
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=42,
        eval_metric='logloss',
    )

    print("  Training XGBoost (600 estimators)...")
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
        description="Train XGBoost model for Leishmania activity prediction"
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
        help="This option requires running RF trainer first to generate data files"
    )

    args = parser.parse_args()

    if args.from_scratch:
        print("Note: XGBoost trainer uses the same data as RF trainer.")
        print("Run 'python leishmania_donovani_activity_trainer.py --from-scratch' first.")
        print("Then run this trainer with --use-existing.")
        return

    # Load data
    active_smiles, inactive_smiles = load_training_data()

    # Prepare fingerprints
    X, y = prepare_fingerprints(active_smiles, inactive_smiles)

    # Train model
    clf, accuracy = train_model(X, y)

    # Save model
    print("\n=== SAVING MODEL ===\n")
    save_sklearn_model(
        model=clf,
        model_name="leishmania_donovani_xgb",
        accuracy=accuracy,
    )


if __name__ == "__main__":
    main()
