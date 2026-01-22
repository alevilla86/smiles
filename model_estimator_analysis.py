"""
Analyze accuracy vs n_estimators for Random Forest and XGBoost models.

Usage:
    python model_estimator_analysis.py --model rf      # Analyze Random Forest
    python model_estimator_analysis.py --model xgb    # Analyze XGBoost
    python model_estimator_analysis.py --model both   # Compare both models
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from constants import MODELS_DIR
from data_loader import load_training_data
from fingerprint_utils import prepare_fingerprints

# Estimator range to test
ESTIMATOR_RANGE = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500]


def evaluate_estimators(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    estimator_range: list[int]
) -> list[dict]:
    """Train models with varying n_estimators and record cross-validation accuracy."""
    print(f"\n=== EVALUATING {model_type.upper()} WITH VARYING n_estimators ===\n")

    results = []

    # Calculate scale_pos_weight for XGBoost
    n_negative = sum(y == 0)
    n_positive = sum(y == 1)
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    for n in estimator_range:
        print(f"  Testing n_estimators={n}...", end=" ", flush=True)

        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        else:  # xgb
            model = XGBClassifier(
                n_estimators=n,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                random_state=42,
                eval_metric='logloss',
            )

        # 5-fold cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

        results.append({
            'n_estimators': n,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        })

        print(f"accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def find_optimal_estimators(results: list[dict], threshold: float = 0.001) -> dict:
    """Find the optimal n_estimators where accuracy gains diminish.

    Returns the point where adding more estimators gives less than threshold improvement.
    """
    best_idx = 0
    best_accuracy = results[0]['mean_accuracy']

    for i, r in enumerate(results):
        if r['mean_accuracy'] > best_accuracy:
            best_accuracy = r['mean_accuracy']
            best_idx = i

    # Find elbow point (where improvement becomes < threshold)
    recommended_idx = 0
    for i in range(1, len(results)):
        improvement = results[i]['mean_accuracy'] - results[i - 1]['mean_accuracy']
        if improvement < threshold:
            recommended_idx = i - 1
            break
        recommended_idx = i

    return {
        'best_n_estimators': results[best_idx]['n_estimators'],
        'best_accuracy': results[best_idx]['mean_accuracy'],
        'recommended_n_estimators': results[recommended_idx]['n_estimators'],
        'recommended_accuracy': results[recommended_idx]['mean_accuracy'],
    }


def plot_results(results: list[dict], model_type: str, optimal: dict) -> str:
    """Generate accuracy vs n_estimators plot and save to file."""
    n_estimators = [r['n_estimators'] for r in results]
    mean_acc = [r['mean_accuracy'] for r in results]
    std_acc = [r['std_accuracy'] for r in results]

    model_name = "Random Forest" if model_type == "rf" else "XGBoost"

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_estimators, mean_acc, yerr=std_acc, marker='o', capsize=5,
                 linewidth=2, markersize=8, label='CV Accuracy')

    # Mark the recommended point
    plt.axvline(x=optimal['recommended_n_estimators'], color='green', linestyle='--',
                linewidth=2, label=f"Recommended: {optimal['recommended_n_estimators']}")

    # Mark the best point
    plt.axvline(x=optimal['best_n_estimators'], color='red', linestyle=':',
                linewidth=2, label=f"Best: {optimal['best_n_estimators']}")

    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title(f'{model_name} - Accuracy vs n_estimators', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    MODELS_DIR.mkdir(exist_ok=True)
    output_path = MODELS_DIR / f"{model_type}_estimator_analysis.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def plot_comparison(rf_results: list[dict], xgb_results: list[dict],
                    rf_optimal: dict, xgb_optimal: dict) -> str:
    """Generate side-by-side comparison plot for both models."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Random Forest plot
    n_est_rf = [r['n_estimators'] for r in rf_results]
    mean_rf = [r['mean_accuracy'] for r in rf_results]
    std_rf = [r['std_accuracy'] for r in rf_results]

    ax1.errorbar(n_est_rf, mean_rf, yerr=std_rf, marker='o', capsize=5,
                 linewidth=2, markersize=8, color='blue')
    ax1.axvline(x=rf_optimal['recommended_n_estimators'], color='green', linestyle='--',
                linewidth=2, label=f"Recommended: {rf_optimal['recommended_n_estimators']}")
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax1.set_title('Random Forest', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # XGBoost plot
    n_est_xgb = [r['n_estimators'] for r in xgb_results]
    mean_xgb = [r['mean_accuracy'] for r in xgb_results]
    std_xgb = [r['std_accuracy'] for r in xgb_results]

    ax2.errorbar(n_est_xgb, mean_xgb, yerr=std_xgb, marker='o', capsize=5,
                 linewidth=2, markersize=8, color='orange')
    ax2.axvline(x=xgb_optimal['recommended_n_estimators'], color='green', linestyle='--',
                linewidth=2, label=f"Recommended: {xgb_optimal['recommended_n_estimators']}")
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax2.set_title('XGBoost', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Model Comparison: Accuracy vs n_estimators', fontsize=16)
    plt.tight_layout()

    # Save plot
    MODELS_DIR.mkdir(exist_ok=True)
    output_path = MODELS_DIR / "comparison_estimator_analysis.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def print_summary(model_type: str, optimal: dict):
    """Print analysis summary."""
    model_name = "Random Forest" if model_type == "rf" else "XGBoost"
    print(f"\n{model_name} Analysis:")
    print(f"  Best accuracy: {optimal['best_accuracy']:.2%} at n_estimators={optimal['best_n_estimators']}")
    print(f"  Recommended: {optimal['recommended_n_estimators']} estimators "
          f"({optimal['recommended_accuracy']:.2%} accuracy, diminishing returns beyond)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy vs n_estimators for Random Forest and XGBoost"
    )
    parser.add_argument(
        "--model",
        choices=["rf", "xgb", "both"],
        required=True,
        help="Model type to analyze: rf (Random Forest), xgb (XGBoost), or both"
    )

    args = parser.parse_args()

    # Load data
    active_smiles, inactive_smiles = load_training_data()

    # Prepare fingerprints
    X, y = prepare_fingerprints(active_smiles, inactive_smiles)

    if args.model == "rf":
        results = evaluate_estimators(X, y, "rf", ESTIMATOR_RANGE)
        optimal = find_optimal_estimators(results)
        plot_path = plot_results(results, "rf", optimal)
        print_summary("rf", optimal)
        print(f"\nPlot saved: {plot_path}")

    elif args.model == "xgb":
        results = evaluate_estimators(X, y, "xgb", ESTIMATOR_RANGE)
        optimal = find_optimal_estimators(results)
        plot_path = plot_results(results, "xgb", optimal)
        print_summary("xgb", optimal)
        print(f"\nPlot saved: {plot_path}")

    else:  # both
        rf_results = evaluate_estimators(X, y, "rf", ESTIMATOR_RANGE)
        rf_optimal = find_optimal_estimators(rf_results)
        rf_plot = plot_results(rf_results, "rf", rf_optimal)

        xgb_results = evaluate_estimators(X, y, "xgb", ESTIMATOR_RANGE)
        xgb_optimal = find_optimal_estimators(xgb_results)
        xgb_plot = plot_results(xgb_results, "xgb", xgb_optimal)

        comparison_plot = plot_comparison(rf_results, xgb_results, rf_optimal, xgb_optimal)

        print_summary("rf", rf_optimal)
        print_summary("xgb", xgb_optimal)
        print(f"\nPlots saved:")
        print(f"  {rf_plot}")
        print(f"  {xgb_plot}")
        print(f"  {comparison_plot}")


if __name__ == "__main__":
    main()
