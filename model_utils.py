"""
Model saving utilities with metadata (timestamp, accuracy).

Provides standardized model saving with human-readable filenames containing
the model name, timestamp, and test accuracy.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib

from constants import MODELS_DIR


def generate_model_filename(
    model_name: str,
    accuracy: float,
    extension: str = ".pkl",
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Generate a standardized model filename with metadata.

    Format: {model_name}_{YYYY-MM-DD_HH-MM-SS}_acc{accuracy:.4f}{extension}

    Args:
        model_name: Base name for the model (e.g., "leishmania_donovani_rf")
        accuracy: Test accuracy as a float (e.g., 0.9523)
        extension: File extension including dot (e.g., ".pkl", ".pth")
        timestamp: Optional datetime, uses current time if not provided

    Returns:
        Formatted filename string

    Example:
        >>> generate_model_filename("leishmania_donovani_rf", 0.9523)
        'leishmania_donovani_rf_2025-01-22_14-30-45_acc0.9523.pkl'
    """
    if timestamp is None:
        timestamp = datetime.now()

    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    accuracy_str = f"acc{accuracy:.4f}"

    return f"{model_name}_{timestamp_str}_{accuracy_str}{extension}"


def save_sklearn_model(
    model,
    model_name: str,
    accuracy: float,
    models_dir: Path = MODELS_DIR,
) -> Path:
    """
    Save a scikit-learn model with metadata in the filename.

    Args:
        model: The trained scikit-learn model
        model_name: Base name for the model (e.g., "leishmania_donovani_rf")
        accuracy: Test accuracy as a float
        models_dir: Directory to save the model

    Returns:
        Path to the saved model file
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_model_filename(model_name, accuracy, extension=".pkl")
    filepath = models_dir / filename

    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")

    return filepath


def save_pytorch_model(
    model,
    model_name: str,
    accuracy: float,
    models_dir: Path = MODELS_DIR,
) -> Path:
    """
    Save a PyTorch model with metadata in the filename.

    Args:
        model: The trained PyTorch model (nn.Module)
        model_name: Base name for the model (e.g., "leishmania_donovani_vae_ae")
        accuracy: Test accuracy as a float
        models_dir: Directory to save the model

    Returns:
        Path to the saved model file
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_model_filename(model_name, accuracy, extension=".pth")
    filepath = models_dir / filename

    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")

    return filepath


def get_latest_model(model_name_prefix: str, models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """
    Find the most recent model file matching a given prefix.

    Args:
        model_name_prefix: Prefix to match (e.g., "leishmania_donovani_rf")
        models_dir: Directory to search

    Returns:
        Path to the most recent model, or None if not found
    """
    if not models_dir.exists():
        return None

    matching_files = list(models_dir.glob(f"{model_name_prefix}_*.pkl")) + \
                     list(models_dir.glob(f"{model_name_prefix}_*.pth"))

    if not matching_files:
        return None

    # Sort by modification time, most recent first
    matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return matching_files[0]


@dataclass
class ModelInfo:
    """Information about a saved model parsed from its filename."""
    path: Path
    name: str
    timestamp: datetime
    accuracy: float
    model_type: str  # "rf" for Random Forest, "vae_ae" or "vae_clf" for VAE

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        type_labels = {
            "rf": "Random Forest",
            "vae_ae": "VAE (Autoencoder)",
            "vae_clf": "VAE (Classifier)",
            "xgb": "XGBoost",
        }
        type_label = type_labels.get(self.model_type, self.model_type.upper())
        date_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"{type_label} - {date_str} - Accuracy: {self.accuracy:.2%}"


def parse_model_filename(filepath: Path) -> Optional[ModelInfo]:
    """
    Parse model metadata from a standardized filename.

    Expected format: {name}_{YYYY-MM-DD_HH-MM-SS}_acc{accuracy}.{ext}
    Example: leishmania_donovani_rf_2025-01-22_14-30-45_acc0.9523.pkl

    Returns:
        ModelInfo if parsing succeeds, None otherwise
    """
    import re

    filename = filepath.stem  # Remove extension

    # Pattern: name_YYYY-MM-DD_HH-MM-SS_accX.XXXX
    pattern = r"^(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_acc(\d+\.\d+)$"
    match = re.match(pattern, filename)

    if not match:
        return None

    name = match.group(1)
    timestamp_str = match.group(2)
    accuracy_str = match.group(3)

    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        accuracy = float(accuracy_str)
    except ValueError:
        return None

    # Determine model type from name
    if name.endswith("_rf"):
        model_type = "rf"
    elif name.endswith("_vae_ae"):
        model_type = "vae_ae"
    elif name.endswith("_vae_clf"):
        model_type = "vae_clf"
    elif name.endswith("_xgb"):
        model_type = "xgb"
    else:
        model_type = "unknown"

    return ModelInfo(
        path=filepath,
        name=name,
        timestamp=timestamp,
        accuracy=accuracy,
        model_type=model_type,
    )


def list_available_models(
    models_dir: Path = MODELS_DIR,
    model_type_filter: Optional[str] = None,
) -> list[ModelInfo]:
    """
    List all available models with their metadata.

    Args:
        models_dir: Directory to search for models
        model_type_filter: Optional filter for model type ("rf", "vae_ae", "vae_clf")

    Returns:
        List of ModelInfo sorted by timestamp (newest first)
    """
    if not models_dir.exists():
        return []

    models = []

    for filepath in models_dir.glob("*"):
        if filepath.suffix not in (".pkl", ".pth"):
            continue

        info = parse_model_filename(filepath)
        if info is None:
            continue

        if model_type_filter and info.model_type != model_type_filter:
            continue

        models.append(info)

    # Sort by timestamp, newest first
    models.sort(key=lambda m: m.timestamp, reverse=True)

    return models


def get_available_rf_models(models_dir: Path = MODELS_DIR) -> list[ModelInfo]:
    """Get all available Random Forest models."""
    return list_available_models(models_dir, model_type_filter="rf")


def get_available_vae_models(models_dir: Path = MODELS_DIR) -> list[tuple[ModelInfo, ModelInfo]]:
    """
    Get all available VAE model pairs (autoencoder + classifier).

    Returns pairs where both ae and clf models exist with matching timestamps.
    """
    ae_models = {m.timestamp: m for m in list_available_models(models_dir, "vae_ae")}
    clf_models = {m.timestamp: m for m in list_available_models(models_dir, "vae_clf")}

    pairs = []
    for timestamp in ae_models:
        if timestamp in clf_models:
            pairs.append((ae_models[timestamp], clf_models[timestamp]))

    # Sort by timestamp, newest first
    pairs.sort(key=lambda p: p[0].timestamp, reverse=True)

    return pairs


def get_available_xgb_models(models_dir: Path = MODELS_DIR) -> list[ModelInfo]:
    """Get all available XGBoost models."""
    return list_available_models(models_dir, model_type_filter="xgb")
