"""
Predictor factory for Leishmania activity prediction.

Provides a unified interface for different prediction models (Random Forest, VAE).
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from constants import (
    PredictorType,
    RF_MODEL_PREFIX,
    VAE_AE_MODEL_PREFIX,
    VAE_CLF_MODEL_PREFIX,
)
from model_utils import get_latest_model

RDLogger.DisableLog('rdApp.*')


def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint vector (2048 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class BasePredictor(ABC):
    """Abstract base class for Leishmania activity predictors."""

    @abstractmethod
    def predict(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Predict activity for a list of SMILES strings.

        Returns:
            DataFrame with columns: SMILES, Probabilidad de ser activo
        """
        pass


class RandomForestPredictor(BasePredictor):
    """Random Forest-based predictor using Morgan fingerprints."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = get_latest_model(RF_MODEL_PREFIX)
            if model_path is None:
                raise FileNotFoundError(f"No RF model found with prefix '{RF_MODEL_PREFIX}'")
        self.model = joblib.load(model_path)

    def predict(self, smiles_list: List[str]) -> pd.DataFrame:
        results = []
        for smiles in smiles_list:
            fp = get_fingerprint(smiles)
            if fp is not None:
                prob = self.model.predict_proba([fp])[0][1]
                results.append({"SMILES": smiles, "Probabilidad de ser activo": prob})
            else:
                results.append({"SMILES": smiles, "Probabilidad de ser activo": None})
        return pd.DataFrame(results)


class VAEPredictor(BasePredictor):
    """VAE + Classifier predictor using latent space representations."""

    def __init__(
        self,
        ae_path: Optional[str] = None,
        clf_path: Optional[str] = None,
    ):
        # Import model classes here to avoid circular imports
        from leishmania_semi_supervised import Autoencoder, ActivityClassifier

        if ae_path is None:
            ae_path = get_latest_model(VAE_AE_MODEL_PREFIX)
            if ae_path is None:
                raise FileNotFoundError(f"No VAE autoencoder model found with prefix '{VAE_AE_MODEL_PREFIX}'")

        if clf_path is None:
            clf_path = get_latest_model(VAE_CLF_MODEL_PREFIX)
            if clf_path is None:
                raise FileNotFoundError(f"No VAE classifier model found with prefix '{VAE_CLF_MODEL_PREFIX}'")

        self.ae = Autoencoder()
        self.ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
        self.ae.eval()

        self.clf = ActivityClassifier(input_dim=256)
        self.clf.load_state_dict(torch.load(clf_path, map_location="cpu", weights_only=True))
        self.clf.eval()

    def predict(self, smiles_list: List[str]) -> pd.DataFrame:
        results = []
        for smiles in smiles_list:
            fp = get_fingerprint(smiles)
            if fp is None:
                results.append({"SMILES": smiles, "Probabilidad de ser activo": None})
                continue

            fp_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                latent = self.ae.encode(fp_tensor)
                logit = self.clf(latent)
                prob = torch.sigmoid(logit).item()

            results.append({"SMILES": smiles, "Probabilidad de ser activo": prob})
        return pd.DataFrame(results)


def create_predictor(
    predictor_type: PredictorType = PredictorType.RANDOM_FOREST,
    model_path: Optional[str] = None,
    vae_ae_path: Optional[str] = None,
    vae_clf_path: Optional[str] = None,
) -> BasePredictor:
    """
    Factory function to create the appropriate predictor.

    Args:
        predictor_type: Type of predictor to create (RANDOM_FOREST or VAE)
        model_path: Path to Random Forest model (optional, uses default if not provided)
        vae_ae_path: Path to VAE autoencoder model (optional)
        vae_clf_path: Path to VAE classifier model (optional)

    Returns:
        An instance of the requested predictor
    """
    if predictor_type == PredictorType.RANDOM_FOREST:
        if model_path:
            return RandomForestPredictor(model_path=model_path)
        return RandomForestPredictor()
    elif predictor_type == PredictorType.VAE:
        if vae_ae_path and vae_clf_path:
            return VAEPredictor(ae_path=vae_ae_path, clf_path=vae_clf_path)
        return VAEPredictor()
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
