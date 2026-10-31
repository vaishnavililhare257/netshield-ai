"""
Random Forest Supervised Baseline Classifier for Network Intrusion Detection.
Serves as an empirical baseline against unsupervised VAE anomaly detection.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RF_DEFAULTS, RF_MODEL_FILE


class RandomForestBaseline:
    """
    Supervised Random Forest baseline model.
    Learns explicitly from labelled training records (Normal vs Attack).
    """
    def __init__(
        self,
        n_estimators: int = RF_DEFAULTS["n_estimators"],
        max_depth: Optional[int] = RF_DEFAULTS["max_depth"],
        min_samples_split: int = RF_DEFAULTS["min_samples_split"],
        random_state: int = RF_DEFAULTS["random_state"],
        class_weight: str = RF_DEFAULTS["class_weight"]
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
            "class_weight": class_weight,
            "n_jobs": -1
        }
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None) -> "RandomForestBaseline":
        """Fits Random Forest classifier on labelled data."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (0: Normal, 1: Attack)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities [P(Normal), P(Attack)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importances(self, top_k: int = 15) -> Dict[str, float]:
        """Returns top-k most important features identified by Random Forest."""
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        return {self.feature_names[i]: float(importances[i]) for i in indices}

    def save(self, filepath: Optional[Path] = None):
        """Serializes trained model artifact using joblib."""
        path = Path(filepath or RF_MODEL_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "RandomForestBaseline":
        """Loads serialized Random Forest artifact from disk."""
        path = Path(filepath or RF_MODEL_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Random Forest artifact not found at: {path}")
        return joblib.load(path)
