"""
Unified inference engine for NetShield AI.
Loads saved artifacts, preprocesses uploaded CSV records, executes model inference,
and returns structured predictions ready for dashboard display and CSV export.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREPROCESSOR_FILE, VAE_MODEL_FILE, RF_MODEL_FILE
from src.preprocessing import NIDSPreprocessor
from src.train_vae import load_trained_vae
from src.train_random_forest import load_trained_random_forest


class NetShieldInferenceEngine:
    """
    Production inference engine orchestrating feature preprocessing and model predictions.
    """
    def __init__(self):
        self.preprocessor: Optional[NIDSPreprocessor] = None
        self.vae_model = None
        self.vae_config = None
        self.rf_model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Attempts to load saved preprocessor and models if present."""
        if PREPROCESSOR_FILE.exists():
            try:
                self.preprocessor = NIDSPreprocessor.load(PREPROCESSOR_FILE)
            except Exception as e:
                print(f"[WARN] Failed loading preprocessor: {e}")
                
        if VAE_MODEL_FILE.exists():
            try:
                self.vae_model, self.vae_config = load_trained_vae()
            except Exception as e:
                print(f"[WARN] Failed loading VAE: {e}")
                
        if RF_MODEL_FILE.exists():
            try:
                self.rf_model = load_trained_random_forest()
            except Exception as e:
                print(f"[WARN] Failed loading Random Forest: {e}")

    def is_vae_ready(self) -> bool:
        return self.preprocessor is not None and self.vae_model is not None

    def is_rf_ready(self) -> bool:
        return self.preprocessor is not None and self.rf_model is not None

    def predict_vae(
        self,
        input_df: pd.DataFrame,
        custom_threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs VAE unsupervised anomaly detection on input dataframe.
        
        Returns:
            (annotated_dataframe, summary_statistics)
        """
        if not self.is_vae_ready():
            raise RuntimeError("VAE model or preprocessor artifacts are not available. Please train VAE first.")
            
        clean_df = input_df.copy()
        
        # Transform features
        X_trans = self.preprocessor.transform(clean_df)
        
        # Anomaly scoring
        scores = self.vae_model.get_reconstruction_error(X_trans)
        threshold = custom_threshold if custom_threshold is not None else self.vae_model.threshold
        
        preds = (scores > threshold).astype(int)
        labels = np.where(preds == 1, "Suspicious", "Normal")
        
        # Annotate results
        annotated_df = clean_df.copy()
        annotated_df["anomaly_score"] = np.round(scores, 5)
        annotated_df["anomaly_threshold"] = round(threshold, 5)
        annotated_df["prediction_code"] = preds
        annotated_df["predicted_status"] = labels
        
        suspicious_cnt = int(np.sum(preds == 1))
        normal_cnt = int(np.sum(preds == 0))
        total = len(clean_df)
        
        stats = {
            "total_records": total,
            "suspicious_records": suspicious_cnt,
            "normal_records": normal_cnt,
            "suspicious_percentage": round(100.0 * suspicious_cnt / max(total, 1), 2),
            "threshold_used": round(threshold, 5),
            "min_score": round(float(np.min(scores)), 5),
            "max_score": round(float(np.max(scores)), 5),
            "mean_score": round(float(np.mean(scores)), 5)
        }
        return annotated_df, stats

    def predict_rf(self, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs Random Forest supervised classifier on input dataframe.
        """
        if not self.is_rf_ready():
            raise RuntimeError("Random Forest model or preprocessor artifacts are not available. Please train RF first.")
            
        clean_df = input_df.copy()
        X_trans = self.preprocessor.transform(clean_df)
        
        preds = self.rf_model.predict(X_trans)
        proba = self.rf_model.predict_proba(X_trans)
        attack_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        labels = np.where(preds == 1, "Attack", "Normal")
        
        annotated_df = clean_df.copy()
        annotated_df["rf_prediction_code"] = preds
        annotated_df["rf_predicted_class"] = labels
        annotated_df["rf_attack_probability"] = np.round(attack_prob, 4)
        
        attack_cnt = int(np.sum(preds == 1))
        normal_cnt = int(np.sum(preds == 0))
        total = len(clean_df)
        
        stats = {
            "total_records": total,
            "predicted_attacks": attack_cnt,
            "predicted_normal": normal_cnt,
            "attack_percentage": round(100.0 * attack_cnt / max(total, 1), 2)
        }
        return annotated_df, stats
