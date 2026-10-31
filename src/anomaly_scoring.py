"""
Anomaly scoring and thresholding module for VAE Intrusion Detection.
Calculates per-record reconstruction loss, applies heuristic thresholds,
and performs sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from models.vae import VAEAnomalyDetector


def compute_anomaly_scores(
    vae: VAEAnomalyDetector,
    x: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Computes continuous anomaly score for each network record.
    Score S(x) = MSE(x, x_hat)
    """
    return vae.get_reconstruction_error(x, device=device)


def classify_anomalies(
    scores: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies heuristic decision boundary:
      If Score > Threshold -> 1 (Suspicious / Anomaly)
      Else                 -> 0 (Normal)
      
    Returns:
      (binary_predictions, diagnostic_stats)
    """
    predictions = (scores > threshold).astype(int)
    suspicious_count = int(np.sum(predictions == 1))
    normal_count = int(np.sum(predictions == 0))
    total = len(scores)
    
    stats = {
        "total_records": total,
        "suspicious_count": suspicious_count,
        "normal_count": normal_count,
        "suspicious_pct": round(100.0 * suspicious_count / max(total, 1), 2),
        "normal_pct": round(100.0 * normal_count / max(total, 1), 2),
        "threshold_applied": float(threshold),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_std": float(np.std(scores))
    }
    return predictions, stats


def analyze_threshold_sensitivity(
    scores: np.ndarray,
    quantiles: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Performs sensitivity analysis showing anomaly detection rate across different threshold quantiles.
    """
    if quantiles is None:
        quantiles = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
        
    results = []
    total = len(scores)
    for q in quantiles:
        thresh = float(np.quantile(scores, q))
        flagged = int(np.sum(scores > thresh))
        results.append({
            "Quantile": f"{int(q * 100)}th",
            "Threshold_Value": round(thresh, 5),
            "Flagged_Suspicious": flagged,
            "Flagged_Percentage": round(100.0 * flagged / max(total, 1), 2)
        })
    return pd.DataFrame(results)
