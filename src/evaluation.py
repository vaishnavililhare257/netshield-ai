"""
Comprehensive academic evaluation module for Network Intrusion Detection.
Calculates Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, FPR, and Latency.
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METRICS_FILE


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard cybersecurity and machine learning classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # True Positive Rate (Recall) = TP / (TP + FN)
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist()
    }
    
    # Continuous curve metrics if scores/probabilities provided
    if y_scores is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_scores)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_scores)), 4)
            
            # ROC curve points (subsampled for clean plotting)
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
            p_arr, r_arr, _ = precision_recall_curve(y_true, y_scores)
            
            # Downsample for JSON serialization efficiency
            step = max(len(fpr_arr) // 100, 1)
            metrics["roc_curve"] = {
                "fpr": fpr_arr[::step].tolist(),
                "tpr": tpr_arr[::step].tolist()
            }
            step_pr = max(len(p_arr) // 100, 1)
            metrics["pr_curve"] = {
                "precision": p_arr[::step_pr].tolist(),
                "recall": r_arr[::step_pr].tolist()
            }
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
            
    return metrics


def measure_inference_latency(
    predict_func,
    x_sample: np.ndarray,
    n_runs: int = 10
) -> Dict[str, float]:
    """
    Measures empirical inference latency in milliseconds per batch and per record.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_func(x_sample)
        times.append(time.perf_counter() - t0)
        
    avg_total_sec = float(np.mean(times))
    per_record_ms = (avg_total_sec / len(x_sample)) * 1000.0
    
    return {
        "avg_batch_time_ms": round(avg_total_sec * 1000.0, 2),
        "latency_per_record_ms": round(per_record_ms, 4),
        "throughput_records_per_sec": round(len(x_sample) / max(avg_total_sec, 1e-6), 1)
    }


def evaluate_per_category_recall(
    y_true_cats: np.ndarray,
    y_pred_binary: np.ndarray
) -> pd.DataFrame:
    """
    Computes detection recall for each individual attack category.
    """
    df = pd.DataFrame({"attack_cat": y_true_cats, "detected": y_pred_binary})
    grouped = df.groupby("attack_cat")
    
    records = []
    for cat, group in grouped:
        total = len(group)
        detected = group["detected"].sum()
        recall = round(float(detected / total), 4) if total > 0 else 0.0
        records.append({
            "Attack_Category": str(cat),
            "Total_Instances": int(total),
            "Detected_Instances": int(detected),
            "Detection_Recall": recall
        })
        
    return pd.DataFrame(records).sort_values(by="Total_Instances", ascending=False)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def save_metrics_summary(metrics_dict: Dict[str, Any], filepath: Optional[Path] = None):
    """Persists evaluation comparison results to JSON artifact."""
    try:
        path = Path(filepath or METRICS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics_dict, f, cls=NpEncoder, indent=4)
    except Exception as e:
        print(f"[WARN] Could not persist metrics summary: {e}")




def load_metrics_summary(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Loads saved evaluation comparison results."""
    path = Path(filepath or METRICS_FILE)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)
