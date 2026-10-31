"""
Training pipeline for Random Forest Supervised Baseline.
Trains on labelled training data (Normal vs Attack) and exports artifacts.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RF_DEFAULTS, RF_MODEL_FILE
from models.random_forest import RandomForestBaseline


def train_random_forest_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_estimators: int = RF_DEFAULTS["n_estimators"],
    max_depth: Optional[int] = RF_DEFAULTS["max_depth"],
    min_samples_split: int = RF_DEFAULTS["min_samples_split"],
    random_state: int = RF_DEFAULTS["random_state"],
    class_weight: str = RF_DEFAULTS["class_weight"]
) -> Tuple[RandomForestBaseline, Dict[str, Any]]:
    """
    Fits the Random Forest baseline on labelled training data.
    """
    rf = RandomForestBaseline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        class_weight=class_weight
    )
    
    start_time = time.time()
    rf.fit(x_train, y_train, feature_names=feature_names)
    training_time = round(time.time() - start_time, 3)
    
    # Save model artifact
    rf.save(RF_MODEL_FILE)
    
    top_importances = rf.get_feature_importances(top_k=15)
    
    results = {
        "training_time": training_time,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": class_weight,
        "top_features": top_importances
    }
    
    return rf, results


def load_trained_random_forest() -> RandomForestBaseline:
    """Loads serialized Random Forest model artifact."""
    return RandomForestBaseline.load(RF_MODEL_FILE)
