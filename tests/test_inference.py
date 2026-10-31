"""
Unit tests for unified inference engine and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import NIDSPreprocessor, prepare_datasets_for_training
from src.train_vae import train_vae_pipeline
from src.train_random_forest import train_random_forest_pipeline
from src.inference import NetShieldInferenceEngine
from data.download_unsw import generate_benchmark_subset


@pytest.fixture(scope="module")
def trained_environment(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("artifacts")
    raw_df = generate_benchmark_subset(tmp_dir / "bench.csv", num_records=300, random_seed=42)
    
    prepared = prepare_datasets_for_training(raw_df, max_normal_train=180)
    
    # Save preprocessor
    prepared["preprocessor"].save()
    
    # Train small VAE
    train_vae_pipeline(
        prepared["x_vae_train"],
        prepared["x_vae_val"],
        latent_dim=8,
        epochs=2,
        batch_size=64
    )
    
    # Train small RF
    train_random_forest_pipeline(
        prepared["x_rf_train"],
        prepared["y_rf_train"],
        n_estimators=10,
        max_depth=5
    )
    
    return raw_df


def test_inference_engine_prediction(trained_environment):
    engine = NetShieldInferenceEngine()
    
    assert engine.is_vae_ready()
    assert engine.is_rf_ready()
    
    # Run VAE inference
    annotated_vae, stats_vae = engine.predict_vae(trained_environment.head(50))
    assert "anomaly_score" in annotated_vae.columns
    assert "predicted_status" in annotated_vae.columns
    assert stats_vae["total_records"] == 50
    assert stats_vae["suspicious_records"] >= 0
    
    # Run RF inference
    annotated_rf, stats_rf = engine.predict_rf(trained_environment.head(50))
    assert "rf_predicted_class" in annotated_rf.columns
    assert "rf_attack_probability" in annotated_rf.columns
    assert stats_rf["total_records"] == 50


def test_inference_engine_missing_columns():
    engine = NetShieldInferenceEngine()
    invalid_df = pd.DataFrame({"dummy_col1": [1, 2], "dummy_col2": [3, 4]})
    
    with pytest.raises(ValueError) as excinfo:
        engine.predict_vae(invalid_df)
    assert "missing required features" in str(excinfo.value).lower()
