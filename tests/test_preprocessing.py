"""
Unit tests for data loader and preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import NIDSPreprocessor, prepare_datasets_for_training
from data.download_unsw import generate_benchmark_subset


@pytest.fixture
def sample_dataset(tmp_path):
    csv_path = tmp_path / "test_unsw.csv"
    df = generate_benchmark_subset(csv_path, num_records=200, random_seed=42)
    return df


def test_preprocessor_fit_transform(sample_dataset):
    preprocessor = NIDSPreprocessor(scaler_type="standard")
    transformed = preprocessor.fit_transform(sample_dataset)
    
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_dataset)
    assert transformed.shape[1] > 0
    # Check no NaNs or Infs
    assert not np.isnan(transformed).any()
    assert not np.isinf(transformed).any()
    # Check target columns are excluded
    assert "label" not in preprocessor.feature_names_in_
    assert "attack_cat" not in preprocessor.feature_names_in_
    assert "id" not in preprocessor.feature_names_in_


def test_prepare_datasets_for_training(sample_dataset):
    prepared = prepare_datasets_for_training(sample_dataset, val_ratio=0.2, max_normal_train=150)
    
    assert "x_vae_train" in prepared
    assert "x_vae_val" in prepared
    assert "x_rf_train" in prepared
    assert "y_rf_train" in prepared
    
    # VAE train must have only normal records
    assert prepared["x_vae_train"].shape[0] > 0
    assert prepared["x_vae_val"].shape[0] > 0
    assert prepared["x_rf_train"].shape[0] == len(sample_dataset)
    assert len(prepared["y_rf_train"]) == len(sample_dataset)


def test_preprocessor_serialization(sample_dataset, tmp_path):
    preprocessor = NIDSPreprocessor()
    preprocessor.fit(sample_dataset)
    
    save_path = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_path)
    assert save_path.exists()
    
    loaded = NIDSPreprocessor.load(save_path)
    assert loaded.is_fitted
    assert loaded.numeric_cols_ == preprocessor.numeric_cols_
    assert loaded.categorical_cols_ == preprocessor.categorical_cols_
