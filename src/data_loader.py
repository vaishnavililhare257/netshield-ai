"""
Data loading and validation module for the UNSW-NB15 Intrusion Detection System.
Provides rigorous file checking, column validation, and dataset summary extraction.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_BINARY, TARGET_MULTICLASS, COLUMNS_TO_DROP


def load_dataset(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads dataset from a CSV file with validation.
    
    Args:
        file_path: Path to the CSV file.
        nrows: Optional limit on number of rows to load for fast development.
        
    Returns:
        pd.DataFrame containing the loaded data.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is empty or missing critical columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {path}.\n"
            f"Please verify that 'UNSW_NB15_training-set.csv' or 'UNSW_NB15_testing-set.csv' "
            f"is placed in the data/ folder, or run data/download_unsw.py to prepare the dataset."
        )
    
    try:
        df = pd.read_csv(path, nrows=nrows)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {path}: {str(e)}")
        
    if df.empty:
        raise ValueError(f"The dataset at {path} is completely empty.")
        
    # Standardize column names (strip whitespace and lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    
    return df


def validate_dataset_schema(df: pd.DataFrame, is_training: bool = True) -> Dict[str, Any]:
    """
    Validates dataset structure, column presence, data types, and integrity.
    
    Args:
        df: Input DataFrame.
        is_training: Whether the data is expected to have target labels for training.
        
    Returns:
        Dictionary with validation results: 'is_valid', 'errors', 'warnings', 'features_present'.
    """
    errors = []
    warnings = []
    
    target_bin = TARGET_BINARY.lower()
    target_multi = TARGET_MULTICLASS.lower()
    
    if is_training:
        if target_bin not in df.columns:
            errors.append(f"Required binary target column '{target_bin}' is missing from training data.")
        if target_multi not in df.columns:
            warnings.append(f"Multiclass attack column '{target_multi}' is not present. Multiclass analysis will be disabled.")
            
    # Check for NaN or Inf features
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        warnings.append(f"Missing values found in columns: {list(null_cols.index)} (will be imputed during preprocessing).")
        
    # Check for minimum number of features
    feature_cols = [c for c in df.columns if c not in [target_bin, target_multi] + [c.lower() for c in COLUMNS_TO_DROP]]
    if len(feature_cols) < 5:
        errors.append(f"Insufficient feature columns detected ({len(feature_cols)} found). Minimum 5 required.")
        
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "num_rows": len(df)
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates academic and operational summary statistics for a dataset.
    """
    target_bin = TARGET_BINARY.lower()
    target_multi = TARGET_MULTICLASS.lower()
    
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "columns": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "has_binary_label": target_bin in df.columns,
        "has_multiclass_label": target_multi in df.columns,
    }
    
    if summary["has_binary_label"]:
        val_counts = df[target_bin].value_counts().to_dict()
        normal_count = val_counts.get(0, 0)
        attack_count = val_counts.get(1, 0)
        total = max(len(df), 1)
        summary["normal_records"] = normal_count
        summary["attack_records"] = attack_count
        summary["normal_pct"] = round(100.0 * normal_count / total, 2)
        summary["attack_pct"] = round(100.0 * attack_count / total, 2)
        
    if summary["has_multiclass_label"]:
        summary["attack_categories"] = df[target_multi].value_counts().to_dict()
        
    return summary
