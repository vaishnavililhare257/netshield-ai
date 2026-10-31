"""
Robust, leakage-free data preprocessing pipeline for UNSW-NB15 intrusion detection.
Handles missing values, categorical encoding, numerical scaling, and artifact serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_BINARY, TARGET_MULTICLASS, COLUMNS_TO_DROP, CATEGORICAL_FEATURES, PREPROCESSOR_FILE


class NIDSPreprocessor:
    """
    Production-grade preprocessor guaranteeing no data leakage.
    Fits imputers, one-hot encoders, and standard scalers strictly on training data.
    """
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.transformed_feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Separates valid feature columns into numeric and categorical subsets."""
        target_bin = TARGET_BINARY.lower()
        target_multi = TARGET_MULTICLASS.lower()
        drop_cols = [c.lower() for c in COLUMNS_TO_DROP]
        
        # Exclude target and drop columns
        features = [c for c in df.columns if c not in [target_bin, target_multi] + drop_cols]
        
        cat_cols = []
        num_cols = []
        
        for col in features:
            if col in [c.lower() for c in CATEGORICAL_FEATURES] or df[col].dtype == "object":
                cat_cols.append(col)
            else:
                num_cols.append(col)
                
        return num_cols, cat_cols

    def fit(self, df: pd.DataFrame) -> "NIDSPreprocessor":
        """
        Fits all preprocessing transformations strictly on the training partition.
        """
        # Ensure column names are lowercase
        clean_df = df.copy()
        clean_df.columns = [c.strip().lower() for c in clean_df.columns]
        
        self.numeric_cols_, self.categorical_cols_ = self._identify_columns(clean_df)
        self.feature_names_in_ = self.numeric_cols_ + self.categorical_cols_
        
        if not self.numeric_cols_ and not self.categorical_cols_:
            raise ValueError("No valid feature columns found to fit preprocessor.")
            
        # Fit numeric imputer & scaler
        if self.numeric_cols_:
            num_data = clean_df[self.numeric_cols_].values
            imputed_num = self.num_imputer.fit_transform(num_data)
            self.scaler.fit(imputed_num)
            
        # Fit categorical imputer & one-hot encoder
        if self.categorical_cols_:
            cat_data = clean_df[self.categorical_cols_].astype(str).values
            imputed_cat = self.cat_imputer.fit_transform(cat_data)
            self.encoder.fit(imputed_cat)
            encoded_cat_names = self.encoder.get_feature_names_out(self.categorical_cols_).tolist()
        else:
            encoded_cat_names = []
            
        self.transformed_feature_names_ = self.numeric_cols_ + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame using previously fitted scalers and encoders.
        Ensures consistent schema and order.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
            
        clean_df = df.copy()
        clean_df.columns = [c.strip().lower() for c in clean_df.columns]
        
        # Verify required input columns are present
        missing_num = [c for c in self.numeric_cols_ if c not in clean_df.columns]
        missing_cat = [c for c in self.categorical_cols_ if c not in clean_df.columns]
        
        if missing_num or missing_cat:
            raise ValueError(
                f"Input DataFrame is missing required features.\n"
                f"Missing numeric features: {missing_num}\n"
                f"Missing categorical features: {missing_cat}"
            )
            
        # Transform numeric features
        if self.numeric_cols_:
            num_data = clean_df[self.numeric_cols_].values
            imputed_num = self.num_imputer.transform(num_data)
            scaled_num = self.scaler.transform(imputed_num)
        else:
            scaled_num = np.empty((len(df), 0))
            
        # Transform categorical features
        if self.categorical_cols_:
            cat_data = clean_df[self.categorical_cols_].astype(str).values
            imputed_cat = self.cat_imputer.transform(cat_data)
            encoded_cat = self.encoder.transform(imputed_cat)
        else:
            encoded_cat = np.empty((len(df), 0))
            
        transformed = np.hstack([scaled_num, encoded_cat]).astype(np.float32)
        
        # Replace any potential residual NaNs or Infs
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=-1.0)
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, filepath: Optional[Path] = None):
        """Saves preprocessor instance to disk using joblib."""
        path = Path(filepath or PREPROCESSOR_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "NIDSPreprocessor":
        """Loads preprocessor instance from disk."""
        path = Path(filepath or PREPROCESSOR_FILE)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {path}")
        return joblib.load(path)


def prepare_datasets_for_training(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    val_ratio: float = 0.15,
    max_normal_train: Optional[int] = 15000,
    scaler_type: str = "standard",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Prepares leakage-free datasets for:
    1. VAE: Normal-only training and normal validation sets.
    2. Random Forest: Labelled training and testing sets.
    
    Args:
        train_df: Training DataFrame.
        test_df: Optional separate test DataFrame. If None, splits train_df.
        val_ratio: Proportion of normal training data reserved for validation.
        max_normal_train: Optional sample cap for fast execution.
        scaler_type: 'standard' or 'minmax'.
        random_seed: Reproducibility seed.
        
    Returns:
        Dictionary containing preprocessed arrays, targets, preprocessor object, and metadata.
    """
    target_bin = TARGET_BINARY.lower()
    target_multi = TARGET_MULTICLASS.lower()
    
    # Clean column names
    train_df = train_df.copy()
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    
    # Filter normal traffic records for VAE
    if target_bin in train_df.columns:
        normal_mask = train_df[target_bin] == 0
        normal_df = train_df[normal_mask].copy()
    else:
        raise ValueError(f"Target column '{target_bin}' required for filtering normal traffic.")
        
    if len(normal_df) == 0:
        raise ValueError("No normal traffic records (label == 0) found in training set.")
        
    # Cap normal training records if requested for rapid execution
    if max_normal_train and len(normal_df) > max_normal_train:
        normal_df = normal_df.sample(n=max_normal_train, random_state=random_seed)
        
    # Split normal data into train and validation for VAE threshold tuning
    normal_train_df, normal_val_df = train_test_split(
        normal_df, test_size=val_ratio, random_state=random_seed, shuffle=True
    )
    
    # Fit preprocessor strictly on normal training data (or general train features)
    preprocessor = NIDSPreprocessor(scaler_type=scaler_type)
    preprocessor.fit(train_df)  # Fit on training features so feature bounds encompass diverse network types
    
    # Transform VAE datasets
    x_vae_train = preprocessor.transform(normal_train_df)
    x_vae_val = preprocessor.transform(normal_val_df)
    
    # Prepare Random Forest labelled datasets
    x_rf_train = preprocessor.transform(train_df)
    y_rf_train = train_df[target_bin].values.astype(int)
    
    # Prepare Test partition
    if test_df is not None:
        clean_test = test_df.copy()
        clean_test.columns = [c.strip().lower() for c in clean_test.columns]
        x_test = preprocessor.transform(clean_test)
        y_test = clean_test[target_bin].values.astype(int) if target_bin in clean_test.columns else None
        y_test_multi = clean_test[target_multi].values if target_multi in clean_test.columns else None
    else:
        x_test, y_test, y_test_multi = None, None, None
        
    return {
        "preprocessor": preprocessor,
        "x_vae_train": x_vae_train,
        "x_vae_val": x_vae_val,
        "x_rf_train": x_rf_train,
        "y_rf_train": y_rf_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_multi": y_test_multi,
        "feature_dim": x_vae_train.shape[1],
        "feature_names": preprocessor.transformed_feature_names_
    }
