import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset File Paths
TRAIN_DATA_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_DATA_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"
SAMPLE_DATA_PATH = DATA_DIR / "UNSW_NB15_sample.csv"

# Metadata & Feature Definitions
ID_COLUMN = "id"
TARGET_BINARY = "label"       # 0: Normal, 1: Attack
TARGET_MULTICLASS = "attack_cat"

# Inappropriate columns causing leakage or strictly metadata identifiers
COLUMNS_TO_DROP = [ID_COLUMN]

# Known Categorical Features in UNSW-NB15
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Model Artifact Filenames
VAE_MODEL_FILE = ARTIFACTS_DIR / "vae_model.pt"
VAE_CONFIG_FILE = ARTIFACTS_DIR / "vae_config.json"
RF_MODEL_FILE = ARTIFACTS_DIR / "random_forest_model.joblib"
RBM_MODEL_FILE = ARTIFACTS_DIR / "rbm_model.pt"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_FILE = ARTIFACTS_DIR / "evaluation_metrics.json"

# Default Model Hyperparameters
VAE_DEFAULTS = {
    "hidden_dim1": 64,
    "hidden_dim2": 32,
    "latent_dim": 16,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 10,
    "beta": 1.0,                       # KL-divergence weight
    "threshold_quantile": 0.95,         # Quantile on normal validation reconstruction error
    "random_seed": 42
}

RF_DEFAULTS = {
    "n_estimators": 100,
    "max_depth": 18,
    "min_samples_split": 5,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1
}

RBM_DEFAULTS = {
    "n_hidden": 32,
    "learning_rate": 0.01,
    "k_steps": 1,                      # CD-1 (Contrastive Divergence 1)
    "batch_size": 128,
    "epochs": 10,
    "random_seed": 42
}

# Attack Categories in UNSW-NB15
ATTACK_CATEGORIES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"
]
