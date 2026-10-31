# NetShield AI — Model & Preprocessing Artifacts Registry

This directory contains persistent model checkpoints, preprocessing states, and evaluation metric logs produced by the NetShield AI pipeline.

## Artifact Inventory

| File | Type | Description |
| :--- | :--- | :--- |
| `preprocessor.joblib` | Scikit-Learn / Joblib | Serialized `NIDSPreprocessor` object storing fitted `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, and feature ordering to guarantee zero leakage. |
| `vae_model.pt` | PyTorch State Dict | Trained PyTorch weights for `VAEAnomalyDetector` (Encoder, Latent projections, Decoder). |
| `vae_config.json` | JSON Metadata | Model hyperparameter configurations, calibrated validation threshold, training duration, and feature dimensions. |
| `random_forest_model.joblib` | Scikit-Learn / Joblib | Serialized `RandomForestClassifier` baseline model. |
| `rbm_model.pt` | PyTorch Checkpoint | Trained Restricted Boltzmann Machine weights, biases, and contrastive divergence parameters. |
| `evaluation_metrics.json` | JSON Benchmark | Recorded evaluation scores: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, FPR, and Latency. |

## Serialization Integrity
All artifacts are loaded dynamically during inference via `src/inference.py`. Re-running training in `2_⚙️_Model_Training` or via CLI automatically updates these serialized files.
