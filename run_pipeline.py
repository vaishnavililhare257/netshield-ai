"""
End-to-End Pipeline Execution Script for NetShield AI.
Executes dataset verification, preprocessing, VAE training, RF training, and evaluation.
Generates genuine saved artifacts and prints comparative evaluation metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, PREPROCESSOR_FILE, VAE_CONFIG_FILE
from data.download_unsw import ensure_data_available, check_dataset_status
from src.data_loader import load_dataset
from src.preprocessing import prepare_datasets_for_training, NIDSPreprocessor
from src.train_vae import train_vae_pipeline
from src.train_random_forest import train_random_forest_pipeline
from src.train_rbm import train_rbm_pipeline
from src.evaluation import evaluate_binary_classifier, measure_inference_latency, evaluate_per_category_recall, save_metrics_summary


def run_full_pipeline():
    print("=" * 70)
    print("NETSHIELD AI — END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Dataset Verification
    print("\n[STEP 1/5] Checking Dataset Availability...")
    ensure_data_available()
    status = check_dataset_status()
    print(f"  Training Set: {TRAIN_DATA_PATH} ({status['train_size_mb']} MB)")
    print(f"  Testing Set:  {TEST_DATA_PATH} ({status['test_size_mb']} MB)")

    # 2. Loading Datasets
    print("\n[STEP 2/5] Loading Datasets...")
    train_df = load_dataset(TRAIN_DATA_PATH)
    test_df = load_dataset(TEST_DATA_PATH)
    print(f"  Loaded Train: {len(train_df):,} records | Test: {len(test_df):,} records")

    # 3. Leakage-free Preprocessing
    print("\n[STEP 3/5] Preprocessing and Isolating Normal Traffic for VAE...")
    prepared = prepare_datasets_for_training(
        train_df=train_df,
        test_df=test_df,
        val_ratio=0.15,
        max_normal_train=12000,
        scaler_type="standard"
    )
    # Save preprocessor artifact
    prepared["preprocessor"].save()
    print(f"  Preprocessor saved to {PREPROCESSOR_FILE}")
    print(f"  Normal VAE Train: {prepared['x_vae_train'].shape[0]:,} samples | Val: {prepared['x_vae_val'].shape[0]:,} samples")
    print(f"  Labelled RF Train: {prepared['x_rf_train'].shape[0]:,} samples")
    print(f"  Feature Dimensions: {prepared['feature_dim']} numerical inputs")

    # 4. Train Models
    print("\n[STEP 4/5] Training Deep Learning and Machine Learning Models...")
    
    # Train VAE (Normal traffic only)
    print("  -> Training Unsupervised VAE (Normal-Only)...")
    vae, vae_res = train_vae_pipeline(
        x_train=prepared["x_vae_train"],
        x_val=prepared["x_vae_val"],
        hidden_dim1=64,
        hidden_dim2=32,
        latent_dim=16,
        epochs=8,
        batch_size=256,
        beta=1.0,
        threshold_quantile=0.95
    )
    print(f"     [OK] VAE Trained in {vae_res['training_time']}s | Threshold: {vae_res['threshold']:.5f}")

    # Train Random Forest (Labelled data)
    print("  -> Training Supervised Random Forest Baseline (100 Trees)...")
    rf, rf_res = train_random_forest_pipeline(
        x_train=prepared["x_rf_train"],
        y_train=prepared["y_rf_train"],
        feature_names=prepared["feature_names"],
        n_estimators=100,
        max_depth=16
    )
    print(f"     [OK] Random Forest Trained in {rf_res['training_time']}s")

    # Train RBM Extension
    print("  -> Training RBM Module (Contrastive Divergence CD-1)...")
    rbm, rbm_res = train_rbm_pipeline(
        x_train=prepared["x_vae_train"][:3000],
        n_hidden=32,
        learning_rate=0.01,
        epochs=6,
        batch_size=128
    )
    print(f"     [OK] RBM Trained in {rbm_res['training_time']}s")

    # 5. Held-out Test Evaluation
    print("\n[STEP 5/5] Evaluating on Held-Out Test Records...")
    x_test = prepared["x_test"]
    y_test = prepared["y_test"]
    y_cats = prepared["y_test_multi"]

    # Evaluate VAE
    vae_scores = vae.get_reconstruction_error(x_test)
    vae_preds = (vae_scores > vae.threshold).astype(int)
    vae_metrics = evaluate_binary_classifier(y_test, vae_preds, y_scores=vae_scores, model_name="VAE (Unsupervised)")
    vae_latency = measure_inference_latency(lambda x: vae.get_reconstruction_error(x), x_test[:200])
    vae_metrics.update(vae_latency)

    # Evaluate RF
    rf_preds = rf.predict(x_test)
    rf_proba = rf.predict_proba(x_test)
    rf_scores = rf_proba[:, 1] if rf_proba.shape[1] > 1 else rf_proba[:, 0]
    rf_metrics = evaluate_binary_classifier(y_test, rf_preds, y_scores=rf_scores, model_name="Random Forest (Supervised)")
    rf_latency = measure_inference_latency(lambda x: rf.predict(x), x_test[:200])
    rf_metrics.update(rf_latency)

    # Save summary
    results_to_save = {
        "vae_metrics": vae_metrics,
        "rf_metrics": rf_metrics
    }
    save_metrics_summary(results_to_save)

    # Print Comparative Results Table
    print("\n" + "=" * 70)
    print("FINAL ACADEMIC EVALUATION RESULTS ON HELD-OUT TEST DATA")
    print("=" * 70)
    
    summary_df = pd.DataFrame([
        {
            "Metric": "Paradigm",
            "VAE Anomaly Detector": "Unsupervised (Normal Only)",
            "Random Forest": "Supervised (Labelled)"
        },
        {
            "Metric": "Accuracy",
            "VAE Anomaly Detector": f"{vae_metrics['accuracy'] * 100:.2f}%",
            "Random Forest": f"{rf_metrics['accuracy'] * 100:.2f}%"
        },
        {
            "Metric": "Precision",
            "VAE Anomaly Detector": f"{vae_metrics['precision'] * 100:.2f}%",
            "Random Forest": f"{rf_metrics['precision'] * 100:.2f}%"
        },
        {
            "Metric": "Recall (Detection Rate)",
            "VAE Anomaly Detector": f"{vae_metrics['recall'] * 100:.2f}%",
            "Random Forest": f"{rf_metrics['recall'] * 100:.2f}%"
        },
        {
            "Metric": "F1-Score",
            "VAE Anomaly Detector": f"{vae_metrics['f1_score']:.4f}",
            "Random Forest": f"{rf_metrics['f1_score']:.4f}"
        },
        {
            "Metric": "ROC-AUC",
            "VAE Anomaly Detector": f"{vae_metrics.get('roc_auc', 'N/A')}",
            "Random Forest": f"{rf_metrics.get('roc_auc', 'N/A')}"
        },
        {
            "Metric": "False Positive Rate (FPR)",
            "VAE Anomaly Detector": f"{vae_metrics['false_positive_rate'] * 100:.2f}%",
            "Random Forest": f"{rf_metrics['false_positive_rate'] * 100:.2f}%"
        },
        {
            "Metric": "Latency per Record",
            "VAE Anomaly Detector": f"{vae_metrics['latency_per_record_ms']:.4f} ms",
            "Random Forest": f"{rf_metrics['latency_per_record_ms']:.4f} ms"
        }
    ])

    print(summary_df.to_string(index=False))
    print("=" * 70)
    print("\nAll model weights and preprocessors successfully saved to netshield-ai/artifacts/")
    print("Launch Dashboard via: streamlit run app.py")


if __name__ == "__main__":
    run_full_pipeline()
