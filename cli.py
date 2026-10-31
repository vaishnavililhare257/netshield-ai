"""
NetShield AI — Terminal Command Line Interface (CLI)
Provides command-line commands to inspect traffic, run inference, evaluate models,
and launch the web dashboard without needing a web browser.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SAMPLE_DATA_PATH, TEST_DATA_PATH, METRICS_FILE
from src.inference import NetShieldInferenceEngine
import json


def print_banner():
    banner = r"""
  ==============================================================
    _   _      _   ____  _     _      _     _      _    ___ 
   | \ | | ___| |_/ ___|| |__ (_) ___| | __| |    / \  |_ _|
   |  \| |/ _ \ __\___ \| '_ \| |/ _ \ |/ _` |   / _ \  | | 
   | |\  |  __/ |_ ___) | | | | |  __/ | (_| |  / ___ \ | | 
   |_| \_|\___|\__|____/|_| |_|_|\___|_|\__,_| /_/   \_\___|
     Deep Learning & Machine Learning Network Intrusion Detection
  ==============================================================
    """
    print(banner)


def run_predict(file_path: str = None, rows: int = 10, threshold: float = None):
    target_path = Path(file_path) if file_path else SAMPLE_DATA_PATH
    if not target_path.exists():
        print(f"[ERROR] Target file does not exist: {target_path}")
        return

    print(f"\n[+] Loading traffic records from: {target_path}")
    df = pd.read_csv(target_path)
    total_records = len(df)
    print(f"[+] Total records loaded: {total_records:,}")

    engine = NetShieldInferenceEngine()
    if not engine.is_vae_ready() or not engine.is_rf_ready():
        print("[!] Pretrained models not found in artifacts/. Please run training first:")
        print("    python run_pipeline.py  OR  python cli.py train")
        return

    # Run VAE anomaly detection
    vae_annotated, vae_stats = engine.predict_vae(df, custom_threshold=threshold)
    # Run Random Forest baseline
    rf_annotated, rf_stats = engine.predict_rf(df)

    print("\n" + "=" * 65)
    print("  NETSHIELD INFERENCE SUMMARY REPORT (TERMINAL)")
    print("=" * 65)
    print(f"  Total Network Records Inspected : {vae_stats['total_records']:,}")
    print(f"  VAE Reconstruction Threshold   : {vae_stats['threshold_used']:.4f}")
    print(f"  VAE Flagged Suspicious         : {vae_stats['suspicious_records']:,} ({vae_stats['suspicious_percentage']}%)")
    print(f"  VAE Flagged Normal             : {vae_stats['normal_records']:,}")
    print(f"  RF Predicted Attacks           : {rf_stats['predicted_attacks']:,} ({rf_stats['attack_percentage']}%)")
    print(f"  RF Predicted Normal            : {rf_stats['predicted_normal']:,}")
    print("=" * 65)

    display_cols = ["id", "proto", "service", "state", "sbytes", "dbytes"]
    available_cols = [c for c in display_cols if c in df.columns]
    
    sample_view = df[available_cols].copy()
    sample_view["Anomaly_Score"] = vae_annotated["anomaly_score"]
    sample_view["VAE_Status"] = vae_annotated["predicted_status"]
    sample_view["RF_Class"] = rf_annotated["rf_predicted_class"]
    sample_view["RF_Attack_Prob"] = (rf_annotated["rf_attack_probability"] * 100).round(1).astype(str) + "%"

    print(f"\n[+] First {min(rows, total_records)} Inspected Flow Records:\n")
    print(sample_view.head(rows).to_string(index=False))
    print("\n" + "-" * 65)


def run_metrics():
    if not METRICS_FILE.exists():
        print("[!] No metrics found. Run 'python run_pipeline.py' first.")
        return

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    vae_m = data.get("vae_metrics", {})
    rf_m = data.get("rf_metrics", {})

    print("\n" + "=" * 68)
    print("  NETSHIELD AI — MODEL COMPARATIVE BENCHMARK METRICS (TERMINAL)")
    print("=" * 68)
    print(f"  {'Metric':<28} | {'VAE (Unsupervised)':<16} | {'Random Forest':<15}")
    print("  " + "-" * 64)

    rows = [
        ("Paradigm", "Normal-Only VAE", "Supervised RF"),
        ("Accuracy", f"{vae_m.get('accuracy', 0)*100:.2f}%", f"{rf_m.get('accuracy', 0)*100:.2f}%"),
        ("Precision", f"{vae_m.get('precision', 0)*100:.2f}%", f"{rf_m.get('precision', 0)*100:.2f}%"),
        ("Recall (Detection Rate)", f"{vae_m.get('recall', 0)*100:.2f}%", f"{rf_m.get('recall', 0)*100:.2f}%"),
        ("F1-Score", f"{vae_m.get('f1_score', 0):.4f}", f"{rf_m.get('f1_score', 0):.4f}"),
        ("ROC-AUC", f"{vae_m.get('roc_auc', 'N/A')}", f"{rf_m.get('roc_auc', 'N/A')}"),
        ("PR-AUC", f"{vae_m.get('pr_auc', 'N/A')}", f"{rf_m.get('pr_auc', 'N/A')}"),
        ("False Positive Rate (FPR)", f"{vae_m.get('false_positive_rate', 0)*100:.2f}%", f"{rf_m.get('false_positive_rate', 0)*100:.2f}%"),
        ("Latency per Record", f"{vae_m.get('latency_per_record_ms', 0):.4f} ms", f"{rf_m.get('latency_per_record_ms', 0):.4f} ms"),
        ("Throughput (Records/sec)", f"{vae_m.get('throughput_records_per_sec', 0):,.0f}", f"{rf_m.get('throughput_records_per_sec', 0):,.0f}"),
    ]

    for label, vae_val, rf_val in rows:
        print(f"  {label:<28} | {vae_val:<16} | {rf_val:<15}")
    print("=" * 68 + "\n")



def run_interactive():
    print_banner()
    while True:
        print("\nSelect an action:")
        print("  [1] Run Inference on sample traffic (CLI prediction)")
        print("  [2] View Comparative Model Benchmark Metrics")
        print("  [3] Retrain Models & Pipeline (run_pipeline.py)")
        print("  [4] Run Automated Unit Tests (pytest)")
        print("  [5] Launch Streamlit Web Dashboard")
        print("  [q] Quit")
        choice = input("\nEnter choice (1-5 or q): ").strip().lower()

        if choice == "1":
            run_predict()
        elif choice == "2":
            run_metrics()
        elif choice == "3":
            from run_pipeline import run_full_pipeline
            run_full_pipeline()
        elif choice == "4":
            os.system(f"{sys.executable} -m pytest tests/")
        elif choice == "5":
            print("\n[+] Launching Streamlit dashboard...")
            os.system(f"{sys.executable} -m streamlit run app.py")
        elif choice in ("q", "quit", "exit"):
            print("Exiting NetShield AI CLI.")
            break
        else:
            print("[!] Invalid option. Please choose 1-5 or q.")


def main():
    parser = argparse.ArgumentParser(
        description="NetShield AI Terminal Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run model inference on CSV records")
    predict_parser.add_argument("--file", "-f", type=str, default=None, help="Path to input CSV file")
    predict_parser.add_argument("--rows", "-r", type=int, default=10, help="Number of records to display")
    predict_parser.add_argument("--threshold", "-t", type=float, default=None, help="Custom VAE anomaly threshold")

    # Metrics command
    subparsers.add_parser("metrics", help="Display comparative evaluation benchmark metrics")

    # Train command
    subparsers.add_parser("train", help="Run end-to-end training pipeline")

    # Test command
    subparsers.add_parser("test", help="Run unit test suite")

    # Serve command
    subparsers.add_parser("serve", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    if args.command == "predict":
        print_banner()
        run_predict(file_path=args.file, rows=args.rows, threshold=args.threshold)
    elif args.command == "metrics":
        print_banner()
        run_metrics()
    elif args.command == "train":
        print_banner()
        from run_pipeline import run_full_pipeline
        run_full_pipeline()
    elif args.command == "test":
        print_banner()
        os.system(f"{sys.executable} -m pytest tests/")
    elif args.command == "serve":
        print_banner()
        print("[+] Launching Streamlit web application on http://localhost:8501 ...")
        os.system(f"{sys.executable} -m streamlit run app.py")
    else:
        # Default to predict demo if non-interactive or interactive if terminal
        if sys.stdin and sys.stdin.isatty():
            run_interactive()
        else:
            print_banner()
            run_predict()


if __name__ == "__main__":
    main()
