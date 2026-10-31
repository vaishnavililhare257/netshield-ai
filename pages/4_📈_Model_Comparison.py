"""
NetShield AI — Academic Model Comparison & Benchmarking
Evaluates VAE Unsupervised Anomaly Detection vs Random Forest Supervised Baseline
on identical held-out test records. Computes ROC-AUC, PR-AUC, Confusion Matrices, and Latency.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEST_DATA_PATH, VAE_MODEL_FILE, RF_MODEL_FILE, PREPROCESSOR_FILE
from src.ui_theme import apply_cyber_theme, render_header, render_metric_card
from src.data_loader import load_dataset
from src.preprocessing import NIDSPreprocessor
from src.train_vae import load_trained_vae
from src.train_random_forest import load_trained_random_forest
from src.evaluation import evaluate_binary_classifier, measure_inference_latency, evaluate_per_category_recall, save_metrics_summary
from data.download_unsw import ensure_data_available

apply_cyber_theme()

render_header(
    title="Model Comparison & Evaluation Benchmarks",
    subtitle="Rigorous empirical comparison between Unsupervised VAE and Supervised Random Forest on held-out test traffic."
)

if not VAE_MODEL_FILE.exists() or not RF_MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
    st.warning("⚠️ Both the VAE and Random Forest models must be trained before running model comparison.")
    st.info("Please visit `2_⚙️_Model_Training` to train both models first.")
    st.stop()

if not TEST_DATA_PATH.exists():
    st.warning("Testing dataset not found.")
    if st.button("Generate Benchmark Test Set"):
        ensure_data_available()
        st.rerun()
    st.stop()

st.subheader("1. Run Benchmark Evaluation on Test Records")
test_records_count = st.slider("Select number of test records to evaluate:", 500, 15000, 3000, step=500)

eval_btn = st.button("📊 Run Evaluation & Latency Benchmark", type="primary")

if eval_btn or "comparison_results" in st.session_state:
    if eval_btn:
        with st.spinner("Executing evaluation pipeline on test data..."):
            test_df = load_dataset(TEST_DATA_PATH, nrows=test_records_count)
            preprocessor = NIDSPreprocessor.load(PREPROCESSOR_FILE)
            
            x_test = preprocessor.transform(test_df)
            y_test = test_df["label"].values.astype(int)
            y_cats = test_df["attack_cat"].values if "attack_cat" in test_df.columns else None

            # Load models
            vae, vae_cfg = load_trained_vae()
            rf = load_trained_random_forest()

            # 1. VAE Anomaly Scoring & Predictions
            vae_scores = vae.get_reconstruction_error(x_test)
            vae_preds = (vae_scores > vae.threshold).astype(int)
            vae_metrics = evaluate_binary_classifier(y_test, vae_preds, y_scores=vae_scores, model_name="VAE Anomaly Detector")
            vae_latency = measure_inference_latency(lambda x: vae.get_reconstruction_error(x), x_test[:200])
            vae_metrics["training_time"] = vae_cfg.get("training_time_sec", "N/A")
            vae_metrics.update(vae_latency)

            # 2. Random Forest Predictions
            rf_preds = rf.predict(x_test)
            rf_proba = rf.predict_proba(x_test)
            rf_scores = rf_proba[:, 1] if rf_proba.shape[1] > 1 else rf_proba[:, 0]
            rf_metrics = evaluate_binary_classifier(y_test, rf_preds, y_scores=rf_scores, model_name="Random Forest Baseline")
            rf_latency = measure_inference_latency(lambda x: rf.predict(x), x_test[:200])
            rf_metrics["training_time"] = getattr(rf, "training_time", "N/A")
            rf_metrics.update(rf_latency)

            # 3. Per category recall
            per_cat_vae = evaluate_per_category_recall(y_cats, vae_preds) if y_cats is not None else None
            per_cat_rf = evaluate_per_category_recall(y_cats, rf_preds) if y_cats is not None else None

            comparison_data = {
                "vae_metrics": vae_metrics,
                "rf_metrics": rf_metrics,
                "per_cat_vae": per_cat_vae,
                "per_cat_rf": per_cat_rf
            }
            st.session_state["comparison_results"] = comparison_data
            try:
                save_metrics_summary(comparison_data)
            except Exception as e:
                pass

    else:
        comparison_data = st.session_state["comparison_results"]
        vae_metrics = comparison_data["vae_metrics"]
        rf_metrics = comparison_data["rf_metrics"]
        per_cat_vae = comparison_data.get("per_cat_vae")
        per_cat_rf = comparison_data.get("per_cat_rf")

    st.markdown("---")
    st.subheader("2. Metric Comparison Summary")

    # High-level comparison table
    summary_table = pd.DataFrame([
        {
            "Model Paradigm": "VAE (Unsupervised)",
            "Accuracy": f"{vae_metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{vae_metrics['precision'] * 100:.2f}%",
            "Recall (Detection Rate)": f"{vae_metrics['recall'] * 100:.2f}%",
            "F1-Score": f"{vae_metrics['f1_score']:.4f}",
            "ROC-AUC": f"{vae_metrics.get('roc_auc', 'N/A')}",
            "False Positive Rate": f"{vae_metrics['false_positive_rate'] * 100:.2f}%",
            "Latency / Record": f"{vae_metrics['latency_per_record_ms']:.4f} ms"
        },
        {
            "Model Paradigm": "Random Forest (Supervised)",
            "Accuracy": f"{rf_metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{rf_metrics['precision'] * 100:.2f}%",
            "Recall (Detection Rate)": f"{rf_metrics['recall'] * 100:.2f}%",
            "F1-Score": f"{rf_metrics['f1_score']:.4f}",
            "ROC-AUC": f"{rf_metrics.get('roc_auc', 'N/A')}",
            "False Positive Rate": f"{rf_metrics['false_positive_rate'] * 100:.2f}%",
            "Latency / Record": f"{rf_metrics['latency_per_record_ms']:.4f} ms"
        }
    ])
    st.dataframe(summary_table, use_container_width=True)

    # Detailed Tabs: ROC / PR Curves, Confusion Matrices, Category Breakdown, Trade-off Discussion
    t_roc, t_cm, t_cats, t_disc = st.tabs([
        "📈 ROC & Precision-Recall Curves",
        "🔲 Confusion Matrices",
        "🎯 Per-Attack Category Recall",
        "⚖️ Academic Trade-off Discussion"
    ])

    with t_roc:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Receiver Operating Characteristic (ROC)")
            fig_roc = go.Figure()
            if "roc_curve" in vae_metrics and vae_metrics["roc_curve"]:
                fig_roc.add_trace(go.Scatter(
                    x=vae_metrics["roc_curve"]["fpr"],
                    y=vae_metrics["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"VAE (AUC = {vae_metrics.get('roc_auc', 'N/A')})",
                    line=dict(color="#38bdf8", width=2)
                ))
            if "roc_curve" in rf_metrics and rf_metrics["roc_curve"]:
                fig_roc.add_trace(go.Scatter(
                    x=rf_metrics["roc_curve"]["fpr"],
                    y=rf_metrics["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"Random Forest (AUC = {rf_metrics.get('roc_auc', 'N/A')})",
                    line=dict(color="#10b981", width=2)
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="#64748b", dash="dash")))
            fig_roc.update_layout(template="plotly_dark", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)")
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_c2:
            st.markdown("#### Precision-Recall Curve")
            fig_pr = go.Figure()
            if "pr_curve" in vae_metrics and vae_metrics["pr_curve"]:
                fig_pr.add_trace(go.Scatter(
                    x=vae_metrics["pr_curve"]["recall"],
                    y=vae_metrics["pr_curve"]["precision"],
                    mode="lines",
                    name=f"VAE (PR-AUC = {vae_metrics.get('pr_auc', 'N/A')})",
                    line=dict(color="#38bdf8", width=2)
                ))
            if "pr_curve" in rf_metrics and rf_metrics["pr_curve"]:
                fig_pr.add_trace(go.Scatter(
                    x=rf_metrics["pr_curve"]["recall"],
                    y=rf_metrics["pr_curve"]["precision"],
                    mode="lines",
                    name=f"Random Forest (PR-AUC = {rf_metrics.get('pr_auc', 'N/A')})",
                    line=dict(color="#10b981", width=2)
                ))
            fig_pr.update_layout(template="plotly_dark", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

    with t_cm:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### VAE Confusion Matrix")
            cm_v = vae_metrics["confusion_matrix"]
            fig_cm_v = px.imshow(
                cm_v,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Normal (0)", "Suspicious (1)"],
                y=["Normal (0)", "Attack (1)"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm_v.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cm_v, use_container_width=True)
            st.caption(f"FPR: {vae_metrics['false_positive_rate']*100:.2f}% | FP: {vae_metrics['false_positives']} | FN: {vae_metrics['false_negatives']}")

        with col_m2:
            st.markdown("#### Random Forest Confusion Matrix")
            cm_r = rf_metrics["confusion_matrix"]
            fig_cm_r = px.imshow(
                cm_r,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Normal (0)", "Attack (1)"],
                y=["Normal (0)", "Attack (1)"],
                text_auto=True,
                color_continuous_scale="Greens"
            )
            fig_cm_r.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cm_r, use_container_width=True)
            st.caption(f"FPR: {rf_metrics['false_positive_rate']*100:.2f}% | FP: {rf_metrics['false_positives']} | FN: {rf_metrics['false_negatives']}")

    with t_cats:
        st.markdown("#### Per-Attack Category Detection Recall Comparison")
        if per_cat_vae is not None and per_cat_rf is not None:
            merged_cat = pd.merge(
                per_cat_vae[["Attack_Category", "Total_Instances", "Detection_Recall"]].rename(columns={"Detection_Recall": "VAE_Recall"}),
                per_cat_rf[["Attack_Category", "Detection_Recall"]].rename(columns={"Detection_Recall": "RF_Recall"}),
                on="Attack_Category"
            )
            st.dataframe(merged_cat, use_container_width=True)
            
            fig_cats = px.bar(
                merged_cat.melt(id_vars=["Attack_Category", "Total_Instances"], value_vars=["VAE_Recall", "RF_Recall"], var_name="Model", value_name="Recall"),
                x="Attack_Category",
                y="Recall",
                color="Model",
                barmode="group",
                color_discrete_map={"VAE_Recall": "#38bdf8", "RF_Recall": "#10b981"},
                title="Detection Recall by Attack Family"
            )
            fig_cats.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cats, use_container_width=True)
        else:
            st.info("Attack category labels are not present in this test subset.")

    with t_disc:
        st.subheader("Academic Comparison & Key Findings")
        st.markdown("""
        ### Unsupervised VAE vs Supervised Random Forest Analysis

        1. **Zero-Day Attack Generalization:**
           - **VAE Advantage:** Does not require historical attack signatures. If a novel zero-day attack produces traffic features 
             outside the normal manifold, the VAE flags it through reconstruction failure.
           - **Random Forest Limitation:** May fail to classify unseen attack patterns if their feature distributions differ 
             from the known attack classes encountered during training.

        2. **Detection Precision & False Positive Rate (FPR):**
           - **Supervised Advantage:** Random Forest achieves higher precision and lower FPR because it explicitly maps 
             the non-linear decision boundary separating normal and malicious clusters.
           - **VAE Challenge:** Because threshold selection is heuristic (based on normal validation error quantiles), 
             a 95th percentile threshold by definition yields a ~5% baseline false alarm rate on benign variations.

        3. **Inference Latency:**
           - Both models exhibit sub-millisecond per-packet inference times, making them suitable for near-line network flow monitoring.
        """)
