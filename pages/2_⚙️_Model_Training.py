"""
NetShield AI — Interactive Model Training Laboratory
Enables interactive training of the Unsupervised Variational Autoencoder,
the Supervised Random Forest baseline, and the Restricted Boltzmann Machine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, VAE_DEFAULTS, RF_DEFAULTS, RBM_DEFAULTS,
    VAE_MODEL_FILE, RF_MODEL_FILE, RBM_MODEL_FILE, PREPROCESSOR_FILE
)
from src.ui_theme import apply_cyber_theme, render_header, render_metric_card
from src.data_loader import load_dataset
from src.preprocessing import prepare_datasets_for_training, NIDSPreprocessor
from src.train_vae import train_vae_pipeline
from src.train_random_forest import train_random_forest_pipeline
from src.train_rbm import train_rbm_pipeline
from data.download_unsw import check_dataset_status, ensure_data_available

apply_cyber_theme()

render_header(
    title="Model Training Laboratory",
    subtitle="Configure hyperparameters, train models on real UNSW-NB15 partitions, and serialize artifacts."
)

# Verify dataset readiness
ds_status = check_dataset_status()
if not ds_status["train_exists"]:
    st.warning("⚠️ UNSW-NB15 training dataset is not detected in `data/`.")
    if st.button("Generate Benchmark Training Set Now"):
        with st.spinner("Preparing dataset..."):
            ensure_data_available()
            st.success("Dataset ready!")
            st.rerun()
    st.stop()

# Model Selection Tabs
tabs = st.tabs([
    "🛡️ 1. Train Variational Autoencoder (VAE)",
    "🌲 2. Train Random Forest Baseline",
    "🧠 3. Train Restricted Boltzmann Machine (RBM)"
])

# =====================================================================
# TAB 1: VARIATIONAL AUTOENCODER TRAINING
# =====================================================================
with tabs[0]:
    st.subheader("Unsupervised VAE Anomaly Detection Training")
    st.markdown("""
    **Core DL-II Principle:** The VAE is trained **exclusively on Normal (benign) network traffic**. 
    It learns a compressed probabilistic manifold $q_\phi(z|x)$ parameterized by latent mean $\mu$ and variance $\sigma^2$.
    During inference, attack traffic produces high reconstruction error.
    """)

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        st.markdown("#### Hyperparameter Configuration")
        v_latent_dim = st.selectbox("Latent Dimension ($z$)", [8, 16, 32], index=1, help="Bottleneck representation dimension")
        v_epochs = st.slider("Training Epochs", 2, 30, 8, step=1)
        v_batch_size = st.selectbox("Batch Size", [64, 128, 256, 512], index=2)
        v_lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        v_beta = st.slider(r"Beta ($\beta$) KL Weight", 0.1, 5.0, 1.0, step=0.1, help="Higher beta enforces stricter prior regularization")
        v_quantile = st.slider("Validation Threshold Quantile", 0.80, 0.99, 0.95, step=0.01, help="Threshold heuristic on normal validation reconstruction error")
        v_max_records = st.number_input("Max Normal Training Samples", min_value=1000, max_value=50000, value=12000, step=1000)

        train_vae_btn = st.button("🚀 Start VAE Training", key="btn_train_vae")

    with col_v2:
        if train_vae_btn:
            status_box = st.empty()
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()

            try:
                status_box.info("Loading training and testing datasets...")
                train_df = load_dataset(TRAIN_DATA_PATH)
                test_df = load_dataset(TEST_DATA_PATH) if TEST_DATA_PATH.exists() else None

                status_box.info("Executing leakage-free preprocessing & normal traffic filtering...")
                prepared = prepare_datasets_for_training(
                    train_df=train_df,
                    test_df=test_df,
                    val_ratio=0.15,
                    max_normal_train=int(v_max_records),
                    scaler_type="standard"
                )

                # Persist preprocessor
                prepared["preprocessor"].save()
                st.session_state["preprocessor"] = prepared["preprocessor"]

                status_box.info(f"Training VAE on {len(prepared['x_vae_train']):,} normal records across {v_epochs} epochs...")

                epoch_history = {"epoch": [], "train_loss": [], "recon_loss": [], "val_loss": []}

                def update_progress(ep, total_eps, tr_loss, recon_l, val_l):
                    progress_bar.progress(ep / total_eps)
                    epoch_history["epoch"].append(ep)
                    epoch_history["train_loss"].append(tr_loss)
                    epoch_history["recon_loss"].append(recon_l)
                    epoch_history["val_loss"].append(val_l)
                    
                    df_plot = pd.DataFrame(epoch_history)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_plot["epoch"], y=df_plot["train_loss"], mode="lines+markers", name="Total Train Loss", line=dict(color="#38bdf8", width=2)))
                    fig.add_trace(go.Scatter(x=df_plot["epoch"], y=df_plot["recon_loss"], mode="lines", name="Reconstruction MSE", line=dict(color="#10b981", dash="dot")))
                    fig.add_trace(go.Scatter(x=df_plot["epoch"], y=df_plot["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#f59e0b", width=2)))
                    fig.update_layout(template="plotly_dark", title="Real-Time VAE Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                trained_vae, results = train_vae_pipeline(
                    x_train=prepared["x_vae_train"],
                    x_val=prepared["x_vae_val"],
                    hidden_dim1=64,
                    hidden_dim2=32,
                    latent_dim=v_latent_dim,
                    learning_rate=v_lr,
                    batch_size=v_batch_size,
                    epochs=v_epochs,
                    beta=v_beta,
                    threshold_quantile=v_quantile,
                    progress_callback=update_progress
                )

                status_box.success(f"✅ VAE Training Successfully Completed in {results['training_time']} seconds!")
                
                # Display final threshold and KPI metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_metric_card("Reconstruction Threshold", f"{results['threshold']:.5f}", f"{int(v_quantile*100)}th quantile on normal val", "success")
                with m2:
                    render_metric_card("Training Duration", f"{results['training_time']} s", f"{v_epochs} epochs @ batch {v_batch_size}")
                with m3:
                    render_metric_card("Compute Device", results["device"].upper(), "Hardware accelerator")

            except Exception as e:
                status_box.error(f"Training failed: {e}")
        else:
            if VAE_MODEL_FILE.exists():
                st.success("✅ A trained VAE model artifact (`vae_model.pt`) already exists. You can retrain with new parameters or proceed directly to Anomaly Detection.")
            else:
                st.info("Configure parameters on the left and click 'Start VAE Training'.")

# =====================================================================
# TAB 2: RANDOM FOREST BASELINE TRAINING
# =====================================================================
with tabs[1]:
    st.subheader("Supervised Random Forest Classifier Baseline")
    st.markdown("""
    **Supervised Baseline:** Unlike the VAE, Random Forest trains on **both Normal and Attack labels**.
    It builds an ensemble of orthogonal decision trees to distinguish known attacks from benign traffic.
    """)

    col_r1, col_r2 = st.columns([1, 2])

    with col_r1:
        st.markdown("#### Hyperparameter Configuration")
        rf_trees = st.slider("Number of Trees (n_estimators)", 20, 200, 100, step=10)
        rf_depth = st.slider("Max Tree Depth", 5, 30, 16, step=1)
        rf_class_weight = st.selectbox("Class Weighting", ["balanced", "balanced_subsample", None], index=0)
        rf_max_samples = st.number_input("Max Labelled Training Samples", min_value=1000, max_value=50000, value=15000, step=1000)

        train_rf_btn = st.button("🚀 Start Random Forest Training", key="btn_train_rf")

    with col_r2:
        if train_rf_btn:
            status_box_rf = st.empty()
            try:
                status_box_rf.info("Preparing labelled training partitions...")
                train_df = load_dataset(TRAIN_DATA_PATH)
                if len(train_df) > int(rf_max_samples):
                    train_df = train_df.sample(n=int(rf_max_samples), random_state=42)

                # Ensure preprocessor is fitted
                if PREPROCESSOR_FILE.exists():
                    preprocessor = NIDSPreprocessor.load(PREPROCESSOR_FILE)
                else:
                    preprocessor = NIDSPreprocessor()
                    preprocessor.fit(train_df)
                    preprocessor.save(PREPROCESSOR_FILE)

                x_rf = preprocessor.transform(train_df)
                y_rf = train_df["label"].values.astype(int)

                status_box_rf.info(f"Training Random Forest ({rf_trees} trees, max_depth={rf_depth})...")
                rf_model, rf_results = train_random_forest_pipeline(
                    x_train=x_rf,
                    y_train=y_rf,
                    feature_names=preprocessor.transformed_feature_names_,
                    n_estimators=rf_trees,
                    max_depth=rf_depth,
                    class_weight=rf_class_weight
                )

                status_box_rf.success(f"✅ Random Forest Training Completed in {rf_results['training_time']} seconds!")

                st.markdown("#### Top 10 Most Discriminative Features")
                top_feats = rf_results["top_features"]
                feat_df = pd.DataFrame(list(top_feats.items()), columns=["Feature", "Importance"]).head(10).sort_values(by="Importance", ascending=True)
                fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
                fig_feat.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_feat, use_container_width=True)

            except Exception as e:
                status_box_rf.error(f"Random Forest training failed: {e}")
        else:
            if RF_MODEL_FILE.exists():
                st.success("✅ A trained Random Forest artifact (`random_forest_model.joblib`) is already available.")
            else:
                st.info("Click 'Start Random Forest Training' to train the supervised baseline.")

# =====================================================================
# TAB 3: RESTRICTED BOLTZMANN MACHINE (RBM)
# =====================================================================
with tabs[2]:
    st.subheader("Restricted Boltzmann Machine (RBM) Syllabus Extension")
    st.markdown("""
    **Deep Learning-II Concept:** RBMs are energy-based undirected bipartite graphical models. 
    They are trained via **Contrastive Divergence (CD-k)** using alternating Gibbs sampling 
    between visible units ($v$) and hidden representations ($h$).
    """)

    col_rb1, col_rb2 = st.columns([1, 2])

    with col_rb1:
        st.markdown("#### RBM Configuration")
        rbm_hidden = st.selectbox("Hidden Units (Features)", [16, 32, 64], index=1)
        rbm_epochs = st.slider("RBM Epochs", 2, 20, 6, step=1)
        rbm_lr = st.select_slider("CD-1 Learning Rate", options=[0.001, 0.005, 0.01, 0.05], value=0.01)
        rbm_batch_size = st.selectbox("RBM Batch Size", [64, 128, 256], index=1)

        train_rbm_btn = st.button("🚀 Start RBM Training", key="btn_train_rbm")

    with col_rb2:
        if train_rbm_btn:
            status_box_rbm = st.empty()
            chart_placeholder_rbm = st.empty()
            try:
                status_box_rbm.info("Loading normal traffic records for RBM feature learning...")
                train_df = load_dataset(TRAIN_DATA_PATH)
                normal_records = train_df[train_df["label"] == 0].head(8000)

                if PREPROCESSOR_FILE.exists():
                    preprocessor = NIDSPreprocessor.load(PREPROCESSOR_FILE)
                else:
                    preprocessor = NIDSPreprocessor()
                    preprocessor.fit(train_df)

                x_norm_raw = preprocessor.transform(normal_records)

                rbm_history = {"epoch": [], "recon_error": [], "free_energy": []}

                def update_rbm_progress(ep, total_eps, recon_e, free_en):
                    rbm_history["epoch"].append(ep)
                    rbm_history["recon_error"].append(recon_e)
                    rbm_history["free_energy"].append(free_en)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=rbm_history["epoch"], y=rbm_history["recon_error"], mode="lines+markers", name="Reconstruction MSE", line=dict(color="#f43f5e", width=2)))
                    fig.update_layout(template="plotly_dark", title="RBM Contrastive Divergence Reconstruction Error", xaxis_title="Epoch", yaxis_title="MSE Error")
                    chart_placeholder_rbm.plotly_chart(fig, use_container_width=True)

                rbm_model, rbm_res = train_rbm_pipeline(
                    x_train=x_norm_raw,
                    n_hidden=rbm_hidden,
                    learning_rate=rbm_lr,
                    k_steps=1,
                    batch_size=rbm_batch_size,
                    epochs=rbm_epochs,
                    progress_callback=update_rbm_progress
                )

                status_box_rbm.success(f"✅ RBM Contrastive Divergence Training Completed in {rbm_res['training_time']} seconds!")
            except Exception as e:
                status_box_rbm.error(f"RBM training failed: {e}")
        else:
            if RBM_MODEL_FILE.exists():
                st.success("✅ Trained RBM artifact (`rbm_model.pt`) is available.")
            else:
                st.info("Train the RBM to observe Gibbs Sampling and Contrastive Divergence diagnostics.")
