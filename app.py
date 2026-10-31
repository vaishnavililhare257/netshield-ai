"""
NetShield AI — Intelligent Network Intrusion Detection
Main Application Entry Point & Executive Overview Dashboard
"""

import streamlit as st
import torch
import json
from pathlib import Path

# Setup page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="NetShield AI | Deep Learning NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_DATA_PATH,
    VAE_MODEL_FILE, VAE_CONFIG_FILE, RF_MODEL_FILE, RBM_MODEL_FILE,
    PREPROCESSOR_FILE, METRICS_FILE
)
from src.ui_theme import apply_cyber_theme, render_header, render_metric_card
from data.download_unsw import check_dataset_status, ensure_data_available

# Apply global dark cybersecurity aesthetics
apply_cyber_theme()


def get_system_hardware():
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
    return "GPU (CUDA)" if has_cuda else "CPU", device_name


def main():
    # Sidebar: System Environment & Model Registry Status
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("NetShield AI")
    st.sidebar.caption("Unsupervised VAE + Supervised Baseline NIDS")
    st.sidebar.markdown("---")

    # Real Hardware Status
    hw_type, hw_name = get_system_hardware()
    st.sidebar.markdown(f"**Compute Device:** `{hw_type}`")
    st.sidebar.caption(f"{hw_name}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Artifact Status")

    # Dataset status
    ds_status = check_dataset_status()
    if ds_status["train_exists"] and ds_status["test_exists"]:
        st.sidebar.markdown("📊 **Dataset:** <span class='status-badge badge-ready'>Ready</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"Train: {ds_status['train_size_mb']}MB | Test: {ds_status['test_size_mb']}MB")
    else:
        st.sidebar.markdown("📊 **Dataset:** <span class='status-badge badge-warning'>Missing Full CSVs</span>", unsafe_allow_html=True)
        if st.sidebar.button("Generate Benchmark Subset"):
            with st.spinner("Generating UNSW-NB15 benchmark subset..."):
                ensure_data_available()
                st.sidebar.success("Benchmark data generated!")
                st.rerun()

    # Preprocessor status
    if PREPROCESSOR_FILE.exists():
        st.sidebar.markdown("⚙️ **Preprocessor:** <span class='status-badge badge-ready'>Fitted</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("⚙️ **Preprocessor:** <span class='status-badge badge-alert'>Not Fitted</span>", unsafe_allow_html=True)

    # VAE status
    if VAE_MODEL_FILE.exists() and VAE_CONFIG_FILE.exists():
        with open(VAE_CONFIG_FILE, "r") as f:
            v_cfg = json.load(f)
        st.sidebar.markdown("🛡️ **VAE Model:** <span class='status-badge badge-ready'>Trained</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"Latent Dim: {v_cfg.get('latent_dim', 16)} | Thresh: {v_cfg.get('threshold', 0):.4f}")
    else:
        st.sidebar.markdown("🛡️ **VAE Model:** <span class='status-badge badge-alert'>Untrained</span>", unsafe_allow_html=True)

    # Random Forest status
    if RF_MODEL_FILE.exists():
        st.sidebar.markdown("🌲 **Random Forest:** <span class='status-badge badge-ready'>Trained</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("🌲 **Random Forest:** <span class='status-badge badge-alert'>Untrained</span>", unsafe_allow_html=True)

    # RBM status
    if RBM_MODEL_FILE.exists():
        st.sidebar.markdown("🧠 **RBM Module:** <span class='status-badge badge-ready'>Trained</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("🧠 **RBM Module:** <span class='status-badge badge-warning'>Optional / Ready</span>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Navigation Tip:**\n"
        "Use the sidebar pages to navigate:\n"
        "1. **Dataset Explorer**: Inspect & upload traffic.\n"
        "2. **Model Training**: Train VAE, RF, and RBM.\n"
        "3. **Anomaly Detection**: Run VAE reconstruction.\n"
        "4. **Model Comparison**: Evaluation metrics & ROC.\n"
        "5. **DL-II Concepts**: Mathematical formulation & viva prep."
    )

    # Main Hero Banner
    render_header(
        title="NetShield AI — Intelligent Network Intrusion Detection",
        subtitle="Unsupervised Variational Autoencoder Anomaly Detection & Supervised Baseline for UNSW-NB15 Traffic Analysis"
    )

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tr_exist = ds_status["train_exists"]
        render_metric_card(
            label="Dataset Status",
            value="Ready" if tr_exist else "Pending",
            subtext=f"{ds_status['train_size_mb']} MB training data" if tr_exist else "Click sidebar to initialize",
            status="success" if tr_exist else "danger"
        )
    with col2:
        vae_ready = VAE_MODEL_FILE.exists()
        render_metric_card(
            label="VAE Anomaly Detector",
            value="Active" if vae_ready else "Untrained",
            subtext="Trained on Normal records only",
            status="success" if vae_ready else "danger"
        )
    with col3:
        rf_ready = RF_MODEL_FILE.exists()
        render_metric_card(
            label="Supervised Baseline",
            value="Active" if rf_ready else "Untrained",
            subtext="Random Forest (100 Trees)",
            status="success" if rf_ready else "danger"
        )
    with col4:
        render_metric_card(
            label="Hardware Engine",
            value=hw_type,
            subtext=hw_name[:22],
            status="normal"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Academic & Technical Overview Section
    tabs = st.tabs(["📌 Project Overview", "🏗️ System Architecture", "🔬 Research Methodology", "🚀 Quick Start Guide"])

    with tabs[0]:
        st.subheader("B.Tech CSE (AIML) Deep Learning-II Project Scope")
        st.markdown("""
        Modern computer networks encounter diverse, evolving cyber-attacks such as Denial of Service (DoS), 
        Backdoors, Exploits, Fuzzers, and Worms. Traditional rule-based Intrusion Detection Systems (IDS) 
        suffer from high false-negative rates when faced with **zero-day attacks**.

        **NetShield AI** implements a two-fold paradigm:
        1. **Primary Deep Learning Model (Unsupervised VAE):**
           Learns the manifold of benign network traffic ($P(x | \\text{Normal})$). During inference, anomalous traffic 
           deviates from the learned latent distribution, resulting in high reconstruction error $S(x) = \\frac{1}{D} \\sum_{j=1}^D (x_j - \\hat{x}_j)^2$.
        2. **Supervised Baseline (Random Forest):**
           A discriminative ensemble model trained on labelled benign and attack samples to establish a performance benchmark.
        3. **Syllabus Extension (Restricted Boltzmann Machine):**
           Energy-based model trained with Contrastive Divergence (CD-1) to demonstrate representation learning and visible reconstruction.
        """)

        st.markdown("### Primary Dataset: UNSW-NB15")
        st.markdown("""
        Created by the Cyber Range Lab of ACCS (Australian Centre for Cyber Security), UNSW-NB15 comprises 
        **45 features** capturing raw packet headers, flow statistics, content indicators, and behavioral metrics 
        across 9 attack families and normal traffic.
        """)

    with tabs[1]:
        st.subheader("End-to-End Pipeline Architecture")
        st.markdown("""
        The system enforces strict **data leakage prevention**:
        - Column `id` is dropped to avoid identifier bias.
        - Target labels (`label`, `attack_cat`) are isolated from feature vectors.
        - Imputers, encoders, and scalers are fitted **strictly** on the training partition.
        - VAE anomaly thresholds are derived exclusively from **normal validation data quantiles**, never test labels.
        """)

        st.code("""
        [ UNSW-NB15 Network Traffic CSV ]
                       │
                       ▼
        [ Data Cleaning & Schema Validation ] (data_loader.py)
                       │
                       ▼
        [ Feature Encoding & Standard Scaling ] (preprocessing.py)
                       │
          ┌────────────┴───────────────────────────┐
          │ (Normal-only Records)                  │ (Labelled Records)
          ▼                                        ▼
    [ VAE Model Training ]               [ Random Forest Training ]
    - Encoder: D -> 64 -> 32             - 100 Estimators
    - Latent Bottleneck: mu, logvar      - Balanced Class Weights
    - Reparameterization: z = mu + eps*std
    - Decoder: 32 -> 64 -> D
          │                                        │
          ▼                                        ▼
    [ Validation Thresholding (95th %ile) ]        │
          │                                        │
          └────────────┬───────────────────────────┘
                       │
                       ▼
        [ Unified Evaluation & Latency Benchmark ] (evaluation.py)
                       │
                       ▼
        [ Interactive Cybersecurity Dashboard ] (Streamlit App)
        """, language="text")

    with tabs[2]:
        st.subheader("Key Deep Learning Mathematical Foundations")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 1. VAE Objective Function (ELBO)")
            st.latex(r"\mathcal{L}_{\text{VAE}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))")
            st.markdown("""
            - **Reconstruction Loss**: Mean Squared Error measuring fidelity between original flow $x$ and decoder reconstruction $\hat{x}$.
            - **KL Divergence**: Regularizes the approximate posterior $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ toward standard prior $\mathcal{N}(0, I)$.
            - **Beta Weight**: Controls the degree of latent regularization vs reconstruction sharpness.
            """)

        with col_m2:
            st.markdown("#### 2. Reparameterization Trick")
            st.latex(r"z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)")
            st.markdown("""
            Direct sampling from $\mathcal{N}(\mu, \sigma^2)$ creates a stochastic node with no derivative with respect to $\mu$ and $\sigma$.
            The reparameterization trick isolates the stochasticity into an external parameter-free random variable $\epsilon$, 
            allowing backpropagation via standard automatic differentiation.
            """)

    with tabs[3]:
        st.subheader("Quick Start & Execution Steps")
        st.markdown("""
        Follow these steps to run the complete project:
        1. **Ensure Dataset is Ready**: If you do not have the full 150MB UNSW-NB15 files yet, click **Generate Benchmark Subset** in the sidebar.
        2. **Navigate to `2_⚙️_Model_Training`**:
           - Click **Train VAE Anomaly Detector** to train the deep learning model on normal traffic.
           - Click **Train Random Forest Baseline** to train the supervised classifier.
           - (Optional) Train the RBM module to observe Contrastive Divergence.
        3. **Navigate to `3_🛡️_Anomaly_Detection`**:
           - Upload test records or use sample traffic to obtain real-time anomaly predictions, reconstruction scores, and CSV exports.
        4. **Navigate to `4_📈_Model_Comparison`**:
           - Review side-by-side ROC curves, Precision-Recall curves, confusion matrices, and latency measurements.
        5. **Navigate to `5_🧠_DL_II_Concepts`**:
           - Explore latent space visualizers, loss breakdown, and oral viva questions with complete technical answers.
        """)

    st.markdown("---")
    st.caption("NetShield AI — Academic Research & Educational Prototype | B.Tech CSE (AIML) Deep Learning-II Laboratory")


if __name__ == "__main__":
    main()
