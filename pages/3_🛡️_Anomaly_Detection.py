"""
NetShield AI — Real-Time Anomaly Detection & Threat Scoring
Executes VAE reconstruction error inference on network traffic records,
applies anomaly decision thresholds, and provides CSV export capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEST_DATA_PATH, SAMPLE_DATA_PATH, VAE_MODEL_FILE, PREPROCESSOR_FILE
from src.ui_theme import apply_cyber_theme, render_header, render_metric_card
from src.data_loader import load_dataset
from src.inference import NetShieldInferenceEngine
from data.download_unsw import check_dataset_status, ensure_data_available

apply_cyber_theme()

render_header(
    title="Real-Time Anomaly Detection & Threat Scoring",
    subtitle="Evaluate network traffic flows using unsupervised VAE reconstruction error and dynamic thresholding."
)

# Check model availability
if not VAE_MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
    st.warning("⚠️ VAE Model or Preprocessor artifacts are not found. Please train the VAE first in '2_⚙️_Model_Training'.")
    st.stop()

# Initialize unified inference engine
try:
    engine = NetShieldInferenceEngine()
except Exception as e:
    st.error(f"Failed to load inference engine: {e}")
    st.stop()

st.subheader("1. Select Traffic Data for Anomaly Scoring")

data_source = st.radio(
    "Choose input traffic source:",
    ["Use Existing Test Dataset (data/UNSW_NB15_testing-set.csv)", "Upload Custom Network Traffic CSV", "Load Quick Benchmark Sample"],
    horizontal=True
)

traffic_df = None

if data_source == "Use Existing Test Dataset (data/UNSW_NB15_testing-set.csv)":
    if not TEST_DATA_PATH.exists():
        st.warning("Test dataset not found.")
        if st.button("Generate Benchmark Test Set"):
            ensure_data_available()
            st.rerun()
    else:
        sample_size = st.slider("Number of records to scan:", 100, 10000, 1500, step=100)
        traffic_df = load_dataset(TEST_DATA_PATH, nrows=sample_size)

elif data_source == "Upload Custom Network Traffic CSV":
    uploaded = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
    if uploaded is not None:
        try:
            traffic_df = pd.read_csv(uploaded)
            traffic_df.columns = [c.strip().lower() for c in traffic_df.columns]
            st.success(f"Loaded {len(traffic_df)} records from uploaded file.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

elif data_source == "Load Quick Benchmark Sample":
    ensure_data_available()
    traffic_df = load_dataset(SAMPLE_DATA_PATH, nrows=500)

if traffic_df is not None and not traffic_df.empty:
    st.markdown("---")
    st.subheader("2. Anomaly Scoring & Threshold Configuration")

    default_thresh = float(engine.vae_model.threshold)

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown(f"**Model Calibrated Threshold:** `{default_thresh:.5f}`")
        custom_thresh = st.number_input(
            "Decision Threshold (Reconstruction MSE)",
            min_value=0.00001,
            max_value=100.0,
            value=default_thresh,
            format="%.5f",
            help="Packets with reconstruction error exceeding this threshold are flagged as Suspicious."
        )
        run_detection = st.button("🔍 Run Anomaly Detection", type="primary")

    with col_t2:
        st.info(
            "💡 **How VAE Detects Anomalies:**\n"
            "Because the VAE was trained strictly on benign traffic patterns, benign packets reconstruct with low MSE. "
            "Attacks introduce anomalous combinations of packet lengths, TTLs, and flow rates that the VAE cannot compress faithfully, "
            "triggering an anomaly score spike above the threshold."
        )

    if run_detection or "latest_annotated_vae" in st.session_state:
        if run_detection:
            with st.spinner("Executing VAE inference and calculating reconstruction errors..."):
                try:
                    annotated_df, stats = engine.predict_vae(traffic_df, custom_threshold=custom_thresh)
                    st.session_state["latest_annotated_vae"] = annotated_df
                    st.session_state["latest_vae_stats"] = stats
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()
        else:
            annotated_df = st.session_state["latest_annotated_vae"]
            stats = st.session_state["latest_vae_stats"]

        st.markdown("---")
        st.subheader("3. Threat Detection Analytics")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Total Scanned Records", f"{stats['total_records']:,}")
        with m2:
            render_metric_card("Suspicious / Anomalies", f"{stats['suspicious_records']:,}", f"{stats['suspicious_percentage']}% of scanned", status="danger")
        with m3:
            render_metric_card("Normal Traffic", f"{stats['normal_records']:,}", f"{100 - stats['suspicious_percentage']:.2f}% benign", status="success")
        with m4:
            render_metric_card("Applied Threshold", f"{stats['threshold_used']:.5f}", f"Max Score: {stats['max_score']:.4f}")

        # Visualizations: Score distribution with threshold marker
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Reconstruction Score Distribution vs Decision Boundary")
            fig_hist = px.histogram(
                annotated_df,
                x="anomaly_score",
                nbins=50,
                color="predicted_status",
                color_discrete_map={"Normal": "#10b981", "Suspicious": "#ef4444"},
                title="Packet Reconstruction Error Spread"
            )
            fig_hist.add_vline(
                x=stats["threshold_used"],
                line_dash="dash",
                line_color="#38bdf8",
                line_width=2,
                annotation_text=f"Threshold ({stats['threshold_used']:.4f})",
                annotation_position="top right"
            )
            fig_hist.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_g2:
            st.markdown("#### Sequential Packet Anomaly Scores")
            sub_sample = annotated_df.head(400).reset_index()
            fig_scatter = px.scatter(
                sub_sample,
                x="index",
                y="anomaly_score",
                color="predicted_status",
                color_discrete_map={"Normal": "#10b981", "Suspicious": "#ef4444"},
                title="Packet Timeline Flow Anomaly Profile"
            )
            fig_scatter.add_hline(y=stats["threshold_used"], line_dash="dash", line_color="#38bdf8")
            fig_scatter.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Flagged Records Table & CSV Export
        st.markdown("---")
        st.subheader("4. Suspicious Records Audit & Export")

        filter_view = st.radio("Display Filter:", ["Suspicious Records Only", "All Records", "Normal Records Only"], horizontal=True)

        if filter_view == "Suspicious Records Only":
            display_df = annotated_df[annotated_df["predicted_status"] == "Suspicious"]
        elif filter_view == "Normal Records Only":
            display_df = annotated_df[annotated_df["predicted_status"] == "Normal"]
        else:
            display_df = annotated_df

        st.dataframe(display_df.head(100), use_container_width=True)

        csv_data = annotated_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Predictions CSV (with Anomaly Scores)",
            data=csv_data,
            file_name="netshield_vae_predictions.csv",
            mime="text/csv"
        )
