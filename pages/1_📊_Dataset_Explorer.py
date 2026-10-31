"""
NetShield AI — Dataset Explorer
Allows users to inspect UNSW-NB15 records, analyze distributions, review attack categories,
and upload custom network traffic datasets with schema validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_DATA_PATH, TARGET_BINARY, TARGET_MULTICLASS
from src.ui_theme import apply_cyber_theme, render_header, render_metric_card
from src.data_loader import load_dataset, get_dataset_summary, validate_dataset_schema
from data.download_unsw import check_dataset_status, ensure_data_available

apply_cyber_theme()

render_header(
    title="Dataset Explorer & Traffic Profiler",
    subtitle="Inspect UNSW-NB15 flow characteristics, distributions, missing values, and attack class balance."
)

# Source selector: Existing local files vs Custom User Upload
st.subheader("1. Select Dataset Source")
source_option = st.radio(
    "Choose data source to analyze:",
    ["UNSW-NB15 Training Set (Local)", "UNSW-NB15 Testing Set (Local)", "Upload Custom Network Traffic CSV"],
    horizontal=True
)

df = None
selected_path_name = ""

if source_option == "UNSW-NB15 Training Set (Local)":
    if not TRAIN_DATA_PATH.exists():
        st.warning(f"Training dataset not found at `{TRAIN_DATA_PATH}`.")
        if st.button("Generate Benchmark Training Set Now"):
            ensure_data_available()
            st.rerun()
    else:
        selected_path_name = str(TRAIN_DATA_PATH)
        max_rows = st.slider("Maximum records to load for inspection:", 500, 50000, 5000, step=500)
        try:
            with st.spinner("Loading training data..."):
                df = load_dataset(TRAIN_DATA_PATH, nrows=max_rows)
        except Exception as e:
            st.error(f"Error loading training data: {e}")

elif source_option == "UNSW-NB15 Testing Set (Local)":
    if not TEST_DATA_PATH.exists():
        st.warning(f"Testing dataset not found at `{TEST_DATA_PATH}`.")
        if st.button("Generate Benchmark Testing Set Now"):
            ensure_data_available()
            st.rerun()
    else:
        selected_path_name = str(TEST_DATA_PATH)
        max_rows = st.slider("Maximum records to load for inspection:", 500, 30000, 3000, step=500)
        try:
            with st.spinner("Loading testing data..."):
                df = load_dataset(TEST_DATA_PATH, nrows=max_rows)
        except Exception as e:
            st.error(f"Error loading testing data: {e}")

elif source_option == "Upload Custom Network Traffic CSV":
    uploaded_file = st.file_uploader("Upload a network flow CSV (e.g., UNSW-NB15 formatted)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.strip().lower() for c in df.columns]
            selected_path_name = uploaded_file.name
            st.success(f"Successfully loaded uploaded file: `{uploaded_file.name}` ({len(df)} records)")
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")

if df is not None and not df.empty:
    st.markdown("---")
    st.subheader(f"2. Dataset Summary ({selected_path_name})")
    
    summary = get_dataset_summary(df)
    schema_val = validate_dataset_schema(df, is_training=False)

    # Metric summary row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Rows", f"{summary['total_records']:,}", f"Features: {summary['total_columns']}")
    with c2:
        render_metric_card("Memory Usage", f"{summary['memory_mb']} MB", "Pandas in-memory footprint")
    with c3:
        if summary.get("has_binary_label"):
            norm_pct = summary.get("normal_pct", 0)
            render_metric_card("Normal Traffic", f"{summary.get('normal_records', 0):,}", f"{norm_pct}% of total")
        else:
            render_metric_card("Target Labels", "Unlabelled", "Pure inference dataset")
    with c4:
        if summary.get("has_binary_label"):
            atk_pct = summary.get("attack_pct", 0)
            render_metric_card("Attack Traffic", f"{summary.get('attack_records', 0):,}", f"{atk_pct}% of total", status="danger")
        else:
            render_metric_card("Validation", "Valid Schema" if schema_val["is_valid"] else "Invalid", f"{len(schema_val['warnings'])} warnings")

    # Data Schema & Warnings
    if schema_val["warnings"]:
        for w in schema_val["warnings"]:
            st.info(f"ℹ️ {w}")
    if schema_val["errors"]:
        for err in schema_val["errors"]:
            st.error(f"⚠️ {err}")

    # Tabs for in-depth data exploration
    t1, t2, t3, t4 = st.tabs(["📋 Data Preview", "📊 Class Distributions", "📈 Feature Visualizer", "🔍 Missing Values & Types"])

    with t1:
        st.markdown("#### First 10 Records of Network Flow Data")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Includes packet flow durations, source/dest byte counts, TTLs, TCP window sizes, jitter, and state flags.")

    with t2:
        if summary.get("has_binary_label") or summary.get("has_multiclass_label"):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if summary.get("has_binary_label"):
                    st.markdown("#### Binary Class Distribution (0: Normal, 1: Attack)")
                    counts_df = pd.DataFrame({
                        "Traffic Type": ["Normal (0)", "Attack (1)"],
                        "Count": [summary.get("normal_records", 0), summary.get("attack_records", 0)]
                    })
                    fig_pie = px.pie(
                        counts_df,
                        names="Traffic Type",
                        values="Count",
                        color="Traffic Type",
                        color_discrete_map={"Normal (0)": "#10b981", "Attack (1)": "#ef4444"},
                        hole=0.45
                    )
                    fig_pie.update_layout(template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col_b2:
                if summary.get("has_multiclass_label"):
                    st.markdown("#### Attack Categories Breakdown")
                    atk_cats = summary.get("attack_categories", {})
                    cat_df = pd.DataFrame(list(atk_cats.items()), columns=["Attack Family", "Count"]).sort_values(by="Count", ascending=True)
                    fig_bar = px.bar(
                        cat_df,
                        x="Count",
                        y="Attack Family",
                        orientation="h",
                        color="Attack Family",
                        color_discrete_sequence=px.colors.sequential.Tealgrn
                    )
                    fig_bar.update_layout(template="plotly_dark", showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Uploaded dataset does not contain label columns. Only feature analysis is available.")

    with t3:
        st.markdown("#### Feature Distribution Profiler")
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["id", "label"]]
        if numeric_cols:
            selected_feature = st.selectbox("Select numerical feature to analyze:", numeric_cols, index=0)
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                fig_hist = px.histogram(
                    df,
                    x=selected_feature,
                    nbins=40,
                    color="label" if "label" in df.columns else None,
                    color_discrete_map={0: "#10b981", 1: "#ef4444"} if "label" in df.columns else None,
                    marginal="box",
                    title=f"Distribution of '{selected_feature}'"
                )
                fig_hist.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            with c_f2:
                st.markdown(f"##### Summary Statistics for `{selected_feature}`")
                st.dataframe(df[selected_feature].describe().to_frame().T, use_container_width=True)
                st.markdown("""
                **Cybersecurity Significance:**
                Network traffic features frequently follow heavy-tailed or exponential distributions 
                (e.g., packet duration `dur`, transaction bytes `sbytes`, time-to-live `sttl`). 
                Standardization and proper imputation ensure robust VAE reconstruction without unbounded numerical explosion.
                """)
        else:
            st.warning("No numeric features found to visualize.")

    with t4:
        st.markdown("#### Missing Values & Column Data Types")
        null_s = df.isnull().sum()
        dtype_s = df.dtypes
        audit_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(t) for t in dtype_s],
            "Missing Count": null_s.values,
            "Missing %": [round(100.0 * v / len(df), 2) for v in null_s.values]
        })
        st.dataframe(audit_df, use_container_width=True)
else:
    st.info("Select or initialize a dataset above to start exploring.")
