"""
Cybersecurity visual theme and shared UI components for NetShield AI Streamlit Dashboard.
"""

import streamlit as st


def apply_cyber_theme():
    """Injects high-end cybersecurity dark mode CSS styling."""
    custom_css = """
    <style>
        /* Global App Styling */
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Main Container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }

        /* Top Banner & Header */
        .cyber-header {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .cyber-title {
            color: #38bdf8;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }

        .cyber-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 15px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: rgba(56, 189, 248, 0.5);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }
        .metric-subtext {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Status Pills */
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-ready {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge-alert {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Section Cards */
        .cyber-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(51, 65, 85, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: #ffffff;
            font-weight: 600;
            border-radius: 8px;
            border: 1px solid #38bdf8;
            padding: 0.5rem 1.25rem;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
            border-color: #7dd3fc;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
            transform: scale(1.01);
        }

        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid rgba(51, 65, 85, 0.6);
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Renders consistent glowing cybersecurity top header."""
    html = f"""
    <div class="cyber-header">
        <div class="cyber-title">🛡️ {title}</div>
        <div class="cyber-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", status: str = "normal"):
    """Renders a styled metric card."""
    border_color = "rgba(56, 189, 248, 0.3)"
    if status == "success":
        border_color = "rgba(16, 185, 129, 0.4)"
    elif status == "danger":
        border_color = "rgba(239, 68, 68, 0.4)"
        
    html = f"""
    <div class="metric-card" style="border-color: {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
