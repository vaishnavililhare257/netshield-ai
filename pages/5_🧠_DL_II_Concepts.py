"""
NetShield AI — Deep Learning-II Concepts & Academic Viva Preparation
Interactive educational visualizers explaining VAE architectures, mathematical formulations,
the Reparameterization Trick, KL Divergence, RBM Gibbs Sampling, and university viva Q&A.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ui_theme import apply_cyber_theme, render_header

apply_cyber_theme()

render_header(
    title="Deep Learning-II Concepts & Viva Defense Guide",
    subtitle="Interactive syllabus demonstrations: VAE Manifolds, Reparameterization Trick, ELBO Loss, and RBMs."
)

c_tabs = st.tabs([
    "📐 1. VAE Mathematical Architecture",
    "🎲 2. The Reparameterization Trick",
    "⚖️ 3. Reconstruction vs KL Divergence",
    "🧠 4. RBM & Contrastive Divergence",
    "🎓 5. University Viva Q&A Guide"
])

# =====================================================================
# TAB 1: VAE MATHEMATICAL ARCHITECTURE
# =====================================================================
with c_tabs[0]:
    st.subheader("Variational Autoencoder (VAE) Pipeline")
    st.markdown("""
    Unlike classical deterministic autoencoders that memorize discrete points, a **Variational Autoencoder** is a 
    **directed probabilistic generative model**. It maps each input data point $x$ not to a fixed vector, but to 
    parameters of a probability distribution $q_\phi(z|x) = \mathcal{N}(\mu(x), \Sigma(x))$.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Architectural Specifications")
        st.code("""
        [ Input Features: D dimensions ]
                       │
                       ▼
        [ Encoder Layer 1: Linear(D -> 64) + ReLU ]
                       │
                       ▼
        [ Encoder Layer 2: Linear(64 -> 32) + ReLU ]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        [ Mean mu ]       [ Log-Variance logvar ]
         Linear(32 -> K)   Linear(32 -> K)
             │                   │
             └─────────┬─────────┘
                       │
                       ▼  <-- eps ~ N(0, I)
        [ Reparameterization: z = mu + std * eps ]
                       │
                       ▼
        [ Decoder Layer 1: Linear(K -> 32) + ReLU ]
                       │
                       ▼
        [ Decoder Layer 2: Linear(32 -> 64) + ReLU ]
                       │
                       ▼
        [ Output Layer: Linear(64 -> D) (Linear activation) ]
        """, language="text")

    with col2:
        st.markdown("#### Latent Manifold Simulation")
        st.caption("Interactive 2D projection of normal vs anomalous flow latent embeddings.")
        
        # Interactive latent embedding scatter
        np.random.seed(42)
        n_pts = 300
        z_norm_x = np.random.normal(0, 1.0, n_pts)
        z_norm_y = np.random.normal(0, 1.0, n_pts)

        z_atk_x = np.random.normal(3.2, 1.4, int(n_pts * 0.4))
        z_atk_y = np.random.normal(-2.8, 1.5, int(n_pts * 0.4))

        latent_sim_df = pd.DataFrame({
            "z_1": np.concatenate([z_norm_x, z_atk_x]),
            "z_2": np.concatenate([z_norm_y, z_atk_y]),
            "Traffic Class": ["Normal Traffic (Learned Prior)"] * n_pts + ["Anomalous Traffic (Out of Distribution)"] * len(z_atk_x)
        })

        fig_lat = px.scatter(
            latent_sim_df,
            x="z_1",
            y="z_2",
            color="Traffic Class",
            color_discrete_map={"Normal Traffic (Learned Prior)": "#38bdf8", "Anomalous Traffic (Out of Distribution)": "#ef4444"},
            title="VAE Latent Space Distribution"
        )
        fig_lat.update_layout(template="plotly_dark", margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_lat, use_container_width=True)

# =====================================================================
# TAB 2: REPARAMETERIZATION TRICK
# =====================================================================
with c_tabs[1]:
    st.subheader("The Reparameterization Trick: Enabling Backpropagation")
    st.markdown("""
    In a standard autoencoder, the encoder produces a deterministic code $z = f(x)$. 
    In a variational autoencoder, $z$ is a stochastic random variable sampled from $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$.

    **The Problem:** The sampling operation $z \sim \mathcal{N}(\mu, \sigma^2)$ is non-differentiable. 
    Standard backpropagation cannot compute $\\frac{\\partial z}{\\partial \\mu}$ or $\\frac{\\partial z}{\\partial \\sigma}$.
    """)

    st.markdown("### The Solution: Coordinate Transformation")
    st.latex(r"z = \mu(x) + \sigma(x) \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)")

    st.markdown("""
    By expressing $z$ as a deterministic function of $\mu$, $\sigma$, and an external stochastic noise tensor $\epsilon$:
    1. The random noise $\epsilon$ does not depend on network weights $\phi$.
    2. The gradients with respect to $\mu$ and $\sigma$ flow smoothly:
    """)
    st.latex(r"\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon")

    # Interactive widget demonstrating sampling stability
    st.markdown("#### Interactive Reparameterization Simulator")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sim_mu = st.slider("Simulated Mean (mu)", -3.0, 3.0, 0.5, step=0.1)
        sim_logvar = st.slider("Simulated Log-Variance (logvar)", -2.0, 2.0, 0.0, step=0.1)
        sim_std = np.exp(0.5 * sim_logvar)
        st.caption(f"Computed Standard Deviation (sigma): `{sim_std:.4f}`")

    with c_s2:
        eps_samples = np.random.normal(0, 1, 1000)
        z_samples = sim_mu + sim_std * eps_samples
        fig_z = px.histogram(x=z_samples, nbins=35, title=f"Sampled Latent Vector (mu={sim_mu}, sigma={sim_std:.2f})")
        fig_z.update_layout(template="plotly_dark", margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_z, use_container_width=True)

# =====================================================================
# TAB 3: RECONSTRUCTION VS KL DIVERGENCE
# =====================================================================
with c_tabs[2]:
    st.subheader(r"ELBO Loss & The Beta ($\beta$) Hyperparameter")
    st.markdown("""
    The Variational Autoencoder optimizes the **Evidence Lower Bound (ELBO)**:
    """)
    st.latex(r"\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{Reconstruction}} + \beta \cdot \mathcal{L}_{\text{KL}}")

    st.markdown("""
    ### 1. Reconstruction Loss ($\mathcal{L}_{\text{Recon}}$)
    Measures how faithfully the decoder reconstructs the network feature vector:
    """)
    st.latex(r"\mathcal{L}_{\text{Recon}} = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2")

    st.markdown("""
    ### 2. Kullback-Leibler Divergence ($\mathcal{L}_{\text{KL}}$)
    Measures the relative entropy between the encoder's approximate posterior $q_\phi(z|x)$ and the standard Gaussian prior $p(z) = \mathcal{N}(0, I)$:
    """)
    st.latex(r"\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{k=1}^K \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)")

    st.markdown("""
    ### The Role of Beta ($\beta$):
    - **$\beta \to 0$:** Model behaves like a standard autoencoder. Reconstructions become sharper, but latent space exhibits clusters and "holes", impairing zero-day generalization.
    - **$\beta = 1.0$:** Standard VAE formulation.
    - **$\beta > 1.0$ ($\beta$-VAE):** Enforces stronger disentanglement of latent variables, forcing each latent dimension to represent independent network flow characteristics.
    """)

# =====================================================================
# TAB 4: RBM & CONTRASTIVE DIVERGENCE
# =====================================================================
with c_tabs[3]:
    st.subheader("Restricted Boltzmann Machines (RBM) & Contrastive Divergence")
    st.markdown("""
    A **Restricted Boltzmann Machine** is an undirected energy-based neural network consisting of two layers:
    1. **Visible layer ($v$):** Observes normalized network traffic features.
    2. **Hidden layer ($h$):** Stochastic latent feature detectors.
    3. **Restriction:** No intra-layer connections exist (bipartite graph).
    """)

    st.latex(r"E(v, h) = - \sum_{i} a_i v_i - \sum_{j} b_j h_j - \sum_{i, j} v_i W_{ij} h_j")

    st.markdown("""
    ### Contrastive Divergence (CD-k) Algorithm:
    Exact maximum likelihood learning in an RBM requires computing the partition function $Z = \sum_{v, h} e^{-E(v, h)}$, 
    which involves an intractable sum over $2^{V+H}$ states.

    Geoffrey Hinton's **Contrastive Divergence (CD-1)** approximates the gradient via $k=1$ step of alternating Gibbs sampling:
    1. **Positive Phase:** Clamp training vector $v^{(0)}$ onto visible units. Compute $p(h^{(0)}=1|v^{(0)}) = \sigma(v^{(0)} W + b)$.
    2. **Negative Phase:** Sample visible reconstruction $v^{(1)} \sim p(v|h^{(0)}) = \sigma(h^{(0)} W^T + a)$. Sample hidden reconstruction $h^{(1)} \sim p(h|v^{(1)})$.
    3. **Weight Update:** $\Delta W = \eta \left( v^{(0)} (h^{(0)})^T - v^{(1)} (h^{(1)})^T \right)$.
    """)

# =====================================================================
# TAB 5: UNIVERSITY VIVA Q&A GUIDE
# =====================================================================
with c_tabs[4]:
    st.subheader("B.Tech DL-II University Viva & Defense Guide")
    st.markdown("Comprehensive technical answers to expected examiner questions.")

    viva_qa = [
        (
            "Q1: Why is an unsupervised VAE preferred over a supervised classifier for Network Intrusion Detection?",
            "Supervised classifiers (like Random Forest or MLP) require labelled attack data and are inherently biased toward known attack signatures. In real-world enterprise networks, zero-day attacks and polymorphic exploits emerge continuously. An unsupervised VAE is trained solely on benign (normal) traffic. When a novel, unseen attack occurs, its flow pattern deviates from the learned normal manifold, resulting in high reconstruction error. Thus, VAEs can flag zero-day attacks without having ever seen them during training."
        ),
        (
            "Q2: Why can we NOT apply backpropagation directly through the latent layer of a VAE without the reparameterization trick?",
            "In a VAE, the latent vector z is drawn from a stochastic distribution N(mu, sigma^2). The sampling operation z ~ q(z|x) is a non-deterministic stochastic node with no analytical derivative with respect to encoder parameters mu and sigma. Backpropagation requires deterministic computational graphs. The reparameterization trick resolves this by rewriting z = mu + sigma * epsilon, where epsilon ~ N(0, I) is an independent external random variable. The stochasticity is pushed outside the parameter graph, making z differentiable with dz/dmu = 1 and dz/dsigma = epsilon."
        ),
        (
            "Q3: How do you choose the anomaly detection threshold in this project without causing data leakage?",
            "The threshold is selected using a quantile heuristic (e.g., the 95th percentile) calculated strictly on the reconstruction errors of the NORMAL VALIDATION traffic set. It is NEVER tuned using test labels or attack records. Setting the threshold at the 95th percentile on normal validation data establishes an expected baseline false-positive rate of approximately 5% under nominal conditions."
        ),
        (
            "Q4: Why does our VAE decoder use a Linear output layer rather than a Sigmoid activation?",
            "In computer vision applications (like MNIST or CelebA), pixel intensities are strictly bounded in [0, 1], so a Sigmoid output with Binary Cross-Entropy loss is valid. However, tabular network intrusion features (e.g., sload, packet duration, TTL, byte count) are continuous variables normalized using StandardScaler to mean 0 and unit variance (Z-scores). Standardized z-scores take both positive and negative values (e.g., -2.5 to +3.5). Applying a Sigmoid would artificially compress outputs to (0, 1), completely destroying continuous reconstruction fidelity. Hence, a Linear output with Mean Squared Error (MSE) loss is mathematically necessary."
        ),
        (
            "Q5: What are the differences between an RBM and a Variational Autoencoder?",
            "1. Architecture: An RBM is an undirected energy-based graphical model with a bipartite graph between visible and hidden units. A VAE is a directed probabilistic neural network consisting of an encoder and decoder. 2. Objective: An RBM minimizes energy and maximizes visible configuration likelihood via Contrastive Divergence (CD-k) using Gibbs sampling. A VAE maximizes the Evidence Lower Bound (ELBO) balancing reconstruction loss and KL divergence using gradient backpropagation."
        ),
        (
            "Q6: What columns were removed from UNSW-NB15 to prevent data leakage?",
            "The 'id' column was dropped because sequential identifiers can cause models to memorize row index order rather than genuine flow physics. Additionally, 'label' and 'attack_cat' were strictly excluded from the feature space and used only as target labels for evaluation and baseline comparison."
        )
    ]

    for q, a in viva_qa:
        with st.expander(f"📌 {q}"):
            st.markdown(f"**Answer:** {a}")
