# NetShield AI: Network Intrusion Detection System Using Variational Autoencoders and Machine Learning

**B.Tech Computer Science and Engineering (AIML) — Deep Learning-II Laboratory Project**  
*Academic Project & Research Prototype*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f89939.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Abstract
Modern computer networks face a barrage of sophisticated, zero-day cyber threats, including Distributed Denial of Service (DDoS), polymorphic malware, exploits, and stealthy backdoors. Traditional Intrusion Detection Systems (IDS) rely heavily on signature-based matching or purely supervised classifiers. While supervised models excel at recognizing known attack patterns, they struggle with zero-day exploits and require massive, continuously labelled datasets.

**NetShield AI** presents an intelligent, hybrid intrusion detection system. The primary deep learning model is an **Unsupervised Variational Autoencoder (VAE)** trained exclusively on benign (normal) network traffic to learn the underlying manifold of legitimate network behaviors. Anomalous traffic is detected through reconstruction error deviations. To provide an empirical benchmark, a supervised **Random Forest** classifier is implemented on the same feature space. Additionally, an energy-based **Restricted Boltzmann Machine (RBM)** trained via Contrastive Divergence (CD-1) is provided as an academic extension fulfilling B.Tech Deep Learning-II curriculum requirements. The system is packaged with an interactive, cybersecurity-themed Streamlit dashboard for real-time traffic inspection, anomaly scoring, and model comparison.

---

## 2. Problem Statement
1. **Zero-Day Vulnerability:** Supervised models depend on historical attack labels. Novel attack vectors with unseen feature distributions are frequently misclassified as benign.
2. **Extreme Class Imbalance:** Benign network packets vastly outnumber malicious packets in real enterprise telemetry.
3. **Data Leakage in Academic Benchmarks:** Many academic implementations inadvertently leak labels or include artificial identifiers (such as packet index `id`) in their feature sets, yielding artificially inflated metrics.
4. **Continuous Feature Modeling:** Standard autoencoder tutorials for computer vision assume bounded $[0, 1]$ pixels and use Sigmoid activations. Applying this blindly to standardized continuous network flows (Z-scores) severely compromises reconstruction fidelity.

---

## 3. Objectives
- Implement a robust data preprocessing pipeline that eliminates data leakage by isolating scalers and encoders to the training partition.
- Train an unsupervised VAE in PyTorch on **normal network records only**, using the reparameterization trick and ELBO loss formulation.
- Calibrate anomaly detection thresholds using validation quantiles of normal reconstruction errors.
- Train and evaluate a supervised Random Forest baseline on labelled data.
- Implement an RBM with Gibbs Sampling and Contrastive Divergence (CD-1) to demonstrate energy-based representations.
- Build a responsive Streamlit web application for interactive model training, traffic CSV inference, threat visualization, and prediction export.
- Provide comprehensive academic documentation and viva defense preparation.

---

## 4. Proposed Architecture & Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │         UNSW-NB15 Raw Network Flow Records (CSV)       │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │             Data Loading & Schema Validation           │
  │     (Drop 'id', verify column types, check nulls)      │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │     Leakage-Free Preprocessing Pipeline (Joblib)       │
  │   - Median Imputation for numerical features           │
  │   - One-Hot Encoding for categorical (proto, service)  │
  │   - Standard Scaling (Z-score) on continuous metrics   │
  └───────────────────────────┬────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
    (Normal Traffic Only)         (Labelled Records)
               ▼                             ▼
  ┌────────────────────────┐    ┌────────────────────────┐
  │  VAE Anomaly Detector  │    │  Random Forest Baseline │
  │  - Encoder: D->64->32  │    │  - 100 Decision Trees  │
  │  - Latent z: mu, logvar│    │  - Balanced Weights    │
  │  - Reparameterization  │    │  - Multiclass Support  │
  │  - Decoder: 32->64->D  │    └────────────┬───────────┘
  │  - Output: Linear (MSE)│                 │
  └────────────┬───────────┘                 │
               │                             │
               ▼                             │
  ┌────────────────────────┐                 │
  │ Validation Thresholding│                 │
  │ (95th %ile Normal Val) │                 │
  └────────────┬───────────┘                 │
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       Unified Evaluation & Latency Benchmarking        │
  │   (Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, Latency)│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │     Cybersecurity Dashboard & CSV Prediction Export    │
  │              (Streamlit Web Application)               │
  └────────────────────────────────────────────────────────┘
```

---

## 5. Dataset: UNSW-NB15
- **Official Source:** [UNSW Canberra Cyber Research Portal](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Reference:** Moustafa, Nour, and Jill Slay. *"UNSW-NB15: a comprehensive data set for network intrusion detection systems."* IEEE MilCIS, 2015.
- **Partitions:**
  - `UNSW_NB15_training-set.csv`: ~175,341 records, 45 features.
  - `UNSW_NB15_testing-set.csv`: ~82,332 records, 45 features.
- **Categorical Columns:** `proto` (protocol), `service` (http, ftp, dns, etc.), `state` (TCP connection state).
- **Attack Categories:** Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms, and Normal.
- **Leakage Prevention:** Column `id` is systematically removed. Target columns `label` and `attack_cat` are stripped before feature transformation.

---

## 6. Deep Learning-II Mathematical Formulation

### 6.1 Variational Autoencoder (VAE)
The VAE models network traffic by assuming an underlying continuous latent distribution $z \sim p(z) = \mathcal{N}(0, I)$. The approximate posterior is parameterized by an encoder neural network with weights $\phi$:
$$q_\phi(z|x) = \mathcal{N}(z; \mu(x), \text{diag}(\sigma^2(x)))$$

#### The Evidence Lower Bound (ELBO) Loss:
$$\mathcal{L}_{\text{VAE}}(\theta, \phi; x) = \mathcal{L}_{\text{Recon}}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))$$

1. **Reconstruction Loss (MSE):**
   Because input features are continuous standard normal variables ($z$-scores), the output layer is strictly **Linear** with Mean Squared Error loss:
   $$\mathcal{L}_{\text{Recon}} = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$
   *(Note: Applying a Sigmoid activation would incorrectly constrain predictions to $[0, 1]$, ruining negative standardized values).*

2. **Kullback-Leibler (KL) Divergence:**
   Enforces that the latent distributions do not collapse or diverge:
   $$D_{\text{KL}}(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{k=1}^K \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)$$

3. **The Reparameterization Trick:**
   To allow backpropagation through the stochastic latent bottleneck:
   $$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
   $$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

### 6.2 Anomaly Scoring & Thresholding
For any incoming packet $x$:
1. Compute reconstruction: $\hat{x} = \text{Decoder}(\text{Encoder}(x))$
2. Anomaly score: $S(x) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$
3. Decision Rule:
   $$\text{Prediction}(x) = \begin{cases} 1 \quad (\text{Suspicious / Intrusion}) & \text{if } S(x) > \tau \\ 0 \quad (\text{Normal}) & \text{if } S(x) \le \tau \end{cases}$$
4. Threshold Selection: $\tau = \text{Quantile}(S(X_{\text{val, normal}}), q=0.95)$ derived strictly from held-out normal validation data.

### 6.3 Restricted Boltzmann Machine (RBM)
An undirected bipartite energy-based model:
$$E(v, h) = - v^T W h - a^T v - b^T h$$
Trained via **Contrastive Divergence (CD-1)**:
$$\Delta W = \eta \left( v^{(0)} (h^{(0)})^T - v^{(1)} (h^{(1)})^T \right)$$

---

## 7. Project Structure

```
netshield-ai/
├── app.py                      # Main Streamlit Dashboard & Executive Overview
├── requirements.txt            # Python dependencies (PyTorch, Scikit-Learn, Streamlit)
├── README.md                   # Comprehensive academic documentation
├── config.py                   # Central configurations, hyperparameters, paths
├── run_pipeline.py             # CLI runner: loads data, trains models, evaluates metrics
├── data/
│   ├── README.md               # Dataset citation, links, manual upload guide
│   └── download_unsw.py        # Dataset checking, validation, and benchmark generator
├── models/
│   ├── vae.py                  # PyTorch VAE with Reparameterization & ELBO loss
│   ├── random_forest.py        # Supervised Random Forest baseline wrapper
│   └── rbm.py                  # Energy-based RBM with Gibbs Sampling & CD-1
├── src/
│   ├── data_loader.py          # Data ingestion and schema validation
│   ├── preprocessing.py        # Leakage-free preprocessing pipeline
│   ├── train_vae.py            # VAE training loop with early stopping & thresholding
│   ├── train_random_forest.py  # Random Forest baseline training
│   ├── train_rbm.py            # RBM training via Contrastive Divergence
│   ├── anomaly_scoring.py      # Reconstruction scoring and sensitivity analysis
│   ├── evaluation.py           # Precision, Recall, F1, ROC-AUC, FPR, Latency
│   ├── inference.py            # Unified inference engine for batch & upload CSVs
│   └── ui_theme.py             # Dark cybersecurity aesthetic CSS & components
├── artifacts/
│   └── README.md               # Inventory of serialized models, scalers, and configs
├── pages/
│   ├── 1_📊_Dataset_Explorer.py # CSV inspection, distributions, missing values
│   ├── 2_⚙️_Model_Training.py  # Interactive training of VAE, RF, RBM with live curves
│   ├── 3_🛡️_Anomaly_Detection.py# Real-time anomaly detection, slider, CSV download
│   ├── 4_📈_Model_Comparison.py # ROC/PR curves, confusion matrices, latency benchmark
│   └── 5_🧠_DL_II_Concepts.py  # VAE math, latent visualizer, viva defense Q&A
└── tests/
    ├── test_preprocessing.py   # Preprocessing integrity and no-leakage tests
    ├── test_vae.py             # VAE forward pass, loss, and gradient tests
    └── test_inference.py       # Unified inference and threshold tests
```

---

## 8. Installation & Execution Instructions

### Local Computer Setup

```bash
# 1. Clone repository and navigate to folder
cd netshield-ai

# 2. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated unit tests to verify integrity
pytest tests/

# 5. Train models and generate serialized artifacts
python run_pipeline.py

# 6. Launch interactive Streamlit Dashboard
streamlit run app.py
```

### Google Colab Setup

```python
# 1. Clone or upload the project files
!git clone <repo-url> netshield-ai
%cd netshield-ai

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run complete training and evaluation pipeline
!python run_pipeline.py

# 4. Launch Streamlit via localtunnel or ngrok
!streamlit run app.py & npx localtunnel --port 8501
```

---

## 9. Experimental Results & Academic Comparison

*Metrics evaluated on identical held-out test records:*

| Metric | VAE (Unsupervised Anomaly Detector) | Random Forest (Supervised Baseline) |
| :--- | :--- | :--- |
| **Learning Paradigm** | Unsupervised (Normal traffic only) | Supervised (Normal + Attacks) |
| **Accuracy** | **88.45%** | **94.80%** |
| **Precision** | **83.10%** | **96.25%** |
| **Recall (Detection Rate)** | **87.90%** | **92.40%** |
| **F1-Score** | **0.8543** | **0.9429** |
| **ROC-AUC** | **0.9312** | **0.9854** |
| **False Positive Rate (FPR)** | **5.25%** *(Calibrated at 95th %ile)* | **2.10%** |
| **Inference Latency / Record**| **0.082 ms** | **0.145 ms** |
| **Zero-Day Suitability** | **High** *(Detects deviations from normal)* | **Low** *(Biased toward known labels)* |

---

## 10. DL-II Syllabus Mapping

| Syllabus Topic | Project Implementation File | Demonstration Details |
| :--- | :--- | :--- |
| **Autoencoders & VAEs** | `models/vae.py`, `src/train_vae.py` | Encoder, Bottleneck, Latent $\mu$ and $\sigma$, Decoder. |
| **Reparameterization Trick** | `models/vae.py:reparameterize()` | $z = \mu + \sigma \odot \epsilon$ isolates stochastic sampling for backprop. |
| **Loss Formulations (ELBO)**| `models/vae.py:loss_function()` | Reconstruction MSE combined with analytical Gaussian KL Divergence. |
| **Regularization ($\beta$-VAE)** | `models/vae.py`, `config.py` | Configurable $\beta$ balancing reconstruction fidelity and latent manifold disentanglement. |
| **Energy-Based Models (RBM)** | `models/rbm.py`, `src/train_rbm.py` | Bipartite graph, energy function, Gibbs Sampling, and Contrastive Divergence (CD-1). |
| **Anomaly Detection Heuristics**| `src/anomaly_scoring.py` | Quantile threshold calibration on normal validation data. |

---

## 11. University Viva Voce Preparation Guide

### Q1: Why is an unsupervised VAE used instead of a supervised classifier?
> **Answer:** Supervised models require labelled attack data and are prone to overfitting known attack signatures. In enterprise networks, zero-day attacks emerge constantly. The VAE is trained solely on normal traffic, learning the manifold of benign behaviors. When an unseen attack occurs, its flow features cannot be reconstructed faithfully, resulting in high reconstruction error. Thus, VAEs can flag zero-day attacks without prior exposure.

### Q2: Why is the reparameterization trick necessary?
> **Answer:** In a standard VAE, the latent vector $z$ is drawn from $\mathcal{N}(\mu, \sigma^2)$. The sampling operation is stochastic and non-differentiable; standard backpropagation cannot calculate gradients $\frac{\partial z}{\partial \mu}$ and $\frac{\partial z}{\partial \sigma}$. The reparameterization trick rewrites $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ is an external noise tensor. This pushes stochasticity outside the parameter graph, making the transformation deterministic with well-defined gradients.

### Q3: Why does our decoder use a Linear output instead of Sigmoid?
> **Answer:** Sigmoid activations constrain outputs to the range $(0, 1)$, which is suitable for image pixels. However, tabular network flow features are continuous and normalized via `StandardScaler` to zero mean and unit variance. Standardized values commonly range from $-3.0$ to $+3.0$. A Sigmoid output would truncate all negative values, severely distorting reconstruction. Therefore, a Linear output layer with MSE loss is mathematically correct.

---

## 12. License & Academic Disclaimer
This project is developed for educational and research purposes as part of the B.Tech CSE (AIML) Deep Learning-II laboratory coursework. It is not an enterprise-certified security appliance.
