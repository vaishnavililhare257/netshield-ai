"""
Training pipeline for Variational Autoencoder (VAE) on Normal-Only network traffic.
Adheres to unsupervised anomaly detection principles with validation-based thresholding.
"""

import time
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VAE_DEFAULTS, VAE_MODEL_FILE, VAE_CONFIG_FILE
from models.vae import VAEAnomalyDetector


def train_vae_pipeline(
    x_train: np.ndarray,
    x_val: np.ndarray,
    hidden_dim1: int = VAE_DEFAULTS["hidden_dim1"],
    hidden_dim2: int = VAE_DEFAULTS["hidden_dim2"],
    latent_dim: int = VAE_DEFAULTS["latent_dim"],
    learning_rate: float = VAE_DEFAULTS["learning_rate"],
    batch_size: int = VAE_DEFAULTS["batch_size"],
    epochs: int = VAE_DEFAULTS["epochs"],
    beta: float = VAE_DEFAULTS["beta"],
    threshold_quantile: float = VAE_DEFAULTS["threshold_quantile"],
    random_seed: int = VAE_DEFAULTS["random_seed"],
    progress_callback: Optional[Callable[[int, int, float, float, float], None]] = None
) -> Tuple[VAEAnomalyDetector, Dict[str, Any]]:
    """
    Trains the VAE strictly on normal traffic features.
    
    Returns:
        (trained_vae, training_results_dict)
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = x_train.shape[1]
    
    # Instantiate VAE
    vae = VAEAnomalyDetector(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        latent_dim=latent_dim,
        beta=beta
    ).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    
    # DataLoaders
    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    history = {
        "train_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "val_loss": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0
        
        for batch in train_loader:
            x_b = batch[0].to(device)
            optimizer.zero_grad()
            
            recon_b, mu_b, logvar_b = vae(x_b)
            loss, recon_l, kl_l = vae.loss_function(recon_b, x_b, mu_b, logvar_b)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
            num_batches += 1
            
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_recon = epoch_recon / max(num_batches, 1)
        avg_kl = epoch_kl / max(num_batches, 1)
        
        # Validation pass
        vae.eval()
        with torch.no_grad():
            val_recon, val_mu, val_logvar = vae(val_tensor)
            v_loss, _, _ = vae.loss_function(val_recon, val_tensor, val_mu, val_logvar)
            avg_val_loss = v_loss.item()
            
        history["train_loss"].append(round(avg_train_loss, 5))
        history["recon_loss"].append(round(avg_recon, 5))
        history["kl_loss"].append(round(avg_kl, 5))
        history["val_loss"].append(round(avg_val_loss, 5))
        
        if progress_callback:
            progress_callback(epoch, epochs, avg_train_loss, avg_recon, avg_val_loss)
            
    training_time = round(time.time() - start_time, 3)
    
    # Calculate Anomaly Threshold strictly from Normal Validation Scores
    vae.eval()
    val_scores = vae.get_reconstruction_error(x_val, device=str(device))
    threshold = vae.set_threshold_from_validation(val_scores, quantile=threshold_quantile)
    
    # Save Model Weights and Configuration
    VAE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), VAE_MODEL_FILE)
    
    config_data = {
        "input_dim": input_dim,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "latent_dim": latent_dim,
        "beta": beta,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "training_time_sec": training_time,
        "device": str(device),
        "val_score_min": float(np.min(val_scores)),
        "val_score_mean": float(np.mean(val_scores)),
        "val_score_max": float(np.max(val_scores)),
    }
    
    with open(VAE_CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)
        
    results = {
        "history": history,
        "threshold": threshold,
        "training_time": training_time,
        "device": str(device),
        "config": config_data
    }
    
    return vae, results


def load_trained_vae(device: Optional[str] = None) -> Tuple[VAEAnomalyDetector, Dict[str, Any]]:
    """Loads trained VAE model weights and configuration artifact."""
    if not VAE_MODEL_FILE.exists() or not VAE_CONFIG_FILE.exists():
        raise FileNotFoundError("VAE model or config artifact not found. Please train the model first.")
        
    with open(VAE_CONFIG_FILE, "r") as f:
        config = json.load(f)
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    vae = VAEAnomalyDetector(
        input_dim=config["input_dim"],
        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        latent_dim=config["latent_dim"],
        beta=config["beta"]
    )
    vae.load_state_dict(torch.load(VAE_MODEL_FILE, map_location=device))
    vae.threshold = config["threshold"]
    vae.to(device)
    vae.eval()
    return vae, config
