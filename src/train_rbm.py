"""
Training pipeline for Restricted Boltzmann Machine (RBM) via Contrastive Divergence (CD-1).
Demonstrates Energy-based unsupervised feature learning and reconstruction diagnostics.
"""

import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RBM_DEFAULTS, RBM_MODEL_FILE
from models.rbm import RestrictedBoltzmannMachine


def train_rbm_pipeline(
    x_train: np.ndarray,
    n_hidden: int = RBM_DEFAULTS["n_hidden"],
    learning_rate: float = RBM_DEFAULTS["learning_rate"],
    k_steps: int = RBM_DEFAULTS["k_steps"],
    batch_size: int = RBM_DEFAULTS["batch_size"],
    epochs: int = RBM_DEFAULTS["epochs"],
    random_seed: int = RBM_DEFAULTS["random_seed"],
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Tuple[RestrictedBoltzmannMachine, Dict[str, Any]]:
    """
    Trains an RBM on normal traffic representations mapped to [0, 1].
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Scale input strictly into [0, 1] for Bernoulli visible units
    minmax = MinMaxScaler()
    x_norm = minmax.fit_transform(x_train).astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_visible = x_norm.shape[1]
    
    rbm = RestrictedBoltzmannMachine(
        num_visible=num_visible,
        num_hidden=n_hidden,
        k_steps=k_steps,
        learning_rate=learning_rate
    ).to(device)
    
    dataset = TensorDataset(torch.tensor(x_norm, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    history = {
        "recon_error": [],
        "free_energy": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_recon = 0.0
        epoch_energy = 0.0
        batches = 0
        
        for batch in loader:
            v_batch = batch[0].to(device)
            recon_err, free_en = rbm.contrastive_divergence(v_batch)
            epoch_recon += recon_err
            epoch_energy += free_en
            batches += 1
            
        avg_recon = epoch_recon / max(batches, 1)
        avg_energy = epoch_energy / max(batches, 1)
        
        history["recon_error"].append(round(avg_recon, 6))
        history["free_energy"].append(round(avg_energy, 4))
        
        if progress_callback:
            progress_callback(epoch, epochs, avg_recon, avg_energy)
            
    training_time = round(time.time() - start_time, 3)
    
    # Save model artifact
    rbm.save(RBM_MODEL_FILE)
    
    results = {
        "training_time": training_time,
        "history": history,
        "n_hidden": n_hidden,
        "k_steps": k_steps,
        "minmax_scaler": minmax
    }
    
    return rbm, results
