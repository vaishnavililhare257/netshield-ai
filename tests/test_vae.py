"""
Unit tests for VAE Anomaly Detector architecture and loss formulation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vae import VAEAnomalyDetector


def test_vae_architecture_forward():
    batch_size = 32
    input_dim = 40
    latent_dim = 16
    
    vae = VAEAnomalyDetector(
        input_dim=input_dim,
        hidden_dim1=64,
        hidden_dim2=32,
        latent_dim=latent_dim,
        beta=1.0
    )
    
    x = torch.randn(batch_size, input_dim)
    recon_x, mu, logvar = vae(x)
    
    assert recon_x.shape == (batch_size, input_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


def test_vae_loss_and_gradients():
    input_dim = 30
    vae = VAEAnomalyDetector(input_dim=input_dim, latent_dim=8)
    
    x = torch.randn(16, input_dim)
    recon_x, mu, logvar = vae(x)
    
    total_loss, recon_loss, kl_loss = vae.loss_function(recon_x, x, mu, logvar)
    
    assert torch.isfinite(total_loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)
    assert total_loss.item() > 0
    
    # Backpropagation check
    total_loss.backward()
    for param in vae.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_reconstruction_error_and_threshold():
    input_dim = 25
    vae = VAEAnomalyDetector(input_dim=input_dim, latent_dim=8)
    vae.eval()
    
    val_data = np.random.randn(50, input_dim).astype(np.float32)
    scores = vae.get_reconstruction_error(val_data)
    
    assert len(scores) == 50
    assert (scores >= 0).all()
    
    threshold = vae.set_threshold_from_validation(scores, quantile=0.90)
    assert threshold > 0
    
    preds, p_scores = vae.predict(val_data)
    assert len(preds) == 50
    assert set(np.unique(preds)).issubset({0, 1})
