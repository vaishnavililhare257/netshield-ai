"""
Variational Autoencoder (VAE) for Unsupervised Network Intrusion Anomaly Detection.
PyTorch implementation strictly following Deep Learning-II specifications.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional


class VAEAnomalyDetector(nn.Module):
    """
    Variational Autoencoder (VAE) architecture for network traffic anomaly detection.
    
    Architecture:
      Encoder:
        Input (D) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> [mu (latent_dim), logvar (latent_dim)]
      Reparameterization Trick:
        z = mu + std * epsilon, where epsilon ~ N(0, I)
      Decoder:
        z (latent_dim) -> Linear(32) -> ReLU -> Linear(64) -> ReLU -> Output (D) [Linear, no sigmoid]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        latent_dim: int = 16,
        beta: float = 1.0
    ):
        super(VAEAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        self.beta = beta
        self.threshold: float = 0.0

        # Encoder Layers
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.encoder_relu2 = nn.ReLU()

        # Latent Mean & Log-Variance Projections
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # Decoder Layers
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoder_relu2 = nn.ReLU()
        self.decoder_out = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input feature vector to latent distribution parameters (mu, logvar)."""
        h = self.encoder_relu1(self.encoder_fc1(x))
        h = self.encoder_relu2(self.encoder_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick:
        z = mu + exp(0.5 * logvar) * eps
        Allows gradient backpropagation through stochastic latent variable.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation/inference, use deterministic mean
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs original input dimensions from latent representation."""
        h = self.decoder_relu1(self.decoder_fc1(z))
        h = self.decoder_relu2(self.decoder_fc2(h))
        recon = self.decoder_out(h)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE pass: x -> encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates ELBO (Evidence Lower Bound) loss:
        Total Loss = Reconstruction Loss (MSE) + beta * KL Divergence
        """
        # Reconstruction loss (per-sample MSE averaged across feature dimension)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")

        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total_loss = recon_loss + self.beta * kl_div
        return total_loss, recon_loss, kl_div

    @torch.no_grad()
    def get_reconstruction_error(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Computes per-record mean squared reconstruction error.
        Anomaly Score S(x) = (1/D) * sum_{j=1}^D (x_j - x_hat_j)^2
        """
        self.eval()
        tensor_x = torch.tensor(x, dtype=torch.float32).to(device)
        recon_x, _, _ = self.forward(tensor_x)
        
        # Per-record squared error averaged across features
        sq_err = (tensor_x - recon_x).pow(2)
        sample_mse = torch.mean(sq_err, dim=1).cpu().numpy()
        return sample_mse

    def set_threshold_from_validation(self, val_scores: np.ndarray, quantile: float = 0.95):
        """
        Sets anomaly decision boundary using a specified quantile
        on normal validation traffic reconstruction errors.
        """
        self.threshold = float(np.quantile(val_scores, quantile))
        return self.threshold

    def predict(self, x: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns binary anomaly predictions (0: Normal, 1: Suspicious) and continuous anomaly scores.
        """
        scores = self.get_reconstruction_error(x, device=device)
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores
