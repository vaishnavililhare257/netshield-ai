"""
Restricted Boltzmann Machine (RBM) for Deep Learning-II Syllabus Demonstration.
Implements Energy-based modeling, Gibbs Sampling, and Contrastive Divergence (CD-k).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RBM_DEFAULTS, RBM_MODEL_FILE


class RestrictedBoltzmannMachine(nn.Module):
    """
    Restricted Boltzmann Machine (RBM).
    An undirected energy-based graphical model with a bipartite graph between visible and hidden units.
    
    Energy Function:
      E(v, h) = - v^T W h - a^T v - b^T h
      where:
        v: visible units (input features scaled to [0, 1])
        h: latent stochastic hidden features
        W: weight matrix (visible_dim x hidden_dim)
        a: visible bias
        b: hidden bias
    """
    def __init__(
        self,
        num_visible: int,
        num_hidden: int = RBM_DEFAULTS["n_hidden"],
        k_steps: int = RBM_DEFAULTS["k_steps"],
        learning_rate: float = RBM_DEFAULTS["learning_rate"]
    ):
        super(RestrictedBoltzmannMachine, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k_steps = k_steps
        self.learning_rate = learning_rate

        # Initialize weights with standard Gaussian scaled by fan-in
        self.weights = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.05)
        self.visible_bias = nn.Parameter(torch.zeros(num_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))

    def sample_hidden(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes conditional probability P(h_j = 1 | v) = sigmoid(v * W_j + b_j)
        and draws Bernoulli stochastic samples.
        """
        p_h = torch.sigmoid(torch.matmul(v, self.weights) + self.hidden_bias)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes conditional probability P(v_i = 1 | h) = sigmoid(h * W_i^T + a_i)
        and draws visible reconstructions.
        """
        p_v = torch.sigmoid(torch.matmul(h, self.weights.t()) + self.visible_bias)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def gibbs_step(self, v_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one alternating Gibbs sampling step:
        v -> h ~ P(h|v) -> v' ~ P(v|h) -> h' ~ P(h|v')
        """
        p_h0, h0 = self.sample_hidden(v_init)
        p_v1, v1 = self.sample_visible(h0)
        p_h1, h1 = self.sample_hidden(v1)
        return p_h0, h0, p_v1, p_h1

    def contrastive_divergence(self, v_data: torch.Tensor) -> Tuple[float, float]:
        """
        Performs Contrastive Divergence (CD-k) weight update.
        Approximates the log-likelihood gradient using k-step Gibbs sampling.
        """
        batch_size = v_data.size(0)

        # Positive Phase (Data driven)
        p_h0, h0 = self.sample_hidden(v_data)

        # Negative Phase (Reconstruction / Dream phase via Gibbs sampling)
        v_k = v_data
        h_k = h0
        for _ in range(self.k_steps):
            p_vk, v_k = self.sample_visible(h_k)
            p_hk, h_k = self.sample_hidden(v_k)

        # Compute empirical gradients
        # Positive association: v_data^T * p(h|v_data)
        pos_assoc = torch.matmul(v_data.t(), p_h0)
        # Negative association: v_k^T * p(h|v_k)
        neg_assoc = torch.matmul(v_k.t(), p_hk)

        # Update weights and biases
        with torch.no_grad():
            self.weights += self.learning_rate * ((pos_assoc - neg_assoc) / batch_size)
            self.visible_bias += self.learning_rate * torch.mean(v_data - v_k, dim=0)
            self.hidden_bias += self.learning_rate * torch.mean(p_h0 - p_hk, dim=0)

        # Compute reconstruction error diagnostic
        recon_error = torch.mean((v_data - v_k) ** 2).item()
        
        # Free energy diagnostic
        free_energy = torch.mean(self.free_energy(v_data)).item()
        return recon_error, free_energy

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the free energy of a visible configuration:
        F(v) = - v^T a - sum_j log(1 + exp(v * W_j + b_j))
        Lower free energy indicates patterns more consistent with the training distribution.
        """
        v_bias_term = torch.matmul(v, self.visible_bias)
        wx_b = torch.matmul(v, self.weights) + self.hidden_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(torch.clamp(wx_b, -20, 20))), dim=1)
        return -v_bias_term - hidden_term

    @torch.no_grad()
    def reconstruct(self, v: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Performs one visible reconstruction pass."""
        tensor_v = torch.tensor(v, dtype=torch.float32).to(device)
        p_h, _ = self.sample_hidden(tensor_v)
        p_v, _ = self.sample_visible(p_h)
        return p_v.cpu().numpy()

    @torch.no_grad()
    def get_anomaly_scores(self, v: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Computes per-record reconstruction error under RBM."""
        v_recon = self.reconstruct(v, device=device)
        return np.mean((v - v_recon) ** 2, axis=1)

    def save(self, filepath: Optional[Path] = None):
        """Saves RBM state dictionary and configuration."""
        path = Path(filepath or RBM_MODEL_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "num_visible": self.num_visible,
            "num_hidden": self.num_hidden,
            "k_steps": self.k_steps,
            "learning_rate": self.learning_rate
        }, path)

    @classmethod
    def load(cls, filepath: Optional[Path] = None, device: str = "cpu") -> "RestrictedBoltzmannMachine":
        """Loads RBM from serialized checkpoint."""
        path = Path(filepath or RBM_MODEL_FILE)
        if not path.exists():
            raise FileNotFoundError(f"RBM checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location=device)
        rbm = cls(
            num_visible=checkpoint["num_visible"],
            num_hidden=checkpoint["num_hidden"],
            k_steps=checkpoint["k_steps"],
            learning_rate=checkpoint["learning_rate"]
        )
        rbm.load_state_dict(checkpoint["state_dict"])
        rbm.to(device)
        rbm.eval()
        return rbm
