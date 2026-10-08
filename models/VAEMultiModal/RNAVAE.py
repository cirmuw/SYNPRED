"""RNA encoder/decoder primitives used by the publication model."""

import torch
from torch import nn


class RNAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, output="distribution"):
        super().__init__()
        self.output = output
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.shared_fc(x)
        if self.output != "distribution":
            return hidden
        return self.mean(hidden), self.logvar(hidden)


class RNADecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.decoder(z)


def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def kl_divergence(mu_q, logvar_q, mu_p=None, logvar_p=None):
    mu_p = torch.zeros_like(mu_q) if mu_p is None else mu_p
    logvar_p = torch.zeros_like(logvar_q) if logvar_p is None else logvar_p
    return 0.5 * torch.sum(
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).square()) / torch.exp(logvar_p)
        - 1,
        dim=1,
    )
