"""A thin, architecture-independent variational encoder/decoder wrapper."""

from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _distribution(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize common encoder outputs to ``(mu, logvar)``."""
    if isinstance(output, dict):
        mu = output.get("mu", output.get("mean"))
        logvar = output.get("logvar", output.get("log_var", output.get("log_sigma")))
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        mu, logvar = output[:2]
    else:
        raise TypeError("Encoder must return (mu, logvar) or a dict containing them.")
    if mu is None or logvar is None:
        raise ValueError("Encoder output is missing mu/logvar.")
    return mu, logvar


class DisentangledAutoencoder(nn.Module):
    """Wrap any encoder and decoder in the shared publication training API.

    The encoder must return ``(mu, logvar)`` (or a dict with those keys), and
    the decoder must accept one latent tensor. Latent coordinates are ordered
    as ``[class, style, auxiliary]``. Only the class block receives the
    supervised classification loss; KL is applied to the complete latent.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        class_latent_dim: int,
        num_classes: int = 2,
        style_latent_dim: int = 0,
        auxiliary_latent_dim: int = 0,
    ):
        super().__init__()
        dims = [class_latent_dim, style_latent_dim, auxiliary_latent_dim]
        if latent_dim <= 0 or sum(dims) != latent_dim:
            raise ValueError("latent_dim must equal class + style + auxiliary dimensions.")
        if class_latent_dim <= 0 or num_classes < 2:
            raise ValueError("A positive class latent and at least two classes are required.")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = int(latent_dim)
        self.class_latent_dim = int(class_latent_dim)
        self.style_latent_dim = int(style_latent_dim)
        self.auxiliary_latent_dim = int(auxiliary_latent_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(class_latent_dim, num_classes)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x, target=None, labels=None, sample=True):
        mu, logvar = _distribution(self.encoder(x))
        z = self.reparameterize(mu, logvar) if sample else mu
        reconstruction = self.decoder(z)
        output: Dict[str, torch.Tensor] = {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "class_logits": self.classifier(mu[:, : self.class_latent_dim]),
        }
        if target is not None:
            output["loss_reconstruction"] = F.mse_loss(reconstruction, target)
        if labels is not None:
            output["loss_classification"] = F.cross_entropy(output["class_logits"], labels.long())
        output["loss_kl"] = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
        return output
