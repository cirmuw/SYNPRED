"""Task-agnostic encoder/decoder training API."""

from .model import DisentangledAutoencoder
from .trainer import GenericTrainer, TrainerConfig, build_model, default_batch_adapter

__all__ = [
    "DisentangledAutoencoder",
    "GenericTrainer",
    "TrainerConfig",
    "default_batch_adapter",
    "build_model",
]
