"""Reusable training regiment for arbitrary encoder/decoder architectures."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from .model import DisentangledAutoencoder


BatchAdapter = Callable[[object], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def default_batch_adapter(batch):
    """Adapter for ``(x, target, label)`` tuples or common dict keys."""
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("T0")))
        target = batch.get("target", batch.get("target_T0", x))
        labels = batch.get("label", batch.get("labels", batch.get("pcr")))
    else:
        if len(batch) == 2:
            x, labels = batch
            target = x
        else:
            x, target, labels = batch[:3]
    if x is None or labels is None:
        raise KeyError("Batch adapter could not find x and labels.")
    return x, target, labels


@dataclass
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    beta: float = 1e-4
    classification_weight: float = 1.0
    warmup_epochs: int = 0
    device: str = "cuda"
    seed: int = 42
    checkpoint_name: str = "best.pt"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GenericTrainer:
    """Train/evaluate a :class:`DisentangledAutoencoder`.

    The model, dataloaders, and batch adapter are the only task-specific
    pieces. The trainer performs reconstruction + KL + supervised latent
    classification, tracks validation loss/AUC, and saves the best model.
    """

    def __init__(self, model, config: TrainerConfig, batch_adapter: BatchAdapter = default_batch_adapter):
        self.model = model
        self.config = config
        self.batch_adapter = batch_adapter
        requested = config.device
        self.device = torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _step(self, batch, train: bool, epoch: int) -> Dict[str, float]:
        x, target, labels = self.batch_adapter(batch)
        x, target, labels = x.to(self.device), target.to(self.device), labels.to(self.device)
        with torch.set_grad_enabled(train):
            out = self.model(x, target=target, labels=labels, sample=train)
            beta = 0.0 if epoch < self.config.warmup_epochs else self.config.beta
            loss = (
                out["loss_reconstruction"]
                + beta * out["loss_kl"]
                + self.config.classification_weight * out["loss_classification"]
            )
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        return {"loss": float(loss.detach()), "reconstruction": float(out["loss_reconstruction"].detach()),
                "kl": float(out["loss_kl"].detach()), "classification": float(out["loss_classification"].detach())}

    def _run_epoch(self, loader, epoch: int, train: bool):
        self.model.train(train)
        values = []
        scores, labels = [], []
        for batch in loader:
            values.append(self._step(batch, train=train, epoch=epoch))
            with torch.no_grad():
                x, _, y = self.batch_adapter(batch)
                logits = self.model(x.to(self.device), sample=False)["class_logits"]
                scores.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy())
                labels.extend(y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y))
        mean = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
        pred = (np.asarray(scores) >= 0.5).astype(int)
        mean["accuracy"] = float(accuracy_score(labels, pred))
        mean["auc"] = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        return mean

    def fit(self, train_loader, val_loader=None, output_dir: Optional[str] = None):
        seed_everything(self.config.seed)
        output = Path(output_dir) if output_dir else None
        if output:
            output.mkdir(parents=True, exist_ok=True)
        best = float("inf")
        history = []
        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            val_metrics = self._run_epoch(val_loader, epoch, train=False) if val_loader else train_metrics
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)
            if output and val_metrics["loss"] < best:
                best = val_metrics["loss"]
                torch.save(self.model.state_dict(), output / self.config.checkpoint_name)
            if output:
                (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True))
        return history


def build_model(encoder, decoder, latent_dim, class_latent_dim, num_classes=2,
                style_latent_dim=0, auxiliary_latent_dim=0):
    """Convenience factory for the public API."""
    return DisentangledAutoencoder(
        encoder=encoder, decoder=decoder, latent_dim=latent_dim,
        class_latent_dim=class_latent_dim, num_classes=num_classes,
        style_latent_dim=style_latent_dim, auxiliary_latent_dim=auxiliary_latent_dim,
    )
