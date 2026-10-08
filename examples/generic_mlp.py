"""Minimal example: replace the two MLPs with any encoder and decoder."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generic import GenericTrainer, TrainerConfig, build_model


class Encoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, z):
        return self.net(z)


def main():
    x = torch.randn(256, 32)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = build_model(Encoder(), Decoder(), latent_dim=16, class_latent_dim=4,
                        style_latent_dim=12, num_classes=2)
    trainer = GenericTrainer(model, TrainerConfig(epochs=5, device="cpu"))
    trainer.fit(loader, val_loader=loader, output_dir="results/generic_mlp")


if __name__ == "__main__":
    main()
