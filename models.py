\
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """
    Dense autoencoder for tabular anomaly detection.

    The default architecture mirrors the uploaded notebook:
    input -> 96 -> 64 -> 48 -> 16 -> latent(4) -> 16 -> 48 -> 64 -> 96 -> input
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (96, 64, 48, 16),
        latent_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer size.")

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.Tanh())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def build_model(model_config: dict, input_dim: int) -> nn.Module:
    model_type = model_config.get("type", "AutoEncoder")
    if model_type.lower() != "autoencoder":
        raise ValueError(f"Unsupported model type: {model_type}")

    hidden_dims = model_config.get("hidden_dims", [96, 64, 48, 16])
    latent_dim = model_config.get("latent_dim", 4)
    dropout = model_config.get("dropout", 0.1)

    return AutoEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    )
