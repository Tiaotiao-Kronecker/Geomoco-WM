"""Shared neural-network helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Small MLP used by the initial baselines."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for out_dim in hidden_dims:
            in_dim = prev_dim
            layers.extend([nn.Linear(in_dim, out_dim), activation()])
            prev_dim = out_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
    """Sample a Gaussian latent with the reparameterization trick."""

    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn_like(std)
