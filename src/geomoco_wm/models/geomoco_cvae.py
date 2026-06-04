"""Conditional GeoMoCo motion VAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from geomoco_wm.models.common import MLP, reparameterize


@dataclass(frozen=True)
class CVAEOutput:
    reconstruction: Tensor
    latent: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor


class GeoMoCoCVAE(nn.Module):
    """Conditional distribution over future geometric motion latents."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.posterior = MLP(context_dim + motion_dim, hidden_dims, latent_dim * 2)
        self.prior = MLP(context_dim, hidden_dims, latent_dim * 2)
        self.decoder = MLP(context_dim + latent_dim, hidden_dims, motion_dim)

    def encode_posterior(self, context: Tensor, motion: Tensor) -> tuple[Tensor, Tensor]:
        stats = self.posterior(torch.cat([context, motion], dim=-1))
        return stats.chunk(2, dim=-1)

    def encode_prior(self, context: Tensor) -> tuple[Tensor, Tensor]:
        stats = self.prior(context)
        return stats.chunk(2, dim=-1)

    def decode(self, context: Tensor, latent: Tensor) -> Tensor:
        return self.decoder(torch.cat([context, latent], dim=-1))

    def sample_prior(self, context: Tensor) -> tuple[Tensor, Tensor]:
        mean, logvar = self.encode_prior(context)
        latent = reparameterize(mean, logvar)
        return self.decode(context, latent), latent

    def forward(self, context: Tensor, motion: Tensor) -> CVAEOutput:
        posterior_mean, posterior_logvar = self.encode_posterior(context, motion)
        prior_mean, prior_logvar = self.encode_prior(context)
        latent = reparameterize(posterior_mean, posterior_logvar)
        reconstruction = self.decode(context, latent)
        return CVAEOutput(
            reconstruction=reconstruction,
            latent=latent,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )
