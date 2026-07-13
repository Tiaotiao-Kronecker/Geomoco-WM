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


@dataclass(frozen=True)
class VisualCVAEOutput:
    posterior_reconstruction: Tensor
    prior_mean_reconstruction: Tensor
    latent: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor
    condition: Tensor


class VisualConditionedGeoMoCoCVAE(nn.Module):
    """Visual-grounded conditional VAE over future EEF-delta chunks."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        visual_token_dim: int,
        visual_token_count: int,
        conditioning_dim: int = 0,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] = (512, 512),
        query_dim: int = 384,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if visual_token_dim <= 0:
            raise ValueError("visual_token_dim must be positive")
        if visual_token_count <= 0:
            raise ValueError("visual_token_count must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.visual_token_dim = visual_token_dim
        self.visual_token_count = visual_token_count
        self.conditioning_dim = conditioning_dim
        self.latent_dim = latent_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        base_dim = context_dim + conditioning_dim
        condition_dim = base_dim + query_dim
        self.condition_dim = condition_dim
        self.query_net = MLP(base_dim, (query_dim,), query_dim)
        self.visual_proj = (
            nn.Identity()
            if visual_token_dim == query_dim
            else nn.Linear(visual_token_dim, query_dim)
        )
        self.attention = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.posterior = MLP(condition_dim + motion_dim, hidden_dims, latent_dim * 2)
        self.prior = MLP(condition_dim, hidden_dims, latent_dim * 2)
        self.decoder = MLP(condition_dim + latent_dim, hidden_dims, motion_dim)

    def condition(
        self,
        context: Tensor,
        visual_features: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        base = self._base_features(context, conditioning)
        visual_tokens = self._visual_tokens(visual_features)
        query = self.query_net(base).unsqueeze(1)
        keys_values = self.visual_proj(visual_tokens.to(dtype=context.dtype))
        grounded, _ = self.attention(query, keys_values, keys_values, need_weights=False)
        return torch.cat([base, grounded.squeeze(1)], dim=-1)

    def encode_posterior(self, condition: Tensor, motion: Tensor) -> tuple[Tensor, Tensor]:
        stats = self.posterior(torch.cat([condition, motion], dim=-1))
        return stats.chunk(2, dim=-1)

    def encode_prior(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        stats = self.prior(condition)
        return stats.chunk(2, dim=-1)

    def decode(self, condition: Tensor, latent: Tensor) -> Tensor:
        return self.decoder(torch.cat([condition, latent], dim=-1))

    def sample_prior(
        self,
        context: Tensor,
        visual_features: Tensor,
        conditioning: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        condition = self.condition(context, visual_features, conditioning)
        mean, logvar = self.encode_prior(condition)
        latent = reparameterize(mean, logvar)
        return self.decode(condition, latent), latent

    def prior_mean_prediction(
        self,
        context: Tensor,
        visual_features: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        condition = self.condition(context, visual_features, conditioning)
        prior_mean, _ = self.encode_prior(condition)
        return self.decode(condition, prior_mean)

    def forward(
        self,
        context: Tensor,
        visual_features: Tensor,
        motion: Tensor,
        conditioning: Tensor | None = None,
    ) -> VisualCVAEOutput:
        condition = self.condition(context, visual_features, conditioning)
        posterior_mean, posterior_logvar = self.encode_posterior(condition, motion)
        prior_mean, prior_logvar = self.encode_prior(condition)
        latent = reparameterize(posterior_mean, posterior_logvar)
        posterior_reconstruction = self.decode(condition, latent)
        prior_mean_reconstruction = self.decode(condition, prior_mean)
        return VisualCVAEOutput(
            posterior_reconstruction=posterior_reconstruction,
            prior_mean_reconstruction=prior_mean_reconstruction,
            latent=latent,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            condition=condition,
        )

    def _base_features(self, context: Tensor, conditioning: Tensor | None) -> Tensor:
        if self.conditioning_dim == 0:
            if conditioning is not None and conditioning.shape[-1] != 0:
                raise ValueError("conditioning was provided but conditioning_dim is 0")
            return context
        if conditioning is None:
            raise ValueError("conditioning is required when conditioning_dim is positive")
        if conditioning.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                "conditioning batch shape must match context batch shape: "
                f"{conditioning.shape[:-1]} vs {context.shape[:-1]}"
            )
        if conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"conditioning dim must be {self.conditioning_dim}, got {conditioning.shape[-1]}"
            )
        return torch.cat([context, conditioning.to(dtype=context.dtype)], dim=-1)

    def _visual_tokens(self, visual_features: Tensor) -> Tensor:
        expected_dim = self.visual_token_count * self.visual_token_dim
        if visual_features.shape[-1] != expected_dim:
            raise ValueError(
                f"visual feature dim must be {expected_dim}, got {visual_features.shape[-1]}"
            )
        return visual_features.reshape(
            *visual_features.shape[:-1],
            self.visual_token_count,
            self.visual_token_dim,
        )


def gaussian_kl_divergence(
    posterior_mean: Tensor,
    posterior_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
    free_bits: float = 0.0,
) -> Tensor:
    """Mean KL(q || p) for diagonal Gaussian distributions."""

    if free_bits < 0.0:
        raise ValueError("free_bits must be non-negative")
    posterior_var = torch.exp(posterior_logvar)
    prior_var = torch.exp(prior_logvar)
    kl = (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / prior_var
        - 1.0
    )
    kl_per_dim = 0.5 * kl
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=-1).mean()
