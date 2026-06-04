"""Deterministic GeoMoCo-AE baseline."""

from __future__ import annotations

from torch import Tensor, nn

from geomoco_wm.models.common import MLP


class GeoMoCoAE(nn.Module):
    """Encode and reconstruct future SE(3) motion chunks deterministically."""

    def __init__(
        self,
        motion_dim: int,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.encoder = MLP(motion_dim, hidden_dims, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dims, motion_dim)

    def encode(self, motion: Tensor) -> Tensor:
        return self.encoder(motion)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, motion: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(motion)
        reconstruction = self.decode(latent)
        return reconstruction, latent
