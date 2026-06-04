"""Adapter boundary for ZipMo motion-latent baselines."""

from __future__ import annotations

from typing import Any


class ZipMoAdapter:
    """Normalize ZipMo latents into the shared action-decoder interface."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def encode_motion(self, batch: Any) -> Any:
        raise NotImplementedError("Connect ZipMo latent extraction here.")
