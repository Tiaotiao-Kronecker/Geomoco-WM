"""Adapter boundary for AMPLIFY motion-token baselines."""

from __future__ import annotations

from typing import Any


class AMPLIFYAdapter:
    """Normalize AMPLIFY outputs into the shared action-decoder interface."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def encode_motion(self, batch: Any) -> Any:
        raise NotImplementedError("Connect AMPLIFY token extraction here.")
