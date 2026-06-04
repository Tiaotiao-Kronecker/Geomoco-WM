"""Minimal motion-chunk dataset interface.

The first concrete exporter should convert LIBERO demonstrations into chunks
with aligned observation context, future SE(3) motion, and optional actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MotionChunk:
    """One training example for GeoMoCo-style world-motion learning."""

    context: Mapping[str, Any]
    se3_motion: Any
    actions: Any | None = None
    task: str | None = None


class MotionChunkDataset:
    """Thin dataset wrapper around already-materialized motion chunks."""

    def __init__(self, chunks: Sequence[MotionChunk]) -> None:
        self._chunks = list(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> MotionChunk:
        return self._chunks[index]
