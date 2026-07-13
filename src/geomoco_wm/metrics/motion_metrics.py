"""Metrics for flattened future EEF-delta motion chunks."""

from __future__ import annotations

from typing import Any

import torch


def future_motion_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    step_dim: int = 6,
) -> dict[str, float]:
    """Return flat and coordinate-split metrics for future EEF deltas.

    The exported LIBERO windows currently store future EEF deltas as 6D rows:
    position delta in dims 0:3 and orientation-coordinate delta in dims 3:6.
    The orientation part is intentionally reported as coordinate error here,
    not as an SO(3) geodesic, because the observation representation is not the
    same controller rotvec action representation audited for action metrics.
    """

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.shape[-1] % step_dim != 0:
        raise ValueError(f"last dimension {pred.shape[-1]} is not divisible by step_dim={step_dim}")

    error = pred - target
    metrics = {
        "mse": _to_float(torch.mean(error.square())),
        "mae": _to_float(torch.mean(error.abs())),
        "l2": _to_float(torch.linalg.vector_norm(error, dim=-1).mean()),
    }

    rows = error.reshape(*error.shape[:-1], -1, step_dim)
    if step_dim >= 3:
        translation_error = rows[..., :3]
        metrics["translation_mse"] = _to_float(torch.mean(translation_error.square()))
        metrics["translation_mae"] = _to_float(torch.mean(translation_error.abs()))
        metrics["translation_l2"] = _to_float(
            torch.linalg.vector_norm(translation_error, dim=-1).mean()
        )
    if step_dim >= 6:
        orientation_error = rows[..., 3:6]
        metrics["orientation_coord_mse"] = _to_float(torch.mean(orientation_error.square()))
        metrics["orientation_coord_mae"] = _to_float(torch.mean(orientation_error.abs()))
        metrics["orientation_coord_l2"] = _to_float(
            torch.linalg.vector_norm(orientation_error, dim=-1).mean()
        )
    return metrics


def _to_float(value: torch.Tensor | Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
