"""Per-window action metrics for bootstrap reliability audits."""

from __future__ import annotations

from typing import Any

import torch


def per_window_action_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> list[dict[str, float]]:
    """Return loss-style action metrics for each window in a batch."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"pred and target must be [B,H,A], got {pred.shape}")

    error = pred - target
    rows: dict[str, torch.Tensor] = {
        "mse": error.square().mean(dim=(1, 2)),
        "mae": error.abs().mean(dim=(1, 2)),
    }
    action_dim = int(error.shape[-1])
    if action_dim >= 3:
        translation_error = error[..., :3]
        rows["translation_mse"] = translation_error.square().mean(dim=(1, 2))
        rows["translation_mae"] = translation_error.abs().mean(dim=(1, 2))
        rows["translation_l2"] = torch.linalg.vector_norm(translation_error, dim=-1).mean(dim=1)
    if action_dim >= 6:
        rotation_error = error[..., 3:6]
        se3_error = error[..., :6]
        rows["rotation_mse"] = rotation_error.square().mean(dim=(1, 2))
        rows["rotation_mae"] = rotation_error.abs().mean(dim=(1, 2))
        rows["rotation_l2"] = torch.linalg.vector_norm(rotation_error, dim=-1).mean(dim=1)
        rows["se3_mse"] = se3_error.square().mean(dim=(1, 2))
        rows["se3_mae"] = se3_error.abs().mean(dim=(1, 2))
        rows["se3_l2"] = torch.linalg.vector_norm(se3_error, dim=-1).mean(dim=1)
    if action_dim > 6:
        gripper_error = error[..., 6:]
        rows["gripper_mse"] = gripper_error.square().mean(dim=(1, 2))
        rows["gripper_mae"] = gripper_error.abs().mean(dim=(1, 2))

    return _tensor_columns_to_records(rows)


def window_metadata_records(
    batch: dict[str, Any],
    event_labels: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return serializable metadata rows aligned to a batch."""

    batch_size = _batch_size_from_values(batch["window_id"])
    rows: list[dict[str, Any]] = []
    for row in range(batch_size):
        window_id = _batch_string_at(batch["window_id"], row)
        label = event_labels.get(window_id) if event_labels is not None else None
        record = {
            "window_id": window_id,
            "episode_id": _batch_string_at(batch["episode_id"], row),
            "task_id": _batch_string_at(batch["task_id"], row),
            "suite_name": _batch_string_at(batch["suite_name"], row),
        }
        if isinstance(label, dict):
            for key in (
                "event_type",
                "event_mode",
                "timing_bin",
                "event_step",
                "close_step",
                "open_step",
            ):
                record[key] = label.get(key)
        elif isinstance(label, str):
            record["event_mode"] = label
            record["event_type"] = label.split("::", 1)[0]
        rows.append(record)
    return rows


def merge_window_metric_records(
    metadata: list[dict[str, Any]],
    metrics: list[dict[str, float]],
    *,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Merge aligned metadata and metric rows."""

    if len(metadata) != len(metrics):
        raise ValueError(f"metadata/metrics length mismatch: {len(metadata)} vs {len(metrics)}")
    output: list[dict[str, Any]] = []
    for meta, values in zip(metadata, metrics, strict=True):
        row = dict(meta)
        for key, value in values.items():
            metric_key = f"{prefix}_{key}" if prefix else key
            row[metric_key] = value
        output.append(row)
    return output


def _tensor_columns_to_records(columns: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    if not columns:
        return []
    detached = {
        key: value.detach().cpu().to(dtype=torch.float64).tolist()
        for key, value in columns.items()
    }
    first = next(iter(detached.values()))
    rows: list[dict[str, float]] = []
    for index in range(len(first)):
        rows.append({key: float(values[index]) for key, values in detached.items()})
    return rows


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _batch_size_from_values(values: object) -> int:
    if isinstance(values, (list, tuple)):
        return len(values)
    return int(len(values))  # type: ignore[arg-type]
