"""Action metrics for normalized LIBERO OSC_POSE action chunks."""

from __future__ import annotations

import math
from typing import Any

import torch

from geomoco_wm.data.action_semantics import default_libero_osc_pose_action_semantics


def action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return flat, split, physical-scale, and SO(3)-geodesic action metrics."""

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    error = pred - target
    metrics = {
        "mse": _to_float(torch.mean(error.square())),
        "mae": _to_float(torch.mean(error.abs())),
    }
    action_dim = int(error.shape[-1])
    if action_dim >= 3:
        translation_error = error[..., :3]
        metrics["translation_mse"] = _to_float(torch.mean(translation_error.square()))
        metrics["translation_mae"] = _to_float(torch.mean(translation_error.abs()))
        metrics["translation_l2"] = _to_float(
            torch.linalg.vector_norm(translation_error, dim=-1).mean()
        )

        translation_scale = _scale_tensor(
            default_libero_osc_pose_action_semantics().translation_scale_m,
            pred,
        )
        translation_m_error = translation_error * translation_scale
        metrics["translation_m_mse"] = _to_float(torch.mean(translation_m_error.square()))
        metrics["translation_m_mae"] = _to_float(torch.mean(translation_m_error.abs()))
        metrics["translation_m_l2"] = _to_float(
            torch.linalg.vector_norm(translation_m_error, dim=-1).mean()
        )
    if action_dim >= 6:
        rotation_error = error[..., 3:6]
        se3_error = error[..., :6]
        metrics["rotation_mse"] = _to_float(torch.mean(rotation_error.square()))
        metrics["rotation_mae"] = _to_float(torch.mean(rotation_error.abs()))
        metrics["rotation_l2"] = _to_float(torch.linalg.vector_norm(rotation_error, dim=-1).mean())
        metrics["se3_mse"] = _to_float(torch.mean(se3_error.square()))
        metrics["se3_mae"] = _to_float(torch.mean(se3_error.abs()))
        metrics["se3_l2"] = _to_float(torch.linalg.vector_norm(se3_error, dim=-1).mean())

        rotation_scale = _scale_tensor(
            default_libero_osc_pose_action_semantics().rotation_scale_rad,
            pred,
        )
        pred_rotvec = pred[..., 3:6] * rotation_scale
        target_rotvec = target[..., 3:6] * rotation_scale
        rotation_rotvec_error = pred_rotvec - target_rotvec
        rotation_geodesic = rotation_geodesic_angle(pred_rotvec, target_rotvec)
        metrics["rotation_rotvec_rad_mse"] = _to_float(torch.mean(rotation_rotvec_error.square()))
        metrics["rotation_rotvec_rad_mae"] = _to_float(torch.mean(rotation_rotvec_error.abs()))
        metrics["rotation_rotvec_rad_l2"] = _to_float(
            torch.linalg.vector_norm(rotation_rotvec_error, dim=-1).mean()
        )
        metrics["rotation_geodesic_rad"] = _to_float(rotation_geodesic.mean())
        metrics["rotation_geodesic_deg"] = _to_float(rotation_geodesic.mean() * (180.0 / math.pi))
    if action_dim > 6:
        gripper_error = error[..., 6:]
        metrics["gripper_mse"] = _to_float(torch.mean(gripper_error.square()))
        metrics["gripper_mae"] = _to_float(torch.mean(gripper_error.abs()))
    return metrics


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors to rotation matrices with Rodrigues' formula."""

    if rotvec.shape[-1] != 3:
        raise ValueError(f"rotvec last dimension must be 3, got {rotvec.shape}")

    rotvec = rotvec.to(dtype=torch.float64) if rotvec.dtype == torch.float16 else rotvec
    skew = _skew(rotvec)
    skew2 = skew @ skew
    theta2 = torch.sum(rotvec.square(), dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    theta2_matrix = theta2.unsqueeze(-1)
    theta_matrix = theta.unsqueeze(-1)
    small = theta2_matrix < 1e-12
    theta_safe = torch.clamp(theta_matrix, min=1e-8)
    theta2_safe = torch.clamp(theta2_matrix, min=1e-16)
    a = torch.where(
        small,
        1.0 - theta2_matrix / 6.0 + theta2_matrix.square() / 120.0,
        torch.sin(theta_matrix) / theta_safe,
    )
    b = torch.where(
        small,
        0.5 - theta2_matrix / 24.0 + theta2_matrix.square() / 720.0,
        (1.0 - torch.cos(theta_matrix)) / theta2_safe,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye.expand(rotvec.shape[:-1] + (3, 3)) + a * skew + b * skew2


def rotation_geodesic_angle(pred_rotvec: torch.Tensor, target_rotvec: torch.Tensor) -> torch.Tensor:
    """Return SO(3) geodesic angles between two batches of rotation vectors."""

    if pred_rotvec.shape != target_rotvec.shape:
        raise ValueError(
            "pred_rotvec and target_rotvec must have the same shape, "
            f"got {pred_rotvec.shape} vs {target_rotvec.shape}"
        )
    pred_matrix = rotvec_to_matrix(pred_rotvec)
    target_matrix = rotvec_to_matrix(target_rotvec)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[..., 0])
    x = vector[..., 0]
    y = vector[..., 1]
    z = vector[..., 2]
    rows = (
        torch.stack((zero, -z, y), dim=-1),
        torch.stack((z, zero, -x), dim=-1),
        torch.stack((-y, x, zero), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def _scale_tensor(values: tuple[float, float, float], reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _to_float(value: torch.Tensor | Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
