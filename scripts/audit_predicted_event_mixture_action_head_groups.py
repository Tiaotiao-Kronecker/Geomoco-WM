#!/usr/bin/env python3
"""Group stress audit for predicted event-mixture action heads."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.predicted_event_mixture import event_label_is_transition  # noqa: E402
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.data.action_semantics import default_libero_osc_pose_action_semantics  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics, rotation_geodesic_angle  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_predicted_event_mixture_action_head import _load_action_head  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _average_metrics,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import (  # noqa: E402
    _apply_future_input_control,
    _boundary_index_predicted_actions,
    _event_mode_for_record,
    _gripper_boundary_step_targets,
    _load_event_label_records,
    _oracle_boundary_step_routed_actions,
    _predicted_boundary_step_routed_actions,
    _predicted_event_future_inputs,
    _replace_action_gripper,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate 3.2a group stress audit for event-aware predicted-event "
            "mixture action heads."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Action-head model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument("--num-eval-passes", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--event-top-m", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--predicted-boundary-thresholds", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    predicted_boundary_thresholds = _parse_thresholds(args.predicted_boundary_thresholds)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    cvae_checkpoint_path = Path(metrics["checkpoint"]).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        max_windows=int(metrics["dataset"]["num_windows"]),
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
    conditioner = _conditioner_from_metrics(metrics["conditioning"])
    visual_token_config = _resolve_visual_token_config(
        dataset,
        cvae_metrics["visual_token_config"]["visual_token_count"],
        cvae_metrics["visual_token_config"]["visual_token_dim"],
    )
    cvae = _load_model(
        cvae_checkpoint,
        context_dim=int(metrics["dataset"]["context_dim"]),
        motion_dim=int(metrics["dataset"]["motion_dim"]),
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim + len(metrics["cvae_event_classes"]),
        device=device,
    )
    _freeze_module(cvae)
    event_probe, probe_metrics, probe_conditioner = _load_event_probe(
        metrics["event_probe_checkpoint"],
        device,
    )
    _freeze_module(event_probe)
    action_head = _load_action_head(checkpoint, metrics, device)
    event_audit_json = args.event_mode_audit_json or _checkpoint_event_audit_json(cvae_metrics)
    if event_audit_json is None:
        raise ValueError("--event-mode-audit-json is required when absent from cVAE checkpoint")
    event_labels = _load_event_label_records(event_audit_json)
    include_oracle_boundary_step_routed = (
        metrics.get("gripper_step_residual_mode") == "event_step"
        and metrics.get("gripper_step_target_mode") == "boundary_start"
    )
    include_boundary_index_pred = metrics.get("gripper_boundary_index_mode") == "boundary_index"

    _, val_indices = _split_indices(
        dataset,
        float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"]),
        int(metrics["seed"]),
        metrics["split_by"],
    )
    batch_size = int(args.batch_size or metrics["batch_size"])
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    num_samples = int(args.num_samples or metrics["num_samples"])
    event_top_m = int(args.event_top_m or metrics["event_top_m"])
    event_candidate_policy = str(metrics.get("event_candidate_policy", "topk"))
    transition_reserve_threshold = float(metrics.get("transition_reserve_threshold", 0.0))
    future_input_control = str(metrics.get("future_input_control") or "real")

    pass_reports: list[dict[str, Any]] = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_reports.append(
            _audit_pass(
                action_head,
                cvae,
                event_probe,
                val_loader,
                device,
                conditioner,
                probe_conditioner,
                event_labels=event_labels,
                cvae_event_classes=tuple(str(value) for value in metrics["cvae_event_classes"]),
                probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
                probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
                event_top_m=event_top_m,
                num_samples=num_samples,
                event_candidate_policy=event_candidate_policy,
                transition_reserve_threshold=transition_reserve_threshold,
                sample_feature_mode=str(metrics.get("sample_feature_mode", "none")),
                future_input_control=future_input_control,
                include_oracle_boundary_step_routed=include_oracle_boundary_step_routed,
                include_boundary_index_pred=include_boundary_index_pred,
                predicted_boundary_thresholds=predicted_boundary_thresholds,
                max_batches=args.max_batches,
            )
        )

    output = {
        "checkpoint": str(checkpoint_path),
        "source_cvae_checkpoint": str(cvae_checkpoint_path),
        "event_probe_checkpoint": metrics["event_probe_checkpoint"],
        "event_mode_audit_json": str(Path(event_audit_json).expanduser().resolve()),
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "num_samples": num_samples,
        "event_top_m": event_top_m,
        "event_candidate_policy": event_candidate_policy,
        "transition_reserve_threshold": transition_reserve_threshold,
        "batch_size": batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "dataset": metrics["dataset"],
        "split_by": metrics["split_by"],
        "input_mode": metrics["input_mode"],
        "sample_feature_mode": metrics.get("sample_feature_mode", "none"),
        "future_input_control": future_input_control,
        "include_oracle_boundary_step_routed": include_oracle_boundary_step_routed,
        "include_boundary_index_pred": include_boundary_index_pred,
        "predicted_boundary_thresholds": list(predicted_boundary_thresholds),
        "pass_reports": pass_reports,
        "mean_report": _mean_reports(pass_reports),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "overall": output["mean_report"]["overall"],
                "worst_groups": output["mean_report"]["worst_groups"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@torch.no_grad()
def _audit_pass(
    action_head: MotionPriorActionHead,
    cvae: Any,
    event_probe: Any,
    loader: Any,
    device: torch.device,
    conditioner: Any,
    probe_conditioner: Any,
    *,
    event_labels: dict[str, dict[str, Any]],
    cvae_event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    future_input_control: str,
    include_oracle_boundary_step_routed: bool,
    include_boundary_index_pred: bool,
    predicted_boundary_thresholds: tuple[float, ...],
    max_batches: int | None,
) -> dict[str, Any]:
    action_head.eval()
    cvae.eval()
    event_probe.eval()
    overall_totals: dict[str, float] = {}
    overall_count = 0
    group_totals: dict[str, dict[str, float]] = defaultdict(dict)
    group_counts: dict[str, int] = defaultdict(int)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _batch_conditioning_for_action_head(batch, conditioner, device)
        future_inputs, sample_features = _predicted_event_future_inputs(
            cvae,
            event_probe,
            batch,
            context,
            conditioning,
            device,
            probe_conditioner,
            event_classes=cvae_event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=event_top_m,
            num_samples=num_samples,
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
        )
        future_inputs, sample_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            future_input_control,
        )
        output = action_head.forward_with_aux(context, future_inputs, conditioning, sample_features)
        pred_actions = output["actions"]
        if pred_actions is None:
            raise ValueError("action head output is missing actions")
        batch_metrics = action_metrics(pred_actions, actions)
        routed_actions = output.get("routed_actions")
        routed_batch_metrics = (
            {f"routed_{key}": value for key, value in action_metrics(routed_actions, actions).items()}
            if routed_actions is not None
            else {}
        )
        step_routed_actions = output.get("step_routed_actions")
        step_routed_batch_metrics = (
            {
                f"step_routed_{key}": value
                for key, value in action_metrics(step_routed_actions, actions).items()
            }
            if step_routed_actions is not None
            else {}
        )
        trajectory_routed_actions = output.get("trajectory_routed_actions")
        trajectory_routed_batch_metrics = (
            {
                f"trajectory_routed_{key}": value
                for key, value in action_metrics(trajectory_routed_actions, actions).items()
            }
            if trajectory_routed_actions is not None
            else {}
        )
        temporal_actions = output.get("temporal_actions")
        temporal_action_batch_metrics = (
            {
                f"temporal_action_{key}": value
                for key, value in action_metrics(temporal_actions, actions).items()
            }
            if temporal_actions is not None
            else {}
        )
        oracle_step_routed_actions = None
        if include_oracle_boundary_step_routed:
            step_residuals = output.get("gripper_step_residuals")
            if step_residuals is None:
                raise ValueError("oracle boundary audit requires gripper_step_residuals")
            step_targets = _gripper_boundary_step_targets(
                batch,
                event_labels,
                horizon=int(pred_actions.shape[1]),
                device=device,
            )
            oracle_step_routed_actions = _oracle_boundary_step_routed_actions(
                pred_actions,
                step_targets,
                step_residuals,
            )
        oracle_step_routed_batch_metrics = (
            {
                f"oracle_step_routed_{key}": value
                for key, value in action_metrics(oracle_step_routed_actions, actions).items()
            }
            if oracle_step_routed_actions is not None
            else {}
        )
        predicted_boundary_actions = _predicted_boundary_actions_from_output(
            pred_actions,
            output,
            include=include_oracle_boundary_step_routed,
            thresholds=predicted_boundary_thresholds,
        )
        boundary_index_actions = _boundary_index_actions_from_output(
            pred_actions,
            output,
            include=include_boundary_index_pred,
        )
        predicted_boundary_batch_metrics = {
            prefix: {
                f"{prefix}_{key}": value
                for key, value in action_metrics(routed_actions, actions).items()
            }
            for prefix, routed_actions in predicted_boundary_actions.items()
        }
        boundary_index_batch_metrics = (
            {
                f"boundary_index_pred_{key}": value
                for key, value in action_metrics(boundary_index_actions, actions).items()
            }
            if boundary_index_actions is not None
            else {}
        )
        batch_size = int(context.shape[0])
        for key, value in batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in routed_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in step_routed_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in trajectory_routed_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in temporal_action_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in oracle_step_routed_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for batch_metric in predicted_boundary_batch_metrics.values():
            for key, value in batch_metric.items():
                overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        for key, value in boundary_index_batch_metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        overall_count += batch_size

        per_item = _per_item_action_metrics(pred_actions, actions)
        routed_per_item = (
            _per_item_action_metrics(routed_actions, actions)
            if routed_actions is not None
            else None
        )
        step_routed_per_item = (
            _per_item_action_metrics(step_routed_actions, actions)
            if step_routed_actions is not None
            else None
        )
        trajectory_routed_per_item = (
            _per_item_action_metrics(trajectory_routed_actions, actions)
            if trajectory_routed_actions is not None
            else None
        )
        temporal_action_per_item = (
            _per_item_action_metrics(temporal_actions, actions)
            if temporal_actions is not None
            else None
        )
        oracle_step_routed_per_item = (
            _per_item_action_metrics(oracle_step_routed_actions, actions)
            if oracle_step_routed_actions is not None
            else None
        )
        predicted_boundary_per_item = {
            prefix: _per_item_action_metrics(routed_actions, actions)
            for prefix, routed_actions in predicted_boundary_actions.items()
        }
        boundary_index_per_item = (
            _per_item_action_metrics(boundary_index_actions, actions)
            if boundary_index_actions is not None
            else None
        )
        aux_per_item = None
        if output["aux_gripper"] is not None:
            aux_actions = _replace_action_gripper(pred_actions, output["aux_gripper"])
            aux_per_item = _per_item_action_metrics(aux_actions, actions)
        for row in range(batch_size):
            window_id = _batch_string_at(batch["window_id"], row)
            suite = _batch_string_at(batch["suite_name"], row)
            task = _batch_string_at(batch["task_id"], row)
            event_mode = _event_mode_for_record(event_labels.get(window_id)) or "unknown"
            event_family = _event_family(event_mode)
            transition_group = "transition" if event_label_is_transition(event_mode) else "sustain"
            groups = (
                "all",
                f"suite/{suite}",
                f"task/{suite}/{task}",
                f"event_mode/{event_mode}",
                f"event_family/{event_family}",
                f"transition_group/{transition_group}",
            )
            for group in groups:
                group_counts[group] += 1
                totals = group_totals[group]
                for key, values in per_item.items():
                    totals[key] = totals.get(key, 0.0) + float(values[row])
                if routed_per_item is not None:
                    for key, values in routed_per_item.items():
                        routed_key = f"routed_{key}"
                        totals[routed_key] = totals.get(routed_key, 0.0) + float(values[row])
                if step_routed_per_item is not None:
                    for key, values in step_routed_per_item.items():
                        step_routed_key = f"step_routed_{key}"
                        totals[step_routed_key] = totals.get(step_routed_key, 0.0) + float(
                            values[row]
                        )
                if trajectory_routed_per_item is not None:
                    for key, values in trajectory_routed_per_item.items():
                        trajectory_key = f"trajectory_routed_{key}"
                        totals[trajectory_key] = totals.get(trajectory_key, 0.0) + float(
                            values[row]
                        )
                if temporal_action_per_item is not None:
                    for key, values in temporal_action_per_item.items():
                        temporal_key = f"temporal_action_{key}"
                        totals[temporal_key] = totals.get(temporal_key, 0.0) + float(
                            values[row]
                        )
                if oracle_step_routed_per_item is not None:
                    for key, values in oracle_step_routed_per_item.items():
                        oracle_key = f"oracle_step_routed_{key}"
                        totals[oracle_key] = totals.get(oracle_key, 0.0) + float(values[row])
                for prefix, per_item_values in predicted_boundary_per_item.items():
                    for key, values in per_item_values.items():
                        predicted_key = f"{prefix}_{key}"
                        totals[predicted_key] = totals.get(predicted_key, 0.0) + float(
                            values[row]
                        )
                if boundary_index_per_item is not None:
                    for key, values in boundary_index_per_item.items():
                        boundary_index_key = f"boundary_index_pred_{key}"
                        totals[boundary_index_key] = totals.get(
                            boundary_index_key,
                            0.0,
                        ) + float(values[row])
                if aux_per_item is not None:
                    for key, values in aux_per_item.items():
                        aux_key = f"aux_replaced_{key}"
                        totals[aux_key] = totals.get(aux_key, 0.0) + float(values[row])
    return _finalize_report(overall_totals, overall_count, group_totals, group_counts)


def _predicted_boundary_actions_from_output(
    pred_actions: torch.Tensor,
    output: dict[str, torch.Tensor | None],
    *,
    include: bool,
    thresholds: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if not include:
        return {}
    step_logits = output.get("gripper_step_logits")
    step_residuals = output.get("gripper_step_residuals")
    if step_logits is None or step_residuals is None:
        raise ValueError("predicted boundary routing requires logits and residuals")
    routed = {
        "pred_boundary_argmax": _predicted_boundary_step_routed_actions(
            pred_actions,
            step_logits,
            step_residuals,
            threshold=None,
        )
    }
    for threshold in thresholds:
        key = f"pred_boundary_t{_threshold_key(threshold)}"
        routed[key] = _predicted_boundary_step_routed_actions(
            pred_actions,
            step_logits,
            step_residuals,
            threshold=threshold,
        )
    return routed


def _boundary_index_actions_from_output(
    pred_actions: torch.Tensor,
    output: dict[str, torch.Tensor | None],
    *,
    include: bool,
) -> torch.Tensor | None:
    if not include:
        return None
    boundary_logits = output.get("gripper_boundary_index_logits")
    step_residuals = output.get("gripper_step_residuals")
    if boundary_logits is None or step_residuals is None:
        raise ValueError("boundary-index routing requires logits and step residuals")
    return _boundary_index_predicted_actions(pred_actions, boundary_logits, step_residuals)


def _batch_conditioning_for_action_head(
    batch: dict[str, object],
    conditioner: Any,
    device: torch.device,
) -> torch.Tensor | None:
    from train_future_motion_predictor import _batch_conditioning

    return _batch_conditioning(batch, conditioner, device, include_visual=False)


def _per_item_action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    error = pred - target
    metrics = {
        "mse": error.square().mean(dim=(1, 2)).detach().cpu(),
        "mae": error.abs().mean(dim=(1, 2)).detach().cpu(),
    }
    if error.shape[-1] >= 3:
        translation = error[..., :3]
        metrics["translation_mse"] = translation.square().mean(dim=(1, 2)).detach().cpu()
        translation_scale = torch.as_tensor(
            default_libero_osc_pose_action_semantics().translation_scale_m,
            dtype=pred.dtype,
            device=pred.device,
        )
        translation_m = translation * translation_scale
        metrics["translation_m_mse"] = translation_m.square().mean(dim=(1, 2)).detach().cpu()
    if error.shape[-1] >= 6:
        rotation = error[..., 3:6]
        metrics["rotation_mse"] = rotation.square().mean(dim=(1, 2)).detach().cpu()
        rotation_scale = torch.as_tensor(
            default_libero_osc_pose_action_semantics().rotation_scale_rad,
            dtype=pred.dtype,
            device=pred.device,
        )
        rotation_geodesic = rotation_geodesic_angle(
            pred[..., 3:6] * rotation_scale,
            target[..., 3:6] * rotation_scale,
        )
        metrics["rotation_geodesic_deg"] = (
            rotation_geodesic.mean(dim=1) * (180.0 / math.pi)
        ).detach().cpu()
    if error.shape[-1] > 6:
        gripper = error[..., 6:]
        metrics["gripper_mse"] = gripper.square().mean(dim=(1, 2)).detach().cpu()
    return metrics


def _finalize_report(
    overall_totals: dict[str, float],
    overall_count: int,
    group_totals: dict[str, dict[str, float]],
    group_counts: dict[str, int],
) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int]] = {}
    for group, totals in sorted(group_totals.items()):
        count = group_counts[group]
        row: dict[str, float | int] = {"count": count}
        row.update({key: value / count for key, value in sorted(totals.items())})
        groups[group] = row
    return {
        "overall": _average_metrics(overall_totals, overall_count),
        "groups": groups,
        "worst_groups": _worst_groups(groups, metric="mse", min_count=25, limit=12),
    }


def _mean_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"overall": {}, "groups": {}, "worst_groups": []}
    overall_keys = set().union(*(report["overall"].keys() for report in reports))
    overall = {
        key: _mean_optional([report["overall"].get(key) for report in reports])
        for key in sorted(overall_keys)
    }
    group_names = set().union(*(report["groups"].keys() for report in reports))
    groups: dict[str, dict[str, float | int]] = {}
    for group_name in sorted(group_names):
        rows = [report["groups"][group_name] for report in reports if group_name in report["groups"]]
        metric_keys = set().union(*(row.keys() for row in rows)) - {"count"}
        row_out: dict[str, float | int] = {
            "count": int(round(_mean_optional([row.get("count") for row in rows]) or 0.0))
        }
        row_out.update(
            {
                key: _mean_optional([row.get(key) for row in rows])
                for key in sorted(metric_keys)
            }
        )
        groups[group_name] = row_out
    return {
        "overall": overall,
        "groups": groups,
        "worst_groups": _worst_groups(groups, metric="mse", min_count=25, limit=12),
    }


def _worst_groups(
    groups: dict[str, dict[str, float | int]],
    *,
    metric: str,
    min_count: int,
    limit: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for group, values in groups.items():
        count = int(values.get("count", 0))
        value = values.get(metric)
        if count < min_count or value is None:
            continue
        rows.append({"group": group, "count": count, metric: float(value)})
    rows.sort(key=lambda row: float(row[metric]), reverse=True)
    return rows[:limit]


def _mean_optional(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _event_family(event_mode: str) -> str:
    if "::" in event_mode:
        return event_mode.split("::", 1)[0]
    return event_mode


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive when provided")
    if args.event_top_m is not None and args.event_top_m <= 0:
        raise ValueError("--event-top-m must be positive when provided")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")
    _parse_thresholds(args.predicted_boundary_thresholds)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("predicted boundary thresholds must be between 0 and 1")
    return thresholds


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
