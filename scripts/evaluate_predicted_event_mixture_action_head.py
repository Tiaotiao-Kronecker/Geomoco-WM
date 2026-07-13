#!/usr/bin/env python3
"""Repeated evaluation for predicted event-mixture action heads."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import _evaluate  # noqa: E402
from train_predicted_event_mixture_action_head import _load_training_event_labels  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated validation eval for Gate 3.1e action-head checkpoints."
    )
    parser.add_argument("--checkpoint", required=True, help="Action-head model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-eval-passes", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--event-top-m", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--per-window-output-jsonl",
        default=None,
        help="Optional JSONL with one validation row per window for bootstrap CI.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if args.per_window_output_jsonl is not None and args.num_eval_passes != 1:
        raise ValueError("--per-window-output-jsonl requires --num-eval-passes 1")
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])
    _seed_everything(base_seed)

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        max_windows=int(metrics["dataset"]["num_windows"]),
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
    cvae_checkpoint = torch.load(
        Path(metrics["checkpoint"]).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    cvae_metrics = cvae_checkpoint["metrics"]
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
    model = _load_action_head(checkpoint, metrics, device)
    loss_weight_mode = str(metrics.get("loss_weight_mode", "none"))
    transition_loss_weight = float(metrics.get("transition_loss_weight", 1.0))
    aux_gripper_loss_weight = float(metrics.get("aux_gripper_loss_weight", 0.0))
    gripper_residual_mode = str(metrics.get("gripper_residual_mode", "none"))
    gripper_residual_loss_weight = float(metrics.get("gripper_residual_loss_weight", 0.0))
    gripper_route_loss_weight = float(metrics.get("gripper_route_loss_weight", 0.0))
    gripper_step_residual_mode = str(metrics.get("gripper_step_residual_mode", "none"))
    gripper_step_residual_loss_weight = float(
        metrics.get("gripper_step_residual_loss_weight", 0.0)
    )
    gripper_step_loss_weight = float(metrics.get("gripper_step_loss_weight", 0.0))
    gripper_step_target_mode = str(metrics.get("gripper_step_target_mode", "command_state"))
    gripper_step_positive_loss_weight = float(
        metrics.get("gripper_step_positive_loss_weight", 1.0)
    )
    gripper_step_oracle_boundary_residual_loss_weight = float(
        metrics.get("gripper_step_oracle_boundary_residual_loss_weight", 0.0)
    )
    gripper_step_residual_blend = str(
        metrics.get("gripper_step_residual_blend", "all_classes")
    )
    gripper_boundary_index_mode = str(metrics.get("gripper_boundary_index_mode", "none"))
    gripper_boundary_index_loss_weight = float(
        metrics.get("gripper_boundary_index_loss_weight", 0.0)
    )
    gripper_trajectory_residual_mode = str(
        metrics.get("gripper_trajectory_residual_mode", "none")
    )
    gripper_trajectory_residual_loss_weight = float(
        metrics.get("gripper_trajectory_residual_loss_weight", 0.0)
    )
    event_time_conditioning_mode = str(metrics.get("event_time_conditioning_mode", "none"))
    event_time_loss_weight = float(metrics.get("event_time_loss_weight", 0.0))
    temporal_action_decoder_mode = str(metrics.get("temporal_action_decoder_mode", "none"))
    temporal_action_loss_weight = float(metrics.get("temporal_action_loss_weight", 0.0))
    flow_action_decoder_mode = str(metrics.get("flow_action_decoder_mode", "none"))
    flow_action_loss_weight = float(metrics.get("flow_action_loss_weight", 0.0))
    flow_matching_loss_weight = float(metrics.get("flow_matching_loss_weight", 0.0))
    sample_score_mode = str(metrics.get("sample_score_mode", "none"))
    sample_score_loss_weight = float(metrics.get("sample_score_loss_weight", 0.0))
    sample_score_target = str(metrics.get("sample_score_target", "motion_regret"))
    sample_score_loss_type = str(metrics.get("sample_score_loss_type", "soft_ce"))
    sample_score_temperature = float(metrics.get("sample_score_temperature", 0.05))
    event_audit_json = metrics.get("event_mode_audit_json") or _checkpoint_event_audit_json(
        cvae_metrics
    )
    event_labels = _load_training_event_labels(
        event_audit_json,
        loss_weight_mode=loss_weight_mode,
        gripper_residual_mode=gripper_residual_mode,
        gripper_route_loss_weight=gripper_route_loss_weight,
        gripper_step_residual_mode=gripper_step_residual_mode,
        gripper_boundary_index_mode=gripper_boundary_index_mode,
        gripper_trajectory_residual_mode=gripper_trajectory_residual_mode,
        event_time_conditioning_mode=event_time_conditioning_mode,
        temporal_action_decoder_mode=temporal_action_decoder_mode,
        flow_action_decoder_mode=flow_action_decoder_mode,
    )
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
    pass_metrics: list[dict[str, float | None]] = []
    per_window_records: list[dict[str, Any]] | None = (
        [] if args.per_window_output_jsonl is not None else None
    )
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_metrics.append(
            _evaluate(
                model,
                cvae,
                event_probe,
                val_loader,
                device,
                conditioner,
                probe_conditioner,
                event_labels,
                event_classes=tuple(str(value) for value in metrics["cvae_event_classes"]),
                probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
                probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
                event_top_m=event_top_m,
                num_samples=num_samples,
                event_candidate_policy=event_candidate_policy,
                transition_reserve_threshold=transition_reserve_threshold,
                sample_feature_mode=str(metrics.get("sample_feature_mode", "none")),
                future_input_control=future_input_control,
                loss_weight_mode=loss_weight_mode,
                transition_loss_weight=transition_loss_weight,
                aux_gripper_loss_weight=aux_gripper_loss_weight,
                gripper_residual_mode=gripper_residual_mode,
                gripper_residual_loss_weight=gripper_residual_loss_weight,
                gripper_route_loss_weight=gripper_route_loss_weight,
                gripper_step_residual_mode=gripper_step_residual_mode,
                gripper_step_residual_loss_weight=gripper_step_residual_loss_weight,
                gripper_step_loss_weight=gripper_step_loss_weight,
                gripper_step_target_mode=gripper_step_target_mode,
                gripper_step_positive_loss_weight=gripper_step_positive_loss_weight,
                gripper_step_oracle_boundary_residual_loss_weight=(
                    gripper_step_oracle_boundary_residual_loss_weight
                ),
                gripper_boundary_index_mode=gripper_boundary_index_mode,
                gripper_boundary_index_loss_weight=gripper_boundary_index_loss_weight,
                gripper_trajectory_residual_mode=gripper_trajectory_residual_mode,
                gripper_trajectory_residual_loss_weight=(
                    gripper_trajectory_residual_loss_weight
                ),
                event_time_conditioning_mode=event_time_conditioning_mode,
                event_time_loss_weight=event_time_loss_weight,
                temporal_action_decoder_mode=temporal_action_decoder_mode,
                temporal_action_loss_weight=temporal_action_loss_weight,
                flow_action_decoder_mode=flow_action_decoder_mode,
                flow_action_loss_weight=flow_action_loss_weight,
                flow_matching_loss_weight=flow_matching_loss_weight,
                sample_score_mode=sample_score_mode,
                sample_score_loss_weight=sample_score_loss_weight,
                sample_score_target=sample_score_target,
                sample_score_loss_type=sample_score_loss_type,
                sample_score_temperature=sample_score_temperature,
                per_window_records=per_window_records if eval_pass == 0 else None,
            )
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "source_cvae_checkpoint": metrics["checkpoint"],
        "event_probe_checkpoint": metrics["event_probe_checkpoint"],
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "num_samples": num_samples,
        "event_top_m": event_top_m,
        "event_candidate_policy": event_candidate_policy,
        "transition_reserve_threshold": transition_reserve_threshold,
        "future_input_control": future_input_control,
        "batch_size": batch_size,
        "device": str(device),
        "loss_weight_mode": loss_weight_mode,
        "transition_loss_weight": transition_loss_weight,
        "aux_gripper_loss_weight": aux_gripper_loss_weight,
        "gripper_residual_mode": gripper_residual_mode,
        "gripper_residual_loss_weight": gripper_residual_loss_weight,
        "gripper_route_loss_weight": gripper_route_loss_weight,
        "gripper_step_residual_mode": gripper_step_residual_mode,
        "gripper_step_residual_loss_weight": gripper_step_residual_loss_weight,
        "gripper_step_loss_weight": gripper_step_loss_weight,
        "gripper_step_target_mode": gripper_step_target_mode,
        "gripper_step_positive_loss_weight": gripper_step_positive_loss_weight,
        "gripper_step_residual_blend": gripper_step_residual_blend,
        "gripper_step_oracle_boundary_residual_loss_weight": (
            gripper_step_oracle_boundary_residual_loss_weight
        ),
        "gripper_boundary_index_mode": gripper_boundary_index_mode,
        "gripper_boundary_index_loss_weight": gripper_boundary_index_loss_weight,
        "gripper_trajectory_residual_mode": gripper_trajectory_residual_mode,
        "gripper_trajectory_residual_loss_weight": (
            gripper_trajectory_residual_loss_weight
        ),
        "event_time_conditioning_mode": event_time_conditioning_mode,
        "event_time_loss_weight": event_time_loss_weight,
        "temporal_action_decoder_mode": temporal_action_decoder_mode,
        "temporal_action_loss_weight": temporal_action_loss_weight,
        "flow_action_decoder_mode": flow_action_decoder_mode,
        "flow_action_loss_weight": flow_action_loss_weight,
        "flow_matching_loss_weight": flow_matching_loss_weight,
        "sample_score_mode": sample_score_mode,
        "sample_score_loss_weight": sample_score_loss_weight,
        "sample_score_target": sample_score_target,
        "sample_score_loss_type": sample_score_loss_type,
        "sample_score_temperature": sample_score_temperature,
        "pass_metrics": pass_metrics,
        "mean_metrics": _mean_metrics(pass_metrics),
        "std_metrics": _std_metrics(pass_metrics),
    }
    if args.per_window_output_jsonl is not None:
        per_window_path = Path(args.per_window_output_jsonl).expanduser().resolve()
        per_window_path.parent.mkdir(parents=True, exist_ok=True)
        assert per_window_records is not None
        per_window_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_window_records),
            encoding="utf-8",
        )
        output["per_window_output_jsonl"] = str(per_window_path)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "mean_metrics": output["mean_metrics"]}, indent=2))


def _load_action_head(
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
) -> MotionPriorActionHead:
    config = metrics["model_config"]
    model = MotionPriorActionHead(
        context_dim=int(config["context_dim"]),
        motion_dim=int(config["motion_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        conditioning_dim=int(config["conditioning_dim"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        token_dim=int(config["token_dim"]),
        num_heads=int(config["num_heads"]),
        temporal_layers=int(config["temporal_layers"]),
        set_aggregator=str(config.get("set_aggregator", "context_attention")),
        set_query_count=int(config.get("set_query_count", 4)),
        sample_feature_dim=int(config.get("sample_feature_dim", 0)),
        aux_gripper_head=bool(config.get("aux_gripper_head", False)),
        gripper_residual_mode=str(config.get("gripper_residual_mode", "none")),
        gripper_route_count=int(config.get("gripper_route_count", 3)),
        gripper_step_residual_mode=str(config.get("gripper_step_residual_mode", "none")),
        gripper_step_class_count=int(config.get("gripper_step_class_count", 3)),
        gripper_step_residual_blend=str(
            config.get("gripper_step_residual_blend", "all_classes")
        ),
        gripper_boundary_index_mode=str(config.get("gripper_boundary_index_mode", "none")),
        gripper_trajectory_residual_mode=str(
            config.get("gripper_trajectory_residual_mode", "none")
        ),
        event_time_conditioning_mode=str(
            config.get("event_time_conditioning_mode", "none")
        ),
        temporal_action_decoder_mode=str(config.get("temporal_action_decoder_mode", "none")),
        flow_action_decoder_mode=str(config.get("flow_action_decoder_mode", "none")),
        sample_score_mode=str(config.get("sample_score_mode", "none")),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    output: dict[str, float | None] = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = sum(float(value) for value in values) / len(values) if values else None
    return output


def _std_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    means = _mean_metrics(rows)
    output: dict[str, float | None] = {}
    for key, mean in means.items():
        if mean is None:
            output[key] = None
            continue
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if len(values) <= 1:
            output[key] = 0.0
            continue
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        output[key] = variance**0.5
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
