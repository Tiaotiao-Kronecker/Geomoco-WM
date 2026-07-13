#!/usr/bin/env python3
"""Train a post-hoc residual adapter over a frozen Gate 3.4 action head."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.predicted_event_mixture import (  # noqa: E402
    event_label_is_transition,
    map_event_probabilities,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import (  # noqa: E402
    MotionPriorActionHead,
    PostHocActionResidualAdapter,
)
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _batch_conditioning,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_event_mode_probe import _batch_features  # noqa: E402
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import (  # noqa: E402
    _add_metric_values,
    _apply_future_input_control,
    _batch_string_at,
    _event_mode_for_record,
    _finalize_metric_values,
    _load_training_event_labels,
    _predicted_event_future_inputs,
    _weighted_action_loss,
)


GATE34_TEMPORAL_BASELINE_MSE = 0.034262
GATE34_TEMPORAL_TRANSITION_MSE = 0.131311


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gate 3.5b post-hoc residual adapters over frozen Gate 3.4."
    )
    parser.add_argument("--frozen-action-head-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-hidden-dims", default="256,256")
    parser.add_argument("--adapter-step-dim", type=int, default=32)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument(
        "--residual-gate-mode",
        default="none",
        choices=["none", "predicted_transition_prob", "oracle_transition"],
        help=(
            "Optionally gate the post-hoc residual. predicted_transition_prob "
            "uses deployable event-probe probabilities; oracle_transition is a "
            "diagnostic upper bound only."
        ),
    )
    parser.add_argument(
        "--residual-gate-threshold",
        type=float,
        default=None,
        help="Optional threshold that binarizes the transition gate.",
    )
    parser.add_argument(
        "--residual-leak",
        type=float,
        default=0.0,
        help="Minimum residual gate value outside transition-likely windows.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--selection-metric",
        default="adapter_mse",
        choices=[
            "adapter_mse",
            "adapter_weighted_loss",
            "adapter_transition_mse",
            "adapter_gripper_mse",
            "decoder_gain_mse",
        ],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    frozen_path = Path(args.frozen_action_head_checkpoint).expanduser().resolve()
    device = _resolve_device(args.device) if not args.dry_run else torch.device("cpu")
    checkpoint = torch.load(frozen_path, map_location=device, weights_only=False)
    frozen_metrics = checkpoint["metrics"]
    seed = int(args.seed if args.seed is not None else frozen_metrics["seed"])
    _seed_everything(seed)
    adapter_hidden_dims = _parse_hidden_dims(args.adapter_hidden_dims)

    dataset, cvae, event_probe, conditioner, probe_conditioner, cvae_metrics, probe_metrics = (
        _load_frozen_stack(
            frozen_metrics,
            device,
            max_windows=args.max_windows,
        )
    )
    event_classes = tuple(str(value) for value in frozen_metrics["cvae_event_classes"])
    event_labels = _load_event_labels(frozen_metrics, cvae_metrics)
    frozen_model = _load_action_head(checkpoint, frozen_metrics, device)
    _freeze_module(frozen_model)
    feature_dim = _feature_dim(frozen_metrics)
    action_dim = int(frozen_metrics["model_config"]["action_dim"])
    horizon = int(frozen_metrics["model_config"]["horizon"])
    adapter = PostHocActionResidualAdapter(
        feature_dim=feature_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=adapter_hidden_dims,
        step_dim=args.adapter_step_dim,
        dropout=args.adapter_dropout,
    ).to(device)
    train_indices, val_indices = _split_indices(
        dataset,
        float(frozen_metrics["train_size"])
        / float(frozen_metrics["train_size"] + frozen_metrics["val_size"]),
        int(frozen_metrics["seed"]),
        str(frozen_metrics["split_by"]),
    )
    batch_size = int(args.batch_size or frozen_metrics["batch_size"])
    train_loader = _make_loader(dataset, train_indices, batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "frozen_action_head_checkpoint": str(frozen_path),
                    "dataset": dataset.spec().to_dict(),
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "batch_size": batch_size,
                    "seed": seed,
                    "feature_dim": feature_dim,
                    "adapter_config": _adapter_config(
                        feature_dim=feature_dim,
                        action_dim=action_dim,
                        horizon=horizon,
                        hidden_dims=adapter_hidden_dims,
                        step_dim=args.adapter_step_dim,
                        dropout=args.adapter_dropout,
                        residual_gate_mode=args.residual_gate_mode,
                        residual_gate_threshold=args.residual_gate_threshold,
                        residual_leak=args.residual_leak,
                    ),
                    "frozen_sample_feature_mode": frozen_metrics["sample_feature_mode"],
                    "frozen_event_candidate_policy": frozen_metrics.get(
                        "event_candidate_policy",
                        "topk",
                    ),
                    "frozen_transition_reserve_threshold": frozen_metrics.get(
                        "transition_reserve_threshold",
                        0.0,
                    ),
                    "frozen_future_input_control": frozen_metrics.get(
                        "future_input_control",
                        "real",
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_metric = float("inf")
    history: list[dict[str, float | int | None]] = []
    eval_kwargs = _eval_kwargs(
        frozen_metrics,
        probe_metrics,
        event_classes,
        event_labels,
        residual_gate_mode=args.residual_gate_mode,
        residual_gate_threshold=args.residual_gate_threshold,
        residual_leak=args.residual_leak,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            frozen_model,
            adapter,
            cvae,
            event_probe,
            train_loader,
            optimizer,
            device,
            conditioner,
            probe_conditioner,
            **eval_kwargs,
        )
        val_metrics = _evaluate(
            frozen_model,
            adapter,
            cvae,
            event_probe,
            val_loader,
            device,
            conditioner,
            probe_conditioner,
            **eval_kwargs,
        )
        row: dict[str, float | int | None] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate = val_metrics.get(args.selection_metric)
        selection_value = (
            -float(candidate)
            if args.selection_metric == "decoder_gain_mse" and candidate is not None
            else float(candidate)
            if candidate is not None
            else None
        )
        if selection_value is not None and selection_value < best_metric:
            best_metric = selection_value
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        adapter.load_state_dict(best_state)
    final_metrics = _evaluate(
        frozen_model,
        adapter,
        cvae,
        event_probe,
        val_loader,
        device,
        conditioner,
        probe_conditioner,
        **eval_kwargs,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "input_mode": "posthoc_residual_adapter_over_frozen_predicted_event_mixture_action_head",
        "gate": "3.5c" if args.residual_gate_mode != "none" else "3.5b",
        "device": str(device),
        "dataset": dataset.spec().to_dict(),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": seed,
        "split_by": frozen_metrics["split_by"],
        "motion_mode": frozen_metrics["motion_mode"],
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_value": (
            -best_metric if args.selection_metric == "decoder_gain_mse" and best_state else best_metric
        )
        if best_state is not None
        else None,
        "history": history,
        "final_action_metrics": final_metrics,
        "gate34_temporal_baseline_mse": GATE34_TEMPORAL_BASELINE_MSE,
        "gate34_temporal_transition_mse": GATE34_TEMPORAL_TRANSITION_MSE,
        "frozen_action_head_checkpoint": str(frozen_path),
        "frozen_action_head_metrics": _frozen_summary(frozen_metrics),
        "checkpoint": frozen_metrics["checkpoint"],
        "event_probe_checkpoint": frozen_metrics["event_probe_checkpoint"],
        "event_top_m": frozen_metrics["event_top_m"],
        "num_samples": frozen_metrics["num_samples"],
        "event_candidate_policy": frozen_metrics.get("event_candidate_policy", "topk"),
        "transition_reserve_threshold": frozen_metrics.get("transition_reserve_threshold", 0.0),
        "sample_feature_mode": frozen_metrics["sample_feature_mode"],
        "future_input_control": frozen_metrics.get("future_input_control") or "real",
        "sample_feature_dim": frozen_metrics["sample_feature_dim"],
        "conditioning": frozen_metrics["conditioning"],
        "cvae_event_classes": list(event_classes),
        "visual_feature_cache": frozen_metrics["visual_feature_cache"],
        "event_mode_audit_json": str(Path(_event_audit_json(frozen_metrics, cvae_metrics)).resolve()),
        "adapter_config": _adapter_config(
            feature_dim=feature_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=adapter_hidden_dims,
            step_dim=args.adapter_step_dim,
            dropout=args.adapter_dropout,
            residual_gate_mode=args.residual_gate_mode,
            residual_gate_threshold=args.residual_gate_threshold,
            residual_leak=args.residual_leak,
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": adapter.state_dict(), "metrics": metrics},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "best_epoch": best_epoch,
                    "final_action_metrics": final_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.adapter_step_dim <= 0:
        raise ValueError("--adapter-step-dim must be positive")
    if args.adapter_dropout < 0.0:
        raise ValueError("--adapter-dropout must be non-negative")
    if args.residual_gate_threshold is not None and not 0.0 <= args.residual_gate_threshold <= 1.0:
        raise ValueError("--residual-gate-threshold must be between 0 and 1")
    if not 0.0 <= args.residual_leak <= 1.0:
        raise ValueError("--residual-leak must be between 0 and 1")


def _load_frozen_stack(
    frozen_metrics: dict[str, Any],
    device: torch.device,
    *,
    max_windows: int | None,
) -> tuple[
    OracleActionWindowDataset,
    torch.nn.Module,
    torch.nn.Module,
    CategoricalConditioner,
    CategoricalConditioner,
    dict[str, Any],
    dict[str, Any],
]:
    dataset = OracleActionWindowDataset(
        frozen_metrics["dataset"]["windows_jsonl"],
        max_windows=max_windows or int(frozen_metrics["dataset"]["num_windows"]),
        motion_mode=str(frozen_metrics["motion_mode"]),
        visual_feature_cache_path=frozen_metrics["visual_feature_cache"],
    )
    cvae_checkpoint = torch.load(
        Path(frozen_metrics["checkpoint"]).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    cvae_metrics = cvae_checkpoint["metrics"]
    conditioner = _conditioner_from_metrics(frozen_metrics["conditioning"])
    visual_token_config = _resolve_visual_token_config(
        dataset,
        cvae_metrics["visual_token_config"]["visual_token_count"],
        cvae_metrics["visual_token_config"]["visual_token_dim"],
    )
    cvae = _load_model(
        cvae_checkpoint,
        context_dim=int(frozen_metrics["dataset"]["context_dim"]),
        motion_dim=int(frozen_metrics["dataset"]["motion_dim"]),
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim + len(frozen_metrics["cvae_event_classes"]),
        device=device,
    )
    _freeze_module(cvae)
    event_probe, probe_metrics, probe_conditioner = _load_event_probe(
        frozen_metrics["event_probe_checkpoint"],
        device,
    )
    _freeze_module(event_probe)
    return dataset, cvae, event_probe, conditioner, probe_conditioner, cvae_metrics, probe_metrics


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


def _load_event_labels(
    frozen_metrics: dict[str, Any],
    cvae_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    event_audit_json = _event_audit_json(frozen_metrics, cvae_metrics)
    return _load_training_event_labels(
        event_audit_json,
        loss_weight_mode=str(frozen_metrics.get("loss_weight_mode", "none")),
        temporal_action_decoder_mode="sequence_mlp",
    )


def _event_audit_json(
    frozen_metrics: dict[str, Any],
    cvae_metrics: dict[str, Any],
) -> str:
    event_audit_json = frozen_metrics.get("event_mode_audit_json") or _checkpoint_event_audit_json(
        cvae_metrics
    )
    if event_audit_json is None:
        raise ValueError("event audit JSON is required for post-hoc adapter metrics")
    return str(event_audit_json)


def _feature_dim(metrics: dict[str, Any]) -> int:
    config = metrics["model_config"]
    token_dim = int(config["token_dim"])
    return token_dim * 3 + 3


def _eval_kwargs(
    frozen_metrics: dict[str, Any],
    probe_metrics: dict[str, Any],
    event_classes: tuple[str, ...],
    event_labels: dict[str, Any] | None,
    *,
    residual_gate_mode: str,
    residual_gate_threshold: float | None,
    residual_leak: float,
) -> dict[str, Any]:
    return {
        "event_labels": event_labels,
        "event_classes": event_classes,
        "probe_class_names": tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
        "probe_input_variant": str(probe_metrics["probe"]["input_variant"]),
        "event_top_m": int(frozen_metrics["event_top_m"]),
        "num_samples": int(frozen_metrics["num_samples"]),
        "event_candidate_policy": str(frozen_metrics.get("event_candidate_policy", "topk")),
        "transition_reserve_threshold": float(
            frozen_metrics.get("transition_reserve_threshold", 0.0)
        ),
        "sample_feature_mode": str(frozen_metrics["sample_feature_mode"]),
        "future_input_control": str(frozen_metrics.get("future_input_control") or "real"),
        "loss_weight_mode": str(frozen_metrics.get("loss_weight_mode", "none")),
        "transition_loss_weight": float(frozen_metrics.get("transition_loss_weight", 1.0)),
        "residual_gate_mode": residual_gate_mode,
        "residual_gate_threshold": residual_gate_threshold,
        "residual_leak": residual_leak,
    }


def _run_epoch(
    frozen_model: MotionPriorActionHead,
    adapter: PostHocActionResidualAdapter,
    cvae: torch.nn.Module,
    event_probe: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    **kwargs: Any,
) -> dict[str, float | None]:
    adapter.train()
    frozen_model.eval()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch_metrics, batch_counts, loss = _adapter_batch(
            frozen_model,
            adapter,
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            require_loss=True,
            **kwargs,
        )
        optimizer.zero_grad(set_to_none=True)
        if loss is None:
            raise RuntimeError("training batch did not return a loss")
        loss.backward()
        optimizer.step()
        _add_metric_values(totals, counts, batch_metrics, batch_counts)
    return _finalize_metric_values(totals, counts)


@torch.no_grad()
def _evaluate(
    frozen_model: MotionPriorActionHead,
    adapter: PostHocActionResidualAdapter,
    cvae: torch.nn.Module,
    event_probe: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    **kwargs: Any,
) -> dict[str, float | None]:
    adapter.eval()
    frozen_model.eval()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch_metrics, batch_counts, _ = _adapter_batch(
            frozen_model,
            adapter,
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            require_loss=False,
            **kwargs,
        )
        _add_metric_values(totals, counts, batch_metrics, batch_counts)
    return _finalize_metric_values(totals, counts)


def _adapter_batch(
    frozen_model: MotionPriorActionHead,
    adapter: PostHocActionResidualAdapter,
    cvae: torch.nn.Module,
    event_probe: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    *,
    event_labels: dict[str, Any] | None,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    future_input_control: str,
    loss_weight_mode: str,
    transition_loss_weight: float,
    residual_gate_mode: str,
    residual_gate_threshold: float | None,
    residual_leak: float,
    require_loss: bool,
) -> tuple[dict[str, float], dict[str, int], torch.Tensor | None]:
    context = batch["context"].to(device)
    actions = batch["actions"].to(device)
    action_conditioning = _batch_conditioning(
        batch,
        conditioner,
        device,
        include_visual=False,
    )
    gate_values = _residual_gate_values(
        event_probe,
        batch,
        context,
        device,
        probe_conditioner,
        probe_class_names=probe_class_names,
        probe_input_variant=probe_input_variant,
        event_classes=event_classes,
        event_labels=event_labels,
        residual_gate_mode=residual_gate_mode,
        residual_gate_threshold=residual_gate_threshold,
        residual_leak=residual_leak,
    )
    future_inputs, sample_features = _predicted_event_future_inputs(
        cvae,
        event_probe,
        batch,
        context,
        action_conditioning,
        device,
        probe_conditioner,
        event_classes=event_classes,
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
    with torch.no_grad():
        frozen_output = frozen_model.forward_with_aux(
            context,
            future_inputs,
            action_conditioning,
            sample_features,
        )
        features = frozen_output["features"]
        temporal_actions = frozen_output["temporal_actions"]
        if features is None or temporal_actions is None:
            raise ValueError("frozen Gate 3.4 checkpoint must return features and temporal_actions")
        features = features.detach()
        temporal_actions = temporal_actions.detach()
    adapter_output = adapter(features, temporal_actions)
    adapter_actions = adapter_output["adapter_actions"]
    raw_residual = adapter_output["adapter_residual"]
    if gate_values is not None:
        adapter_actions = temporal_actions + gate_values.reshape(-1, 1, 1) * raw_residual
    adapter_loss, adapter_loss_metrics, adapter_loss_counts = _weighted_action_loss(
        adapter_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    _, temporal_loss_metrics, temporal_loss_counts = _weighted_action_loss(
        temporal_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    batch_size = int(context.shape[0])
    metrics = {
        f"adapter_{key}": value
        for key, value in action_metrics(adapter_actions.detach(), actions).items()
    }
    counts = {key: batch_size for key in metrics}
    for key, value in adapter_loss_metrics.items():
        metrics[f"adapter_{key}"] = value
        counts[f"adapter_{key}"] = adapter_loss_counts[key]
    temporal_metrics = {
        f"frozen_temporal_{key}": value
        for key, value in action_metrics(temporal_actions, actions).items()
    }
    metrics.update(temporal_metrics)
    counts.update({key: batch_size for key in temporal_metrics})
    for key, value in temporal_loss_metrics.items():
        metrics[f"frozen_temporal_{key}"] = value
        counts[f"frozen_temporal_{key}"] = temporal_loss_counts[key]
    residual_target = actions - temporal_actions
    residual = adapter_actions - temporal_actions
    metrics["adapter_residual_mse"] = float(
        (residual - residual_target).square().mean().detach().cpu()
    )
    metrics["adapter_residual_abs_mean"] = float(residual.abs().mean().detach().cpu())
    metrics["adapter_raw_residual_mse"] = float(
        (raw_residual - residual_target).square().mean().detach().cpu()
    )
    metrics["adapter_raw_residual_abs_mean"] = float(raw_residual.abs().mean().detach().cpu())
    metrics["decoder_gain_mse"] = metrics["frozen_temporal_mse"] - metrics["adapter_mse"]
    counts["adapter_residual_mse"] = batch_size
    counts["adapter_residual_abs_mean"] = batch_size
    counts["adapter_raw_residual_mse"] = batch_size
    counts["adapter_raw_residual_abs_mean"] = batch_size
    if gate_values is not None:
        metrics["adapter_residual_gate_mean"] = float(gate_values.mean().detach().cpu())
        metrics["adapter_residual_gate_min"] = float(gate_values.min().detach().cpu())
        metrics["adapter_residual_gate_max"] = float(gate_values.max().detach().cpu())
        counts["adapter_residual_gate_mean"] = batch_size
        counts["adapter_residual_gate_min"] = batch_size
        counts["adapter_residual_gate_max"] = batch_size
    counts["decoder_gain_mse"] = batch_size
    if "frozen_temporal_transition_mse" in metrics and "adapter_transition_mse" in metrics:
        metrics["decoder_gain_transition_mse"] = (
            metrics["frozen_temporal_transition_mse"] - metrics["adapter_transition_mse"]
        )
        counts["decoder_gain_transition_mse"] = adapter_loss_counts["transition_mse"]
    loss = adapter_loss if require_loss else None
    return metrics, counts, loss


def _adapter_config(
    *,
    feature_dim: int,
    action_dim: int,
    horizon: int,
    hidden_dims: tuple[int, ...],
    step_dim: int,
    dropout: float,
    residual_gate_mode: str,
    residual_gate_threshold: float | None,
    residual_leak: float,
) -> dict[str, Any]:
    return {
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "hidden_dims": list(hidden_dims),
        "step_dim": step_dim,
        "dropout": dropout,
        "zero_init_output": True,
        "residual_gate_mode": residual_gate_mode,
        "residual_gate_threshold": residual_gate_threshold,
        "residual_leak": residual_leak,
    }


def _residual_gate_values(
    event_probe: torch.nn.Module,
    batch: dict[str, object],
    context: torch.Tensor,
    device: torch.device,
    probe_conditioner: CategoricalConditioner,
    *,
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_classes: tuple[str, ...],
    event_labels: dict[str, Any] | None,
    residual_gate_mode: str,
    residual_gate_threshold: float | None,
    residual_leak: float,
) -> torch.Tensor | None:
    if residual_gate_mode == "none":
        return None
    if residual_gate_mode == "predicted_transition_prob":
        source_probs = torch.softmax(
            event_probe(_batch_features(batch, probe_conditioner, device, probe_input_variant)),
            dim=-1,
        )
        event_probs = map_event_probabilities(source_probs, probe_class_names, event_classes)
        transition_mask = torch.tensor(
            [event_label_is_transition(label) for label in event_classes],
            dtype=torch.bool,
            device=device,
        )
        gate = event_probs[:, transition_mask].sum(dim=-1)
    elif residual_gate_mode == "oracle_transition":
        if event_labels is None:
            raise ValueError("oracle_transition residual gate requires event labels")
        values = [
            event_label_is_transition(
                _event_mode_for_record(
                    event_labels.get(_batch_string_at(batch["window_id"], row))
                )
            )
            for row in range(int(context.shape[0]))
        ]
        gate = torch.tensor(values, dtype=context.dtype, device=device)
    else:
        raise ValueError(f"unsupported residual_gate_mode {residual_gate_mode!r}")
    if residual_gate_threshold is not None:
        gate = (gate >= residual_gate_threshold).to(dtype=context.dtype)
    else:
        gate = gate.to(dtype=context.dtype)
    return residual_leak + (1.0 - residual_leak) * gate


def _frozen_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    final = metrics.get("final_action_metrics", {})
    return {
        "seed": metrics.get("seed"),
        "sample_feature_mode": metrics.get("sample_feature_mode"),
        "future_input_control": metrics.get("future_input_control") or "real",
        "best_epoch": metrics.get("best_epoch"),
        "best_selection_metric": metrics.get("best_selection_metric"),
        "best_selection_value": metrics.get("best_selection_value"),
        "final_temporal_action_mse": final.get("temporal_action_mse"),
        "final_temporal_action_transition_mse": final.get("temporal_action_transition_mse"),
        "final_temporal_action_gripper_mse": final.get("temporal_action_gripper_mse"),
        "final_mse": final.get("mse"),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
