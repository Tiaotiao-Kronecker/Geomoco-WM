#!/usr/bin/env python3
"""Train a deterministic context -> future EEF-delta predictor."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.window_dataset import MOTION_MODES, OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.metrics.motion_metrics import future_motion_metrics  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from geomoco_wm.models.future_motion_predictor import (  # noqa: E402
    FutureMotionPredictor,
    StepwiseVisualCrossAttentionFutureMotionPredictor,
    VisualCrossAttentionFutureMotionPredictor,
)


@dataclass(frozen=True)
class CategoricalConditioner:
    condition_on: str
    vocab: tuple[str, ...]
    index_by_label: dict[str, int]

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_on": self.condition_on,
            "dim": self.dim,
            "vocab": list(self.vocab),
            "encoding": "one_hot" if self.dim else "none",
            "vocab_source": "full dataset metadata labels",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the first deterministic learned future-motion prior."
    )
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default="outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl",
        help="Input windows.jsonl produced by scripts/export_libero_windows.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/future_motion_predictor/gate2_deterministic_smoke",
        help="Output directory for metrics.json and model.pt.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--split-by",
        default="episode",
        choices=["window", "episode"],
        help="Use episode-level splits for real comparisons.",
    )
    parser.add_argument(
        "--motion-mode",
        default="future_delta",
        choices=MOTION_MODES,
        help="Prediction target stored in the exported window dataset.",
    )
    parser.add_argument(
        "--action-decoder-checkpoint",
        default=None,
        help="Optional oracle action-decoder checkpoint for downstream action metrics.",
    )
    parser.add_argument(
        "--action-aware-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional training loss weight for frozen-action-decoder MSE. "
            "Requires --action-decoder-checkpoint when positive."
        ),
    )
    parser.add_argument(
        "--condition-on",
        default="none",
        choices=["none", "suite", "task", "suite_task"],
        help="Optional categorical metadata one-hot appended to the future-motion prior input.",
    )
    parser.add_argument(
        "--visual-feature-cache",
        default=None,
        help="Optional HDF5 visual feature cache aligned to windows.jsonl.",
    )
    parser.add_argument(
        "--visual-fusion",
        default="mlp_conditioning",
        choices=["mlp_conditioning", "cross_attention", "stepwise_cross_attention"],
        help="How to fuse visual features when a visual cache is provided.",
    )
    parser.add_argument("--visual-token-count", type=int, default=None)
    parser.add_argument("--visual-token-dim", type=int, default=None)
    parser.add_argument("--visual-query-dim", type=int, default=384)
    parser.add_argument("--visual-num-heads", type=int, default=4)
    parser.add_argument(
        "--future-step-dim",
        type=int,
        default=6,
        help="Step width for stepwise_cross_attention future-motion decoding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print shapes only.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch JSON logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _seed_everything(args.seed)
    dataset = OracleActionWindowDataset(
        args.windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=args.motion_mode,
        visual_feature_cache_path=args.visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_conditioner(dataset, args.condition_on)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "dataset": spec.to_dict(),
                    "conditioning": conditioner.to_dict(),
                    "action_aware_loss_weight": args.action_aware_loss_weight,
                    "visual_fusion": args.visual_fusion,
                    "total_conditioning_dim": _conditioning_dim(
                        conditioner,
                        spec.visual_dim,
                        args.visual_fusion,
                    ),
                    "visual_token_config": _resolve_visual_token_config(
                        dataset,
                        args.visual_token_count,
                        args.visual_token_dim,
                    )
                    if _uses_visual_attention(args.visual_fusion)
                    else None,
                    "future_step_dim": args.future_step_dim,
                },
                indent=2,
            )
        )
        return

    device = _resolve_device(args.device)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, args.split_by)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    visual_token_config = (
        _resolve_visual_token_config(dataset, args.visual_token_count, args.visual_token_dim)
        if _uses_visual_attention(args.visual_fusion)
        else None
    )
    model = _build_model(
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        conditioner_dim=conditioner.dim,
        visual_dim=spec.visual_dim,
        visual_fusion=args.visual_fusion,
        visual_token_config=visual_token_config,
        visual_query_dim=args.visual_query_dim,
        visual_num_heads=args.visual_num_heads,
        future_step_dim=args.future_step_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    action_decoder_for_loss = None
    action_decoder_config = None
    if args.action_aware_loss_weight < 0.0:
        raise ValueError("--action-aware-loss-weight must be non-negative")
    if args.action_aware_loss_weight > 0.0:
        if not args.action_decoder_checkpoint:
            raise ValueError(
                "--action-decoder-checkpoint is required when "
                "--action-aware-loss-weight is positive"
            )
        action_decoder_for_loss, action_decoder_config = _load_action_decoder(
            args.action_decoder_checkpoint,
            device,
        )
        _freeze_action_decoder(action_decoder_for_loss)

    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer,
            conditioner,
            args.visual_fusion,
            action_decoder_for_loss,
            args.action_aware_loss_weight,
            args.motion_mode,
        )
        val_metrics = _evaluate_motion(
            model,
            val_loader,
            device,
            loss_fn,
            conditioner,
            args.visual_fusion,
            action_decoder_for_loss,
            args.action_aware_loss_weight,
            args.motion_mode,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    downstream_action_metrics = None
    if args.action_decoder_checkpoint:
        if action_decoder_for_loss is None:
            action_decoder, action_decoder_config = _load_action_decoder(
                args.action_decoder_checkpoint,
                device,
            )
        else:
            action_decoder = action_decoder_for_loss
        downstream_action_metrics = _evaluate_predicted_motion_actions(
            model,
            action_decoder,
            val_loader,
            device,
            conditioner,
            args.visual_fusion,
            args.motion_mode,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": spec.to_dict(),
        "device": str(device),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "action_aware_loss_weight": args.action_aware_loss_weight,
        "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
        "motion_mode": args.motion_mode,
        "seed": args.seed,
        "split_by": args.split_by,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(args.visual_feature_cache).expanduser())
        if args.visual_feature_cache
        else None,
        "visual_fusion": args.visual_fusion,
        "visual_token_config": visual_token_config,
        "visual_query_dim": args.visual_query_dim,
        "visual_num_heads": args.visual_num_heads,
        "future_step_dim": args.future_step_dim,
        "total_conditioning_dim": _conditioning_dim(
            conditioner,
            spec.visual_dim,
            args.visual_fusion,
        ),
        "history": history,
        "final": history[-1] if history else {},
        "action_decoder_checkpoint": str(Path(args.action_decoder_checkpoint).expanduser())
        if args.action_decoder_checkpoint
        else None,
        "action_decoder_config": action_decoder_config,
        "predicted_motion_action_metrics": downstream_action_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "final": metrics["final"],
                    "predicted_motion_action_metrics": downstream_action_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _run_epoch(
    model: (
        FutureMotionPredictor
        | VisualCrossAttentionFutureMotionPredictor
        | StepwiseVisualCrossAttentionFutureMotionPredictor
    ),
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    conditioner: CategoricalConditioner,
    visual_fusion: str,
    action_decoder: ActionDecoder | None,
    action_aware_loss_weight: float,
    motion_mode: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        conditioning = _batch_conditioning(
            batch,
            conditioner,
            device,
            include_visual=visual_fusion == "mlp_conditioning",
        )
        optimizer.zero_grad(set_to_none=True)
        pred = _predict_motion(model, context, batch, conditioning, device, visual_fusion)
        motion_loss = loss_fn(pred, motion)
        action_loss = _action_aware_loss(action_decoder, context, pred, batch, device, loss_fn)
        loss = _combined_training_loss(
            motion_loss,
            action_loss,
            action_aware_loss_weight,
        )
        loss.backward()
        optimizer.step()
        batch_size = int(context.shape[0])
        batch_metrics = _prediction_metrics(pred.detach(), motion, motion_mode)
        batch_metrics["motion_loss"] = float(motion_loss.detach().cpu())
        if action_loss is not None:
            batch_metrics["action_loss"] = float(action_loss.detach().cpu())
        batch_metrics["loss"] = float(loss.detach().cpu())
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate_motion(
    model: (
        FutureMotionPredictor
        | VisualCrossAttentionFutureMotionPredictor
        | StepwiseVisualCrossAttentionFutureMotionPredictor
    ),
    loader: DataLoader | None,
    device: torch.device,
    loss_fn: nn.Module,
    conditioner: CategoricalConditioner,
    visual_fusion: str,
    action_decoder: ActionDecoder | None = None,
    action_aware_loss_weight: float = 0.0,
    motion_mode: str = "future_delta",
) -> dict[str, float | None]:
    if loader is None:
        return {"mse": None, "mae": None, "loss": None}
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        conditioning = _batch_conditioning(
            batch,
            conditioner,
            device,
            include_visual=visual_fusion == "mlp_conditioning",
        )
        pred = _predict_motion(model, context, batch, conditioning, device, visual_fusion)
        motion_loss = loss_fn(pred, motion)
        action_loss = _action_aware_loss(action_decoder, context, pred, batch, device, loss_fn)
        loss = _combined_training_loss(
            motion_loss,
            action_loss,
            action_aware_loss_weight,
        )
        batch_size = int(context.shape[0])
        batch_metrics = _prediction_metrics(pred, motion, motion_mode)
        batch_metrics["motion_loss"] = float(motion_loss.cpu())
        if action_loss is not None:
            batch_metrics["action_loss"] = float(action_loss.cpu())
        batch_metrics["loss"] = float(loss.cpu())
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate_predicted_motion_actions(
    motion_model: FutureMotionPredictor,
    action_decoder: ActionDecoder,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    visual_fusion: str,
    motion_mode: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"mse": None, "mae": None}
    motion_model.eval()
    action_decoder.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _batch_conditioning(
            batch,
            conditioner,
            device,
            include_visual=visual_fusion == "mlp_conditioning",
        )
        pred_motion = _predict_motion(motion_model, context, batch, conditioning, device, visual_fusion)
        pred_actions = action_decoder(context, pred_motion)
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions, actions)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


def _prediction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    motion_mode: str,
) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if motion_mode == "future_delta":
        return future_motion_metrics(pred, target)
    if motion_mode == "future_gripper":
        return _flat_gripper_metrics(pred, target)
    if motion_mode == "future_delta_gripper":
        if pred.shape[-1] < 2:
            raise ValueError("future_delta_gripper prediction must have at least 2 dims")
        horizon = pred.shape[-1] // 7
        if horizon <= 0 or pred.shape[-1] != horizon * 7:
            raise ValueError(
                "future_delta_gripper motion dim must be 7 * horizon "
                f"for flattened [6H + H] features, got {pred.shape[-1]}"
            )
        eef_dim = horizon * 6
        metrics = _prefix("flat", _flat_metrics(pred, target))
        metrics.update(_prefix("eef", future_motion_metrics(pred[..., :eef_dim], target[..., :eef_dim])))
        metrics.update(_prefix("gripper", _flat_gripper_metrics(pred[..., eef_dim:], target[..., eef_dim:])))
        metrics["mse"] = metrics["flat_mse"]
        metrics["mae"] = metrics["flat_mae"]
        return metrics
    if motion_mode == "none":
        return _flat_metrics(pred, target)
    raise ValueError(f"unsupported motion_mode {motion_mode!r}")


def _flat_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    return {
        "mse": float(torch.mean(error.square()).detach().cpu()),
        "mae": float(torch.mean(error.abs()).detach().cpu()),
        "l2": float(torch.linalg.vector_norm(error, dim=-1).mean().detach().cpu()),
    }


def _flat_gripper_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = _flat_metrics(pred, target)
    metrics["gripper_mse"] = metrics["mse"]
    metrics["gripper_mae"] = metrics["mae"]
    metrics["gripper_l2"] = metrics["l2"]
    return metrics


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _load_action_decoder(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ActionDecoder, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint.get("metrics", {})
    dataset = metrics.get("dataset", {})
    hidden_dims = tuple(int(value) for value in metrics.get("hidden_dims", (256, 256)))
    model = ActionDecoder(
        context_dim=int(dataset["context_dim"]),
        motion_rep_dim=int(dataset["motion_dim"]),
        action_dim=int(dataset["action_dim"]),
        horizon=int(dataset["horizon"]),
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "context_dim": int(dataset["context_dim"]),
        "motion_dim": int(dataset["motion_dim"]),
        "action_dim": int(dataset["action_dim"]),
        "horizon": int(dataset["horizon"]),
        "hidden_dims": list(hidden_dims),
        "motion_mode": metrics.get("motion_mode"),
        "split_by": metrics.get("split_by"),
    }


def _freeze_action_decoder(action_decoder: ActionDecoder) -> None:
    action_decoder.eval()
    for parameter in action_decoder.parameters():
        parameter.requires_grad_(False)


def _action_aware_loss(
    action_decoder: ActionDecoder | None,
    context: torch.Tensor,
    pred_motion: torch.Tensor,
    batch: dict[str, object],
    device: torch.device,
    loss_fn: nn.Module,
) -> torch.Tensor | None:
    if action_decoder is None:
        return None
    actions = batch["actions"].to(device)
    pred_actions = action_decoder(context, pred_motion)
    return loss_fn(pred_actions, actions)


def _combined_training_loss(
    motion_loss: torch.Tensor,
    action_loss: torch.Tensor | None,
    action_aware_loss_weight: float,
) -> torch.Tensor:
    if action_aware_loss_weight < 0.0:
        raise ValueError("action_aware_loss_weight must be non-negative")
    if action_loss is None or action_aware_loss_weight == 0.0:
        return motion_loss
    return motion_loss + action_aware_loss_weight * action_loss


def _build_model(
    *,
    context_dim: int,
    motion_dim: int,
    hidden_dims: tuple[int, ...],
    conditioner_dim: int,
    visual_dim: int,
    visual_fusion: str,
    visual_token_config: dict[str, int] | None,
    visual_query_dim: int,
    visual_num_heads: int,
    future_step_dim: int,
) -> (
    FutureMotionPredictor
    | VisualCrossAttentionFutureMotionPredictor
    | StepwiseVisualCrossAttentionFutureMotionPredictor
):
    if visual_fusion == "mlp_conditioning":
        return FutureMotionPredictor(
            context_dim=context_dim,
            motion_dim=motion_dim,
            hidden_dims=hidden_dims,
            conditioning_dim=conditioner_dim + visual_dim,
        )
    if visual_fusion == "cross_attention":
        if visual_dim <= 0:
            raise ValueError("cross_attention visual_fusion requires --visual-feature-cache")
        if visual_token_config is None:
            raise ValueError("cross_attention visual_fusion requires visual token config")
        return VisualCrossAttentionFutureMotionPredictor(
            context_dim=context_dim,
            motion_dim=motion_dim,
            visual_token_dim=int(visual_token_config["visual_token_dim"]),
            visual_token_count=int(visual_token_config["visual_token_count"]),
            hidden_dims=hidden_dims,
            conditioning_dim=conditioner_dim,
            query_dim=visual_query_dim,
            num_heads=visual_num_heads,
        )
    if visual_fusion == "stepwise_cross_attention":
        if visual_dim <= 0:
            raise ValueError("stepwise_cross_attention visual_fusion requires --visual-feature-cache")
        if visual_token_config is None:
            raise ValueError("stepwise_cross_attention visual_fusion requires visual token config")
        return StepwiseVisualCrossAttentionFutureMotionPredictor(
            context_dim=context_dim,
            motion_dim=motion_dim,
            visual_token_dim=int(visual_token_config["visual_token_dim"]),
            visual_token_count=int(visual_token_config["visual_token_count"]),
            hidden_dims=hidden_dims,
            conditioning_dim=conditioner_dim,
            query_dim=visual_query_dim,
            num_heads=visual_num_heads,
            future_step_dim=future_step_dim,
        )
    raise ValueError(
        "visual_fusion must be one of: "
        "mlp_conditioning, cross_attention, stepwise_cross_attention"
    )


def _predict_motion(
    model: (
        FutureMotionPredictor
        | VisualCrossAttentionFutureMotionPredictor
        | StepwiseVisualCrossAttentionFutureMotionPredictor
    ),
    context: torch.Tensor,
    batch: dict[str, object],
    conditioning: torch.Tensor | None,
    device: torch.device,
    visual_fusion: str,
) -> torch.Tensor:
    if visual_fusion == "mlp_conditioning":
        return model(context, conditioning)  # type: ignore[misc]
    if _uses_visual_attention(visual_fusion):
        visual = batch.get("visual")
        if not isinstance(visual, torch.Tensor):
            raise ValueError(f"{visual_fusion} visual_fusion requires batch['visual']")
        return model(context, visual.to(device=device, dtype=torch.float32), conditioning)  # type: ignore[misc]
    raise ValueError(
        "visual_fusion must be one of: "
        "mlp_conditioning, cross_attention, stepwise_cross_attention"
    )


def _conditioning_dim(
    conditioner: CategoricalConditioner,
    visual_dim: int,
    visual_fusion: str,
) -> int:
    if visual_fusion == "mlp_conditioning":
        return conditioner.dim + visual_dim
    if _uses_visual_attention(visual_fusion):
        return conditioner.dim
    raise ValueError(
        "visual_fusion must be one of: "
        "mlp_conditioning, cross_attention, stepwise_cross_attention"
    )


def _uses_visual_attention(visual_fusion: str) -> bool:
    return visual_fusion in {"cross_attention", "stepwise_cross_attention"}


def _resolve_visual_token_config(
    dataset: OracleActionWindowDataset,
    visual_token_count: int | None,
    visual_token_dim: int | None,
) -> dict[str, int]:
    metadata = dataset.visual_feature_cache.metadata if dataset.visual_feature_cache is not None else {}
    resolved_count = visual_token_count or metadata.get("visual_token_count")
    resolved_dim = visual_token_dim or metadata.get("visual_token_dim")
    if resolved_count is None or resolved_dim is None:
        raise ValueError(
            "visual token count/dim must be provided by cache metadata or CLI args "
            "for cross_attention fusion"
        )
    resolved = {
        "visual_token_count": int(resolved_count),
        "visual_token_dim": int(resolved_dim),
    }
    if resolved["visual_token_count"] <= 0 or resolved["visual_token_dim"] <= 0:
        raise ValueError("visual token count and dim must be positive")
    expected_dim = resolved["visual_token_count"] * resolved["visual_token_dim"]
    if dataset.visual_dim != expected_dim:
        raise ValueError(
            f"visual_dim {dataset.visual_dim} does not match token config product {expected_dim}"
        )
    return resolved


def _build_conditioner(
    dataset: OracleActionWindowDataset,
    condition_on: str,
) -> CategoricalConditioner:
    if condition_on == "none":
        return CategoricalConditioner(
            condition_on=condition_on,
            vocab=(),
            index_by_label={},
        )
    labels = sorted({_window_condition_label(window, condition_on) for window in dataset.windows})
    return CategoricalConditioner(
        condition_on=condition_on,
        vocab=tuple(labels),
        index_by_label={label: index for index, label in enumerate(labels)},
    )


def _window_condition_label(window: object, condition_on: str) -> str:
    suite_name = str(getattr(window, "suite_name"))
    task_id = str(getattr(window, "task_id"))
    if condition_on == "suite":
        return suite_name
    if condition_on == "task":
        return task_id
    if condition_on == "suite_task":
        return f"{suite_name}::{task_id}"
    raise ValueError("condition_on must be one of: none, suite, task, suite_task")


def _batch_conditioning(
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    device: torch.device,
    *,
    include_visual: bool,
) -> torch.Tensor | None:
    batch_size = int(batch["context"].shape[0])
    parts: list[torch.Tensor] = []
    if conditioner.dim > 0:
        categorical = torch.zeros((batch_size, conditioner.dim), dtype=torch.float32, device=device)
        for row in range(batch_size):
            label = _batch_condition_label(batch, row, conditioner.condition_on)
            try:
                column = conditioner.index_by_label[label]
            except KeyError as exc:
                raise KeyError(f"unknown conditioning label {label!r}") from exc
            categorical[row, column] = 1.0
        parts.append(categorical)

    visual = batch.get("visual") if include_visual else None
    if isinstance(visual, torch.Tensor):
        parts.append(visual.to(device=device, dtype=torch.float32))

    if not parts:
        return None
    return torch.cat(parts, dim=-1)


def _batch_condition_label(batch: dict[str, object], row: int, condition_on: str) -> str:
    suite_name = _batch_string_at(batch["suite_name"], row)
    task_id = _batch_string_at(batch["task_id"], row)
    if condition_on == "suite":
        return suite_name
    if condition_on == "task":
        return task_id
    if condition_on == "suite_task":
        return f"{suite_name}::{task_id}"
    raise ValueError("condition_on must be one of: none, suite, task, suite_task")


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _split_indices(
    dataset: OracleActionWindowDataset,
    train_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[list[int], list[int]]:
    if split_by == "window":
        return _split_window_indices(len(dataset), train_ratio, seed)
    if split_by == "episode":
        return _split_episode_indices(dataset, train_ratio, seed)
    raise ValueError("split_by must be one of: window, episode")


def _split_window_indices(num_items: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError("train_ratio must be in (0, 1]")
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items == 1 or train_ratio == 1.0:
        return indices, []
    train_size = max(1, min(num_items - 1, int(num_items * train_ratio)))
    return indices[:train_size], indices[train_size:]


def _split_episode_indices(
    dataset: OracleActionWindowDataset,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError("train_ratio must be in (0, 1]")
    episode_to_indices: dict[str, list[int]] = {}
    for index, window in enumerate(dataset.windows):
        episode_to_indices.setdefault(window.episode_id, []).append(index)
    episode_ids = sorted(episode_to_indices)
    if len(episode_ids) <= 1:
        return _split_window_indices(len(dataset), train_ratio, seed)
    random.Random(seed).shuffle(episode_ids)
    if train_ratio == 1.0:
        return [index for episode_id in episode_ids for index in episode_to_indices[episode_id]], []
    train_episode_count = max(1, min(len(episode_ids) - 1, int(len(episode_ids) * train_ratio)))
    train_episode_ids = set(episode_ids[:train_episode_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for episode_id in episode_ids:
        target = train_indices if episode_id in train_episode_ids else val_indices
        target.extend(episode_to_indices[episode_id])
    random.Random(seed).shuffle(train_indices)
    random.Random(seed + 1).shuffle(val_indices)
    return train_indices, val_indices


def _average_metrics(totals: dict[str, float], total_count: int) -> dict[str, float | None]:
    if total_count == 0:
        return {"mse": None, "mae": None}
    return {key: value / total_count for key, value in totals.items()}


def _make_loader(
    dataset: OracleActionWindowDataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader | None:
    if not indices:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle)


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer")
    return dims


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
