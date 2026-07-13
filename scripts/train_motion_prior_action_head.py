#!/usr/bin/env python3
"""Train an action head conditioned on frozen GeoMoCo-WM motion-prior samples."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.event_conditioning import (  # noqa: E402
    EventModeConditioner,
    batch_event_mode_conditioning,
    combine_conditioning,
    load_event_mode_conditioner,
)
from geomoco_wm.data.window_dataset import MOTION_MODES, OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_visual_cvae_samples import _load_model, _sample_prior_motions  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _build_conditioner,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


INPUT_MODES = ("context_only", "prior_mean", "sample_set", "gt_future")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gate 3.0a action heads over frozen motion-prior hypotheses."
    )
    parser.add_argument(
        "--input-mode",
        required=True,
        choices=INPUT_MODES,
        help="Which future-motion input the action head receives.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Frozen visual cVAE checkpoint. Required for prior_mean/sample_set.",
    )
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=None,
        help=(
            "Input windows.jsonl. Defaults to checkpoint dataset windows_jsonl when "
            "--checkpoint is provided, otherwise the two-file four-suite slice."
        ),
    )
    parser.add_argument(
        "--visual-feature-cache",
        default=None,
        help="Visual feature cache. Defaults to checkpoint visual cache for cVAE modes.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="512,512")
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument(
        "--set-aggregator",
        default="context_attention",
        choices=["mean_pool", "context_attention", "multi_query_attention"],
        help="How to aggregate the K future-motion sample tokens.",
    )
    parser.add_argument("--set-query-count", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument(
        "--motion-mode",
        default=None,
        choices=MOTION_MODES,
        help="Defaults to checkpoint motion_mode or future_delta_gripper.",
    )
    parser.add_argument(
        "--condition-on",
        default=None,
        choices=["none", "suite", "task", "suite_task"],
        help="Defaults to checkpoint conditioning or suite_task.",
    )
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument(
        "--event-conditioning-mode",
        default=None,
        choices=["none", "oracle", "shuffled"],
        help="Defaults to checkpoint event-conditioning mode.",
    )
    parser.add_argument("--event-class-set", default=None, choices=["stable8", "all_observed"])
    parser.add_argument("--event-shuffle-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    requires_cvae = args.input_mode in {"prior_mean", "sample_set"}
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    checkpoint: dict[str, Any] | None = None
    checkpoint_metrics: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_metrics = checkpoint["metrics"]
    if requires_cvae and checkpoint is None:
        raise ValueError("--checkpoint is required for prior_mean and sample_set input modes")

    windows_jsonl = _resolve_windows_jsonl(args, checkpoint_metrics)
    motion_mode = args.motion_mode or _checkpoint_motion_mode(checkpoint_metrics) or "future_delta_gripper"
    visual_feature_cache = _resolve_visual_feature_cache(args, checkpoint_metrics, requires_cvae)
    condition_on = args.condition_on or _checkpoint_condition_on(checkpoint_metrics) or "suite_task"
    split_by = args.split_by or _checkpoint_split_by(checkpoint_metrics) or "episode"
    event_conditioning_mode = args.event_conditioning_mode or _checkpoint_event_conditioning_mode(
        checkpoint_metrics
    )
    event_class_set = args.event_class_set or _checkpoint_event_class_set(checkpoint_metrics)

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_checkpoint_conditioner(dataset, checkpoint_metrics, condition_on)
    event_conditioner = load_event_mode_conditioner(
        args.event_mode_audit_json or _checkpoint_event_audit_json(checkpoint_metrics),
        mode=event_conditioning_mode,
        class_set=event_class_set,
        shuffle_seed=args.event_shuffle_seed,
    )
    if spec.motion_dim <= 0:
        raise ValueError("motion-mode must expose a positive motion_dim")

    visual_token_config = None
    cvae: VisualConditionedGeoMoCoCVAE | None = None
    device = _resolve_device(args.device) if not args.dry_run else torch.device("cpu")
    if requires_cvae:
        if checkpoint is None or checkpoint_metrics is None:
            raise ValueError("internal error: cVAE mode missing checkpoint")
        visual_token_config = _resolve_visual_token_config(
            dataset,
            checkpoint_metrics["visual_token_config"]["visual_token_count"],
            checkpoint_metrics["visual_token_config"]["visual_token_dim"],
        )
        cvae = _load_model(
            checkpoint,
            context_dim=spec.context_dim,
            motion_dim=spec.motion_dim,
            visual_token_config=visual_token_config,
            conditioner_dim=conditioner.dim + event_conditioner.dim,
            device=device,
        )
        _freeze_module(cvae)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    model = MotionPriorActionHead(
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
        conditioning_dim=conditioner.dim + event_conditioner.dim,
        hidden_dims=hidden_dims,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        temporal_layers=args.temporal_layers,
        set_aggregator=args.set_aggregator,
        set_query_count=args.set_query_count,
        dropout=args.dropout,
    ).to(device)

    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "input_mode": args.input_mode,
                    "dataset": spec.to_dict(),
                    "motion_mode": motion_mode,
                    "conditioning": conditioner.to_dict(),
                    "event_conditioning": event_conditioner.to_dict(),
                    "combined_conditioning_dim": conditioner.dim + event_conditioner.dim,
                    "split_by": split_by,
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                    "visual_feature_cache": str(visual_feature_cache)
                    if visual_feature_cache
                    else None,
                    "visual_token_config": visual_token_config,
                    "num_samples": args.num_samples,
                    "model_config": _model_config(
                        args,
                        hidden_dims,
                        spec,
                        conditioner,
                        event_conditioner,
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_metric = float("inf")
    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            cvae,
            train_loader,
            optimizer,
            loss_fn,
            device,
            conditioner,
            event_conditioner,
            args.input_mode,
            args.num_samples,
        )
        val_metrics = _evaluate(
            model,
            cvae,
            val_loader,
            loss_fn,
            device,
            conditioner,
            event_conditioner,
            args.input_mode,
            args.num_samples,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate_metric = val_metrics.get("mse")
        if candidate_metric is not None and float(candidate_metric) < best_metric:
            best_metric = float(candidate_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    final_action_metrics = _evaluate(
        model,
        cvae,
        val_loader,
        loss_fn,
        device,
        conditioner,
        event_conditioner,
        args.input_mode,
        args.num_samples,
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
        "seed": args.seed,
        "split_by": split_by,
        "motion_mode": motion_mode,
        "input_mode": args.input_mode,
        "num_samples": args.num_samples,
        "conditioning": conditioner.to_dict(),
        "event_conditioning": event_conditioner.to_dict(),
        "combined_conditioning_dim": conditioner.dim + event_conditioner.dim,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser())
        if visual_feature_cache
        else None,
        "visual_token_config": visual_token_config,
        "cvae_frozen": requires_cvae,
        "cvae_config": _cvae_config(checkpoint_metrics),
        "model_config": _model_config(args, hidden_dims, spec, conditioner, event_conditioner),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mse": best_metric if best_state is not None else None,
        "final_action_metrics": final_action_metrics,
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
                    "best_epoch": best_epoch,
                    "final_action_metrics": final_action_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.dropout < 0.0:
        raise ValueError("--dropout must be non-negative")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.token_dim <= 0:
        raise ValueError("--token-dim must be positive")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive")
    if args.token_dim % args.num_heads != 0:
        raise ValueError("--token-dim must be divisible by --num-heads")
    if args.temporal_layers < 0:
        raise ValueError("--temporal-layers must be non-negative")
    if args.set_query_count <= 0:
        raise ValueError("--set-query-count must be positive")


def _run_epoch(
    model: MotionPriorActionHead,
    cvae: VisualConditionedGeoMoCoCVAE | None,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    input_mode: str,
    num_samples: int,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None, "mse": None}
    model.train()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        future_inputs = _future_inputs(cvae, batch, context, conditioning, device, input_mode, num_samples)
        optimizer.zero_grad(set_to_none=True)
        pred_actions = model(context, future_inputs, conditioning)
        loss = loss_fn(pred_actions, actions)
        loss.backward()
        optimizer.step()
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions.detach(), actions)
        batch_metrics["loss"] = float(loss.detach().cpu())
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate(
    model: MotionPriorActionHead,
    cvae: VisualConditionedGeoMoCoCVAE | None,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    input_mode: str,
    num_samples: int,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None, "mse": None}
    model.eval()
    if cvae is not None:
        cvae.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        future_inputs = _future_inputs(cvae, batch, context, conditioning, device, input_mode, num_samples)
        pred_actions = model(context, future_inputs, conditioning)
        loss = loss_fn(pred_actions, actions)
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions, actions)
        batch_metrics["loss"] = float(loss.cpu())
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _future_inputs(
    cvae: VisualConditionedGeoMoCoCVAE | None,
    batch: dict[str, object],
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    device: torch.device,
    input_mode: str,
    num_samples: int,
) -> torch.Tensor | None:
    if input_mode == "context_only":
        return None
    if input_mode == "gt_future":
        motion = batch["motion"].to(device)
        return motion.unsqueeze(1)
    if input_mode not in {"prior_mean", "sample_set"}:
        raise ValueError(f"unsupported input mode {input_mode!r}")
    if cvae is None:
        raise ValueError("cVAE is required for prior_mean and sample_set")
    visual = _batch_visual(batch, device)
    condition = cvae.condition(context, visual, conditioning)
    prior_mean, prior_logvar = cvae.encode_prior(condition)
    if input_mode == "prior_mean":
        return cvae.decode(condition, prior_mean).unsqueeze(1)
    samples = _sample_prior_motions(cvae, condition, prior_mean, prior_logvar, num_samples)
    return samples.permute(1, 0, 2).contiguous()


def _resolve_windows_jsonl(
    args: argparse.Namespace,
    checkpoint_metrics: dict[str, Any] | None,
) -> list[str] | str:
    if args.windows_jsonl:
        return args.windows_jsonl
    if checkpoint_metrics is not None:
        return checkpoint_metrics["dataset"]["windows_jsonl"]
    return "outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl"


def _resolve_visual_feature_cache(
    args: argparse.Namespace,
    checkpoint_metrics: dict[str, Any] | None,
    requires_cvae: bool,
) -> str | None:
    if args.visual_feature_cache:
        return args.visual_feature_cache
    if requires_cvae:
        if checkpoint_metrics is None:
            raise ValueError("internal error: cVAE mode missing checkpoint metrics")
        return str(checkpoint_metrics["visual_feature_cache"])
    return None


def _checkpoint_motion_mode(checkpoint_metrics: dict[str, Any] | None) -> str | None:
    if checkpoint_metrics is None:
        return None
    value = checkpoint_metrics.get(
        "motion_mode",
        checkpoint_metrics.get("dataset", {}).get("motion_mode"),
    )
    return str(value) if value is not None else None


def _checkpoint_condition_on(checkpoint_metrics: dict[str, Any] | None) -> str | None:
    if checkpoint_metrics is None:
        return None
    conditioning = checkpoint_metrics.get("conditioning", {})
    if isinstance(conditioning, dict):
        value = conditioning.get("condition_on")
        return str(value) if value is not None else None
    return None


def _checkpoint_split_by(checkpoint_metrics: dict[str, Any] | None) -> str | None:
    if checkpoint_metrics is None:
        return None
    split_by = checkpoint_metrics.get("split_by")
    return str(split_by) if split_by is not None else None


def _checkpoint_event_conditioning_mode(checkpoint_metrics: dict[str, Any] | None) -> str:
    if checkpoint_metrics is None:
        return "none"
    event_conditioning = checkpoint_metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        return "none"
    return str(event_conditioning.get("mode", "none"))


def _checkpoint_event_class_set(checkpoint_metrics: dict[str, Any] | None) -> str:
    if checkpoint_metrics is None:
        return "stable8"
    event_conditioning = checkpoint_metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        return "stable8"
    return str(event_conditioning.get("class_set", "stable8"))


def _checkpoint_event_audit_json(checkpoint_metrics: dict[str, Any] | None) -> str | None:
    if checkpoint_metrics is None:
        return None
    event_conditioning = checkpoint_metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        return None
    value = event_conditioning.get("event_mode_audit_json")
    return str(value) if value else None


def _build_checkpoint_conditioner(
    dataset: OracleActionWindowDataset,
    checkpoint_metrics: dict[str, Any] | None,
    condition_on: str,
) -> CategoricalConditioner:
    if checkpoint_metrics is not None:
        checkpoint_conditioning = checkpoint_metrics.get("conditioning", {})
        if isinstance(checkpoint_conditioning, dict):
            checkpoint_condition_on = checkpoint_conditioning.get("condition_on")
            checkpoint_vocab = checkpoint_conditioning.get("vocab")
            if checkpoint_condition_on == condition_on and isinstance(checkpoint_vocab, list):
                vocab = tuple(str(value) for value in checkpoint_vocab)
                return CategoricalConditioner(
                    condition_on=condition_on,
                    vocab=vocab,
                    index_by_label={label: index for index, label in enumerate(vocab)},
                )
    return _build_conditioner(dataset, condition_on)


def _model_config(
    args: argparse.Namespace,
    hidden_dims: tuple[int, ...],
    spec: object,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
) -> dict[str, object]:
    return {
        "context_dim": int(spec.context_dim),
        "motion_dim": int(spec.motion_dim),
        "action_dim": int(spec.action_dim),
        "horizon": int(spec.horizon),
        "conditioning_dim": conditioner.dim + event_conditioner.dim,
        "base_conditioning_dim": conditioner.dim,
        "event_conditioning_dim": event_conditioner.dim,
        "hidden_dims": list(hidden_dims),
        "token_dim": args.token_dim,
        "num_heads": args.num_heads,
        "temporal_layers": args.temporal_layers,
        "set_aggregator": args.set_aggregator,
        "set_query_count": args.set_query_count,
        "dropout": args.dropout,
        "aggregation": args.set_aggregator,
    }


def _cvae_config(checkpoint_metrics: dict[str, Any] | None) -> dict[str, object] | None:
    if checkpoint_metrics is None:
        return None
    return {
        "hidden_dims": checkpoint_metrics.get("hidden_dims"),
        "latent_dim": checkpoint_metrics.get("latent_dim"),
        "free_bits": checkpoint_metrics.get("free_bits"),
        "beta_kl": checkpoint_metrics.get("beta_kl"),
        "prior_recon_weight": checkpoint_metrics.get("prior_recon_weight"),
        "action_aware_loss_weight": checkpoint_metrics.get("action_aware_loss_weight"),
    }


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _combined_batch_conditioning(
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    device: torch.device,
) -> torch.Tensor | None:
    base_conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    event_conditioning = batch_event_mode_conditioning(batch, event_conditioner, device)
    return combine_conditioning(base_conditioning, event_conditioning)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
