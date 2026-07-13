#!/usr/bin/env python3
"""Train a GraspGen-style on-generator critic for GeoMoCo candidates."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.predicted_event_mixture import event_label_is_transition  # noqa: E402
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from audit_predicted_event_mixture_action_head_groups import _event_family  # noqa: E402
from audit_predicted_event_mixture_action_head_usage import (  # noqa: E402
    FutureInputBundle,
    SampleVariant,
    _event_entropy,
    _predicted_event_future_input_bundle,
)
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_predicted_event_mixture_action_head import _load_action_head  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _batch_conditioning,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import (  # noqa: E402
    _event_mode_for_record,
    _load_event_label_records,
)


class OnGeneratorCandidateCritic(nn.Module):
    """Score generated future-motion candidates for downstream action value."""

    def __init__(
        self,
        *,
        context_dim: int,
        motion_dim: int,
        conditioning_dim: int,
        sample_feature_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if motion_dim <= 0:
            raise ValueError("motion_dim must be positive")
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if sample_feature_dim < 0:
            raise ValueError("sample_feature_dim must be non-negative")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.conditioning_dim = conditioning_dim
        self.sample_feature_dim = sample_feature_dim
        input_dim = context_dim + motion_dim + conditioning_dim + sample_feature_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims entries must be positive")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        context: torch.Tensor,
        future_inputs: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        sample_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"context must be [B,C], got {context.shape}")
        if future_inputs.ndim != 3:
            raise ValueError(f"future_inputs must be [B,K,M], got {future_inputs.shape}")
        batch_size, sample_count, motion_dim = future_inputs.shape
        if context.shape[0] != batch_size:
            raise ValueError("context and future_inputs batch sizes differ")
        if motion_dim != self.motion_dim:
            raise ValueError(f"future motion dim {motion_dim} != {self.motion_dim}")
        pieces = [
            context.unsqueeze(1).expand(-1, sample_count, -1),
            future_inputs,
        ]
        if self.conditioning_dim > 0:
            if conditioning is None:
                conditioning_piece = context.new_zeros((batch_size, sample_count, self.conditioning_dim))
            else:
                if conditioning.shape != (batch_size, self.conditioning_dim):
                    raise ValueError(
                        "conditioning must be [B,D], "
                        f"got {conditioning.shape} vs {(batch_size, self.conditioning_dim)}"
                    )
                conditioning_piece = conditioning.unsqueeze(1).expand(-1, sample_count, -1)
            pieces.append(conditioning_piece)
        if self.sample_feature_dim > 0:
            if sample_features is None:
                feature_piece = context.new_zeros((batch_size, sample_count, self.sample_feature_dim))
            else:
                expected = (batch_size, sample_count, self.sample_feature_dim)
                if sample_features.shape != expected:
                    raise ValueError(f"sample_features must be {expected}, got {sample_features.shape}")
                feature_piece = sample_features
            pieces.append(feature_piece)
        features = torch.cat(pieces, dim=-1)
        return self.net(features.reshape(batch_size * sample_count, -1)).reshape(
            batch_size,
            sample_count,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate 3.11a on-generator critic audit for GeoMoCo candidates."
    )
    parser.add_argument("--checkpoint", required=True, help="Gate 3.4 action-head model.pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-control",
        default="real",
        choices=[
            "real",
            "mean_repeated",
            "batch_mismatch",
            "rank_prob_only",
            "shuffled_event_identity",
            "zero_sample_features",
        ],
        help="Control applied before action-regret labeling and critic scoring.",
    )
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--event-top-m", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="512,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--critic-loss-type",
        default="soft_ce",
        choices=["soft_ce", "hard_ce", "combined"],
    )
    parser.add_argument("--critic-temperature", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        default="critic_selected_mse",
        choices=[
            "critic_loss",
            "critic_selected_mse",
            "critic_expected_mse",
            "critic_gap_to_oracle",
            "critic_selected_gain_vs_set",
        ],
    )
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device) if not args.dry_run else torch.device("cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    cvae_checkpoint_path = Path(metrics["checkpoint"]).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])
    split_seed = int(metrics["seed"])
    _seed_everything(base_seed)

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        max_windows=int(metrics["dataset"]["num_windows"]),
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
    conditioner = _conditioner_from_metrics(metrics["conditioning"])
    train_ratio = args.train_ratio
    if train_ratio is None:
        train_ratio = float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"])
    split_by = args.split_by or metrics["split_by"]
    train_indices, val_indices = _split_indices(dataset, train_ratio, split_seed, split_by)
    batch_size = int(args.batch_size or metrics["batch_size"])
    num_samples = int(args.num_samples or metrics["num_samples"])
    event_top_m = int(args.event_top_m or metrics["event_top_m"])
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    sample_feature_dim = int(metrics.get("sample_feature_dim", 0))
    sample_feature_mode = str(metrics.get("sample_feature_mode", "none"))
    event_candidate_policy = str(metrics.get("event_candidate_policy", "topk"))
    transition_reserve_threshold = float(metrics.get("transition_reserve_threshold", 0.0))
    event_audit_json = (
        args.event_mode_audit_json
        or metrics.get("event_mode_audit_json")
        or _checkpoint_event_audit_json(cvae_metrics)
    )
    if event_audit_json is None:
        raise ValueError("--event-mode-audit-json is required when absent from checkpoints")
    event_labels = _load_event_label_records(event_audit_json)

    dry_run_summary = {
        "mode": "dry_run",
        "checkpoint": str(checkpoint_path),
        "source_cvae_checkpoint": str(cvae_checkpoint_path),
        "event_probe_checkpoint": metrics["event_probe_checkpoint"],
        "event_mode_audit_json": str(Path(event_audit_json).expanduser()),
        "candidate_control": args.candidate_control,
        "dataset": metrics["dataset"],
        "seed": base_seed,
        "checkpoint_split_seed": split_seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "event_top_m": event_top_m,
        "sample_feature_mode": sample_feature_mode,
        "sample_feature_dim": sample_feature_dim,
        "hidden_dims": hidden_dims,
        "critic_temperature": args.critic_temperature,
        "critic_loss_type": args.critic_loss_type,
        "selection_metric": args.selection_metric,
    }
    if args.dry_run:
        print(json.dumps(dry_run_summary, indent=2, ensure_ascii=False))
        return

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
    _freeze_module(action_head)
    if str(metrics.get("temporal_action_decoder_mode", "none")) == "none":
        raise ValueError("Gate 3.11a requires a checkpoint with temporal_actions")

    critic = OnGeneratorCandidateCritic(
        context_dim=int(metrics["dataset"]["context_dim"]),
        motion_dim=int(metrics["dataset"]["motion_dim"]),
        conditioning_dim=conditioner.dim,
        sample_feature_dim=sample_feature_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = _make_loader(dataset, train_indices, batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_metric = float("inf")
    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            critic,
            action_head,
            cvae,
            event_probe,
            train_loader,
            optimizer,
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
            sample_feature_mode=sample_feature_mode,
            sample_feature_dim=sample_feature_dim,
            candidate_control=args.candidate_control,
            critic_loss_type=args.critic_loss_type,
            critic_temperature=args.critic_temperature,
            max_batches=args.max_train_batches,
            train=True,
        )
        val_metrics, _ = _evaluate(
            critic,
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
            sample_feature_mode=sample_feature_mode,
            sample_feature_dim=sample_feature_dim,
            candidate_control=args.candidate_control,
            critic_loss_type=args.critic_loss_type,
            critic_temperature=args.critic_temperature,
            max_batches=args.max_val_batches,
        )
        row: dict[str, float | int | None] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate_metric = val_metrics.get(args.selection_metric)
        if candidate_metric is not None and float(candidate_metric) < best_metric:
            best_metric = float(candidate_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(critic.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        critic.load_state_dict(best_state)
    final_metrics, final_groups = _evaluate(
        critic,
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
        sample_feature_mode=sample_feature_mode,
        sample_feature_dim=sample_feature_dim,
        candidate_control=args.candidate_control,
        critic_loss_type=args.critic_loss_type,
        critic_temperature=args.critic_temperature,
        max_batches=args.max_val_batches,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "context_dim": int(metrics["dataset"]["context_dim"]),
        "motion_dim": int(metrics["dataset"]["motion_dim"]),
        "conditioning_dim": conditioner.dim,
        "sample_feature_dim": sample_feature_dim,
        "hidden_dims": list(hidden_dims),
        "dropout": args.dropout,
    }
    output = {
        "gate": "3.11a",
        "input_mode": "on_generator_geomoco_candidate_critic",
        "checkpoint": str(checkpoint_path),
        "source_cvae_checkpoint": str(cvae_checkpoint_path),
        "event_probe_checkpoint": metrics["event_probe_checkpoint"],
        "event_mode_audit_json": str(Path(event_audit_json).expanduser().resolve()),
        "dataset": metrics["dataset"],
        "device": str(device),
        "seed": base_seed,
        "checkpoint_split_seed": split_seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "candidate_control": args.candidate_control,
        "num_samples": num_samples,
        "event_top_m": event_top_m,
        "event_candidate_policy": event_candidate_policy,
        "transition_reserve_threshold": transition_reserve_threshold,
        "sample_feature_mode": sample_feature_mode,
        "sample_feature_dim": sample_feature_dim,
        "critic_loss_type": args.critic_loss_type,
        "critic_temperature": args.critic_temperature,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_value": best_metric if best_state is not None else None,
        "model_config": model_config,
        "action_head_model_config": metrics["model_config"],
        "history": history,
        "final_metrics": final_metrics,
        "final_groups": final_groups,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": critic.state_dict(), "metrics": output},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "best_epoch": best_epoch,
                    "key_metrics": _key_metrics(final_metrics),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _run_epoch(
    critic: OnGeneratorCandidateCritic,
    action_head: MotionPriorActionHead,
    cvae: Any,
    event_probe: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
    sample_feature_dim: int,
    candidate_control: str,
    critic_loss_type: str,
    critic_temperature: float,
    max_batches: int | None,
    train: bool,
) -> dict[str, float | None]:
    critic.train(mode=train)
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_metrics, loss = _critic_batch(
            critic,
            action_head,
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            event_labels=event_labels,
            cvae_event_classes=cvae_event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=event_top_m,
            num_samples=num_samples,
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
            sample_feature_dim=sample_feature_dim,
            candidate_control=candidate_control,
            critic_loss_type=critic_loss_type,
            critic_temperature=critic_temperature,
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(batch["context"].shape[0])
        for key, value in batch_metrics.items():
            if key.startswith("_"):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    return _average_metrics(totals, count)


@torch.no_grad()
def _evaluate(
    critic: OnGeneratorCandidateCritic,
    action_head: MotionPriorActionHead,
    cvae: Any,
    event_probe: Any,
    loader: DataLoader,
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
    sample_feature_dim: int,
    candidate_control: str,
    critic_loss_type: str,
    critic_temperature: float,
    max_batches: int | None,
) -> tuple[dict[str, float | None], dict[str, dict[str, float | int]]]:
    critic.eval()
    totals: dict[str, float] = {}
    count = 0
    group_totals: dict[str, dict[str, float]] = defaultdict(dict)
    group_counts: dict[str, int] = defaultdict(int)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_metrics, _ = _critic_batch(
            critic,
            action_head,
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            event_labels=event_labels,
            cvae_event_classes=cvae_event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=event_top_m,
            num_samples=num_samples,
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
            sample_feature_dim=sample_feature_dim,
            candidate_control=candidate_control,
            critic_loss_type=critic_loss_type,
            critic_temperature=critic_temperature,
        )
        batch_size = int(batch["context"].shape[0])
        for key, value in batch_metrics.items():
            if key.startswith("_"):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
        row_metrics = _row_ranking_metrics(
            batch_metrics["_scores"],
            batch_metrics["_regrets"],
            batch_metrics["_set_mse"],
        )
        _add_group_metrics(
            group_totals,
            group_counts,
            batch,
            event_labels,
            row_metrics,
        )
    return _average_metrics(totals, count), _finalize_groups(group_totals, group_counts)


def _critic_batch(
    critic: OnGeneratorCandidateCritic,
    action_head: MotionPriorActionHead,
    cvae: Any,
    event_probe: Any,
    batch: dict[str, object],
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
    sample_feature_dim: int,
    candidate_control: str,
    critic_loss_type: str,
    critic_temperature: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    context = batch["context"].to(device)
    actions = batch["actions"].to(device)
    conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    bundle = _generator_bundle(
        cvae,
        event_probe,
        batch,
        context,
        conditioning,
        device,
        probe_conditioner,
        cvae_event_classes=cvae_event_classes,
        probe_class_names=probe_class_names,
        probe_input_variant=probe_input_variant,
        event_top_m=event_top_m,
        num_samples=num_samples,
        event_candidate_policy=event_candidate_policy,
        transition_reserve_threshold=transition_reserve_threshold,
        sample_feature_mode=sample_feature_mode,
    )
    variant = _apply_candidate_control(
        bundle,
        candidate_control,
        sample_feature_dim=sample_feature_dim,
        event_class_count=len(cvae_event_classes),
    )
    with torch.no_grad():
        regrets = _candidate_temporal_regrets(
            action_head,
            context,
            conditioning,
            variant.future_inputs,
            variant.sample_features,
            actions,
        )
        set_mse = _set_temporal_mse(
            action_head,
            context,
            conditioning,
            variant.future_inputs,
            variant.sample_features,
            actions,
        )
    scores = critic(context, variant.future_inputs, conditioning, variant.sample_features)
    loss, loss_metrics = _ranking_loss(
        scores,
        regrets,
        loss_type=critic_loss_type,
        temperature=critic_temperature,
    )
    metrics = _ranking_metrics(scores.detach(), regrets, set_mse)
    metrics.update(loss_metrics)
    metrics["event_entropy"] = float(_event_entropy(bundle.top_probs).mean())
    # Private tensors used only by the evaluator for group aggregation.
    metrics["_scores"] = scores.detach()
    metrics["_regrets"] = regrets.detach()
    metrics["_set_mse"] = set_mse.detach()
    return metrics, loss


@torch.no_grad()
def _generator_bundle(
    cvae: Any,
    event_probe: Any,
    batch: dict[str, object],
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    device: torch.device,
    probe_conditioner: Any,
    *,
    cvae_event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
) -> FutureInputBundle:
    return _predicted_event_future_input_bundle(
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


def _apply_candidate_control(
    bundle: FutureInputBundle,
    control: str,
    *,
    sample_feature_dim: int,
    event_class_count: int,
) -> SampleVariant:
    future_inputs = bundle.future_inputs
    sample_features = bundle.sample_features
    if control == "real":
        return SampleVariant(future_inputs, sample_features)
    if control == "mean_repeated":
        mean = future_inputs.mean(dim=1, keepdim=True)
        return SampleVariant(mean.expand_as(future_inputs).contiguous(), sample_features)
    if control == "batch_mismatch":
        if future_inputs.shape[0] <= 1:
            return SampleVariant(future_inputs, sample_features)
        return SampleVariant(
            future_inputs.roll(shifts=1, dims=0),
            sample_features.roll(shifts=1, dims=0) if sample_features is not None else None,
        )
    if control == "rank_prob_only":
        return SampleVariant(future_inputs, _rank_prob_only_features(sample_features, event_class_count))
    if control == "shuffled_event_identity":
        return SampleVariant(
            future_inputs,
            _shuffled_event_identity_features(sample_features, event_class_count),
        )
    if control == "zero_sample_features":
        return SampleVariant(
            future_inputs,
            _zero_sample_features(future_inputs, sample_feature_dim),
        )
    raise ValueError(f"unsupported candidate control {control!r}")


def _rank_prob_only_features(
    sample_features: torch.Tensor | None,
    event_class_count: int,
) -> torch.Tensor | None:
    if sample_features is None:
        return None
    if sample_features.shape[-1] < event_class_count + 2:
        return sample_features
    output = sample_features.clone()
    output[..., :event_class_count] = 0.0
    return output


def _shuffled_event_identity_features(
    sample_features: torch.Tensor | None,
    event_class_count: int,
) -> torch.Tensor | None:
    if sample_features is None:
        return None
    if sample_features.shape[0] <= 1 or sample_features.shape[-1] < event_class_count + 2:
        return sample_features
    output = sample_features.clone()
    output[..., :event_class_count] = output[..., :event_class_count].roll(shifts=1, dims=0)
    return output


def _zero_sample_features(
    future_inputs: torch.Tensor,
    sample_feature_dim: int,
) -> torch.Tensor | None:
    if sample_feature_dim <= 0:
        return None
    return future_inputs.new_zeros(
        (future_inputs.shape[0], future_inputs.shape[1], sample_feature_dim)
    )


@torch.no_grad()
def _candidate_temporal_regrets(
    action_head: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for sample_index in range(future_inputs.shape[1]):
        output = action_head.forward_with_aux(
            context,
            future_inputs[:, sample_index : sample_index + 1, :],
            conditioning,
            sample_features[:, sample_index : sample_index + 1, :]
            if sample_features is not None
            else None,
        )
        temporal_actions = output.get("temporal_actions")
        if temporal_actions is None:
            raise ValueError("action head checkpoint does not produce temporal_actions")
        predictions.append(temporal_actions)
    stacked = torch.stack(predictions, dim=1)
    return (stacked - actions.unsqueeze(1)).square().mean(dim=(2, 3))


@torch.no_grad()
def _set_temporal_mse(
    action_head: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
) -> torch.Tensor:
    output = action_head.forward_with_aux(context, future_inputs, conditioning, sample_features)
    temporal_actions = output.get("temporal_actions")
    if temporal_actions is None:
        raise ValueError("action head checkpoint does not produce temporal_actions")
    return (temporal_actions - actions).square().mean(dim=(1, 2))


def _ranking_loss(
    scores: torch.Tensor,
    regrets: torch.Tensor,
    *,
    loss_type: str,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = torch.softmax(-regrets / temperature, dim=-1)
    log_probs = torch.log_softmax(scores, dim=-1)
    soft_ce = -(targets * log_probs).sum(dim=-1).mean()
    best_indices = regrets.argmin(dim=-1)
    hard_ce = F.cross_entropy(scores, best_indices)
    if loss_type == "soft_ce":
        loss = soft_ce
    elif loss_type == "hard_ce":
        loss = hard_ce
    elif loss_type == "combined":
        loss = 0.5 * (soft_ce + hard_ce)
    else:
        raise ValueError(f"unsupported critic loss type {loss_type!r}")
    return loss, {
        "critic_loss": float(loss.detach().cpu()),
        "critic_soft_ce": float(soft_ce.detach().cpu()),
        "critic_hard_ce": float(hard_ce.detach().cpu()),
    }


def _ranking_metrics(
    scores: torch.Tensor,
    regrets: torch.Tensor,
    set_mse: torch.Tensor,
) -> dict[str, float]:
    row_metrics = _row_ranking_metrics(scores, regrets, set_mse)
    return {
        key: float(value.mean().detach().cpu())
        for key, value in row_metrics.items()
    }


def _row_ranking_metrics(
    scores: torch.Tensor,
    regrets: torch.Tensor,
    set_mse: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probs = torch.softmax(scores, dim=-1)
    best_indices = regrets.argmin(dim=-1)
    selected_indices = scores.argmax(dim=-1)
    selected = regrets.gather(1, selected_indices.unsqueeze(-1)).squeeze(-1)
    best = regrets.gather(1, best_indices.unsqueeze(-1)).squeeze(-1)
    mean = regrets.mean(dim=-1)
    expected = (probs * regrets).sum(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "candidate_mean_mse": mean,
        "candidate_oracle_mse": best,
        "set_temporal_mse": set_mse,
        "oracle_gain_vs_set": set_mse - best,
        "candidate_best_vs_mean_gap": mean - best,
        "candidate_oracle_beats_set": (best < set_mse).to(dtype=torch.float32),
        "critic_selected_mse": selected,
        "critic_expected_mse": expected,
        "critic_top1_accuracy": (selected_indices == best_indices).to(dtype=torch.float32),
        "critic_gain_vs_mean": mean - selected,
        "critic_gap_to_oracle": selected - best,
        "critic_selected_gain_vs_set": set_mse - selected,
        "critic_selected_beats_set": (selected < set_mse).to(dtype=torch.float32),
        "critic_entropy": entropy,
    }


def _add_group_metrics(
    group_totals: dict[str, dict[str, float]],
    group_counts: dict[str, int],
    batch: dict[str, object],
    event_labels: dict[str, dict[str, Any]],
    row_metrics: dict[str, torch.Tensor],
) -> None:
    batch_size = int(next(iter(row_metrics.values())).shape[0])
    for row in range(batch_size):
        window_id = _batch_string_at(batch["window_id"], row)
        event_mode = _event_mode_for_record(event_labels.get(window_id)) or "unknown"
        transition_group = (
            "transition"
            if event_mode != "unknown" and event_label_is_transition(event_mode)
            else "sustain"
            if event_mode != "unknown"
            else "unknown"
        )
        groups = (
            "all",
            f"transition_group/{transition_group}",
            f"event_family/{_event_family(event_mode)}",
        )
        for group in groups:
            group_counts[group] += 1
            totals = group_totals[group]
            for key, values in row_metrics.items():
                totals[key] = totals.get(key, 0.0) + float(values[row].detach().cpu())


def _finalize_groups(
    group_totals: dict[str, dict[str, float]],
    group_counts: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for group, totals in sorted(group_totals.items()):
        count = group_counts[group]
        row: dict[str, float | int] = {"count": count}
        row.update({key: value / count for key, value in sorted(totals.items())})
        output[group] = row
    return output


def _average_metrics(totals: dict[str, Any], count: int) -> dict[str, float | None]:
    if count <= 0:
        return {key: None for key in totals if not key.startswith("_")}
    return {
        key: value / count
        for key, value in sorted(totals.items())
        if not key.startswith("_")
    }


def _key_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    keys = (
        "set_temporal_mse",
        "candidate_oracle_mse",
        "oracle_gain_vs_set",
        "candidate_best_vs_mean_gap",
        "candidate_oracle_beats_set",
        "critic_selected_mse",
        "critic_selected_gain_vs_set",
        "critic_gap_to_oracle",
        "critic_gain_vs_mean",
        "critic_top1_accuracy",
    )
    return {key: metrics.get(key) for key in keys}


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive when provided")
    if args.event_top_m is not None and args.event_top_m <= 0:
        raise ValueError("--event-top-m must be positive when provided")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.dropout < 0.0:
        raise ValueError("--dropout must be non-negative")
    if args.critic_temperature <= 0.0:
        raise ValueError("--critic-temperature must be positive")
    if args.train_ratio is not None and not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1 when provided")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive when provided")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max-val-batches must be positive when provided")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
