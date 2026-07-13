#!/usr/bin/env python3
"""Train a lightweight readout over visual cVAE future-motion samples."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.data.action_semantics import (  # noqa: E402
    default_libero_osc_pose_action_semantics,
)
from geomoco_wm.data.event_labels import (  # noqa: E402
    GripperEventConfig,
    GripperEventLabel,
    label_gripper_events_for_windows,
    label_gripper_transition_events,
    previous_gripper_commands_for_windows,
)
from geomoco_wm.metrics.action_metrics import action_metrics, rotation_geodesic_angle  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from geomoco_wm.models.sample_readout import SampleScoreNet, TemporalSampleScoreNet  # noqa: E402
from evaluate_visual_cvae_samples import (  # noqa: E402
    _diversity_metrics,
    _load_model,
    _sample_prior_motions,
)
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _build_conditioner,
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _parse_hidden_dims,
    _prediction_metrics,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


@dataclass(frozen=True)
class StructuredActionErrors:
    translation_m_l2: torch.Tensor
    rotation_geodesic_rad: torch.Tensor
    gripper_mse: torch.Tensor


@dataclass(frozen=True)
class CandidateBatch:
    condition: torch.Tensor
    prior_mean_motion: torch.Tensor
    prior_mean_actions: torch.Tensor
    samples: torch.Tensor
    sample_actions: torch.Tensor
    action_errors: torch.Tensor
    motion_errors: torch.Tensor
    translation_m_l2_errors: torch.Tensor
    rotation_geodesic_errors: torch.Tensor
    gripper_errors: torch.Tensor


@dataclass(frozen=True)
class EventTargetData:
    config: GripperEventConfig
    gt_labels: dict[str, GripperEventLabel]
    previous_commands: dict[str, float | None]
    horizon: int

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "horizon": self.horizon,
            "num_gt_labels": len(self.gt_labels),
            "num_previous_commands": len(self.previous_commands),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gate 2.4d lightweight scorer over cVAE samples."
    )
    parser.add_argument("--checkpoint", required=True, help="Gate 2.4c cVAE checkpoint.")
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=None,
        help="Defaults to the checkpoint dataset windows_jsonl.",
    )
    parser.add_argument(
        "--visual-feature-cache",
        default=None,
        help="Defaults to the checkpoint visual feature cache.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scorer-arch", default="mlp", choices=["mlp", "temporal"])
    parser.add_argument("--scorer-hidden-dims", default="256,128")
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--target-kind",
        default="action",
        choices=["action", "motion", "hybrid", "se3", "se3_gripper"],
        help="Training target used for sample ranking.",
    )
    parser.add_argument("--action-target-weight", type=float, default=1.0)
    parser.add_argument("--motion-target-weight", type=float, default=1.0)
    parser.add_argument("--translation-target-weight", type=float, default=1.0)
    parser.add_argument("--rotation-target-weight", type=float, default=1.0)
    parser.add_argument("--gripper-target-weight", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument(
        "--event-audit-json",
        default=None,
        help="Optional Gate 2.4h-a event audit JSON used for gripper transition semantics.",
    )
    parser.add_argument(
        "--event-target-weight",
        type=float,
        default=0.0,
        help="Weak event-alignment ranking weight added to the base target scores.",
    )
    parser.add_argument("--event-command-threshold", type=float, default=0.5)
    parser.add_argument("--event-close-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument(
        "--hard-negative-target-kind",
        default="none",
        choices=["none", "se3", "se3_gripper"],
        help="Optional structured oracle used to choose action-hard negative samples.",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.0)
    parser.add_argument(
        "--event-hard-negative-weight",
        type=float,
        default=0.0,
        help=(
            "Optional ranking loss against samples with low action error but bad "
            "gripper-transition timing."
        ),
    )
    parser.add_argument("--event-hard-negative-margin", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--action-decoder-checkpoint", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_metrics = checkpoint["metrics"]
    windows_jsonl = args.windows_jsonl or checkpoint_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or checkpoint_metrics["visual_feature_cache"]
    split_by = args.split_by or checkpoint_metrics.get("split_by", "episode")
    condition_on = args.condition_on or checkpoint_metrics["conditioning"]["condition_on"]
    motion_mode = str(
        checkpoint_metrics.get("motion_mode", checkpoint_metrics["dataset"].get("motion_mode"))
    )
    args.motion_mode = motion_mode

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_checkpoint_conditioner(dataset, checkpoint_metrics, condition_on)
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
        conditioner_dim=conditioner.dim,
        device=device,
    )
    _freeze_module(cvae)

    action_decoder_checkpoint = args.action_decoder_checkpoint or checkpoint_metrics.get(
        "action_decoder_checkpoint"
    )
    if not action_decoder_checkpoint:
        raise ValueError("--action-decoder-checkpoint is required for sample scoring")
    action_decoder, action_decoder_config = _load_action_decoder(action_decoder_checkpoint, device)
    if action_decoder_config["motion_mode"] != motion_mode:
        raise ValueError(
            "action decoder motion mode must match cVAE checkpoint motion mode: "
            f"{action_decoder_config['motion_mode']} vs {motion_mode}"
        )
    if int(action_decoder_config["motion_dim"]) != int(spec.motion_dim):
        raise ValueError(
            "action decoder motion dim must match scorer dataset motion dim: "
            f"{action_decoder_config['motion_dim']} vs {spec.motion_dim}"
        )
    _freeze_action_decoder(action_decoder)

    scorer_hidden_dims = _parse_hidden_dims(args.scorer_hidden_dims)
    scorer = _build_scorer(
        args,
        condition_dim=cvae.condition_dim,
        motion_dim=spec.motion_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
        hidden_dims=scorer_hidden_dims,
    ).to(device)
    event_targets = (
        _build_event_targets(dataset, args)
        if args.event_target_weight > 0.0 or args.event_hard_negative_weight > 0.0
        else None
    )

    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "dataset": spec.to_dict(),
                    "conditioning": conditioner.to_dict(),
                    "visual_token_config": visual_token_config,
                    "condition_dim": cvae.condition_dim,
                    "num_samples": args.num_samples,
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "motion_mode": motion_mode,
                    "scorer_arch": args.scorer_arch,
                    "scorer_hidden_dims": list(scorer_hidden_dims),
                    "temporal_config": _temporal_config_dict(args),
                    "target_kind": args.target_kind,
                    "target_weights": _target_weight_dict(args),
                    "event_target_weight": args.event_target_weight,
                    "event_targets": event_targets.to_dict() if event_targets is not None else None,
                    "hard_negative": _hard_negative_config_dict(args),
                    "event_hard_negative": _event_hard_negative_config_dict(args),
                },
                indent=2,
            )
        )
        return

    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = None
    best_metric = float("inf")
    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            scorer,
            cvae,
            action_decoder,
            train_loader,
            optimizer,
            device,
            conditioner,
            args,
            event_targets,
        )
        val_metrics = _evaluate(
            scorer,
            cvae,
            action_decoder,
            val_loader,
            device,
            conditioner,
            args,
            event_targets,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate_metric = val_metrics.get("scorer_argmax_action_mse")
        if candidate_metric is not None and candidate_metric < best_metric:
            best_metric = float(candidate_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(scorer.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        scorer.load_state_dict(best_state)
    final_readout_metrics = _evaluate(
        scorer,
        cvae,
        action_decoder,
        val_loader,
        device,
        conditioner,
        args,
        event_targets,
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
        "num_samples": args.num_samples,
        "motion_mode": motion_mode,
        "seed": args.seed,
        "scorer_arch": args.scorer_arch,
        "scorer_hidden_dims": list(scorer_hidden_dims),
        "temporal_config": _temporal_config_dict(args),
        "dropout": args.dropout,
        "target_kind": args.target_kind,
        "action_target_weight": args.action_target_weight,
        "motion_target_weight": args.motion_target_weight,
        "translation_target_weight": args.translation_target_weight,
        "rotation_target_weight": args.rotation_target_weight,
        "gripper_target_weight": args.gripper_target_weight,
        "target_weights": _target_weight_dict(args),
        "target_temperature": args.target_temperature,
        "selection_temperature": args.selection_temperature,
        "event_target_weight": args.event_target_weight,
        "event_targets": event_targets.to_dict() if event_targets is not None else None,
        "hard_negative": _hard_negative_config_dict(args),
        "event_hard_negative": _event_hard_negative_config_dict(args),
        "hard_negative_target_kind": args.hard_negative_target_kind,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_margin": args.hard_negative_margin,
        "event_hard_negative_weight": args.event_hard_negative_weight,
        "event_hard_negative_margin": args.event_hard_negative_margin,
        "split_by": split_by,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser()),
        "visual_token_config": visual_token_config,
        "cvae_checkpoint": str(checkpoint_path),
        "cvae_config": {
            "condition_dim": cvae.condition_dim,
            "latent_dim": cvae.latent_dim,
            "hidden_dims": checkpoint_metrics["hidden_dims"],
            "free_bits": checkpoint_metrics.get("free_bits"),
            "beta_kl": checkpoint_metrics.get("beta_kl"),
        },
        "action_decoder_checkpoint": str(Path(action_decoder_checkpoint).expanduser()),
        "action_decoder_config": action_decoder_config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_scorer_argmax_action_mse": best_metric if best_state is not None else None,
        "final_readout_metrics": final_readout_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": scorer.state_dict(), "metrics": metrics},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "best_epoch": best_epoch,
                    "final_readout_metrics": final_readout_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
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
    if args.temporal_dim <= 0:
        raise ValueError("--temporal-dim must be positive")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be positive")
    if args.temporal_heads <= 0:
        raise ValueError("--temporal-heads must be positive")
    if args.temporal_dim % args.temporal_heads != 0:
        raise ValueError("--temporal-dim must be divisible by --temporal-heads")
    target_weights = (
        args.action_target_weight,
        args.motion_target_weight,
        args.translation_target_weight,
        args.rotation_target_weight,
        args.gripper_target_weight,
    )
    if any(weight < 0.0 for weight in target_weights):
        raise ValueError("target weights must be non-negative")
    if args.target_kind == "hybrid" and args.action_target_weight + args.motion_target_weight <= 0.0:
        raise ValueError("hybrid target requires a positive action or motion weight")
    if (
        args.target_kind == "se3"
        and args.translation_target_weight + args.rotation_target_weight <= 0.0
    ):
        raise ValueError("se3 target requires a positive translation or rotation weight")
    if (
        args.target_kind == "se3_gripper"
        and (
            args.translation_target_weight
            + args.rotation_target_weight
            + args.gripper_target_weight
            <= 0.0
        )
    ):
        raise ValueError(
            "se3_gripper target requires a positive translation, rotation, or gripper weight"
        )
    if args.target_temperature <= 0.0:
        raise ValueError("--target-temperature must be positive")
    if args.selection_temperature <= 0.0:
        raise ValueError("--selection-temperature must be positive")
    if args.event_target_weight < 0.0:
        raise ValueError("--event-target-weight must be non-negative")
    if args.event_command_threshold <= 0.0:
        raise ValueError("--event-command-threshold must be positive")
    if args.hard_negative_weight < 0.0:
        raise ValueError("--hard-negative-weight must be non-negative")
    if args.hard_negative_margin < 0.0:
        raise ValueError("--hard-negative-margin must be non-negative")
    if args.event_hard_negative_weight < 0.0:
        raise ValueError("--event-hard-negative-weight must be non-negative")
    if args.event_hard_negative_margin < 0.0:
        raise ValueError("--event-hard-negative-margin must be non-negative")
    if args.hard_negative_target_kind == "none" and args.hard_negative_weight > 0.0:
        raise ValueError("--hard-negative-target-kind must not be none when weight is positive")
    if args.event_hard_negative_weight > 0.0 and args.event_target_weight <= 0.0:
        raise ValueError(
            "--event-hard-negative-weight requires a positive --event-target-weight "
            "so the positive sample is event-aware"
        )


def _target_weight_dict(args: argparse.Namespace) -> dict[str, float]:
    return {
        "action": args.action_target_weight,
        "motion": args.motion_target_weight,
        "translation": args.translation_target_weight,
        "rotation": args.rotation_target_weight,
        "gripper": args.gripper_target_weight,
        "event": args.event_target_weight,
    }


def _temporal_config_dict(args: argparse.Namespace) -> dict[str, int]:
    return {
        "temporal_dim": args.temporal_dim,
        "temporal_layers": args.temporal_layers,
        "temporal_heads": args.temporal_heads,
    }


def _hard_negative_config_dict(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "target_kind": args.hard_negative_target_kind,
        "weight": args.hard_negative_weight,
        "margin": args.hard_negative_margin,
    }


def _event_hard_negative_config_dict(args: argparse.Namespace) -> dict[str, float]:
    return {
        "weight": args.event_hard_negative_weight,
        "margin": args.event_hard_negative_margin,
    }


def _build_scorer(
    args: argparse.Namespace,
    *,
    condition_dim: int,
    motion_dim: int,
    action_dim: int,
    horizon: int,
    hidden_dims: tuple[int, ...],
) -> nn.Module:
    if args.scorer_arch == "mlp":
        return SampleScoreNet(
            condition_dim=condition_dim,
            motion_dim=motion_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
    if args.scorer_arch == "temporal":
        return TemporalSampleScoreNet(
            condition_dim=condition_dim,
            motion_dim=motion_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden_dims,
            temporal_dim=args.temporal_dim,
            num_layers=args.temporal_layers,
            num_heads=args.temporal_heads,
            dropout=args.dropout,
        )
    raise ValueError("scorer_arch must be one of: mlp, temporal")


def _build_event_targets(
    dataset: OracleActionWindowDataset,
    args: argparse.Namespace,
) -> EventTargetData:
    config = _event_config_from_args(args)
    return EventTargetData(
        config=config,
        gt_labels=label_gripper_events_for_windows(
            dataset.windows,
            config=config,
            label_mode="transition",
        ),
        previous_commands=previous_gripper_commands_for_windows(dataset.windows),
        horizon=dataset.horizon,
    )


def _event_config_from_args(args: argparse.Namespace) -> GripperEventConfig:
    if args.event_audit_json:
        report = json.loads(Path(args.event_audit_json).expanduser().read_text(encoding="utf-8"))
        config = report["config"]
        return GripperEventConfig(
            command_threshold=float(config["command_threshold"]),
            close_sign=int(config["close_sign"]),
        )
    return GripperEventConfig(
        command_threshold=args.event_command_threshold,
        close_sign=args.event_close_sign,
    )


def _build_checkpoint_conditioner(
    dataset: OracleActionWindowDataset,
    checkpoint_metrics: dict[str, object],
    condition_on: str,
) -> CategoricalConditioner:
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


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _run_epoch(
    scorer: SampleScoreNet,
    cvae: VisualConditionedGeoMoCoCVAE,
    action_decoder: ActionDecoder,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    conditioner: CategoricalConditioner,
    args: argparse.Namespace,
    event_targets: EventTargetData | None = None,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    scorer.train()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        candidates = _make_candidates(cvae, action_decoder, batch, device, conditioner, args.num_samples)
        logits = _score_candidates(scorer, candidates)
        target_scores = _target_scores(
            candidates.action_errors,
            candidates.motion_errors,
            args.target_kind,
            action_weight=args.action_target_weight,
            motion_weight=args.motion_target_weight,
            translation_weight=args.translation_target_weight,
            rotation_weight=args.rotation_target_weight,
            gripper_weight=args.gripper_target_weight,
            translation_m_l2_errors=candidates.translation_m_l2_errors,
            rotation_geodesic_errors=candidates.rotation_geodesic_errors,
            gripper_errors=candidates.gripper_errors,
        )
        event_errors = None
        if event_targets is not None and args.event_target_weight > 0.0:
            event_errors = _event_alignment_errors(candidates.sample_actions, batch, event_targets, device)
            target_scores = target_scores + args.event_target_weight * _negative_standardized_error(
                event_errors
            )
        loss, loss_metrics = _scorer_training_loss(
            logits,
            target_scores,
            candidates,
            args,
            event_errors=event_errors,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(candidates.condition.shape[0])
        batch_metrics = _batch_readout_metrics(
            scorer,
            action_decoder,
            candidates,
            batch,
            device,
            selection_temperature=args.selection_temperature,
            motion_mode=args.motion_mode,
        )
        batch_metrics["loss"] = float(loss.detach().cpu())
        batch_metrics.update(loss_metrics)
        if event_errors is not None:
            batch_metrics.update(_event_training_metrics(logits.detach(), event_errors))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate(
    scorer: SampleScoreNet,
    cvae: VisualConditionedGeoMoCoCVAE,
    action_decoder: ActionDecoder,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    args: argparse.Namespace,
    event_targets: EventTargetData | None = None,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    scorer.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        candidates = _make_candidates(cvae, action_decoder, batch, device, conditioner, args.num_samples)
        logits = _score_candidates(scorer, candidates)
        target_scores = _target_scores(
            candidates.action_errors,
            candidates.motion_errors,
            args.target_kind,
            action_weight=args.action_target_weight,
            motion_weight=args.motion_target_weight,
            translation_weight=args.translation_target_weight,
            rotation_weight=args.rotation_target_weight,
            gripper_weight=args.gripper_target_weight,
            translation_m_l2_errors=candidates.translation_m_l2_errors,
            rotation_geodesic_errors=candidates.rotation_geodesic_errors,
            gripper_errors=candidates.gripper_errors,
        )
        event_errors = None
        if event_targets is not None and args.event_target_weight > 0.0:
            event_errors = _event_alignment_errors(candidates.sample_actions, batch, event_targets, device)
            target_scores = target_scores + args.event_target_weight * _negative_standardized_error(
                event_errors
            )
        loss, loss_metrics = _scorer_training_loss(
            logits,
            target_scores,
            candidates,
            args,
            event_errors=event_errors,
        )
        batch_size = int(candidates.condition.shape[0])
        batch_metrics = _batch_readout_metrics(
            scorer,
            action_decoder,
            candidates,
            batch,
            device,
            selection_temperature=args.selection_temperature,
            motion_mode=args.motion_mode,
        )
        batch_metrics["loss"] = float(loss.cpu())
        batch_metrics.update(loss_metrics)
        if event_errors is not None:
            batch_metrics.update(_event_training_metrics(logits, event_errors))
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _make_candidates(
    cvae: VisualConditionedGeoMoCoCVAE,
    action_decoder: ActionDecoder,
    batch: dict[str, object],
    device: torch.device,
    conditioner: CategoricalConditioner,
    num_samples: int,
) -> CandidateBatch:
    cvae.eval()
    action_decoder.eval()
    context = batch["context"].to(device)
    motion = batch["motion"].to(device)
    actions = batch["actions"].to(device)
    visual = _batch_visual(batch, device)
    conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    condition = cvae.condition(context, visual, conditioning)
    prior_mean, prior_logvar = cvae.encode_prior(condition)
    prior_mean_motion = cvae.decode(condition, prior_mean)
    prior_mean_actions = action_decoder(context, prior_mean_motion)
    samples = _sample_prior_motions(cvae, condition, prior_mean, prior_logvar, num_samples)
    flat_samples = samples.reshape(-1, samples.shape[-1])
    flat_context = (
        context.unsqueeze(0)
        .expand(num_samples, -1, -1)
        .reshape(num_samples * context.shape[0], context.shape[-1])
    )
    flat_actions = action_decoder(flat_context, flat_samples)
    sample_actions = flat_actions.reshape(
        num_samples,
        context.shape[0],
        actions.shape[1],
        actions.shape[2],
    )
    action_errors = (sample_actions - actions.unsqueeze(0)).pow(2).mean(dim=(2, 3))
    motion_errors = (samples - motion.unsqueeze(0)).pow(2).mean(dim=-1)
    structured_errors = _structured_action_errors(sample_actions, actions)
    return CandidateBatch(
        condition=condition,
        prior_mean_motion=prior_mean_motion,
        prior_mean_actions=prior_mean_actions,
        samples=samples,
        sample_actions=sample_actions,
        action_errors=action_errors,
        motion_errors=motion_errors,
        translation_m_l2_errors=structured_errors.translation_m_l2,
        rotation_geodesic_errors=structured_errors.rotation_geodesic_rad,
        gripper_errors=structured_errors.gripper_mse,
    )


def _structured_action_errors(
    sample_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> StructuredActionErrors:
    if sample_actions.ndim != 4:
        raise ValueError(f"sample_actions must have shape [K, B, H, A], got {sample_actions.shape}")
    if target_actions.ndim != 3:
        raise ValueError(f"target_actions must have shape [B, H, A], got {target_actions.shape}")
    if sample_actions.shape[1:] != target_actions.shape:
        raise ValueError(
            "sample_actions trailing shape must match target_actions: "
            f"{sample_actions.shape[1:]} vs {target_actions.shape}"
        )

    error = sample_actions - target_actions.unsqueeze(0)
    num_samples, batch_size = sample_actions.shape[:2]
    zero = torch.zeros(
        (num_samples, batch_size),
        dtype=sample_actions.dtype,
        device=sample_actions.device,
    )
    action_dim = int(sample_actions.shape[-1])
    semantics = default_libero_osc_pose_action_semantics()

    translation_m_l2 = zero
    if action_dim >= 3:
        translation_scale = torch.as_tensor(
            semantics.translation_scale_m,
            dtype=sample_actions.dtype,
            device=sample_actions.device,
        )
        translation_m_l2 = torch.linalg.vector_norm(
            error[..., :3] * translation_scale,
            dim=-1,
        ).mean(dim=-1)

    rotation_geodesic_rad = zero
    if action_dim >= 6:
        rotation_scale = torch.as_tensor(
            semantics.rotation_scale_rad,
            dtype=sample_actions.dtype,
            device=sample_actions.device,
        )
        pred_rotvec = sample_actions[..., 3:6] * rotation_scale
        target_rotvec = target_actions.unsqueeze(0).expand_as(sample_actions)[..., 3:6]
        target_rotvec = target_rotvec * rotation_scale
        rotation_geodesic_rad = rotation_geodesic_angle(pred_rotvec, target_rotvec).mean(dim=-1)

    gripper_mse = zero
    if action_dim > 6:
        gripper_mse = error[..., 6:].square().mean(dim=(2, 3))

    return StructuredActionErrors(
        translation_m_l2=translation_m_l2,
        rotation_geodesic_rad=rotation_geodesic_rad,
        gripper_mse=gripper_mse,
    )


def _score_candidates(
    scorer: SampleScoreNet,
    candidates: CandidateBatch,
) -> torch.Tensor:
    num_samples, batch_size, motion_dim = candidates.samples.shape
    condition = (
        candidates.condition.unsqueeze(0)
        .expand(num_samples, -1, -1)
        .reshape(num_samples * batch_size, candidates.condition.shape[-1])
    )
    motions = candidates.samples.reshape(num_samples * batch_size, motion_dim)
    actions = candidates.sample_actions.reshape(
        num_samples * batch_size,
        candidates.sample_actions.shape[-2],
        candidates.sample_actions.shape[-1],
    )
    return scorer(condition, motions, actions).reshape(num_samples, batch_size)


def _target_scores(
    action_errors: torch.Tensor,
    motion_errors: torch.Tensor,
    target_kind: str,
    *,
    action_weight: float,
    motion_weight: float,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    gripper_weight: float = 1.0,
    translation_m_l2_errors: torch.Tensor | None = None,
    rotation_geodesic_errors: torch.Tensor | None = None,
    gripper_errors: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_kind == "action":
        return _negative_standardized_error(action_errors)
    if target_kind == "motion":
        return _negative_standardized_error(motion_errors)
    if target_kind == "hybrid":
        return (
            action_weight * _negative_standardized_error(action_errors)
            + motion_weight * _negative_standardized_error(motion_errors)
        )
    if target_kind == "se3":
        translation_errors = _require_target_errors(
            translation_m_l2_errors,
            "translation_m_l2_errors",
        )
        rotation_errors = _require_target_errors(
            rotation_geodesic_errors,
            "rotation_geodesic_errors",
        )
        return (
            translation_weight * _negative_standardized_error(translation_errors)
            + rotation_weight * _negative_standardized_error(rotation_errors)
        )
    if target_kind == "se3_gripper":
        translation_errors = _require_target_errors(
            translation_m_l2_errors,
            "translation_m_l2_errors",
        )
        rotation_errors = _require_target_errors(
            rotation_geodesic_errors,
            "rotation_geodesic_errors",
        )
        gripper_target_errors = _require_target_errors(gripper_errors, "gripper_errors")
        return (
            translation_weight * _negative_standardized_error(translation_errors)
            + rotation_weight * _negative_standardized_error(rotation_errors)
            + gripper_weight * _negative_standardized_error(gripper_target_errors)
        )
    raise ValueError("target_kind must be one of: action, motion, hybrid, se3, se3_gripper")


def _event_alignment_errors(
    sample_actions: torch.Tensor,
    batch: dict[str, object],
    event_targets: EventTargetData,
    device: torch.device,
) -> torch.Tensor:
    if sample_actions.ndim != 4:
        raise ValueError(f"sample_actions must have shape [K, B, H, A], got {sample_actions.shape}")
    window_ids = [str(window_id) for window_id in batch["window_id"]]
    if sample_actions.shape[1] != len(window_ids):
        raise ValueError("sample_actions batch size must match batch window ids")
    gt_labels = [event_targets.gt_labels[window_id] for window_id in window_ids]
    previous_commands = [event_targets.previous_commands.get(window_id) for window_id in window_ids]
    errors: list[list[float]] = []
    sample_chunks = sample_actions.detach().cpu().tolist()
    for sample_index in range(sample_actions.shape[0]):
        sample_errors = []
        for chunk, previous_command, gt_label in zip(
            sample_chunks[sample_index],
            previous_commands,
            gt_labels,
        ):
            pred_label = label_gripper_transition_events(
                chunk,
                previous_gripper_command=previous_command,
                config=event_targets.config,
            )
            sample_errors.append(
                _event_alignment_error(
                    pred_label,
                    gt_label,
                    horizon=event_targets.horizon,
                )
            )
        errors.append(sample_errors)
    return torch.tensor(errors, dtype=sample_actions.dtype, device=device)


def _event_alignment_error(
    pred: GripperEventLabel,
    target: GripperEventLabel,
    *,
    horizon: int,
) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    type_penalty = 0.0 if pred.event_type == target.event_type else float(horizon + 1)
    if pred.event_step is None and target.event_step is None:
        step_penalty = 0.0
    elif pred.event_step is None or target.event_step is None:
        step_penalty = float(horizon)
    else:
        step_penalty = float(abs(int(pred.event_step) - int(target.event_step)))
    return type_penalty + step_penalty


def _event_training_metrics(
    logits: torch.Tensor,
    event_errors: torch.Tensor,
) -> dict[str, float]:
    if logits.shape != event_errors.shape:
        raise ValueError(f"logits and event_errors shape mismatch: {logits.shape} vs {event_errors.shape}")
    batch_indices = torch.arange(logits.shape[1], device=logits.device)
    selected_indices = logits.argmax(dim=0)
    oracle_indices = event_errors.argmin(dim=0)
    selected_errors = event_errors[selected_indices, batch_indices]
    oracle_errors = event_errors[oracle_indices, batch_indices]
    selected_rank = (event_errors < selected_errors.unsqueeze(0)).sum(dim=0) + 1
    return {
        "event_scorer_selected_error": float(selected_errors.mean().detach().cpu()),
        "event_oracle_error": float(oracle_errors.mean().detach().cpu()),
        "event_scorer_oracle_rank": float(selected_rank.float().mean().detach().cpu()),
        "event_scorer_oracle_match": float(
            (selected_indices == oracle_indices).float().mean().detach().cpu()
        ),
    }


def _require_target_errors(errors: torch.Tensor | None, name: str) -> torch.Tensor:
    if errors is None:
        raise ValueError(f"{name} is required for structured target scoring")
    return errors


def _negative_standardized_error(errors: torch.Tensor) -> torch.Tensor:
    mean = errors.mean(dim=0, keepdim=True)
    std = errors.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return -(errors - mean) / std


def _standardized_error(errors: torch.Tensor) -> torch.Tensor:
    mean = errors.mean(dim=0, keepdim=True)
    std = errors.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (errors - mean) / std


def _listwise_ranking_loss(
    logits: torch.Tensor,
    target_scores: torch.Tensor,
    target_temperature: float,
) -> torch.Tensor:
    target_probs = torch.softmax(target_scores / target_temperature, dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    return -(target_probs * log_probs).sum(dim=0).mean()


def _scorer_training_loss(
    logits: torch.Tensor,
    target_scores: torch.Tensor,
    candidates: CandidateBatch,
    args: argparse.Namespace,
    *,
    event_errors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    listwise_loss = _listwise_ranking_loss(logits, target_scores, args.target_temperature)
    hard_negative_loss = logits.new_tensor(0.0)
    if args.hard_negative_target_kind != "none" and args.hard_negative_weight > 0.0:
        hard_negative_loss = _hard_negative_ranking_loss(
            logits,
            candidates,
            args.hard_negative_target_kind,
            margin=args.hard_negative_margin,
        )
    event_hard_negative_loss = logits.new_tensor(0.0)
    if args.event_hard_negative_weight > 0.0:
        if event_errors is None:
            raise ValueError("event_errors are required when event hard-negative loss is enabled")
        event_hard_negative_loss = _event_hard_negative_ranking_loss(
            logits,
            target_scores,
            candidates.action_errors,
            event_errors,
            margin=args.event_hard_negative_margin,
        )
    loss = (
        listwise_loss
        + args.hard_negative_weight * hard_negative_loss
        + args.event_hard_negative_weight * event_hard_negative_loss
    )
    return loss, {
        "listwise_loss": float(listwise_loss.detach().cpu()),
        "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
        "event_hard_negative_loss": float(event_hard_negative_loss.detach().cpu()),
    }


def _hard_negative_ranking_loss(
    logits: torch.Tensor,
    candidates: CandidateBatch,
    hard_negative_target_kind: str,
    *,
    margin: float,
) -> torch.Tensor:
    if logits.shape[0] < 2:
        return logits.new_tensor(0.0)
    batch_indices = torch.arange(logits.shape[1], device=logits.device)
    positive_indices = candidates.action_errors.argmin(dim=0)
    hard_scores = _structured_oracle_scores(candidates, hard_negative_target_kind).clone()
    hard_scores[positive_indices, batch_indices] = -torch.inf
    negative_indices = hard_scores.argmax(dim=0)
    positive_logits = logits[positive_indices, batch_indices]
    negative_logits = logits[negative_indices, batch_indices]
    if margin > 0.0:
        return torch.relu(margin - (positive_logits - negative_logits)).mean()
    return torch.nn.functional.softplus(negative_logits - positive_logits).mean()


def _event_hard_negative_ranking_loss(
    logits: torch.Tensor,
    target_scores: torch.Tensor,
    action_errors: torch.Tensor,
    event_errors: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if logits.shape[0] < 2:
        return logits.new_tensor(0.0)
    if not (logits.shape == target_scores.shape == action_errors.shape == event_errors.shape):
        raise ValueError(
            "logits, target_scores, action_errors, and event_errors must have matching shapes"
        )
    batch_indices = torch.arange(logits.shape[1], device=logits.device)
    positive_indices = target_scores.argmax(dim=0)
    negative_scores = _negative_standardized_error(action_errors) + _standardized_error(
        event_errors
    )
    negative_scores = negative_scores.clone()
    negative_scores[positive_indices, batch_indices] = -torch.inf
    negative_indices = negative_scores.argmax(dim=0)
    positive_logits = logits[positive_indices, batch_indices]
    negative_logits = logits[negative_indices, batch_indices]
    if margin > 0.0:
        return torch.relu(margin - (positive_logits - negative_logits)).mean()
    return torch.nn.functional.softplus(negative_logits - positive_logits).mean()


@torch.no_grad()
def _batch_readout_metrics(
    scorer: SampleScoreNet,
    action_decoder: ActionDecoder,
    candidates: CandidateBatch,
    batch: dict[str, object],
    device: torch.device,
    *,
    selection_temperature: float,
    motion_mode: str,
) -> dict[str, float]:
    context = batch["context"].to(device)
    motion = batch["motion"].to(device)
    actions = batch["actions"].to(device)
    logits = _score_candidates(scorer, candidates)
    batch_indices = torch.arange(context.shape[0], device=device)
    oracle_indices = candidates.action_errors.argmin(dim=0)
    scorer_indices = logits.argmax(dim=0)
    oracle_motion = _select_by_indices(candidates.samples, oracle_indices)
    oracle_actions = _select_by_indices(candidates.sample_actions, oracle_indices)
    scorer_motion = _select_by_indices(candidates.samples, scorer_indices)
    scorer_actions = action_decoder(context, scorer_motion)
    weights = torch.softmax(logits / selection_temperature, dim=0)
    soft_motion = (weights.unsqueeze(-1) * candidates.samples).sum(dim=0)
    soft_actions = action_decoder(context, soft_motion)
    flat_sample_actions = candidates.sample_actions.reshape(
        -1,
        candidates.sample_actions.shape[-2],
        candidates.sample_actions.shape[-1],
    )
    repeated_actions = (
        actions.unsqueeze(0)
        .expand(candidates.sample_actions.shape[0], -1, -1, -1)
        .reshape_as(flat_sample_actions)
    )
    flat_samples = candidates.samples.reshape(-1, candidates.samples.shape[-1])
    repeated_motion = (
        motion.unsqueeze(0).expand(candidates.samples.shape[0], -1, -1).reshape_as(flat_samples)
    )
    selected_errors = candidates.action_errors[scorer_indices, batch_indices]
    oracle_errors = candidates.action_errors[oracle_indices, batch_indices]
    selected_rank = (candidates.action_errors < selected_errors.unsqueeze(0)).sum(dim=0) + 1

    metrics: dict[str, float] = {}
    metrics.update(
        _prefix("prior_mean_motion", _prediction_metrics(candidates.prior_mean_motion, motion, motion_mode))
    )
    metrics.update(_prefix("sample_mean_motion", _prediction_metrics(flat_samples, repeated_motion, motion_mode)))
    metrics.update(_prefix("oracle_best_motion", _prediction_metrics(oracle_motion, motion, motion_mode)))
    metrics.update(_prefix("flat_oracle_best_motion", _prediction_metrics(oracle_motion, motion, motion_mode)))
    metrics.update(_prefix("scorer_argmax_motion", _prediction_metrics(scorer_motion, motion, motion_mode)))
    metrics.update(_prefix("scorer_soft_motion", _prediction_metrics(soft_motion, motion, motion_mode)))
    metrics.update(_prefix("prior_mean_action", action_metrics(candidates.prior_mean_actions, actions)))
    metrics.update(_prefix("sample_mean_action", action_metrics(flat_sample_actions, repeated_actions)))
    metrics.update(_prefix("oracle_best_action", action_metrics(oracle_actions, actions)))
    metrics.update(_prefix("flat_oracle_best_action", action_metrics(oracle_actions, actions)))
    metrics.update(_prefix("scorer_argmax_action", action_metrics(scorer_actions, actions)))
    metrics.update(_prefix("scorer_soft_motion_action", action_metrics(soft_actions, actions)))
    metrics.update(
        _structured_oracle_readout_metrics(
            "se3",
            "se3",
            candidates,
            scorer_indices,
            batch_indices,
            motion,
            actions,
            motion_mode,
        )
    )
    metrics.update(
        _structured_oracle_readout_metrics(
            "se3_gripper",
            "se3_gripper",
            candidates,
            scorer_indices,
            batch_indices,
            motion,
            actions,
            motion_mode,
        )
    )
    metrics.update(_diversity_metrics(candidates.samples, candidates.prior_mean_motion))
    metrics["scorer_top1_oracle_match"] = float((scorer_indices == oracle_indices).float().mean().cpu())
    metrics["scorer_top1_flat_oracle_match"] = metrics["scorer_top1_oracle_match"]
    metrics["scorer_selected_oracle_rank"] = float(selected_rank.float().mean().cpu())
    metrics["scorer_selected_flat_oracle_rank"] = metrics["scorer_selected_oracle_rank"]
    metrics["scorer_selected_sample_action_mse"] = float(selected_errors.mean().cpu())
    metrics["oracle_best_sample_action_mse"] = float(oracle_errors.mean().cpu())
    metrics["scorer_action_mse_regret_to_oracle"] = float((selected_errors - oracle_errors).mean().cpu())
    metrics["scorer_flat_action_mse_regret_to_oracle"] = metrics[
        "scorer_action_mse_regret_to_oracle"
    ]
    return metrics


def _structured_oracle_readout_metrics(
    metric_prefix: str,
    target_kind: str,
    candidates: CandidateBatch,
    scorer_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    motion: torch.Tensor,
    actions: torch.Tensor,
    motion_mode: str,
) -> dict[str, float]:
    target_scores = _structured_oracle_scores(candidates, target_kind)
    oracle_indices = target_scores.argmax(dim=0)
    oracle_motion = _select_by_indices(candidates.samples, oracle_indices)
    oracle_actions = _select_by_indices(candidates.sample_actions, oracle_indices)
    selected_scores = target_scores[scorer_indices, batch_indices]
    oracle_scores = target_scores[oracle_indices, batch_indices]
    selected_rank = _rank_selected_scores(target_scores, scorer_indices, batch_indices)

    metrics: dict[str, float] = {}
    metrics.update(
        _prefix(
            f"{metric_prefix}_oracle_best_motion",
            _prediction_metrics(oracle_motion, motion, motion_mode),
        )
    )
    metrics.update(
        _prefix(
            f"{metric_prefix}_oracle_best_action",
            action_metrics(oracle_actions, actions),
        )
    )
    metrics[f"scorer_top1_{metric_prefix}_oracle_match"] = float(
        (scorer_indices == oracle_indices).float().mean().cpu()
    )
    metrics[f"scorer_selected_{metric_prefix}_oracle_rank"] = float(
        selected_rank.float().mean().cpu()
    )
    metrics[f"scorer_{metric_prefix}_score_regret_to_oracle"] = float(
        (oracle_scores - selected_scores).mean().cpu()
    )
    return metrics


def _structured_oracle_scores(candidates: CandidateBatch, target_kind: str) -> torch.Tensor:
    return _target_scores(
        candidates.action_errors,
        candidates.motion_errors,
        target_kind,
        action_weight=1.0,
        motion_weight=1.0,
        translation_weight=1.0,
        rotation_weight=1.0,
        gripper_weight=1.0,
        translation_m_l2_errors=candidates.translation_m_l2_errors,
        rotation_geodesic_errors=candidates.rotation_geodesic_errors,
        gripper_errors=candidates.gripper_errors,
    )


def _rank_selected_scores(
    target_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    selected_scores = target_scores[selected_indices, batch_indices]
    return (target_scores > selected_scores.unsqueeze(0)).sum(dim=0) + 1


def _select_by_indices(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(indices.shape[0], device=values.device)
    return values[indices, batch_indices]


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
