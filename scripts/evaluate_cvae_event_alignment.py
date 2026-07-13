#!/usr/bin/env python3
"""Evaluate gripper-transition event alignment for visual cVAE samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.event_labels import (  # noqa: E402
    GripperEventConfig,
    GripperEventLabel,
    label_gripper_events_for_windows,
    label_gripper_transition_events,
    previous_gripper_commands_for_windows,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.sample_readout import SampleScoreNet  # noqa: E402
from evaluate_visual_cvae_sample_scorer import _build_scorer_from_metrics  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_sample_scorer import (  # noqa: E402
    _build_checkpoint_conditioner,
    _freeze_module,
    _make_candidates,
    _score_candidates,
    _structured_oracle_scores,
)


EVENT_CLASSES = (
    "close_transition",
    "mixed_transition",
    "open_transition",
    "sustain_close",
    "sustain_open",
    "hold",
)
EVENT_INDEX = {event_type: index for index, event_type in enumerate(EVENT_CLASSES)}
TRANSITION_EVENTS = {"close_transition", "mixed_transition", "open_transition"}


@dataclass
class EventReadoutAccumulator:
    """Aggregate event-type and event-timing metrics for one readout."""

    count: int = 0
    event_type_match: int = 0
    gt_transition_count: int = 0
    pred_transition_count: int = 0
    transition_type_match: int = 0
    transition_step_exact: int = 0
    transition_step_within_1: int = 0
    transition_step_within_2: int = 0
    transition_step_abs_error_sum: float = 0.0
    transition_step_abs_error_count: int = 0

    def __post_init__(self) -> None:
        self.confusion = torch.zeros((len(EVENT_CLASSES), len(EVENT_CLASSES)), dtype=torch.long)

    def add_many(
        self,
        pred_labels: Sequence[GripperEventLabel],
        gt_labels: Sequence[GripperEventLabel],
    ) -> None:
        if len(pred_labels) != len(gt_labels):
            raise ValueError("pred_labels and gt_labels must have the same length")
        for pred, target in zip(pred_labels, gt_labels):
            self.add(pred, target)

    def add(self, pred: GripperEventLabel, target: GripperEventLabel) -> None:
        self.count += 1
        pred_type = _normalize_event_type(pred.event_type)
        target_type = _normalize_event_type(target.event_type)
        self.confusion[EVENT_INDEX[target_type], EVENT_INDEX[pred_type]] += 1
        if pred_type == target_type:
            self.event_type_match += 1
        if target_type in TRANSITION_EVENTS:
            self.gt_transition_count += 1
            if pred_type == target_type:
                self.transition_type_match += 1
        if pred_type in TRANSITION_EVENTS:
            self.pred_transition_count += 1
        if target_type in TRANSITION_EVENTS and pred_type == target_type:
            if target.event_step is not None and pred.event_step is not None:
                step_error = abs(int(pred.event_step) - int(target.event_step))
                self.transition_step_abs_error_sum += float(step_error)
                self.transition_step_abs_error_count += 1
                if step_error == 0:
                    self.transition_step_exact += 1
                if step_error <= 1:
                    self.transition_step_within_1 += 1
                if step_error <= 2:
                    self.transition_step_within_2 += 1

    def metrics(self) -> dict[str, Any]:
        classification = _classification_metrics(self.confusion)
        transition_support = max(self.gt_transition_count, 1)
        return {
            "count": self.count,
            "event_type_accuracy": self.event_type_match / self.count if self.count else 0.0,
            "transition_type_accuracy": self.transition_type_match / transition_support,
            "pred_transition_rate": self.pred_transition_count / self.count if self.count else 0.0,
            "gt_transition_rate": self.gt_transition_count / self.count if self.count else 0.0,
            "transition_step_exact": self.transition_step_exact / transition_support,
            "transition_step_within_1": self.transition_step_within_1 / transition_support,
            "transition_step_within_2": self.transition_step_within_2 / transition_support,
            "transition_step_mae_matched": (
                self.transition_step_abs_error_sum / self.transition_step_abs_error_count
                if self.transition_step_abs_error_count
                else None
            ),
            **classification,
        }


@dataclass
class CoverageAccumulator:
    """Aggregate best-case event coverage over K cVAE samples."""

    horizon: int
    count: int = 0
    transition_count: int = 0
    any_event_type_match: int = 0
    any_transition_type_match: int = 0
    any_transition_step_exact: int = 0
    any_transition_step_within_1: int = 0
    any_transition_step_within_2: int = 0
    best_event_error_sum: float = 0.0

    def add(self, candidate_labels: Sequence[GripperEventLabel], target: GripperEventLabel) -> None:
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty")
        self.count += 1
        target_type = _normalize_event_type(target.event_type)
        is_transition = target_type in TRANSITION_EVENTS
        if is_transition:
            self.transition_count += 1
        self.any_event_type_match += int(
            any(_normalize_event_type(label.event_type) == target_type for label in candidate_labels)
        )
        if is_transition:
            matched = [
                label
                for label in candidate_labels
                if _normalize_event_type(label.event_type) == target_type
            ]
            self.any_transition_type_match += int(bool(matched))
            self.any_transition_step_exact += int(
                any(_same_event_step(label, target, tolerance=0) for label in matched)
            )
            self.any_transition_step_within_1 += int(
                any(_same_event_step(label, target, tolerance=1) for label in matched)
            )
            self.any_transition_step_within_2 += int(
                any(_same_event_step(label, target, tolerance=2) for label in matched)
            )
        self.best_event_error_sum += min(
            event_alignment_error(label, target, horizon=self.horizon)
            for label in candidate_labels
        )

    def metrics(self) -> dict[str, float | int]:
        transition_support = max(self.transition_count, 1)
        return {
            "count": self.count,
            "transition_count": self.transition_count,
            "any_event_type_match": self.any_event_type_match / self.count if self.count else 0.0,
            "any_transition_type_match": self.any_transition_type_match / transition_support,
            "any_transition_step_exact": self.any_transition_step_exact / transition_support,
            "any_transition_step_within_1": self.any_transition_step_within_1 / transition_support,
            "any_transition_step_within_2": self.any_transition_step_within_2 / transition_support,
            "best_event_alignment_error": (
                self.best_event_error_sum / self.count if self.count else 0.0
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Gate 2.4h-c cVAE sample event alignment."
    )
    parser.add_argument("--checkpoint", required=True, help="Gate 2.4c cVAE checkpoint.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scorer-checkpoint", default=None)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--action-decoder-checkpoint", default=None)
    parser.add_argument("--event-audit-json", default=None)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument("--close-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    cvae_checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]

    scorer_checkpoint = None
    scorer_checkpoint_path = None
    scorer_metrics = None
    if args.scorer_checkpoint:
        scorer_checkpoint_path = Path(args.scorer_checkpoint).expanduser().resolve()
        scorer_checkpoint = torch.load(scorer_checkpoint_path, map_location=device, weights_only=False)
        scorer_metrics = scorer_checkpoint["metrics"]

    source_metrics = scorer_metrics or cvae_metrics
    windows_jsonl = args.windows_jsonl or source_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or source_metrics["visual_feature_cache"]
    split_by = args.split_by or source_metrics.get("split_by", "episode")
    condition_on = args.condition_on or source_metrics["conditioning"]["condition_on"]
    motion_mode = str(
        source_metrics.get("motion_mode", cvae_metrics.get("motion_mode", source_metrics["dataset"].get("motion_mode")))
    )

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_checkpoint_conditioner(dataset, source_metrics, condition_on)
    visual_token_config = _resolve_visual_token_config(
        dataset,
        cvae_metrics["visual_token_config"]["visual_token_count"],
        cvae_metrics["visual_token_config"]["visual_token_dim"],
    )
    cvae = _load_model(
        cvae_checkpoint,
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim,
        device=device,
    )
    _freeze_module(cvae)

    action_decoder_checkpoint = args.action_decoder_checkpoint or source_metrics.get(
        "action_decoder_checkpoint"
    )
    if not action_decoder_checkpoint:
        raise ValueError("--action-decoder-checkpoint is required")
    action_decoder, action_decoder_config = _load_action_decoder(action_decoder_checkpoint, device)
    if action_decoder_config["motion_mode"] != motion_mode:
        raise ValueError(
            "action decoder motion mode must match cVAE/scorer motion mode: "
            f"{action_decoder_config['motion_mode']} vs {motion_mode}"
        )
    if int(action_decoder_config["motion_dim"]) != int(spec.motion_dim):
        raise ValueError(
            "action decoder motion dim must match event-alignment dataset motion dim: "
            f"{action_decoder_config['motion_dim']} vs {spec.motion_dim}"
        )
    _freeze_action_decoder(action_decoder)

    scorer = None
    if scorer_checkpoint is not None and scorer_metrics is not None:
        scorer = _build_scorer_from_metrics(
            scorer_metrics,
            condition_dim=cvae.condition_dim,
            motion_dim=spec.motion_dim,
            action_dim=spec.action_dim,
            horizon=spec.horizon,
        ).to(device)
        scorer.load_state_dict(scorer_checkpoint["model_state_dict"])
        scorer.eval()

    event_config = _event_config_from_args(args)
    gt_labels = label_gripper_events_for_windows(
        dataset.windows,
        config=event_config,
        label_mode="transition",
    )
    previous_commands = previous_gripper_commands_for_windows(dataset.windows)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    metrics = evaluate_event_alignment(
        cvae=cvae,
        action_decoder=action_decoder,
        scorer=scorer,
        loader=val_loader,
        device=device,
        conditioner=conditioner,
        num_samples=args.num_samples,
        horizon=spec.horizon,
        event_config=event_config,
        gt_labels=gt_labels,
        previous_commands=previous_commands,
    )
    output = {
        "schema_version": "geomoco_wm_cvae_event_alignment_v0",
        "checkpoint": str(cvae_checkpoint_path),
        "scorer_checkpoint": str(scorer_checkpoint_path) if scorer_checkpoint_path else None,
        "action_decoder_checkpoint": str(Path(action_decoder_checkpoint).expanduser()),
        "action_decoder_config": action_decoder_config,
        "dataset": spec.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "motion_mode": motion_mode,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser()),
        "visual_token_config": visual_token_config,
        "num_samples": args.num_samples,
        "event_config": event_config.to_dict(),
        "metrics": metrics,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": metrics}, indent=2))


@torch.no_grad()
def evaluate_event_alignment(
    *,
    cvae: torch.nn.Module,
    action_decoder: torch.nn.Module,
    scorer: SampleScoreNet | None,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    conditioner: Any,
    num_samples: int,
    horizon: int,
    event_config: GripperEventConfig,
    gt_labels: dict[str, GripperEventLabel],
    previous_commands: dict[str, float | None],
) -> dict[str, Any]:
    readouts = {
        "prior_mean": EventReadoutAccumulator(),
        "random_sample_mean": EventReadoutAccumulator(),
        "event_oracle_best": EventReadoutAccumulator(),
        "flat_action_oracle_best": EventReadoutAccumulator(),
        "se3_gripper_oracle_best": EventReadoutAccumulator(),
    }
    if scorer is not None:
        readouts["scorer_argmax"] = EventReadoutAccumulator()
    coverage = CoverageAccumulator(horizon=horizon)
    selected_ranks: dict[str, list[float]] = {
        "flat_action_oracle_best": [],
        "se3_gripper_oracle_best": [],
    }
    if scorer is not None:
        selected_ranks["scorer_argmax"] = []
    selected_errors: dict[str, list[float]] = {key: [] for key in selected_ranks}
    gt_counts: Counter[str] = Counter()

    for batch in loader:
        candidates = _make_candidates(cvae, action_decoder, batch, device, conditioner, num_samples)
        window_ids = [str(window_id) for window_id in batch["window_id"]]
        batch_gt_labels = [gt_labels[window_id] for window_id in window_ids]
        batch_previous = [previous_commands.get(window_id) for window_id in window_ids]
        for label in batch_gt_labels:
            gt_counts[_normalize_event_type(label.event_type)] += 1

        prior_labels = label_action_chunks(
            candidates.prior_mean_actions,
            batch_previous,
            event_config,
        )
        sample_labels = label_sample_action_chunks(
            candidates.sample_actions,
            batch_previous,
            event_config,
        )
        readouts["prior_mean"].add_many(prior_labels, batch_gt_labels)
        for sample_index in range(num_samples):
            readouts["random_sample_mean"].add_many(sample_labels[sample_index], batch_gt_labels)
        horizon = int(candidates.sample_actions.shape[-2])
        event_oracle_indices = event_oracle_indices_for_batch(
            sample_labels,
            batch_gt_labels,
            horizon=horizon,
        )
        event_oracle_labels = select_labels_by_indices(sample_labels, event_oracle_indices)
        readouts["event_oracle_best"].add_many(event_oracle_labels, batch_gt_labels)

        flat_indices = candidates.action_errors.argmin(dim=0).detach().cpu()
        flat_labels = select_labels_by_indices(sample_labels, flat_indices)
        readouts["flat_action_oracle_best"].add_many(flat_labels, batch_gt_labels)
        _add_rank_stats(
            "flat_action_oracle_best",
            selected_ranks,
            selected_errors,
            sample_labels,
            batch_gt_labels,
            flat_indices,
            horizon=horizon,
        )

        se3_scores = _structured_oracle_scores(candidates, "se3_gripper")
        se3_indices = se3_scores.argmax(dim=0).detach().cpu()
        se3_labels = select_labels_by_indices(sample_labels, se3_indices)
        readouts["se3_gripper_oracle_best"].add_many(se3_labels, batch_gt_labels)
        _add_rank_stats(
            "se3_gripper_oracle_best",
            selected_ranks,
            selected_errors,
            sample_labels,
            batch_gt_labels,
            se3_indices,
            horizon=horizon,
        )

        if scorer is not None:
            logits = _score_candidates(scorer, candidates)
            scorer_indices = logits.argmax(dim=0).detach().cpu()
            scorer_labels = select_labels_by_indices(sample_labels, scorer_indices)
            readouts["scorer_argmax"].add_many(scorer_labels, batch_gt_labels)
            _add_rank_stats(
                "scorer_argmax",
                selected_ranks,
                selected_errors,
                sample_labels,
                batch_gt_labels,
                scorer_indices,
                horizon=horizon,
            )

        for sample_column, target_label in zip(_columns(sample_labels), batch_gt_labels):
            coverage.add(sample_column, target_label)

    readout_metrics = {name: accumulator.metrics() for name, accumulator in readouts.items()}
    rank_metrics = {
        name: {
            "selected_event_oracle_rank": _mean(values),
            "selected_event_alignment_error": _mean(selected_errors[name]),
        }
        for name, values in selected_ranks.items()
    }
    return {
        "event_classes": list(EVENT_CLASSES),
        "gt_event_counts": {key: int(value) for key, value in sorted(gt_counts.items())},
        "readouts": readout_metrics,
        "sample_coverage": coverage.metrics(),
        "selected_event_rank": rank_metrics,
    }


def label_action_chunks(
    action_chunks: torch.Tensor,
    previous_commands: Sequence[float | None],
    config: GripperEventConfig,
) -> list[GripperEventLabel]:
    if action_chunks.ndim != 3:
        raise ValueError(f"action_chunks must have shape [B, H, A], got {action_chunks.shape}")
    if action_chunks.shape[0] != len(previous_commands):
        raise ValueError("action_chunks batch size must match previous_commands")
    chunks = action_chunks.detach().cpu().tolist()
    return [
        label_gripper_transition_events(chunk, previous_gripper_command=previous, config=config)
        for chunk, previous in zip(chunks, previous_commands)
    ]


def label_sample_action_chunks(
    sample_actions: torch.Tensor,
    previous_commands: Sequence[float | None],
    config: GripperEventConfig,
) -> list[list[GripperEventLabel]]:
    if sample_actions.ndim != 4:
        raise ValueError(f"sample_actions must have shape [K, B, H, A], got {sample_actions.shape}")
    return [
        label_action_chunks(sample_actions[index], previous_commands, config)
        for index in range(sample_actions.shape[0])
    ]


def event_oracle_indices_for_batch(
    sample_labels: Sequence[Sequence[GripperEventLabel]],
    gt_labels: Sequence[GripperEventLabel],
    *,
    horizon: int,
) -> torch.Tensor:
    if not sample_labels:
        raise ValueError("sample_labels must be non-empty")
    indices: list[int] = []
    for column, target in zip(_columns(sample_labels), gt_labels):
        errors = [event_alignment_error(label, target, horizon=horizon) for label in column]
        indices.append(min(range(len(errors)), key=errors.__getitem__))
    return torch.tensor(indices, dtype=torch.long)


def select_labels_by_indices(
    sample_labels: Sequence[Sequence[GripperEventLabel]],
    indices: torch.Tensor,
) -> list[GripperEventLabel]:
    if indices.ndim != 1:
        raise ValueError("indices must have shape [B]")
    labels: list[GripperEventLabel] = []
    for batch_index, sample_index in enumerate(indices.tolist()):
        labels.append(sample_labels[int(sample_index)][batch_index])
    return labels


def event_alignment_error(
    pred: GripperEventLabel,
    target: GripperEventLabel,
    *,
    horizon: int,
) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    pred_type = _normalize_event_type(pred.event_type)
    target_type = _normalize_event_type(target.event_type)
    type_penalty = 0.0 if pred_type == target_type else float(horizon + 1)
    if pred.event_step is None and target.event_step is None:
        step_penalty = 0.0
    elif pred.event_step is None or target.event_step is None:
        step_penalty = float(horizon)
    else:
        step_penalty = float(abs(int(pred.event_step) - int(target.event_step)))
    return type_penalty + step_penalty


def _event_config_from_args(args: argparse.Namespace) -> GripperEventConfig:
    if args.event_audit_json:
        report = json.loads(Path(args.event_audit_json).expanduser().read_text(encoding="utf-8"))
        config = report["config"]
        return GripperEventConfig(
            command_threshold=float(config["command_threshold"]),
            close_sign=int(config["close_sign"]),
        )
    return GripperEventConfig(
        command_threshold=args.command_threshold,
        close_sign=args.close_sign,
    )


def _add_rank_stats(
    name: str,
    selected_ranks: dict[str, list[float]],
    selected_errors: dict[str, list[float]],
    sample_labels: Sequence[Sequence[GripperEventLabel]],
    gt_labels: Sequence[GripperEventLabel],
    selected_indices: torch.Tensor,
    *,
    horizon: int,
) -> None:
    for batch_index, (column, target, selected_index) in enumerate(
        zip(_columns(sample_labels), gt_labels, selected_indices.tolist())
    ):
        del batch_index
        errors = torch.tensor(
            [event_alignment_error(label, target, horizon=horizon) for label in column],
            dtype=torch.float32,
        )
        selected_error = float(errors[int(selected_index)].item())
        rank = int((errors < selected_error).sum().item()) + 1
        selected_ranks[name].append(float(rank))
        selected_errors[name].append(selected_error)


def _columns(values: Sequence[Sequence[GripperEventLabel]]) -> list[list[GripperEventLabel]]:
    if not values:
        return []
    batch_size = len(values[0])
    return [[values[sample_index][batch_index] for sample_index in range(len(values))] for batch_index in range(batch_size)]


def _classification_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    total = int(confusion.sum().item())
    if total == 0:
        return {
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_recall": {},
            "per_class_precision": {},
            "per_class_f1": {},
            "confusion_matrix": confusion.tolist(),
        }
    recalls: list[float] = []
    f1_values: list[float] = []
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for index, name in enumerate(EVENT_CLASSES):
        tp = float(confusion[index, index].item())
        support = float(confusion[index, :].sum().item())
        predicted = float(confusion[:, index].sum().item())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_recall[name] = recall
        per_class_precision[name] = precision
        per_class_f1[name] = f1
        if support:
            recalls.append(recall)
            f1_values.append(f1)
    return {
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def _normalize_event_type(event_type: str) -> str:
    return event_type if event_type in EVENT_INDEX else "hold"


def _same_event_step(
    pred: GripperEventLabel,
    target: GripperEventLabel,
    *,
    tolerance: int,
) -> bool:
    if pred.event_step is None or target.event_step is None:
        return False
    return abs(int(pred.event_step) - int(target.event_step)) <= tolerance


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
