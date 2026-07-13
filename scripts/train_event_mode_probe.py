#!/usr/bin/env python3
"""Train Gate 3.1b probes for event-mode prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.common import MLP  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _batch_conditioning,
    _build_conditioner,
    _parse_hidden_dims,
    _resolve_device,
)


STABLE_EVENT_MODE_CLASSES = (
    "sustain_open::none",
    "sustain_close::none",
    "transition_close::early",
    "transition_close::middle",
    "transition_close::late",
    "transition_open::early",
    "transition_open::middle",
    "transition_open::late",
)


class EventModeProbeNet(nn.Module):
    """Small MLP classifier for Gate 3.1 event modes."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        layers: list[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(MLP(input_dim, hidden_dims, num_classes))
        if dropout > 0.0:
            layers.insert(-1, nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class EventModeProbeSpec:
    input_variant: str
    input_dim: int
    num_classes: int
    class_names: tuple[str, ...]
    class_set: str
    event_mode_audit_json: str
    dropped_modes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_variant": self.input_variant,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "class_set": self.class_set,
            "event_mode_audit_json": self.event_mode_audit_json,
            "dropped_modes": self.dropped_modes,
        }


class EventModeProbeDataset(Dataset):
    """Filtered event-mode probe dataset backed by exported windows."""

    def __init__(
        self,
        base_dataset: OracleActionWindowDataset,
        *,
        event_mode_audit_json: str | Path,
        class_names: tuple[str, ...],
    ) -> None:
        self.base_dataset = base_dataset
        self.class_names = class_names
        self.class_index = {name: index for index, name in enumerate(class_names)}
        labels_by_window_id = _load_event_mode_labels(event_mode_audit_json)
        self.base_indices: list[int] = []
        self.labels: list[int] = []
        self.dropped_modes: Counter[str] = Counter()
        for base_index, window in enumerate(base_dataset.windows):
            try:
                event_mode = labels_by_window_id[window.window_id]
            except KeyError as exc:
                raise ValueError(f"{window.window_id} is missing from event-mode audit") from exc
            label_index = self.class_index.get(event_mode)
            if label_index is None:
                self.dropped_modes[event_mode] += 1
                continue
            self.base_indices.append(base_index)
            self.labels.append(label_index)
        if not self.base_indices:
            raise ValueError("event-mode probe dataset is empty after class filtering")
        self.windows = [base_dataset.windows[index] for index in self.base_indices]

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base_dataset[self.base_indices[index]])
        item["event_mode_label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Gate 3.1b event-mode probes.")
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=["outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl"],
    )
    parser.add_argument(
        "--event-mode-audit-json",
        default="outputs/event_modes/gate3_1a_event_modes_2files.json",
    )
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-variant",
        required=True,
        choices=[
            "task_only",
            "task_proprio",
            "visual_only",
            "visual_proprio",
            "future_motion_only",
            "proprio_future_motion",
            "visual_proprio_future_motion",
        ],
    )
    parser.add_argument(
        "--motion-mode",
        default="future_delta_gripper",
        choices=["future_delta", "future_gripper", "future_delta_gripper", "none"],
    )
    parser.add_argument("--class-set", default="stable8", choices=["stable8", "all_observed"])
    parser.add_argument("--condition-on", default="suite_task", choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default="episode", choices=["window", "episode"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    needs_visual = "visual" in args.input_variant
    if needs_visual and not args.visual_feature_cache:
        raise ValueError(f"{args.input_variant} requires --visual-feature-cache")
    class_names = _class_names_from_audit(args.event_mode_audit_json, args.class_set)
    base_dataset = OracleActionWindowDataset(
        args.windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=args.motion_mode,
        visual_feature_cache_path=args.visual_feature_cache if needs_visual else None,
    )
    dataset = EventModeProbeDataset(
        base_dataset,
        event_mode_audit_json=args.event_mode_audit_json,
        class_names=class_names,
    )
    conditioner = _build_conditioner(base_dataset, args.condition_on)
    spec = EventModeProbeSpec(
        input_variant=args.input_variant,
        input_dim=_feature_dim(base_dataset, conditioner, args.input_variant),
        num_classes=len(class_names),
        class_names=class_names,
        class_set=args.class_set,
        event_mode_audit_json=str(Path(args.event_mode_audit_json).expanduser().resolve()),
        dropped_modes={key: int(value) for key, value in sorted(dataset.dropped_modes.items())},
    )
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, args.split_by)
    train_counts = _label_counts(dataset, train_indices)
    val_counts = _label_counts(dataset, val_indices)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "dataset": base_dataset.spec().to_dict(),
                    "probe": spec.to_dict(),
                    "conditioning": conditioner.to_dict(),
                    "num_filtered_windows": len(dataset),
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "train_label_counts": _named_counts(train_counts, class_names),
                    "val_label_counts": _named_counts(val_counts, class_names),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    device = _resolve_device(args.device)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    model = EventModeProbeNet(
        input_dim=spec.input_dim,
        num_classes=spec.num_classes,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        layer_norm=not args.no_layer_norm,
    ).to(device)
    class_weights = _class_weights(train_counts, len(class_names), device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int | None | dict[str, float] | list[list[int]]]] = []
    best_state = None
    best_epoch = None
    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            conditioner,
            args.input_variant,
            class_names,
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            conditioner,
            args.input_variant,
            class_names,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate = val_metrics.get("macro_f1")
        if candidate is not None and float(candidate) > best_metric:
            best_metric = float(candidate)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = _evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        conditioner,
        args.input_variant,
        class_names,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": base_dataset.spec().to_dict(),
        "probe": spec.to_dict(),
        "conditioning": conditioner.to_dict(),
        "device": str(device),
        "num_filtered_windows": len(dataset),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_label_counts": _named_counts(train_counts, class_names),
        "val_label_counts": _named_counts(val_counts, class_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
        "dropout": args.dropout,
        "layer_norm": not args.no_layer_norm,
        "split_by": args.split_by,
        "seed": args.seed,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "final": final_metrics,
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
                    "final": final_metrics,
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
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")


def _load_event_mode_labels(event_mode_audit_json: str | Path) -> dict[str, str]:
    report = json.loads(Path(event_mode_audit_json).expanduser().read_text(encoding="utf-8"))
    labels = report.get("window_labels")
    if not isinstance(labels, list):
        raise ValueError("event-mode audit JSON must include window_labels")
    return {str(item["window_id"]): str(item["event_mode"]) for item in labels}


def _class_names_from_audit(event_mode_audit_json: str | Path, class_set: str) -> tuple[str, ...]:
    if class_set == "stable8":
        return STABLE_EVENT_MODE_CLASSES
    report = json.loads(Path(event_mode_audit_json).expanduser().read_text(encoding="utf-8"))
    return tuple(sorted(str(key) for key in report["event_mode_counts"]))


def _split_indices(
    dataset: EventModeProbeDataset,
    train_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    if split_by == "window":
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        split = int(round(len(indices) * train_ratio))
        return sorted(indices[:split]), sorted(indices[split:])
    episode_to_indices: dict[str, list[int]] = {}
    for index, window in enumerate(dataset.windows):
        episode_to_indices.setdefault(window.episode_id, []).append(index)
    episodes = sorted(episode_to_indices)
    rng.shuffle(episodes)
    split = int(round(len(episodes) * train_ratio))
    train_episodes = set(episodes[:split])
    train_indices = [
        index
        for episode_id, indices in episode_to_indices.items()
        if episode_id in train_episodes
        for index in indices
    ]
    val_indices = [
        index
        for episode_id, indices in episode_to_indices.items()
        if episode_id not in train_episodes
        for index in indices
    ]
    return sorted(train_indices), sorted(val_indices)


def _make_loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader | None:
    if not indices:
        return None
    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = SubsetRandomSampler(indices, generator=generator) if shuffle else indices
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _feature_dim(
    dataset: OracleActionWindowDataset,
    conditioner: CategoricalConditioner,
    input_variant: str,
) -> int:
    dim = 0
    if input_variant in {
        "task_proprio",
        "visual_proprio",
        "proprio_future_motion",
        "visual_proprio_future_motion",
    }:
        dim += dataset.context_dim + conditioner.dim
    if input_variant == "task_only":
        dim += conditioner.dim
    if input_variant in {
        "future_motion_only",
        "proprio_future_motion",
        "visual_proprio_future_motion",
    }:
        dim += dataset.motion_dim
    if input_variant in {"visual_only", "visual_proprio", "visual_proprio_future_motion"}:
        dim += dataset.visual_dim
    if dim <= 0:
        raise ValueError(f"input variant {input_variant} produced empty features")
    return dim


def _batch_features(
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    device: torch.device,
    input_variant: str,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    context = batch["context"].to(device)
    conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    base = context if conditioning is None else torch.cat([context, conditioning], dim=-1)
    if input_variant in {
        "task_proprio",
        "visual_proprio",
        "proprio_future_motion",
        "visual_proprio_future_motion",
    }:
        parts.append(base)
    if input_variant == "task_only":
        if conditioning is None:
            raise ValueError("task_only input requires non-empty conditioning")
        parts.append(conditioning)
    if input_variant in {
        "future_motion_only",
        "proprio_future_motion",
        "visual_proprio_future_motion",
    }:
        parts.append(batch["motion"].to(device))
    if input_variant in {"visual_only", "visual_proprio", "visual_proprio_future_motion"}:
        visual = batch.get("visual")
        if not isinstance(visual, torch.Tensor):
            raise ValueError(f"{input_variant} requires batch['visual']")
        parts.append(visual.to(device=device, dtype=torch.float32))
    return torch.cat(parts, dim=-1)


def _run_epoch(
    model: EventModeProbeNet,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    input_variant: str,
    class_names: tuple[str, ...],
) -> dict[str, float | dict[str, float] | list[list[int]] | None]:
    if loader is None:
        return {"loss": None}
    model.train()
    total_loss = 0.0
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
    total_count = 0
    for batch in loader:
        features = _batch_features(batch, conditioner, device, input_variant)
        targets = batch["event_mode_label"].to(device)
        logits = model(features)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        confusion += _confusion_matrix(
            logits.detach().cpu().argmax(dim=-1),
            targets.detach().cpu(),
            len(class_names),
        )
        total_count += batch_size
    metrics = _classification_metrics(confusion, class_names)
    metrics["loss"] = total_loss / total_count
    return metrics


@torch.no_grad()
def _evaluate(
    model: EventModeProbeNet,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    input_variant: str,
    class_names: tuple[str, ...],
) -> dict[str, float | dict[str, float] | list[list[int]] | None]:
    if loader is None:
        return {"loss": None}
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
    total_count = 0
    for batch in loader:
        features = _batch_features(batch, conditioner, device, input_variant)
        targets = batch["event_mode_label"].to(device)
        logits = model(features)
        loss = loss_fn(logits, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.cpu()) * batch_size
        confusion += _confusion_matrix(logits.cpu().argmax(dim=-1), targets.cpu(), len(class_names))
        total_count += batch_size
    metrics = _classification_metrics(confusion, class_names)
    metrics["loss"] = total_loss / total_count
    return metrics


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target_value, pred_value in zip(target.tolist(), pred.tolist()):
        matrix[int(target_value), int(pred_value)] += 1
    return matrix


def _classification_metrics(
    confusion: torch.Tensor,
    class_names: tuple[str, ...],
) -> dict[str, float | dict[str, float] | list[list[int]]]:
    total = int(confusion.sum().item())
    if total == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "transition_binary_f1": 0.0,
            "transition_timing_accuracy": 0.0,
            "confusion_matrix": confusion.tolist(),
        }
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(class_names):
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
        "accuracy": float(confusion.diag().sum().item() / total),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "transition_binary_f1": _transition_binary_f1(confusion, class_names),
        "transition_timing_accuracy": _transition_timing_accuracy(confusion, class_names),
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def _transition_binary_f1(confusion: torch.Tensor, class_names: tuple[str, ...]) -> float:
    transition_indices = {
        index
        for index, name in enumerate(class_names)
        if name.startswith("transition_") or name.startswith("mixed_transition")
    }
    tp = fp = fn = 0.0
    for target_index in range(len(class_names)):
        for pred_index in range(len(class_names)):
            value = float(confusion[target_index, pred_index].item())
            target_transition = target_index in transition_indices
            pred_transition = pred_index in transition_indices
            if target_transition and pred_transition:
                tp += value
            elif not target_transition and pred_transition:
                fp += value
            elif target_transition and not pred_transition:
                fn += value
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _transition_timing_accuracy(confusion: torch.Tensor, class_names: tuple[str, ...]) -> float:
    correct = total = 0.0
    for target_index, target_name in enumerate(class_names):
        if not target_name.startswith("transition_"):
            continue
        target_timing = target_name.split("::", maxsplit=1)[1]
        for pred_index, pred_name in enumerate(class_names):
            value = float(confusion[target_index, pred_index].item())
            if value == 0.0:
                continue
            total += value
            if pred_name.startswith("transition_") and pred_name.endswith(f"::{target_timing}"):
                correct += value
    return correct / total if total else 0.0


def _label_counts(dataset: EventModeProbeDataset, indices: list[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index in indices:
        counts[int(dataset.labels[index])] += 1
    return counts


def _named_counts(counts: Counter[int], class_names: tuple[str, ...]) -> dict[str, int]:
    return {class_names[index]: int(counts.get(index, 0)) for index in range(len(class_names))}


def _class_weights(counts: Counter[int], num_classes: int, device: torch.device) -> torch.Tensor:
    total = sum(counts.values())
    weights = []
    for index in range(num_classes):
        count = counts.get(index, 0)
        weights.append(total / (num_classes * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
