#!/usr/bin/env python3
"""Train small probes for Gate 2.4h visual phase/event prediction."""

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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.event_labels import (  # noqa: E402
    GripperEventConfig,
    label_gripper_events_for_windows,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.common import MLP  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _batch_conditioning,
    _build_conditioner,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _split_indices,
)


EVENT_CLASSES = (
    "close_transition",
    "mixed_transition",
    "open_transition",
    "sustain_close",
    "sustain_open",
)
EVENT_INDEX = {label: index for index, label in enumerate(EVENT_CLASSES)}


class EventProbeNet(nn.Module):
    """Tiny classifier used to probe phase/event information."""

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
class EventProbeSpec:
    input_variant: str
    input_dim: int
    num_classes: int
    class_names: tuple[str, ...]
    label_mode: str
    event_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_variant": self.input_variant,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "label_mode": self.label_mode,
            "event_config": self.event_config,
        }


class EventProbeDataset(Dataset):
    """Dataset wrapper that adds event labels to exported windows."""

    def __init__(
        self,
        base_dataset: OracleActionWindowDataset,
        *,
        event_config: GripperEventConfig,
        label_mode: str,
    ) -> None:
        self.base_dataset = base_dataset
        labels = label_gripper_events_for_windows(
            base_dataset.windows,
            config=event_config,
            label_mode=label_mode,
        )
        self.labels = []
        for window in base_dataset.windows:
            event_type = labels[window.window_id].event_type
            try:
                label_index = EVENT_INDEX[event_type]
            except KeyError as exc:
                raise ValueError(f"unsupported event type {event_type!r}") from exc
            self.labels.append(label_index)

    @property
    def windows(self) -> list[object]:
        return self.base_dataset.windows

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base_dataset[index])
        item["event_label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Gate 2.4h-b phase/event probes.")
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=["outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl"],
    )
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-variant",
        required=True,
        choices=[
            "task_proprio",
            "task_only",
            "visual_only",
            "future_motion_only",
            "proprio_future_motion",
            "visual_proprio",
            "visual_proprio_future_motion",
            "shuffled_visual_proprio_future_motion",
        ],
    )
    parser.add_argument("--event-audit-json", default=None)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument("--close-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--label-mode", default="transition", choices=["transition", "command"])
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
    event_config, label_mode = _event_config_from_args(args)
    needs_visual = "visual" in args.input_variant
    if needs_visual and not args.visual_feature_cache:
        raise ValueError(f"{args.input_variant} requires --visual-feature-cache")
    base_dataset = OracleActionWindowDataset(
        args.windows_jsonl,
        max_windows=args.max_windows,
        motion_mode="future_delta",
        visual_feature_cache_path=args.visual_feature_cache if needs_visual else None,
    )
    dataset = EventProbeDataset(
        base_dataset,
        event_config=event_config,
        label_mode=label_mode,
    )
    conditioner = _build_conditioner(base_dataset, args.condition_on)
    spec = EventProbeSpec(
        input_variant=args.input_variant,
        input_dim=_feature_dim(base_dataset, conditioner, args.input_variant),
        num_classes=len(EVENT_CLASSES),
        class_names=EVENT_CLASSES,
        label_mode=label_mode,
        event_config=event_config.to_dict(),
    )
    train_indices, val_indices = _split_indices(base_dataset, args.train_ratio, args.seed, args.split_by)
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
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "train_label_counts": _named_counts(train_counts),
                    "val_label_counts": _named_counts(val_counts),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    device = _resolve_device(args.device)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    model = EventProbeNet(
        input_dim=spec.input_dim,
        num_classes=spec.num_classes,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        layer_norm=not args.no_layer_norm,
    ).to(device)
    class_weights = _class_weights(train_counts, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int | None]] = []
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
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            conditioner,
            args.input_variant,
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
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": base_dataset.spec().to_dict(),
        "probe": spec.to_dict(),
        "conditioning": conditioner.to_dict(),
        "device": str(device),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_label_counts": _named_counts(train_counts),
        "val_label_counts": _named_counts(val_counts),
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


def _event_config_from_args(args: argparse.Namespace) -> tuple[GripperEventConfig, str]:
    if args.event_audit_json:
        report = json.loads(Path(args.event_audit_json).expanduser().read_text(encoding="utf-8"))
        config = report["config"]
        return (
            GripperEventConfig(
                command_threshold=float(config["command_threshold"]),
                close_sign=int(config["close_sign"]),
            ),
            str(report.get("label_mode", args.label_mode)),
        )
    return (
        GripperEventConfig(
            command_threshold=args.command_threshold,
            close_sign=args.close_sign,
        ),
        args.label_mode,
    )


def _feature_dim(
    dataset: OracleActionWindowDataset,
    conditioner: CategoricalConditioner,
    input_variant: str,
) -> int:
    dim = 0
    if input_variant in {
        "task_proprio",
        "proprio_future_motion",
        "visual_proprio",
        "visual_proprio_future_motion",
        "shuffled_visual_proprio_future_motion",
    }:
        dim += dataset.context_dim + conditioner.dim
    if input_variant == "task_only":
        dim += conditioner.dim
    if input_variant in {
        "future_motion_only",
        "proprio_future_motion",
        "visual_proprio_future_motion",
        "shuffled_visual_proprio_future_motion",
    }:
        dim += dataset.motion_dim
    if input_variant in {
        "visual_only",
        "visual_proprio",
        "visual_proprio_future_motion",
        "shuffled_visual_proprio_future_motion",
    }:
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
        "proprio_future_motion",
        "visual_proprio",
        "visual_proprio_future_motion",
        "shuffled_visual_proprio_future_motion",
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
        "shuffled_visual_proprio_future_motion",
    }:
        parts.append(batch["motion"].to(device))
    if input_variant in {
        "visual_only",
        "visual_proprio",
        "visual_proprio_future_motion",
        "shuffled_visual_proprio_future_motion",
    }:
        visual = batch.get("visual")
        if not isinstance(visual, torch.Tensor):
            raise ValueError(f"{input_variant} requires batch['visual']")
        visual = visual.to(device=device, dtype=torch.float32)
        if input_variant == "shuffled_visual_proprio_future_motion" and visual.shape[0] > 1:
            visual = visual[torch.randperm(visual.shape[0], device=device)]
        parts.append(visual)
    return torch.cat(parts, dim=-1)


def _run_epoch(
    model: EventProbeNet,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    input_variant: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    model.train()
    totals: dict[str, float] = {}
    confusion = torch.zeros((len(EVENT_CLASSES), len(EVENT_CLASSES)), dtype=torch.long)
    total_count = 0
    for batch in loader:
        features = _batch_features(batch, conditioner, device, input_variant)
        targets = batch["event_label"].to(device)
        logits = model(features)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        confusion += _confusion_matrix(logits.detach().cpu().argmax(dim=-1), targets.detach().cpu())
        total_count += batch_size
    metrics = _classification_metrics(confusion)
    metrics["loss"] = totals["loss"] / total_count
    return metrics


@torch.no_grad()
def _evaluate(
    model: EventProbeNet,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
    conditioner: CategoricalConditioner,
    input_variant: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    model.eval()
    totals: dict[str, float] = {}
    confusion = torch.zeros((len(EVENT_CLASSES), len(EVENT_CLASSES)), dtype=torch.long)
    total_count = 0
    for batch in loader:
        features = _batch_features(batch, conditioner, device, input_variant)
        targets = batch["event_label"].to(device)
        logits = model(features)
        loss = loss_fn(logits, targets)
        batch_size = int(targets.shape[0])
        totals["loss"] = totals.get("loss", 0.0) + float(loss.cpu()) * batch_size
        confusion += _confusion_matrix(logits.cpu().argmax(dim=-1), targets.cpu())
        total_count += batch_size
    metrics = _classification_metrics(confusion)
    metrics["loss"] = totals["loss"] / total_count
    return metrics


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((len(EVENT_CLASSES), len(EVENT_CLASSES)), dtype=torch.long)
    for target_value, pred_value in zip(target.tolist(), pred.tolist()):
        matrix[int(target_value), int(pred_value)] += 1
    return matrix


def _classification_metrics(confusion: torch.Tensor) -> dict[str, float | dict[str, float]]:
    total = int(confusion.sum().item())
    if total == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
    num_classes = int(confusion.shape[0])
    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index in range(num_classes):
        name = EVENT_CLASSES[index] if index < len(EVENT_CLASSES) else f"class_{index}"
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
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def _label_counts(dataset: EventProbeDataset, indices: list[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index in indices:
        counts[int(dataset.labels[index])] += 1
    return counts


def _named_counts(counts: Counter[int]) -> dict[str, int]:
    return {EVENT_CLASSES[index]: int(counts.get(index, 0)) for index in range(len(EVENT_CLASSES))}


def _class_weights(counts: Counter[int], device: torch.device) -> torch.Tensor:
    total = sum(counts.values())
    weights = []
    for index in range(len(EVENT_CLASSES)):
        count = counts.get(index, 0)
        weights.append(total / (len(EVENT_CLASSES) * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
