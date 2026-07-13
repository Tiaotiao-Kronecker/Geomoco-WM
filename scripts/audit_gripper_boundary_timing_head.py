#!/usr/bin/env python3
"""Boundary-quality audit for step-wise gripper timing action heads."""

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
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_predicted_event_mixture_action_head import _load_action_head  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import (  # noqa: E402
    GRIPPER_BOUNDARY_STEP_CLASSES,
    _batch_string_at,
    _event_step_value,
    _gripper_boundary_step_targets,
    _load_event_label_records,
    _predicted_event_future_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit boundary-start precision/recall for Gate 3.2f-style "
            "step-wise gripper timing heads."
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
    parser.add_argument("--thresholds", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    thresholds = _parse_thresholds(args.thresholds)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    if metrics.get("gripper_step_target_mode") != "boundary_start":
        raise ValueError("boundary audit requires gripper_step_target_mode=boundary_start")
    if metrics.get("gripper_step_residual_mode") != "event_step":
        raise ValueError("boundary audit requires gripper_step_residual_mode=event_step")
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
    cvae_checkpoint_path = Path(metrics["checkpoint"]).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
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
    action_head = _load_action_head(checkpoint, metrics, device)
    event_audit_json = (
        args.event_mode_audit_json
        or metrics.get("event_mode_audit_json")
        or _checkpoint_event_audit_json(cvae_metrics)
    )
    if event_audit_json is None:
        raise ValueError("--event-mode-audit-json is required when absent from checkpoints")
    event_labels = _load_event_label_records(event_audit_json)

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
                sample_feature_mode=str(metrics.get("sample_feature_mode", "none")),
                thresholds=thresholds,
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
        "batch_size": batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "sample_feature_mode": metrics.get("sample_feature_mode", "none"),
        "gripper_step_classes": list(GRIPPER_BOUNDARY_STEP_CLASSES),
        "thresholds": thresholds,
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
                "mean_report": output["mean_report"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@torch.no_grad()
def _audit_pass(
    action_head: Any,
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
    sample_feature_mode: str,
    thresholds: tuple[float, ...],
    max_batches: int | None,
) -> dict[str, Any]:
    action_head.eval()
    cvae.eval()
    event_probe.eval()
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    window_records: list[dict[str, Any]] = []
    horizon: int | None = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = batch["context"].to(device)
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
            sample_feature_mode=sample_feature_mode,
        )
        output = action_head.forward_with_aux(context, future_inputs, conditioning, sample_features)
        logits = output.get("gripper_step_logits")
        if logits is None:
            raise ValueError("checkpoint did not produce gripper_step_logits")
        probs = torch.softmax(logits, dim=-1)
        targets = _gripper_boundary_step_targets(
            batch,
            event_labels,
            horizon=int(logits.shape[1]),
            device=device,
        )
        horizon = int(logits.shape[1])
        window_offset = sum(tensor.shape[0] for tensor in all_targets)
        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())
        window_records.extend(
            _window_records(
                batch,
                event_labels,
                probs.detach().cpu(),
                targets.detach().cpu(),
                window_offset=window_offset,
            )
        )

    if not all_targets or not all_probs:
        raise ValueError("no validation batches were evaluated")
    targets_cat = torch.cat(all_targets, dim=0)
    probs_cat = torch.cat(all_probs, dim=0)
    if horizon is None:
        raise ValueError("missing horizon")
    return boundary_quality_report(
        probs_cat,
        targets_cat,
        window_records,
        thresholds=thresholds,
        horizon=horizon,
    )


def boundary_quality_report(
    probs: torch.Tensor,
    targets: torch.Tensor,
    window_records: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    horizon: int,
) -> dict[str, Any]:
    if probs.ndim != 3:
        raise ValueError(f"probs must be [B,H,C], got {probs.shape}")
    if targets.shape != probs.shape[:2]:
        raise ValueError(f"targets shape must match probs [B,H], got {targets.shape}")
    if probs.shape[-1] != len(GRIPPER_BOUNDARY_STEP_CLASSES):
        raise ValueError("probs class dimension must match boundary classes")
    flat_targets = targets.reshape(-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    pred = flat_probs.argmax(dim=-1)
    positive_target = flat_targets != 0
    positive_pred = pred != 0
    positive_score = flat_probs[:, 1:].max(dim=-1).values
    report: dict[str, Any] = {
        "overall": {
            "num_windows": int(targets.shape[0]),
            "num_steps": int(flat_targets.numel()),
            "horizon": horizon,
            "accuracy": _accuracy(pred, flat_targets),
            "positive_fraction": _fraction(positive_target),
            "pred_positive_fraction_argmax": _fraction(positive_pred),
            "positive_ap": average_precision(positive_score, positive_target),
            "positive_score_mean": float(positive_score.mean()),
            "positive_score_on_positive_mean": _masked_mean(positive_score, positive_target),
            "positive_score_on_negative_mean": _masked_mean(positive_score, ~positive_target),
        },
        "argmax": {
            "any_boundary": precision_recall_f1(positive_pred, positive_target),
            "classes": {},
        },
        "thresholds": threshold_reports(positive_score, positive_target, thresholds),
        "classes": {},
        "localization": localization_report(probs, window_records),
    }
    for class_index, class_name in enumerate(GRIPPER_BOUNDARY_STEP_CLASSES):
        target_mask = flat_targets == class_index
        pred_mask = pred == class_index
        class_score = flat_probs[:, class_index]
        class_report = {
            "target_fraction": _fraction(target_mask),
            "pred_fraction_argmax": _fraction(pred_mask),
            "ap": average_precision(class_score, target_mask),
            "score_on_target_mean": _masked_mean(class_score, target_mask),
            "score_off_target_mean": _masked_mean(class_score, ~target_mask),
        }
        report["classes"][class_name] = class_report
        report["argmax"]["classes"][class_name] = precision_recall_f1(pred_mask, target_mask)
    return report


def precision_recall_f1(pred_positive: torch.Tensor, target_positive: torch.Tensor) -> dict[str, Any]:
    pred_bool = pred_positive.to(dtype=torch.bool)
    target_bool = target_positive.to(dtype=torch.bool)
    tp = int((pred_bool & target_bool).sum())
    fp = int((pred_bool & ~target_bool).sum())
    fn = int((~pred_bool & target_bool).sum())
    tn = int((~pred_bool & ~target_bool).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_reports(
    scores: torch.Tensor,
    targets: torch.Tensor,
    thresholds: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    return {
        f"{threshold:.3f}": precision_recall_f1(scores >= threshold, targets)
        for threshold in thresholds
    }


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float | None:
    target_bool = targets.to(dtype=torch.bool).reshape(-1)
    positives = int(target_bool.sum())
    if positives == 0:
        return None
    scores_flat = scores.reshape(-1)
    order = torch.argsort(scores_flat, descending=True)
    ranked_targets = target_bool[order].to(dtype=torch.float32)
    cumulative_tp = torch.cumsum(ranked_targets, dim=0)
    ranks = torch.arange(1, ranked_targets.numel() + 1, dtype=torch.float32)
    precision_at_k = cumulative_tp / ranks
    return float((precision_at_k * ranked_targets).sum() / positives)


def localization_report(
    probs: torch.Tensor,
    window_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for class_index, class_name in (
        (GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start"), "close_start"),
        (GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start"), "open_start"),
    ):
        rows = [row for row in window_records if row[f"{class_name}_step"] is not None]
        if not rows:
            reports[class_name] = {
                "count": 0,
                "top1_exact": None,
                "top1_within1": None,
                "mean_abs_step_error": None,
                "mean_true_step_prob": None,
                "mean_best_step_prob": None,
            }
            continue
        exact = 0
        within1 = 0
        abs_errors: list[float] = []
        true_step_probs: list[float] = []
        best_step_probs: list[float] = []
        for row in rows:
            window_index = int(row["window_index"])
            true_step = int(row[f"{class_name}_step"])
            class_probs = probs[window_index, :, class_index]
            pred_step = int(class_probs.argmax())
            error = abs(pred_step - true_step)
            exact += int(error == 0)
            within1 += int(error <= 1)
            abs_errors.append(float(error))
            true_step_probs.append(float(class_probs[true_step]))
            best_step_probs.append(float(class_probs[pred_step]))
        reports[class_name] = {
            "count": len(rows),
            "top1_exact": exact / len(rows),
            "top1_within1": within1 / len(rows),
            "mean_abs_step_error": sum(abs_errors) / len(abs_errors),
            "mean_true_step_prob": sum(true_step_probs) / len(true_step_probs),
            "mean_best_step_prob": sum(best_step_probs) / len(best_step_probs),
        }
    return reports


def _window_records(
    batch: dict[str, object],
    event_labels: dict[str, dict[str, Any]],
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    window_offset: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batch_size = int(targets.shape[0])
    close_name = "close_start"
    open_name = "open_start"
    for row in range(batch_size):
        window_id = _batch_string_at(batch["window_id"], row)
        record = event_labels.get(window_id, {})
        positive_scores = probs[row, :, 1:].max(dim=-1).values
        records.append(
            {
                "window_index": window_offset + len(records),
                "window_id": window_id,
                "event_mode": str(record.get("event_mode", "")),
                f"{close_name}_step": _event_step_value(record, "close_step"),
                f"{open_name}_step": _event_step_value(record, "open_step"),
                "target_positive_steps": int((targets[row] != 0).sum()),
                "max_positive_score": float(positive_scores.max()),
            }
        )
    return records


def _mean_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    return _mean_nested(reports)


def _mean_nested(values: list[Any]) -> Any:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    first = present_values[0]
    if isinstance(first, dict):
        keys = set().union(*(value.keys() for value in present_values if isinstance(value, dict)))
        return {
            key: _mean_nested([value.get(key) for value in present_values if isinstance(value, dict)])
            for key in sorted(keys)
        }
    if isinstance(first, (int, float)):
        return sum(float(value) for value in present_values) / len(present_values)
    return first


def _batch_conditioning_for_action_head(
    batch: dict[str, object],
    conditioner: Any,
    device: torch.device,
) -> torch.Tensor | None:
    from train_future_motion_predictor import _batch_conditioning

    return _batch_conditioning(batch, conditioner, device, include_visual=False)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("--thresholds values must be in [0, 1]")
    return thresholds


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


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred == target).to(dtype=torch.float32).mean())


def _fraction(mask: torch.Tensor) -> float:
    return float(mask.to(dtype=torch.float32).mean())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return float(values[mask].mean())


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
