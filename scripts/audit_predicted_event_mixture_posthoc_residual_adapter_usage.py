#!/usr/bin/env python3
"""Usage audit for Gate 3.5b post-hoc residual adapters."""

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

from audit_predicted_event_mixture_action_head_usage import (  # noqa: E402
    _action_delta_metrics,
    _build_usage_variants,
    _predicted_event_future_input_bundle,
)
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _batch_conditioning,
    _make_loader,
    _resolve_device,
    _split_indices,
)
from train_predicted_event_mixture_posthoc_residual_adapter import (  # noqa: E402
    _load_action_head,
    _load_frozen_stack,
    _residual_gate_values,
)
from evaluate_predicted_event_mixture_posthoc_residual_adapter import _load_adapter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit eval-time sample usage for Gate 3.5b post-hoc adapters."
    )
    parser.add_argument("--checkpoint", required=True, help="Post-hoc adapter model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-eval-passes", type=int, default=3)
    parser.add_argument("--subset-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if args.subset_samples <= 0:
        raise ValueError("--subset-samples must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    adapter_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = adapter_checkpoint["metrics"]
    frozen_path = Path(metrics["frozen_action_head_checkpoint"]).expanduser().resolve()
    frozen_checkpoint = torch.load(frozen_path, map_location=device, weights_only=False)
    frozen_metrics = frozen_checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])

    dataset, cvae, event_probe, conditioner, probe_conditioner, _cvae_metrics, probe_metrics = (
        _load_frozen_stack(
            frozen_metrics,
            device,
            max_windows=int(metrics["dataset"]["num_windows"]),
        )
    )
    frozen_model = _load_action_head(frozen_checkpoint, frozen_metrics, device)
    frozen_model.eval()
    adapter = _load_adapter(adapter_checkpoint, metrics, device)
    adapter.eval()
    _, val_indices = _split_indices(
        dataset,
        float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"]),
        int(frozen_metrics["seed"]),
        str(metrics["split_by"]),
    )
    batch_size = int(args.batch_size or metrics["batch_size"])
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    event_classes = tuple(str(value) for value in frozen_metrics["cvae_event_classes"])
    pass_reports = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_reports.append(
            _audit_pass(
                frozen_model,
                adapter,
                cvae,
                event_probe,
                val_loader,
                device,
                conditioner,
                probe_conditioner,
                event_classes=event_classes,
                probe_class_names=tuple(
                    str(name) for name in probe_metrics["probe"]["class_names"]
                ),
                probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
                event_top_m=int(frozen_metrics["event_top_m"]),
                num_samples=int(frozen_metrics["num_samples"]),
                event_candidate_policy=str(frozen_metrics.get("event_candidate_policy", "topk")),
                transition_reserve_threshold=float(
                    frozen_metrics.get("transition_reserve_threshold", 0.0)
                ),
                sample_feature_mode=str(frozen_metrics["sample_feature_mode"]),
                residual_gate_mode=str(
                    metrics["adapter_config"].get("residual_gate_mode", "none")
                ),
                residual_gate_threshold=metrics["adapter_config"].get(
                    "residual_gate_threshold"
                ),
                residual_leak=float(metrics["adapter_config"].get("residual_leak", 0.0)),
                subset_samples=args.subset_samples,
                max_batches=args.max_batches,
            )
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "frozen_action_head_checkpoint": str(frozen_path),
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "subset_samples": args.subset_samples,
        "batch_size": batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "event_top_m": frozen_metrics["event_top_m"],
        "num_samples": frozen_metrics["num_samples"],
        "event_candidate_policy": frozen_metrics.get("event_candidate_policy", "topk"),
        "transition_reserve_threshold": frozen_metrics.get("transition_reserve_threshold", 0.0),
        "sample_feature_mode": frozen_metrics["sample_feature_mode"],
        "future_input_control": frozen_metrics.get("future_input_control") or "real",
        "residual_gate_mode": metrics["adapter_config"].get("residual_gate_mode", "none"),
        "residual_gate_threshold": metrics["adapter_config"].get("residual_gate_threshold"),
        "residual_leak": metrics["adapter_config"].get("residual_leak", 0.0),
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
                "key_metrics": _key_metrics(output["mean_report"]),
            },
            indent=2,
        )
    )


@torch.no_grad()
def _audit_pass(
    frozen_model: Any,
    adapter: Any,
    cvae: Any,
    event_probe: Any,
    loader: Any,
    device: torch.device,
    conditioner: Any,
    probe_conditioner: Any,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    residual_gate_mode: str,
    residual_gate_threshold: float | None,
    residual_leak: float,
    subset_samples: int,
    max_batches: int | None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
        gate_values = _residual_gate_values(
            event_probe,
            batch,
            context,
            device,
            probe_conditioner,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_classes=event_classes,
            event_labels=None,
            residual_gate_mode=residual_gate_mode,
            residual_gate_threshold=residual_gate_threshold,
            residual_leak=residual_leak,
        )
        bundle = _predicted_event_future_input_bundle(
            cvae,
            event_probe,
            batch,
            context,
            conditioning,
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
        variants = _build_usage_variants(
            bundle.future_inputs,
            bundle.sample_features,
            bundle.rank_slots,
            bundle.top_indices,
            event_classes,
            subset_samples=subset_samples,
        )
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name, variant in variants.items():
            frozen_output = frozen_model.forward_with_aux(
                context,
                variant.future_inputs,
                conditioning,
                variant.sample_features,
            )
            features = frozen_output["features"]
            temporal_actions = frozen_output["temporal_actions"]
            if features is None or temporal_actions is None:
                raise ValueError("frozen model must return features and temporal_actions")
            adapter_output = adapter(features, temporal_actions)
            adapter_actions = adapter_output["adapter_actions"]
            if gate_values is not None:
                adapter_actions = temporal_actions + gate_values.reshape(
                    -1,
                    1,
                    1,
                ) * adapter_output["adapter_residual"]
            outputs[name] = {
                "adapter_actions": adapter_actions,
                "frozen_temporal_actions": temporal_actions,
            }
        batch_metrics = _variant_metrics(outputs, actions)
        batch_metrics.update(_delta_metrics(outputs))
        batch_size = int(actions.shape[0])
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        total_count += batch_size
    if total_count == 0:
        return {}
    return {key: value / total_count for key, value in sorted(totals.items())}


def _variant_metrics(
    outputs: dict[str, dict[str, torch.Tensor]],
    actions: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for variant, output in outputs.items():
        for readout, readout_actions in output.items():
            prefix = readout.replace("_actions", "")
            for key, value in action_metrics(readout_actions, actions).items():
                metrics[f"variant/{variant}/{prefix}/{key}"] = value
    return metrics


def _delta_metrics(outputs: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    original = outputs["original"]
    for variant, output in outputs.items():
        if variant == "original":
            continue
        for readout, original_actions in original.items():
            prefix = readout.replace("_actions", "")
            variant_actions = output[readout]
            for key, value in _action_delta_metrics(original_actions, variant_actions).items():
                metrics[f"delta/original_vs_{variant}/{prefix}/{key}"] = value
    return metrics


def _mean_reports(reports: list[dict[str, float]]) -> dict[str, float | None]:
    if not reports:
        return {}
    keys = set().union(*(report.keys() for report in reports))
    return {
        key: _mean_optional([report.get(key) for report in reports])
        for key in sorted(keys)
    }


def _key_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    keys = (
        "variant/original/adapter/mse",
        "variant/mean_repeated/adapter/mse",
        "variant/permuted/adapter/mse",
        "variant/subset_k4/adapter/mse",
        "variant/batch_mismatch/adapter/mse",
        "delta/original_vs_mean_repeated/adapter/action_l2",
        "delta/original_vs_permuted/adapter/action_l2",
        "delta/original_vs_batch_mismatch/adapter/action_l2",
        "variant/original/frozen_temporal/mse",
        "variant/mean_repeated/frozen_temporal/mse",
        "variant/batch_mismatch/frozen_temporal/mse",
    )
    return {key: metrics.get(key) for key in keys}


def _mean_optional(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
