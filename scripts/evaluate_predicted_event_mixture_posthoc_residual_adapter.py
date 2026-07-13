#!/usr/bin/env python3
"""Repeated evaluation for Gate 3.5b post-hoc residual adapters."""

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

from geomoco_wm.models.motion_prior_action_head import PostHocActionResidualAdapter  # noqa: E402
from train_future_motion_predictor import _make_loader, _resolve_device, _split_indices  # noqa: E402
from train_predicted_event_mixture_posthoc_residual_adapter import (  # noqa: E402
    _eval_kwargs,
    _feature_dim,
    _load_action_head,
    _load_event_labels,
    _load_frozen_stack,
    _evaluate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated validation eval for Gate 3.5b post-hoc residual adapters."
    )
    parser.add_argument("--checkpoint", required=True, help="Post-hoc adapter model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-eval-passes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    adapter_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = adapter_checkpoint["metrics"]
    frozen_path = Path(metrics["frozen_action_head_checkpoint"]).expanduser().resolve()
    frozen_checkpoint = torch.load(frozen_path, map_location=device, weights_only=False)
    frozen_metrics = frozen_checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])
    _seed_everything(base_seed)

    dataset, cvae, event_probe, conditioner, probe_conditioner, cvae_metrics, probe_metrics = (
        _load_frozen_stack(
            frozen_metrics,
            device,
            max_windows=int(metrics["dataset"]["num_windows"]),
        )
    )
    frozen_model = _load_action_head(frozen_checkpoint, frozen_metrics, device)
    adapter = _load_adapter(adapter_checkpoint, metrics, device)
    event_classes = tuple(str(value) for value in frozen_metrics["cvae_event_classes"])
    event_labels = _load_event_labels(frozen_metrics, cvae_metrics)
    _, val_indices = _split_indices(
        dataset,
        float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"]),
        int(frozen_metrics["seed"]),
        str(metrics["split_by"]),
    )
    batch_size = int(args.batch_size or metrics["batch_size"])
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    eval_kwargs = _eval_kwargs(
        frozen_metrics,
        probe_metrics,
        event_classes,
        event_labels,
        residual_gate_mode=str(metrics["adapter_config"].get("residual_gate_mode", "none")),
        residual_gate_threshold=metrics["adapter_config"].get("residual_gate_threshold"),
        residual_leak=float(metrics["adapter_config"].get("residual_leak", 0.0)),
    )
    pass_metrics: list[dict[str, float | None]] = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_metrics.append(
            _evaluate(
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
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "frozen_action_head_checkpoint": str(frozen_path),
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "batch_size": batch_size,
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
        "pass_metrics": pass_metrics,
        "mean_metrics": _mean_metrics(pass_metrics),
        "std_metrics": _std_metrics(pass_metrics),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "mean_metrics": output["mean_metrics"]}, indent=2))


def _load_adapter(
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
) -> PostHocActionResidualAdapter:
    config = metrics["adapter_config"]
    expected_feature_dim = _feature_dim(
        torch.load(
            Path(metrics["frozen_action_head_checkpoint"]).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )["metrics"]
    )
    if int(config["feature_dim"]) != expected_feature_dim:
        raise ValueError(
            "adapter feature_dim does not match frozen checkpoint: "
            f"{config['feature_dim']} vs {expected_feature_dim}"
        )
    adapter = PostHocActionResidualAdapter(
        feature_dim=int(config["feature_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        step_dim=int(config["step_dim"]),
        dropout=float(config["dropout"]),
        zero_init_output=False,
    ).to(device)
    adapter.load_state_dict(checkpoint["model_state_dict"])
    adapter.eval()
    return adapter


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    output: dict[str, float | None] = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = sum(float(value) for value in values) / len(values) if values else None
    return output


def _std_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    means = _mean_metrics(rows)
    output: dict[str, float | None] = {}
    for key, mean in means.items():
        if mean is None:
            output[key] = None
            continue
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if len(values) <= 1:
            output[key] = 0.0
            continue
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        output[key] = variance**0.5
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
