#!/usr/bin/env python3
"""Evaluate trained motion-prior-conditioned action heads."""

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
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import (  # noqa: E402
    _build_checkpoint_conditioner,
    _evaluate,
    _freeze_module,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated validation eval for Gate 3.0a action-head checkpoints."
    )
    parser.add_argument("--checkpoint", required=True, help="Action-head model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-eval-passes", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=None)
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])
    _seed_everything(base_seed)

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics.get("visual_feature_cache"),
    )
    cvae_checkpoint: dict[str, Any] | None = None
    cvae_metrics: dict[str, Any] | None = None
    cvae = None
    if metrics.get("checkpoint"):
        cvae_checkpoint = torch.load(
            Path(metrics["checkpoint"]).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        cvae_metrics = cvae_checkpoint["metrics"]
    conditioner = _build_checkpoint_conditioner(
        dataset,
        cvae_metrics,
        metrics["conditioning"]["condition_on"],
    )
    if cvae_checkpoint is not None and cvae_metrics is not None:
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
            conditioner_dim=conditioner.dim,
            device=device,
        )
        _freeze_module(cvae)

    model = _load_action_head(checkpoint, metrics, device)
    _, val_indices = _split_indices(
        dataset,
        float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"]),
        int(metrics["seed"]),
        metrics["split_by"],
    )
    batch_size = int(args.batch_size or metrics["batch_size"])
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    loss_fn = torch.nn.MSELoss()
    num_samples = int(args.num_samples or metrics["num_samples"])
    pass_metrics: list[dict[str, float | None]] = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_metrics.append(
            _evaluate(
                model,
                cvae,
                val_loader,
                loss_fn,
                device,
                conditioner,
                metrics["input_mode"],
                num_samples,
            )
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "input_mode": metrics["input_mode"],
        "source_cvae_checkpoint": metrics.get("checkpoint"),
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "device": str(device),
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


def _load_action_head(
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    device: torch.device,
) -> MotionPriorActionHead:
    config = metrics["model_config"]
    model = MotionPriorActionHead(
        context_dim=int(config["context_dim"]),
        motion_dim=int(config["motion_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        conditioning_dim=int(config["conditioning_dim"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        token_dim=int(config["token_dim"]),
        num_heads=int(config["num_heads"]),
        temporal_layers=int(config["temporal_layers"]),
        set_aggregator=str(config.get("set_aggregator", "context_attention")),
        set_query_count=int(config.get("set_query_count", 4)),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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
