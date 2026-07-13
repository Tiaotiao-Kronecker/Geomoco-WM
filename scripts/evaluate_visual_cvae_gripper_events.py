#!/usr/bin/env python3
"""Evaluate transition-event fidelity for visual cVAE gripper predictions."""

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

from geomoco_wm.data.event_labels import (  # noqa: E402
    label_gripper_events_for_windows,
    previous_gripper_commands_for_windows,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from evaluate_cvae_event_alignment import EventReadoutAccumulator, label_action_chunks  # noqa: E402
from evaluate_future_gripper_events import (  # noqa: E402
    _event_config_from_args,
    _extract_gripper_prediction,
    _gripper_to_action_chunk,
)
from evaluate_visual_cvae_samples import _build_checkpoint_conditioner, _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _batch_conditioning,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate transition labels from a visual cVAE prior mean."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--event-audit-json", default=None)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument("--close-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--mode", default="prior_mean", choices=["prior_mean", "posterior"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    dataset_config = metrics["dataset"]
    motion_mode = str(metrics.get("motion_mode", dataset_config.get("motion_mode")))
    if motion_mode not in {"future_gripper", "future_delta_gripper"}:
        raise ValueError(
            "visual cVAE gripper-event eval requires motion_mode=future_gripper "
            f"or future_delta_gripper, got {motion_mode}"
        )
    windows_jsonl = args.windows_jsonl or dataset_config["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or metrics.get("visual_feature_cache")
    split_by = args.split_by or metrics.get("split_by", "episode")
    seed = args.seed if args.seed is not None else int(metrics.get("seed", 7))
    batch_size = args.batch_size or int(metrics.get("batch_size", 64))
    _seed_everything(seed)
    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    conditioner = _build_checkpoint_conditioner(metrics)
    visual_token_config = _resolve_visual_token_config(
        dataset,
        metrics["visual_token_config"]["visual_token_count"],
        metrics["visual_token_config"]["visual_token_dim"],
    )
    model = _load_model(
        checkpoint,
        context_dim=dataset.spec().context_dim,
        motion_dim=dataset.spec().motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim,
        device=device,
    )
    event_config = _event_config_from_args(args)
    gt_labels = label_gripper_events_for_windows(
        dataset.windows,
        config=event_config,
        label_mode="transition",
    )
    previous_commands = previous_gripper_commands_for_windows(dataset.windows)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, seed, split_by)
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    accumulator = EventReadoutAccumulator()
    with torch.no_grad():
        for batch in val_loader:
            pred_motion = _predict_motion(model, batch, conditioner, device, args.mode)
            pred_gripper = _extract_gripper_prediction(pred_motion, motion_mode)
            pred_actions = _gripper_to_action_chunk(pred_gripper)
            window_ids = [str(window_id) for window_id in batch["window_id"]]
            pred_labels = label_action_chunks(
                pred_actions,
                [previous_commands.get(window_id) for window_id in window_ids],
                event_config,
            )
            accumulator.add_many(pred_labels, [gt_labels[window_id] for window_id in window_ids])
    output = {
        "schema_version": "geomoco_wm_visual_cvae_gripper_event_eval_v0",
        "checkpoint": str(checkpoint_path),
        "dataset": dataset.spec().to_dict(),
        "device": str(device),
        "seed": seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "motion_mode": motion_mode,
        "mode": args.mode,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser())
        if visual_feature_cache
        else None,
        "event_config": event_config.to_dict(),
        "metrics": accumulator.metrics(),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": output["metrics"]}, indent=2))


def _predict_motion(
    model: torch.nn.Module,
    batch: dict[str, object],
    conditioner: Any,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    context = batch["context"].to(device)
    motion = batch["motion"].to(device)
    visual = _batch_visual(batch, device)
    conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    if mode == "prior_mean":
        return model.prior_mean_prediction(context, visual, conditioning)
    if mode == "posterior":
        return model(context, visual, motion, conditioning).posterior_reconstruction
    raise ValueError("mode must be one of: prior_mean, posterior")


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
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
