#!/usr/bin/env python3
"""Evaluate transition-event fidelity for trained action decoders."""

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
    GripperEventConfig,
    label_gripper_events_for_windows,
    previous_gripper_commands_for_windows,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from evaluate_cvae_event_alignment import EventReadoutAccumulator, label_action_chunks  # noqa: E402
from train_future_motion_predictor import _make_loader, _resolve_device, _split_indices  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate transition-event fidelity for a trained action decoder."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--event-audit-json", default=None)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument("--close-sign", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", type=int, default=None)
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
    windows_jsonl = args.windows_jsonl or dataset_config["windows_jsonl"]
    split_by = args.split_by or metrics.get("split_by", "episode")
    seed = args.seed if args.seed is not None else int(metrics.get("seed", 7))
    batch_size = args.batch_size or int(metrics.get("batch_size", 64))
    _seed_everything(seed)
    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=metrics["motion_mode"],
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
    model = _load_decoder(checkpoint, device)
    model.eval()
    accumulator = EventReadoutAccumulator()
    with torch.no_grad():
        for batch in val_loader:
            context = batch["context"].to(device)
            motion = batch["motion"].to(device)
            pred_actions = model(context, motion)
            window_ids = [str(window_id) for window_id in batch["window_id"]]
            pred_labels = label_action_chunks(
                pred_actions,
                [previous_commands.get(window_id) for window_id in window_ids],
                event_config,
            )
            batch_gt_labels = [gt_labels[window_id] for window_id in window_ids]
            accumulator.add_many(pred_labels, batch_gt_labels)
    output = {
        "schema_version": "geomoco_wm_action_decoder_event_eval_v0",
        "checkpoint": str(checkpoint_path),
        "dataset": dataset.spec().to_dict(),
        "motion_mode": metrics["motion_mode"],
        "device": str(device),
        "seed": seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "event_config": event_config.to_dict(),
        "metrics": accumulator.metrics(),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": output["metrics"]}, indent=2))


def _load_decoder(checkpoint: dict[str, Any], device: torch.device) -> ActionDecoder:
    metrics = checkpoint["metrics"]
    dataset = metrics["dataset"]
    model = ActionDecoder(
        context_dim=int(dataset["context_dim"]),
        motion_rep_dim=int(dataset["motion_dim"]),
        action_dim=int(dataset["action_dim"]),
        horizon=int(dataset["horizon"]),
        hidden_dims=tuple(int(value) for value in metrics["hidden_dims"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


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
