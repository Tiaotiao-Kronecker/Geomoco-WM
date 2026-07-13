#!/usr/bin/env python3
"""Evaluate GT future EEF plus predicted future gripper through an action decoder."""

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
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _build_model,
    _load_action_decoder,
    _make_loader,
    _parse_hidden_dims,
    _predict_motion,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
    _uses_visual_attention,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate action decoding from GT EEF plus predicted gripper."
    )
    parser.add_argument("--gripper-predictor-checkpoint", required=True)
    parser.add_argument("--action-decoder-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--visual-feature-cache", default=None)
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
    predictor_checkpoint_path = Path(args.gripper_predictor_checkpoint).expanduser().resolve()
    predictor_checkpoint = torch.load(predictor_checkpoint_path, map_location=device, weights_only=False)
    predictor_metrics = predictor_checkpoint["metrics"]
    if predictor_metrics.get("motion_mode") != "future_gripper":
        raise ValueError("gripper predictor checkpoint must use motion_mode=future_gripper")
    windows_jsonl = args.windows_jsonl or predictor_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or predictor_metrics.get("visual_feature_cache")
    split_by = args.split_by or predictor_metrics.get("split_by", "episode")
    seed = args.seed if args.seed is not None else int(predictor_metrics.get("seed", 7))
    batch_size = args.batch_size or int(predictor_metrics.get("batch_size", 64))
    _seed_everything(seed)
    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode="future_delta",
        visual_feature_cache_path=visual_feature_cache,
    )
    conditioner = _build_checkpoint_conditioner(predictor_metrics)
    predictor = _load_gripper_predictor(predictor_checkpoint, dataset, conditioner, device)
    predictor.eval()
    action_decoder, action_decoder_config = _load_action_decoder(
        args.action_decoder_checkpoint,
        device,
    )
    if action_decoder_config["motion_mode"] != "future_delta_gripper":
        raise ValueError("action decoder must use motion_mode=future_delta_gripper")
    action_decoder.eval()
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, seed, split_by)
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    totals: dict[str, float] = {}
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            context = batch["context"].to(device)
            gt_eef = batch["motion"].to(device)
            actions = batch["actions"].to(device)
            conditioning = _batch_conditioning(
                batch,
                conditioner,
                device,
                include_visual=predictor_metrics["visual_fusion"] == "mlp_conditioning",
            )
            pred_gripper = _predict_motion(
                predictor,
                context,
                batch,
                conditioning,
                device,
                predictor_metrics["visual_fusion"],
            )
            joint_motion = torch.cat([gt_eef, pred_gripper], dim=-1)
            pred_actions = action_decoder(context, joint_motion)
            batch_metrics = action_metrics(pred_actions, actions)
            batch_size_actual = int(context.shape[0])
            for key, value in batch_metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size_actual
            total_count += batch_size_actual
    output = {
        "schema_version": "geomoco_wm_predicted_gripper_action_bridge_v0",
        "gripper_predictor_checkpoint": str(predictor_checkpoint_path),
        "action_decoder_checkpoint": str(Path(args.action_decoder_checkpoint).expanduser()),
        "action_decoder_config": action_decoder_config,
        "dataset": dataset.spec().to_dict(),
        "device": str(device),
        "seed": seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser())
        if visual_feature_cache
        else None,
        "visual_fusion": predictor_metrics["visual_fusion"],
        "metrics": _average_metrics(totals, total_count),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": output["metrics"]}, indent=2))


def _load_gripper_predictor(
    checkpoint: dict[str, Any],
    dataset: OracleActionWindowDataset,
    conditioner: Any,
    device: torch.device,
) -> torch.nn.Module:
    metrics = checkpoint["metrics"]
    visual_token_config = (
        _resolve_visual_token_config(
            dataset,
            metrics["visual_token_config"]["visual_token_count"],
            metrics["visual_token_config"]["visual_token_dim"],
        )
        if _uses_visual_attention(metrics["visual_fusion"])
        else None
    )
    model = _build_model(
        context_dim=int(metrics["dataset"]["context_dim"]),
        motion_dim=int(metrics["dataset"]["motion_dim"]),
        hidden_dims=_parse_hidden_dims(",".join(str(value) for value in metrics["hidden_dims"])),
        conditioner_dim=conditioner.dim,
        visual_dim=dataset.visual_dim,
        visual_fusion=metrics["visual_fusion"],
        visual_token_config=visual_token_config,
        visual_query_dim=int(metrics["visual_query_dim"]),
        visual_num_heads=int(metrics["visual_num_heads"]),
        future_step_dim=int(metrics["future_step_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _build_checkpoint_conditioner(metrics: dict[str, Any]) -> CategoricalConditioner:
    conditioning = metrics["conditioning"]
    condition_on = str(conditioning["condition_on"])
    vocab = tuple(str(value) for value in conditioning.get("vocab", []))
    return CategoricalConditioner(
        condition_on=condition_on,
        vocab=vocab,
        index_by_label={label: index for index, label in enumerate(vocab)},
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
