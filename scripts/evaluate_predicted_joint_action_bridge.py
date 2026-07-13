#!/usr/bin/env python3
"""Evaluate predicted EEF plus predicted gripper through an action decoder."""

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
    _prediction_metrics,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
    _uses_visual_attention,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate action decoding from predicted EEF plus predicted gripper."
    )
    parser.add_argument("--eef-predictor-checkpoint", required=True)
    parser.add_argument("--gripper-predictor-checkpoint", required=True)
    parser.add_argument("--action-decoder-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--eef-visual-feature-cache", default=None)
    parser.add_argument("--gripper-visual-feature-cache", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--eef-source", default="predicted", choices=["predicted", "gt"])
    parser.add_argument("--gripper-source", default="predicted", choices=["predicted", "gt"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    seed = int(args.seed)
    _seed_everything(seed)

    eef_checkpoint_path = Path(args.eef_predictor_checkpoint).expanduser().resolve()
    gripper_checkpoint_path = Path(args.gripper_predictor_checkpoint).expanduser().resolve()
    eef_checkpoint = torch.load(eef_checkpoint_path, map_location=device, weights_only=False)
    gripper_checkpoint = torch.load(gripper_checkpoint_path, map_location=device, weights_only=False)
    eef_metrics = eef_checkpoint["metrics"]
    gripper_metrics = gripper_checkpoint["metrics"]
    _validate_checkpoint_motion_mode(eef_metrics, "future_delta", "EEF predictor")
    _validate_checkpoint_motion_mode(gripper_metrics, "future_gripper", "gripper predictor")

    windows_jsonl = args.windows_jsonl or _resolve_windows_jsonl(eef_metrics, gripper_metrics)
    eef_visual_cache = args.eef_visual_feature_cache or _checkpoint_visual_cache(eef_metrics)
    gripper_visual_cache = args.gripper_visual_feature_cache or _checkpoint_visual_cache(gripper_metrics)
    split_by = args.split_by or str(eef_metrics.get("split_by") or gripper_metrics.get("split_by") or "episode")
    batch_size = args.batch_size or int(
        eef_metrics.get("batch_size") or gripper_metrics.get("batch_size") or 64
    )

    eef_dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode="future_delta",
        visual_feature_cache_path=eef_visual_cache,
    )
    gripper_dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode="future_gripper",
        visual_feature_cache_path=gripper_visual_cache,
    )
    _validate_dataset_alignment(eef_dataset, gripper_dataset)

    eef_conditioner = _build_checkpoint_conditioner(eef_metrics)
    gripper_conditioner = _build_checkpoint_conditioner(gripper_metrics)
    eef_model = _load_predictor(eef_checkpoint, eef_dataset, eef_conditioner, device)
    gripper_model = _load_predictor(gripper_checkpoint, gripper_dataset, gripper_conditioner, device)
    eef_model.eval()
    gripper_model.eval()

    action_decoder, action_decoder_config = _load_action_decoder(
        args.action_decoder_checkpoint,
        device,
    )
    if action_decoder_config["motion_mode"] != "future_delta_gripper":
        raise ValueError("action decoder must use motion_mode=future_delta_gripper")
    action_decoder.eval()

    train_indices, val_indices = _split_indices(eef_dataset, args.train_ratio, seed, split_by)
    eef_loader = _make_loader(eef_dataset, val_indices, batch_size, shuffle=False)
    gripper_loader = _make_loader(gripper_dataset, val_indices, batch_size, shuffle=False)

    totals: dict[str, float] = {}
    total_count = 0
    with torch.no_grad():
        for eef_batch, gripper_batch in zip(eef_loader, gripper_loader, strict=True):
            _validate_batch_alignment(eef_batch, gripper_batch)
            context = eef_batch["context"].to(device)
            actions = eef_batch["actions"].to(device)
            gt_eef = eef_batch["motion"].to(device)
            gt_gripper = gripper_batch["motion"].to(device)

            pred_eef = (
                gt_eef
                if args.eef_source == "gt"
                else _predict_with_checkpoint_model(
                    eef_model,
                    eef_batch,
                    eef_conditioner,
                    device,
                    eef_metrics,
                )
            )
            pred_gripper = (
                gt_gripper
                if args.gripper_source == "gt"
                else _predict_with_checkpoint_model(
                    gripper_model,
                    gripper_batch,
                    gripper_conditioner,
                    device,
                    gripper_metrics,
                )
            )
            joint_motion = torch.cat([pred_eef, pred_gripper], dim=-1)
            pred_actions = action_decoder(context, joint_motion)
            batch_metrics = _prefixed_prediction_metrics(pred_eef, gt_eef, "future_delta", "eef")
            batch_metrics.update(
                _prefixed_prediction_metrics(pred_gripper, gt_gripper, "future_gripper", "gripper")
            )
            batch_metrics.update(_prefix("action", action_metrics(pred_actions, actions)))
            batch_size_actual = int(context.shape[0])
            for key, value in batch_metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size_actual
            total_count += batch_size_actual

    output = {
        "schema_version": "geomoco_wm_predicted_joint_action_bridge_v0",
        "eef_predictor_checkpoint": str(eef_checkpoint_path),
        "gripper_predictor_checkpoint": str(gripper_checkpoint_path),
        "action_decoder_checkpoint": str(Path(args.action_decoder_checkpoint).expanduser()),
        "action_decoder_config": action_decoder_config,
        "eef_dataset": eef_dataset.spec().to_dict(),
        "gripper_dataset": gripper_dataset.spec().to_dict(),
        "device": str(device),
        "seed": seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "eef_source": args.eef_source,
        "gripper_source": args.gripper_source,
        "eef_conditioning": eef_conditioner.to_dict(),
        "gripper_conditioning": gripper_conditioner.to_dict(),
        "eef_visual_feature_cache": str(Path(eef_visual_cache).expanduser()) if eef_visual_cache else None,
        "gripper_visual_feature_cache": (
            str(Path(gripper_visual_cache).expanduser()) if gripper_visual_cache else None
        ),
        "eef_visual_fusion": _visual_fusion(eef_metrics),
        "gripper_visual_fusion": _visual_fusion(gripper_metrics),
        "metrics": _average_metrics(totals, total_count),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": output["metrics"]}, indent=2))


def _predict_with_checkpoint_model(
    model: torch.nn.Module,
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    device: torch.device,
    metrics: dict[str, Any],
) -> torch.Tensor:
    context = batch["context"].to(device)
    conditioning = _batch_conditioning(
        batch,
        conditioner,
        device,
        include_visual=_visual_fusion(metrics) == "mlp_conditioning",
    )
    return _predict_motion(
        model,
        context,
        batch,
        conditioning,
        device,
        _visual_fusion(metrics),
    )


def _load_predictor(
    checkpoint: dict[str, Any],
    dataset: OracleActionWindowDataset,
    conditioner: CategoricalConditioner,
    device: torch.device,
) -> torch.nn.Module:
    metrics = checkpoint["metrics"]
    visual_token_config = (
        _resolve_visual_token_config(
            dataset,
            (metrics.get("visual_token_config") or {}).get("visual_token_count"),
            (metrics.get("visual_token_config") or {}).get("visual_token_dim"),
        )
        if _uses_visual_attention(_visual_fusion(metrics))
        else None
    )
    model = _build_model(
        context_dim=int(metrics["dataset"]["context_dim"]),
        motion_dim=int(metrics["dataset"]["motion_dim"]),
        hidden_dims=_parse_hidden_dims(",".join(str(value) for value in metrics["hidden_dims"])),
        conditioner_dim=conditioner.dim,
        visual_dim=dataset.visual_dim,
        visual_fusion=_visual_fusion(metrics),
        visual_token_config=visual_token_config,
        visual_query_dim=int(metrics.get("visual_query_dim") or 384),
        visual_num_heads=int(metrics.get("visual_num_heads") or 4),
        future_step_dim=int(metrics.get("future_step_dim") or 6),
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


def _visual_fusion(metrics: dict[str, Any]) -> str:
    return str(metrics.get("visual_fusion") or "mlp_conditioning")


def _resolve_windows_jsonl(
    eef_metrics: dict[str, Any],
    gripper_metrics: dict[str, Any],
) -> list[str]:
    eef_windows = [str(path) for path in eef_metrics["dataset"]["windows_jsonl"]]
    gripper_windows = [str(path) for path in gripper_metrics["dataset"]["windows_jsonl"]]
    if eef_windows != gripper_windows:
        raise ValueError(
            "EEF and gripper checkpoints used different windows_jsonl; "
            "pass --windows-jsonl only if the datasets are intentionally aligned"
        )
    return eef_windows


def _checkpoint_visual_cache(metrics: dict[str, Any]) -> str | None:
    cache = metrics.get("visual_feature_cache") or metrics["dataset"].get("visual_feature_cache")
    return str(cache) if cache else None


def _validate_checkpoint_motion_mode(
    metrics: dict[str, Any],
    expected: str,
    label: str,
) -> None:
    mode = metrics.get("motion_mode") or metrics["dataset"].get("motion_mode")
    if mode != expected:
        raise ValueError(f"{label} must use motion_mode={expected}, got {mode}")


def _validate_dataset_alignment(
    eef_dataset: OracleActionWindowDataset,
    gripper_dataset: OracleActionWindowDataset,
) -> None:
    if len(eef_dataset) != len(gripper_dataset):
        raise ValueError("EEF and gripper datasets must have the same number of windows")
    for index, (eef_window, gripper_window) in enumerate(zip(eef_dataset.windows, gripper_dataset.windows)):
        if eef_window.window_id != gripper_window.window_id:
            raise ValueError(
                "EEF and gripper datasets are not aligned at "
                f"index {index}: {eef_window.window_id} vs {gripper_window.window_id}"
            )


def _validate_batch_alignment(
    eef_batch: dict[str, object],
    gripper_batch: dict[str, object],
) -> None:
    eef_ids = [str(window_id) for window_id in eef_batch["window_id"]]
    gripper_ids = [str(window_id) for window_id in gripper_batch["window_id"]]
    if eef_ids != gripper_ids:
        raise ValueError("EEF and gripper batches are not aligned")


def _prefixed_prediction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    motion_mode: str,
    prefix: str,
) -> dict[str, float]:
    return _prefix(prefix, _prediction_metrics(pred, target, motion_mode))


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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
