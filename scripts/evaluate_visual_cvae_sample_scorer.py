#!/usr/bin/env python3
"""Re-evaluate a trained visual cVAE sample scorer with current readout metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.models.sample_readout import SampleScoreNet, TemporalSampleScoreNet  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_sample_scorer import (  # noqa: E402
    _build_checkpoint_conditioner,
    _evaluate,
    _freeze_module,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a trained Gate 2.4 sample scorer without retraining."
    )
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cvae-checkpoint", default=None)
    parser.add_argument("--action-decoder-checkpoint", default=None)
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    scorer_checkpoint_path = Path(args.scorer_checkpoint).expanduser().resolve()
    scorer_checkpoint = torch.load(scorer_checkpoint_path, map_location=device, weights_only=False)
    scorer_metrics = scorer_checkpoint["metrics"]

    cvae_checkpoint_path = Path(
        args.cvae_checkpoint or scorer_metrics["cvae_checkpoint"]
    ).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]

    windows_jsonl = args.windows_jsonl or scorer_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or scorer_metrics["visual_feature_cache"]
    split_by = args.split_by or scorer_metrics.get("split_by", "episode")
    condition_on = args.condition_on or scorer_metrics["conditioning"]["condition_on"]
    num_samples = args.num_samples or int(scorer_metrics["num_samples"])
    batch_size = args.batch_size or int(scorer_metrics["batch_size"])
    motion_mode = str(
        scorer_metrics.get(
            "motion_mode",
            cvae_metrics.get("motion_mode", scorer_metrics["dataset"].get("motion_mode")),
        )
    )

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_checkpoint_conditioner(dataset, scorer_metrics, condition_on)
    visual_token_config = _resolve_visual_token_config(
        dataset,
        cvae_metrics["visual_token_config"]["visual_token_count"],
        cvae_metrics["visual_token_config"]["visual_token_dim"],
    )
    cvae = _load_model(
        cvae_checkpoint,
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim,
        device=device,
    )
    _freeze_module(cvae)

    action_decoder_checkpoint = (
        args.action_decoder_checkpoint or scorer_metrics["action_decoder_checkpoint"]
    )
    action_decoder, action_decoder_config = _load_action_decoder(action_decoder_checkpoint, device)
    if action_decoder_config["motion_mode"] != motion_mode:
        raise ValueError(
            "action decoder motion mode must match scorer/cVAE motion mode: "
            f"{action_decoder_config['motion_mode']} vs {motion_mode}"
        )
    if int(action_decoder_config["motion_dim"]) != int(spec.motion_dim):
        raise ValueError(
            "action decoder motion dim must match scorer dataset motion dim: "
            f"{action_decoder_config['motion_dim']} vs {spec.motion_dim}"
        )
    _freeze_action_decoder(action_decoder)

    scorer = _build_scorer_from_metrics(
        scorer_metrics,
        condition_dim=cvae.condition_dim,
        motion_dim=spec.motion_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
    ).to(device)
    scorer.load_state_dict(scorer_checkpoint["model_state_dict"])
    scorer.eval()

    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    eval_args = _make_eval_args(scorer_metrics, num_samples)
    metrics = _evaluate(
        scorer,
        cvae,
        action_decoder,
        val_loader,
        device,
        conditioner,
        eval_args,
    )

    output = {
        "scorer_checkpoint": str(scorer_checkpoint_path),
        "cvae_checkpoint": str(cvae_checkpoint_path),
        "action_decoder_checkpoint": str(Path(action_decoder_checkpoint).expanduser()),
        "action_decoder_config": action_decoder_config,
        "dataset": spec.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "motion_mode": motion_mode,
        "conditioning": conditioner.to_dict(),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser()),
        "visual_token_config": visual_token_config,
        "num_samples": num_samples,
        "scorer_arch": scorer_metrics.get("scorer_arch", "mlp"),
        "scorer_target_kind": scorer_metrics["target_kind"],
        "scorer_hidden_dims": list(scorer_metrics["scorer_hidden_dims"]),
        "metrics": metrics,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"output_json": str(output_path), "metrics": metrics}, indent=2))


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")


def _make_eval_args(scorer_metrics: dict[str, Any], num_samples: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_samples=num_samples,
        motion_mode=str(scorer_metrics.get("motion_mode", scorer_metrics["dataset"].get("motion_mode"))),
        target_kind=scorer_metrics["target_kind"],
        action_target_weight=float(scorer_metrics.get("action_target_weight", 1.0)),
        motion_target_weight=float(scorer_metrics.get("motion_target_weight", 1.0)),
        translation_target_weight=float(scorer_metrics.get("translation_target_weight", 1.0)),
        rotation_target_weight=float(scorer_metrics.get("rotation_target_weight", 1.0)),
        gripper_target_weight=float(scorer_metrics.get("gripper_target_weight", 1.0)),
        target_temperature=float(scorer_metrics.get("target_temperature", 1.0)),
        selection_temperature=float(scorer_metrics.get("selection_temperature", 1.0)),
        hard_negative_target_kind=str(scorer_metrics.get("hard_negative_target_kind", "none")),
        hard_negative_weight=float(scorer_metrics.get("hard_negative_weight", 0.0)),
        hard_negative_margin=float(scorer_metrics.get("hard_negative_margin", 0.0)),
        event_target_weight=0.0,
        event_hard_negative_weight=0.0,
        event_hard_negative_margin=0.0,
    )


def _build_scorer_from_metrics(
    scorer_metrics: dict[str, Any],
    *,
    condition_dim: int,
    motion_dim: int,
    action_dim: int,
    horizon: int,
) -> torch.nn.Module:
    hidden_dims = tuple(int(value) for value in scorer_metrics["scorer_hidden_dims"])
    dropout = float(scorer_metrics.get("dropout", 0.0))
    scorer_arch = str(scorer_metrics.get("scorer_arch", "mlp"))
    if scorer_arch == "mlp":
        return SampleScoreNet(
            condition_dim=condition_dim,
            motion_dim=motion_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if scorer_arch == "temporal":
        temporal_config = scorer_metrics.get("temporal_config", {})
        if not isinstance(temporal_config, dict):
            temporal_config = {}
        return TemporalSampleScoreNet(
            condition_dim=condition_dim,
            motion_dim=motion_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden_dims,
            temporal_dim=int(temporal_config.get("temporal_dim", 128)),
            num_layers=int(temporal_config.get("temporal_layers", 2)),
            num_heads=int(temporal_config.get("temporal_heads", 4)),
            dropout=dropout,
        )
    raise ValueError("scorer_arch must be one of: mlp, temporal")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
