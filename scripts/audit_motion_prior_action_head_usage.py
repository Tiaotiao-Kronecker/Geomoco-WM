#!/usr/bin/env python3
"""Audit whether a motion-prior action head uses sample-set diversity."""

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
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_motion_prior_action_head import _load_action_head, _mean_metrics, _std_metrics  # noqa: E402
from evaluate_visual_cvae_samples import _load_model, _sample_prior_motions  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _build_checkpoint_conditioner, _freeze_module  # noqa: E402
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate 3.0c sample-set diversity and action-head usage audit."
    )
    parser.add_argument("--checkpoint", required=True, help="Motion-prior action-head model.pt")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--subset-samples", type=int, default=4)
    parser.add_argument("--num-eval-passes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--event-command-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    if metrics.get("input_mode") != "sample_set":
        raise ValueError("Gate 3.0c audit expects a sample_set action-head checkpoint")
    if not metrics.get("checkpoint"):
        raise ValueError("sample_set action-head checkpoint must reference a frozen cVAE")
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])

    cvae_checkpoint_path = Path(metrics["checkpoint"]).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]
    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
    conditioner = _build_checkpoint_conditioner(
        dataset,
        cvae_metrics,
        metrics["conditioning"]["condition_on"],
    )
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
    action_head = _load_action_head(checkpoint, metrics, device)

    _, val_indices = _split_indices(
        dataset,
        float(metrics["train_size"]) / float(metrics["train_size"] + metrics["val_size"]),
        int(metrics["seed"]),
        metrics["split_by"],
    )
    batch_size = int(args.batch_size or metrics["batch_size"])
    val_loader = _make_loader(dataset, val_indices, batch_size, shuffle=False)
    num_samples = int(args.num_samples or metrics["num_samples"])
    pass_metrics: list[dict[str, float | None]] = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(base_seed + eval_pass)
        pass_metrics.append(
            _audit_pass(
                action_head,
                cvae,
                val_loader,
                device,
                conditioner,
                num_samples=num_samples,
                subset_samples=args.subset_samples,
                max_batches=args.max_batches,
                event_command_threshold=args.event_command_threshold,
            )
        )
    output = {
        "checkpoint": str(checkpoint_path),
        "source_cvae_checkpoint": str(cvae_checkpoint_path),
        "seed": base_seed,
        "num_eval_passes": args.num_eval_passes,
        "num_samples": num_samples,
        "subset_samples": args.subset_samples,
        "batch_size": batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "input_mode": metrics["input_mode"],
        "model_config": metrics["model_config"],
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
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "key_metrics": _key_metrics(output["mean_metrics"]),
            },
            indent=2,
        )
    )


@torch.no_grad()
def _audit_pass(
    action_head: MotionPriorActionHead,
    cvae: VisualConditionedGeoMoCoCVAE,
    loader: Any,
    device: torch.device,
    conditioner: CategoricalConditioner,
    *,
    num_samples: int,
    subset_samples: int,
    max_batches: int | None,
    event_command_threshold: float,
) -> dict[str, float | None]:
    if loader is None:
        return {"variant/original/mse": None}
    action_head.eval()
    cvae.eval()
    totals: dict[str, float] = {}
    total_count = 0
    group_totals: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
        samples = _sample_batch(cvae, batch, context, conditioning, device, num_samples)
        variants = _build_variants(samples, subset_samples)
        predictions = {
            name: action_head(context, variant, conditioning)
            for name, variant in variants.items()
        }
        original = predictions["original"]
        batch_size = int(context.shape[0])
        for name, pred in predictions.items():
            _add_prefixed_metrics(totals, f"variant/{name}", action_metrics(pred, actions), batch_size)
            _accumulate_group_mse(
                group_totals,
                group_counts,
                name,
                pred,
                actions,
                batch,
                event_command_threshold,
            )
            if name != "original":
                _add_prefixed_metrics(
                    totals,
                    f"delta/original_vs_{name}",
                    _action_delta_metrics(original, pred),
                    batch_size,
                )
        sample_metrics = _motion_sample_metrics(samples, action_head.horizon)
        _add_prefixed_metrics(totals, "sample", sample_metrics, batch_size)
        single_metrics = _single_sample_action_usage_metrics(
            action_head,
            context,
            conditioning,
            samples,
            actions,
        )
        _add_prefixed_metrics(totals, "single_sample_head", single_metrics, batch_size)
        total_count += batch_size

    output = _average_metrics(totals, total_count)
    for key, value in sorted(group_totals.items()):
        count = group_counts[key]
        output[f"group/{key}/mse"] = value / count if count else None
    return output


def _sample_batch(
    cvae: VisualConditionedGeoMoCoCVAE,
    batch: dict[str, object],
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    device: torch.device,
    num_samples: int,
) -> torch.Tensor:
    visual = _batch_visual(batch, device)
    condition = cvae.condition(context, visual, conditioning)
    prior_mean, prior_logvar = cvae.encode_prior(condition)
    samples = _sample_prior_motions(cvae, condition, prior_mean, prior_logvar, num_samples)
    return samples.permute(1, 0, 2).contiguous()


def _build_variants(samples: torch.Tensor, subset_samples: int) -> dict[str, torch.Tensor]:
    batch_size, num_samples, _ = samples.shape
    subset_count = min(subset_samples, num_samples)
    permutation = torch.randperm(num_samples, device=samples.device)
    mean = samples.mean(dim=1, keepdim=True)
    variants = {
        "original": samples,
        "permuted": samples[:, permutation, :],
        "mean_repeated": mean.expand(-1, num_samples, -1),
        "mean_single": mean.squeeze(1),
        "first_single": samples[:, 0, :],
        "subset": samples[:, permutation[:subset_count], :],
    }
    if batch_size > 1:
        variants["batch_mismatch"] = samples.roll(shifts=1, dims=0)
    else:
        variants["batch_mismatch"] = samples
    return variants


def _motion_sample_metrics(samples: torch.Tensor, horizon: int) -> dict[str, float]:
    mean = samples.mean(dim=1, keepdim=True)
    centered = samples - mean
    metrics = {
        "motion_variance": _to_float(samples.var(dim=1, unbiased=False).mean()),
        "sample_to_mean_l2": _to_float(torch.linalg.vector_norm(centered, dim=-1).mean()),
        "pair_l2": _to_float(_pairwise_l2(samples).mean()),
    }
    if samples.shape[-1] == horizon * 7:
        eef_dim = horizon * 6
        metrics["eef_variance"] = _to_float(
            samples[..., :eef_dim].var(dim=1, unbiased=False).mean()
        )
        metrics["gripper_variance"] = _to_float(
            samples[..., eef_dim:].var(dim=1, unbiased=False).mean()
        )
        metrics["gripper_pair_l2"] = _to_float(_pairwise_l2(samples[..., eef_dim:]).mean())
    return metrics


@torch.no_grad()
def _single_sample_action_usage_metrics(
    action_head: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    samples: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, float]:
    single_predictions = []
    for sample_index in range(samples.shape[1]):
        single_predictions.append(
            action_head(context, samples[:, sample_index, :], conditioning)
        )
    stacked = torch.stack(single_predictions, dim=1)
    mean_pred = stacked.mean(dim=1, keepdim=True)
    per_sample_error = (stacked - actions.unsqueeze(1)).square().mean(dim=(2, 3))
    return {
        "action_variance": _to_float(stacked.var(dim=1, unbiased=False).mean()),
        "action_to_mean_l2": _to_float(
            torch.linalg.vector_norm(
                (stacked - mean_pred).reshape(stacked.shape[0], stacked.shape[1], -1),
                dim=-1,
            ).mean()
        ),
        "mean_single_sample_action_mse": _to_float(per_sample_error.mean()),
        "best_single_sample_action_mse": _to_float(per_sample_error.min(dim=1).values.mean()),
    }


def _action_delta_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    error = left - right
    return {
        "action_mse": _to_float(error.square().mean()),
        "action_mae": _to_float(error.abs().mean()),
        "action_l2": _to_float(torch.linalg.vector_norm(error.reshape(error.shape[0], -1), dim=-1).mean()),
    }


def _pairwise_l2(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] <= 1:
        return values.new_zeros((values.shape[0],))
    distances = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            distances.append(torch.linalg.vector_norm(values[:, left] - values[:, right], dim=-1))
    return torch.stack(distances, dim=1).mean(dim=1)


def _add_prefixed_metrics(
    totals: dict[str, float],
    prefix: str,
    metrics: dict[str, float],
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + value * batch_size


def _accumulate_group_mse(
    group_totals: dict[str, float],
    group_counts: dict[str, int],
    variant_name: str,
    pred: torch.Tensor,
    actions: torch.Tensor,
    batch: dict[str, object],
    event_command_threshold: float,
) -> None:
    per_sample_mse = (pred - actions).square().mean(dim=(1, 2)).detach().cpu()
    transitions = _has_gripper_transition(actions, event_command_threshold).detach().cpu()
    for row in range(int(actions.shape[0])):
        suite = _batch_string_at(batch["suite_name"], row)
        event_group = "transition" if bool(transitions[row]) else "no_transition"
        for group_name in (f"suite/{suite}/{variant_name}", f"event/{event_group}/{variant_name}"):
            group_totals[group_name] = group_totals.get(group_name, 0.0) + float(per_sample_mse[row])
            group_counts[group_name] = group_counts.get(group_name, 0) + 1


def _has_gripper_transition(actions: torch.Tensor, threshold: float) -> torch.Tensor:
    gripper = actions[..., -1]
    states = torch.zeros_like(gripper, dtype=torch.int64)
    states = torch.where(gripper >= threshold, torch.ones_like(states), states)
    states = torch.where(gripper <= -threshold, -torch.ones_like(states), states)
    if states.shape[1] <= 1:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=states.device)
    left = states[:, :-1]
    right = states[:, 1:]
    changed = (left != right) & (left != 0) & (right != 0)
    return changed.any(dim=1)


def _key_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    keys = (
        "variant/original/mse",
        "variant/mean_repeated/mse",
        "variant/subset/mse",
        "variant/batch_mismatch/mse",
        "delta/original_vs_mean_repeated/action_l2",
        "delta/original_vs_subset/action_l2",
        "delta/original_vs_permuted/action_l2",
        "sample/pair_l2",
        "sample/gripper_pair_l2",
        "single_sample_head/action_to_mean_l2",
        "single_sample_head/best_single_sample_action_mse",
    )
    return {key: metrics.get(key) for key in keys}


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive when provided")
    if args.subset_samples <= 0:
        raise ValueError("--subset-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")
    if args.event_command_threshold <= 0.0:
        raise ValueError("--event-command-threshold must be positive")


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
