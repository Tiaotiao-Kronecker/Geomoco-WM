#!/usr/bin/env python3
"""Audit sample-diversity usage for predicted-event mixture action heads."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.event_conditioning import combine_conditioning  # noqa: E402
from geomoco_wm.data.predicted_event_mixture import (  # noqa: E402
    event_label_is_transition,
    map_event_probabilities,
    rank_uniform_counts,
    select_event_candidates,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from audit_predicted_event_mixture_action_head_groups import (  # noqa: E402
    _event_family,
    _per_item_action_metrics,
)
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _conditioner_from_metrics,
    _event_one_hot,
    _load_event_probe,
    _sample_rank_mixture,
)
from evaluate_predicted_event_mixture_action_head import _load_action_head  # noqa: E402
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_event_mode_probe import _batch_features  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _average_metrics,
    _batch_conditioning,
    _make_loader,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_predicted_event_mixture_action_head import (  # noqa: E402
    _event_mode_for_record,
    _load_event_label_records,
    _rank_sample_feature,
    _sample_features_for_ranks,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


@dataclass(frozen=True)
class SampleVariant:
    future_inputs: torch.Tensor
    sample_features: torch.Tensor | None


@dataclass(frozen=True)
class FutureInputBundle:
    future_inputs: torch.Tensor
    sample_features: torch.Tensor | None
    rank_slots: torch.Tensor
    top_indices: torch.Tensor
    top_probs: torch.Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate 3.4b sample-diversity usage audit for predicted-event action heads."
    )
    parser.add_argument("--checkpoint", required=True, help="Gate 3.4 action-head model.pt.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument("--num-eval-passes", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--event-top-m", type=int, default=None)
    parser.add_argument("--subset-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    cvae_checkpoint_path = Path(metrics["checkpoint"]).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_checkpoint_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]
    base_seed = int(args.seed if args.seed is not None else metrics["seed"])

    dataset = OracleActionWindowDataset(
        metrics["dataset"]["windows_jsonl"],
        max_windows=int(metrics["dataset"]["num_windows"]),
        motion_mode=metrics["motion_mode"],
        visual_feature_cache_path=metrics["visual_feature_cache"],
    )
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
    event_candidate_policy = str(metrics.get("event_candidate_policy", "topk"))
    transition_reserve_threshold = float(metrics.get("transition_reserve_threshold", 0.0))

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
                event_candidate_policy=event_candidate_policy,
                transition_reserve_threshold=transition_reserve_threshold,
                sample_feature_mode=str(metrics.get("sample_feature_mode", "none")),
                subset_samples=args.subset_samples,
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
        "event_candidate_policy": event_candidate_policy,
        "transition_reserve_threshold": transition_reserve_threshold,
        "subset_samples": args.subset_samples,
        "batch_size": batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "dataset": metrics["dataset"],
        "split_by": metrics["split_by"],
        "input_mode": metrics["input_mode"],
        "sample_feature_mode": metrics.get("sample_feature_mode", "none"),
        "temporal_action_decoder_mode": metrics.get("temporal_action_decoder_mode", "none"),
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
                "key_metrics": _key_metrics(output["mean_report"]["overall"]),
            },
            indent=2,
        )
    )


@torch.no_grad()
def _audit_pass(
    action_head: MotionPriorActionHead,
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
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    subset_samples: int,
    max_batches: int | None,
) -> dict[str, Any]:
    action_head.eval()
    cvae.eval()
    event_probe.eval()
    overall_totals: dict[str, float] = {}
    overall_count = 0
    row_records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
        bundle = _predicted_event_future_input_bundle(
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
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
        )
        variants = _build_usage_variants(
            bundle.future_inputs,
            bundle.sample_features,
            bundle.rank_slots,
            bundle.top_indices,
            cvae_event_classes,
            subset_samples=subset_samples,
        )
        outputs = {
            name: action_head.forward_with_aux(
                context,
                variant.future_inputs,
                conditioning,
                variant.sample_features,
            )
            for name, variant in variants.items()
        }
        batch_size = int(context.shape[0])
        row_metrics = _variant_row_metrics(outputs, actions)
        for key, value in _variant_batch_metrics(outputs, actions).items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * batch_size
        original_output = outputs["original"]
        for name, output in outputs.items():
            if name == "original":
                continue
            for key, value in _delta_batch_metrics(original_output, output).items():
                full_key = f"delta/original_vs_{name}/{key}"
                overall_totals[full_key] = overall_totals.get(full_key, 0.0) + value * batch_size
            for key, values in _delta_row_metrics(original_output, output).items():
                row_metrics[f"delta/original_vs_{name}/{key}"] = values

        sample_metrics = _sample_row_metrics(bundle.future_inputs, action_head.horizon)
        single_metrics = _single_sample_value_metrics(
            action_head,
            context,
            conditioning,
            bundle.future_inputs,
            bundle.sample_features,
            actions,
        )
        for key, values in sample_metrics.items():
            overall_totals[f"sample/{key}"] = overall_totals.get(
                f"sample/{key}",
                0.0,
            ) + float(values.mean().detach().cpu()) * batch_size
        for key, values in single_metrics.items():
            overall_totals[f"single_sample/{key}"] = overall_totals.get(
                f"single_sample/{key}",
                0.0,
            ) + float(values.mean().detach().cpu()) * batch_size
        entropy = _event_entropy(bundle.top_probs)
        overall_totals["event/entropy"] = overall_totals.get("event/entropy", 0.0) + (
            float(entropy.mean().detach().cpu()) * batch_size
        )
        overall_count += batch_size

        for row in range(batch_size):
            window_id = _batch_string_at(batch["window_id"], row)
            suite = _batch_string_at(batch["suite_name"], row)
            task = _batch_string_at(batch["task_id"], row)
            event_mode = _event_mode_for_record(event_labels.get(window_id)) or "unknown"
            metrics_for_row = {
                key: float(values[row])
                for key, values in row_metrics.items()
            }
            metrics_for_row.update(
                {
                    f"sample/{key}": float(values[row])
                    for key, values in sample_metrics.items()
                }
            )
            metrics_for_row.update(
                {
                    f"single_sample/{key}": float(values[row])
                    for key, values in single_metrics.items()
                }
            )
            metrics_for_row["event/entropy"] = float(entropy[row])
            row_records.append(
                {
                    "suite": suite,
                    "task": task,
                    "event_mode": event_mode,
                    "event_family": _event_family(event_mode),
                    "transition_group": "transition"
                    if event_label_is_transition(event_mode)
                    else "sustain",
                    "sample_pair_l2": float(sample_metrics["pair_l2"][row]),
                    "best_vs_mean_gap": float(
                        single_metrics.get(
                            "flow_best_vs_mean_gap",
                            single_metrics.get(
                                "temporal_best_vs_mean_gap",
                                single_metrics["base_best_vs_mean_gap"],
                            ),
                        )[row]
                    ),
                    "event_entropy": float(entropy[row]),
                    "metrics": metrics_for_row,
                }
            )

    return {
        "overall": _average_metrics(overall_totals, overall_count),
        "groups": _group_row_records(row_records),
        "worst_groups": [],
    }


@torch.no_grad()
def _predicted_event_future_input_bundle(
    cvae: Any,
    event_probe: Any,
    batch: dict[str, object],
    context: torch.Tensor,
    base_conditioning: torch.Tensor | None,
    device: torch.device,
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
) -> FutureInputBundle:
    visual = _batch_visual(batch, device)
    source_probs = torch.softmax(
        event_probe(_batch_features(batch, probe_conditioner, device, probe_input_variant)),
        dim=-1,
    )
    event_probs = map_event_probabilities(source_probs, probe_class_names, event_classes)
    top_probs, top_indices = select_event_candidates(
        event_probs,
        event_classes,
        top_m=event_top_m,
        policy=event_candidate_policy,
        transition_reserve_threshold=transition_reserve_threshold,
    )
    rank_latent_means: list[torch.Tensor] = []
    rank_logvars: list[torch.Tensor] = []
    rank_conditions: list[torch.Tensor] = []
    rank_sample_features: list[torch.Tensor] = []
    for rank in range(event_top_m):
        event_one_hot = _event_one_hot(top_indices[:, rank], len(event_classes), device)
        cvae_conditioning = combine_conditioning(base_conditioning, event_one_hot)
        condition = cvae.condition(context, visual, cvae_conditioning)
        prior_mean, prior_logvar = cvae.encode_prior(condition)
        rank_latent_means.append(prior_mean)
        rank_logvars.append(prior_logvar)
        rank_conditions.append(condition)
        if sample_feature_mode != "none":
            rank_sample_features.append(
                _rank_sample_feature(
                    event_one_hot,
                    top_probs[:, rank],
                    rank=rank,
                    event_top_m=event_top_m,
                    sample_feature_mode=sample_feature_mode,
                )
            )
    samples = _sample_rank_mixture(
        cvae,
        rank_conditions,
        rank_latent_means,
        rank_logvars,
        num_samples=num_samples,
        top_m=event_top_m,
    )
    sample_features = _sample_features_for_ranks(
        rank_sample_features,
        num_samples=num_samples,
        event_top_m=event_top_m,
    )
    rank_slots = torch.tensor(
        [
            rank
            for rank, count in enumerate(rank_uniform_counts(num_samples, event_top_m))
            for _ in range(count)
        ],
        dtype=torch.long,
        device=device,
    )
    return FutureInputBundle(
        future_inputs=samples.permute(1, 0, 2).contiguous(),
        sample_features=sample_features,
        rank_slots=rank_slots,
        top_indices=top_indices,
        top_probs=top_probs,
    )


def _build_usage_variants(
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    rank_slots: torch.Tensor,
    top_indices: torch.Tensor,
    event_classes: tuple[str, ...],
    *,
    subset_samples: int,
) -> dict[str, SampleVariant]:
    batch_size, num_samples, _ = future_inputs.shape
    subset_count = min(subset_samples, num_samples)
    permutation = torch.randperm(num_samples, device=future_inputs.device)
    mean = future_inputs.mean(dim=1, keepdim=True)
    variants = {
        "original": SampleVariant(future_inputs, sample_features),
        "permuted": SampleVariant(
            future_inputs[:, permutation, :],
            _select_features(sample_features, permutation),
        ),
        "mean_repeated": SampleVariant(
            mean.expand(-1, num_samples, -1).contiguous(),
            sample_features,
        ),
        "subset_k4": SampleVariant(
            future_inputs[:, permutation[:subset_count], :],
            _select_features(sample_features, permutation[:subset_count]),
        ),
    }
    rank1_mask = rank_slots == 0
    rank1_inputs = future_inputs[:, rank1_mask, :]
    rank1_features = _mask_features(sample_features, rank1_mask)
    rank1_mean = rank1_inputs.mean(dim=1, keepdim=True)
    variants["rank1_only"] = SampleVariant(rank1_inputs, rank1_features)
    variants["rank1_repeated"] = SampleVariant(
        rank1_mean.expand(-1, num_samples, -1).contiguous(),
        _repeat_feature_mean(rank1_features, num_samples),
    )
    non_rank1_mask = ~rank1_mask
    if bool(non_rank1_mask.any().item()):
        variants["drop_rank1"] = SampleVariant(
            future_inputs[:, non_rank1_mask, :],
            _mask_features(sample_features, non_rank1_mask),
        )
    variants["transition_rank_repeated"] = _transition_rank_repeated_variant(
        future_inputs,
        sample_features,
        rank_slots,
        top_indices,
        event_classes,
        num_samples=num_samples,
    )
    if batch_size > 1:
        variants["batch_mismatch"] = SampleVariant(
            future_inputs.roll(shifts=1, dims=0),
            sample_features.roll(shifts=1, dims=0) if sample_features is not None else None,
        )
    else:
        variants["batch_mismatch"] = SampleVariant(future_inputs, sample_features)
    return variants


def _transition_rank_repeated_variant(
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    rank_slots: torch.Tensor,
    top_indices: torch.Tensor,
    event_classes: tuple[str, ...],
    *,
    num_samples: int,
) -> SampleVariant:
    batch_size = int(future_inputs.shape[0])
    transition_rank_mask = torch.zeros(
        (batch_size, top_indices.shape[1]),
        dtype=torch.bool,
        device=future_inputs.device,
    )
    for rank in range(top_indices.shape[1]):
        labels = [event_classes[int(index)] for index in top_indices[:, rank].detach().cpu()]
        values = [event_label_is_transition(label) for label in labels]
        transition_rank_mask[:, rank] = torch.tensor(
            values,
            dtype=torch.bool,
            device=future_inputs.device,
        )
    slot_mask = transition_rank_mask[:, rank_slots]
    weights = slot_mask.to(dtype=future_inputs.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    transition_mean = (future_inputs * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(
        -1
    )
    fallback_mean = future_inputs.mean(dim=1, keepdim=True)
    has_transition = slot_mask.any(dim=1).reshape(batch_size, 1, 1)
    repeated = torch.where(has_transition, transition_mean, fallback_mean)
    repeated_features = None
    if sample_features is not None:
        feature_mean = (
            sample_features * weights.unsqueeze(-1)
        ).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        fallback_feature = sample_features.mean(dim=1, keepdim=True)
        repeated_features = torch.where(has_transition, feature_mean, fallback_feature)
        repeated_features = repeated_features.expand(-1, num_samples, -1).contiguous()
    return SampleVariant(
        repeated.expand(-1, num_samples, -1).contiguous(),
        repeated_features,
    )


def _variant_batch_metrics(
    outputs: dict[str, dict[str, torch.Tensor | None]],
    actions: torch.Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, output in outputs.items():
        for readout in _available_readouts(output):
            readout_actions = _readout_actions(output, readout)
            if readout_actions is None:
                continue
            metrics.update(
                {
                    f"variant/{name}/{readout}/{key}": value
                    for key, value in action_metrics(readout_actions, actions).items()
                }
            )
    return metrics


def _variant_row_metrics(
    outputs: dict[str, dict[str, torch.Tensor | None]],
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for name, output in outputs.items():
        for readout in _available_readouts(output):
            readout_actions = _readout_actions(output, readout)
            if readout_actions is None:
                continue
            for key, value in _per_item_action_metrics(readout_actions, actions).items():
                metrics[f"variant/{name}/{readout}/{key}"] = value
    return metrics


def _delta_batch_metrics(
    original_output: dict[str, torch.Tensor | None],
    variant_output: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for readout in _available_readouts(original_output):
        original = _readout_actions(original_output, readout)
        variant = _readout_actions(variant_output, readout)
        if original is None or variant is None:
            continue
        metrics.update(
            {
                f"{readout}/{key}": value
                for key, value in _action_delta_metrics(original, variant).items()
            }
        )
    return metrics


def _delta_row_metrics(
    original_output: dict[str, torch.Tensor | None],
    variant_output: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for readout in _available_readouts(original_output):
        original = _readout_actions(original_output, readout)
        variant = _readout_actions(variant_output, readout)
        if original is None or variant is None:
            continue
        for key, value in _per_item_action_delta_metrics(original, variant).items():
            metrics[f"{readout}/{key}"] = value
    return metrics


def _single_sample_value_metrics(
    action_head: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    base_predictions = []
    temporal_predictions = []
    flow_predictions = []
    for sample_index in range(future_inputs.shape[1]):
        output = action_head.forward_with_aux(
            context,
            future_inputs[:, sample_index : sample_index + 1, :],
            conditioning,
            sample_features[:, sample_index : sample_index + 1, :]
            if sample_features is not None
            else None,
        )
        base_actions = output.get("actions")
        if base_actions is None:
            raise ValueError("action head output is missing actions")
        base_predictions.append(base_actions)
        temporal_actions = output.get("temporal_actions")
        if temporal_actions is not None:
            temporal_predictions.append(temporal_actions)
        flow_actions = output.get("flow_actions")
        if flow_actions is not None:
            flow_predictions.append(flow_actions)
    metrics = _single_prediction_stack_metrics(
        torch.stack(base_predictions, dim=1),
        actions,
        prefix="base",
    )
    if temporal_predictions:
        metrics.update(
            _single_prediction_stack_metrics(
                torch.stack(temporal_predictions, dim=1),
                actions,
                prefix="temporal",
            )
        )
    if flow_predictions:
        metrics.update(
            _single_prediction_stack_metrics(
                torch.stack(flow_predictions, dim=1),
                actions,
                prefix="flow",
            )
        )
    return metrics


def _single_prediction_stack_metrics(
    predictions: torch.Tensor,
    actions: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    per_sample_mse = (predictions - actions.unsqueeze(1)).square().mean(dim=(2, 3))
    mean_mse = per_sample_mse.mean(dim=1)
    best_mse = per_sample_mse.min(dim=1).values
    prediction_mean = predictions.mean(dim=1, keepdim=True)
    action_to_mean_l2 = torch.linalg.vector_norm(
        (predictions - prediction_mean).reshape(predictions.shape[0], predictions.shape[1], -1),
        dim=-1,
    ).mean(dim=1)
    return {
        f"{prefix}_mean_mse": mean_mse.detach().cpu(),
        f"{prefix}_best_mse": best_mse.detach().cpu(),
        f"{prefix}_best_vs_mean_gap": (mean_mse - best_mse).detach().cpu(),
        f"{prefix}_action_to_mean_l2": action_to_mean_l2.detach().cpu(),
    }


def _sample_row_metrics(samples: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
    mean = samples.mean(dim=1, keepdim=True)
    centered = samples - mean
    metrics = {
        "sample_to_mean_l2": torch.linalg.vector_norm(centered, dim=-1).mean(dim=1),
        "pair_l2": _pairwise_l2(samples),
        "motion_variance": samples.var(dim=1, unbiased=False).mean(dim=-1),
    }
    if samples.shape[-1] == horizon * 7:
        eef_dim = horizon * 6
        metrics["gripper_pair_l2"] = _pairwise_l2(samples[..., eef_dim:])
        metrics["gripper_variance"] = samples[..., eef_dim:].var(dim=1, unbiased=False).mean(dim=-1)
    return {key: value.detach().cpu() for key, value in metrics.items()}


def _group_row_records(row_records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    if not row_records:
        return {}
    diversity_labels = _tertile_labels([row["sample_pair_l2"] for row in row_records])
    gap_labels = _tertile_labels([row["best_vs_mean_gap"] for row in row_records])
    entropy_labels = _tertile_labels([row["event_entropy"] for row in row_records])
    group_totals: dict[str, dict[str, float]] = defaultdict(dict)
    group_counts: dict[str, int] = defaultdict(int)
    for row, diversity, gap, entropy in zip(
        row_records,
        diversity_labels,
        gap_labels,
        entropy_labels,
        strict=True,
    ):
        groups = (
            "all",
            f"suite/{row['suite']}",
            f"task/{row['suite']}/{row['task']}",
            f"event_mode/{row['event_mode']}",
            f"event_family/{row['event_family']}",
            f"transition_group/{row['transition_group']}",
            f"sample_diversity/{diversity}",
            f"best_vs_mean_gap/{gap}",
            f"event_entropy/{entropy}",
        )
        for group in groups:
            group_counts[group] += 1
            totals = group_totals[group]
            for key, value in row["metrics"].items():
                totals[key] = totals.get(key, 0.0) + float(value)
    output: dict[str, dict[str, float | int]] = {}
    for group, totals in sorted(group_totals.items()):
        count = group_counts[group]
        row: dict[str, float | int] = {"count": count}
        row.update({key: value / count for key, value in sorted(totals.items())})
        output[group] = row
    return output


def _mean_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"overall": {}, "groups": {}, "worst_groups": []}
    overall_keys = set().union(*(report["overall"].keys() for report in reports))
    overall = {
        key: _mean_optional([report["overall"].get(key) for report in reports])
        for key in sorted(overall_keys)
    }
    group_names = set().union(*(report["groups"].keys() for report in reports))
    groups: dict[str, dict[str, float | int | None]] = {}
    for group_name in sorted(group_names):
        rows = [report["groups"][group_name] for report in reports if group_name in report["groups"]]
        metric_keys = set().union(*(row.keys() for row in rows)) - {"count"}
        row_out: dict[str, float | int | None] = {
            "count": int(round(_mean_optional([row.get("count") for row in rows]) or 0.0))
        }
        row_out.update(
            {
                key: _mean_optional([row.get(key) for row in rows])
                for key in sorted(metric_keys)
            }
        )
        groups[group_name] = row_out
    return {"overall": overall, "groups": groups, "worst_groups": []}


def _key_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    keys = (
        "variant/original/temporal/mse",
        "variant/mean_repeated/temporal/mse",
        "variant/rank1_only/temporal/mse",
        "variant/subset_k4/temporal/mse",
        "variant/drop_rank1/temporal/mse",
        "variant/batch_mismatch/temporal/mse",
        "variant/original/flow/mse",
        "variant/mean_repeated/flow/mse",
        "variant/rank1_only/flow/mse",
        "variant/subset_k4/flow/mse",
        "variant/drop_rank1/flow/mse",
        "variant/batch_mismatch/flow/mse",
        "delta/original_vs_mean_repeated/temporal/action_l2",
        "delta/original_vs_permuted/temporal/action_l2",
        "delta/original_vs_subset_k4/temporal/action_l2",
        "delta/original_vs_mean_repeated/flow/action_l2",
        "delta/original_vs_permuted/flow/action_l2",
        "delta/original_vs_subset_k4/flow/action_l2",
        "sample/pair_l2",
        "sample/gripper_pair_l2",
        "single_sample/temporal_best_vs_mean_gap",
    )
    return {key: metrics.get(key) for key in keys}


def _readout_actions(
    output: dict[str, torch.Tensor | None],
    readout: str,
) -> torch.Tensor | None:
    if readout == "base":
        return output.get("actions")
    if readout == "temporal":
        return output.get("temporal_actions")
    if readout == "flow":
        return output.get("flow_actions")
    raise ValueError(f"unsupported readout {readout!r}")


def _available_readouts(output: dict[str, torch.Tensor | None]) -> tuple[str, ...]:
    if output.get("actions") is None:
        raise ValueError("action head output is missing actions")
    readouts = ["base"]
    if output.get("temporal_actions") is not None:
        readouts.append("temporal")
    if output.get("flow_actions") is not None:
        readouts.append("flow")
    return tuple(readouts)


def _action_delta_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    error = left - right
    return {
        "action_mse": float(error.square().mean().detach().cpu()),
        "action_mae": float(error.abs().mean().detach().cpu()),
        "action_l2": float(
            torch.linalg.vector_norm(error.reshape(error.shape[0], -1), dim=-1)
            .mean()
            .detach()
            .cpu()
        ),
    }


def _per_item_action_delta_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    error = left - right
    return {
        "action_mse": error.square().mean(dim=(1, 2)).detach().cpu(),
        "action_mae": error.abs().mean(dim=(1, 2)).detach().cpu(),
        "action_l2": torch.linalg.vector_norm(error.reshape(error.shape[0], -1), dim=-1)
        .detach()
        .cpu(),
    }


def _pairwise_l2(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] <= 1:
        return values.new_zeros((values.shape[0],))
    distances = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            distances.append(torch.linalg.vector_norm(values[:, left] - values[:, right], dim=-1))
    return torch.stack(distances, dim=1).mean(dim=1)


def _event_entropy(top_probs: torch.Tensor) -> torch.Tensor:
    return -(top_probs * top_probs.clamp_min(1e-12).log()).sum(dim=-1).detach().cpu()


def _tertile_labels(values: list[float]) -> list[str]:
    if not values:
        return []
    sorted_values = sorted(values)
    low = sorted_values[len(sorted_values) // 3]
    high = sorted_values[(2 * len(sorted_values)) // 3]
    labels = []
    for value in values:
        if value <= low:
            labels.append("low")
        elif value <= high:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


def _select_features(
    sample_features: torch.Tensor | None,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if sample_features is None:
        return None
    return sample_features[:, indices, :]


def _mask_features(
    sample_features: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if sample_features is None:
        return None
    return sample_features[:, mask, :]


def _repeat_feature_mean(
    sample_features: torch.Tensor | None,
    num_samples: int,
) -> torch.Tensor | None:
    if sample_features is None:
        return None
    return sample_features.mean(dim=1, keepdim=True).expand(-1, num_samples, -1).contiguous()


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _mean_optional(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive when provided")
    if args.event_top_m is not None and args.event_top_m <= 0:
        raise ValueError("--event-top-m must be positive when provided")
    if args.subset_samples <= 0:
        raise ValueError("--subset-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
