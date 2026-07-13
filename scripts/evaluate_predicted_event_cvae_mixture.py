#!/usr/bin/env python3
"""Evaluate predicted event-mode top-M mixtures for event-conditioned cVAEs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.event_conditioning import combine_conditioning  # noqa: E402
from geomoco_wm.data.predicted_event_mixture import (  # noqa: E402
    event_label_is_transition,
    event_timing_bin,
    map_event_probabilities,
    rank_uniform_counts,
    select_event_candidates,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from evaluate_visual_cvae_samples import (  # noqa: E402
    _batch_sample_metrics,
    _load_model,
    _sample_prior_motions,
)
from train_event_mode_probe import EventModeProbeNet, _batch_features  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _prediction_metrics,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Gate 3.1d predicted-event cVAE top-M mixtures."
    )
    parser.add_argument("--checkpoint", required=True, help="Event-conditioned cVAE model.pt.")
    parser.add_argument("--event-probe-checkpoint", required=True, help="Gate 3.1b probe model.pt.")
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=None,
        help="Defaults to the cVAE checkpoint dataset windows_jsonl.",
    )
    parser.add_argument(
        "--visual-feature-cache",
        default=None,
        help="Defaults to the cVAE checkpoint visual feature cache.",
    )
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--event-candidate-policy",
        default="topk",
        choices=["topk", "transition_reserve"],
        help="Policy for selecting event candidates before cVAE sampling.",
    )
    parser.add_argument(
        "--transition-reserve-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum mapped transition probability required for "
            "transition_reserve candidate replacement."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--action-decoder-checkpoint", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device) if not args.dry_run else torch.device("cpu")

    cvae_path = Path(args.checkpoint).expanduser().resolve()
    cvae_checkpoint = torch.load(cvae_path, map_location=device, weights_only=False)
    cvae_metrics = cvae_checkpoint["metrics"]
    event_classes = _checkpoint_event_classes(cvae_metrics)
    windows_jsonl = args.windows_jsonl or cvae_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or cvae_metrics["visual_feature_cache"]
    motion_mode = str(cvae_metrics.get("motion_mode", cvae_metrics["dataset"]["motion_mode"]))
    split_by = args.split_by or cvae_metrics.get("split_by", "episode")
    condition_on = args.condition_on or cvae_metrics["conditioning"]["condition_on"]
    event_audit_json = args.event_mode_audit_json or _checkpoint_event_audit_json(cvae_metrics)
    if event_audit_json is None:
        raise ValueError("--event-mode-audit-json is required when absent from checkpoint")

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _conditioner_from_metrics(cvae_metrics["conditioning"])
    if condition_on != conditioner.condition_on:
        raise ValueError(
            "condition_on must match cVAE checkpoint conditioning: "
            f"{condition_on} vs {conditioner.condition_on}"
        )
    visual_token_config = _resolve_visual_token_config(
        dataset,
        cvae_metrics["visual_token_config"]["visual_token_count"],
        cvae_metrics["visual_token_config"]["visual_token_dim"],
    )
    event_probe, probe_metrics, probe_conditioner = _load_event_probe(
        args.event_probe_checkpoint,
        device,
    )
    event_labels = _load_event_labels(event_audit_json)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "checkpoint": str(cvae_path),
                    "event_probe_checkpoint": str(Path(args.event_probe_checkpoint).expanduser()),
                    "dataset": spec.to_dict(),
                    "motion_mode": motion_mode,
                    "conditioning": conditioner.to_dict(),
                    "cvae_event_classes": list(event_classes),
                    "probe_event_classes": probe_metrics["probe"]["class_names"],
                    "top_m": args.top_m,
                    "num_samples": args.num_samples,
                    "event_candidate_policy": args.event_candidate_policy,
                    "transition_reserve_threshold": args.transition_reserve_threshold,
                    "rank_sample_counts": list(rank_uniform_counts(args.num_samples, args.top_m)),
                    "split_by": split_by,
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "visual_feature_cache": str(visual_feature_cache),
                    "visual_token_config": visual_token_config,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    cvae = _load_model(
        cvae_checkpoint,
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim + len(event_classes),
        device=device,
    )
    action_decoder = None
    action_decoder_config = None
    action_decoder_checkpoint = args.action_decoder_checkpoint or cvae_metrics.get(
        "action_decoder_checkpoint"
    )
    if action_decoder_checkpoint:
        action_decoder, action_decoder_config = _load_action_decoder(
            action_decoder_checkpoint,
            device,
        )
        if action_decoder_config["motion_mode"] != motion_mode:
            raise ValueError(
                "action decoder motion mode must match cVAE motion mode: "
                f"{action_decoder_config['motion_mode']} vs {motion_mode}"
            )
        _freeze_action_decoder(action_decoder)

    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    metrics, event_report = _evaluate_predicted_mixture(
        cvae,
        event_probe,
        val_loader,
        device,
        conditioner,
        probe_conditioner,
        cvae_event_classes=event_classes,
        probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
        probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
        event_labels=event_labels,
        action_decoder=action_decoder,
        motion_mode=motion_mode,
        num_samples=args.num_samples,
        top_m=args.top_m,
        event_candidate_policy=args.event_candidate_policy,
        transition_reserve_threshold=args.transition_reserve_threshold,
    )
    output = {
        "checkpoint": str(cvae_path),
        "event_probe_checkpoint": str(Path(args.event_probe_checkpoint).expanduser().resolve()),
        "dataset": spec.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "motion_mode": motion_mode,
        "conditioning": conditioner.to_dict(),
        "cvae_event_classes": list(event_classes),
        "event_probe": _probe_summary(probe_metrics),
        "event_mode_audit_json": str(Path(event_audit_json).expanduser().resolve()),
        "visual_feature_cache": str(visual_feature_cache),
        "visual_token_config": visual_token_config,
        "num_samples": args.num_samples,
        "top_m": args.top_m,
        "event_candidate_policy": args.event_candidate_policy,
        "transition_reserve_threshold": args.transition_reserve_threshold,
        "rank_sample_counts": list(rank_uniform_counts(args.num_samples, args.top_m)),
        "action_decoder_checkpoint": str(action_decoder_checkpoint)
        if action_decoder_checkpoint
        else None,
        "action_decoder_config": action_decoder_config,
        "event_prediction": event_report,
        "metrics": metrics,
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
                "event_prediction": event_report,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@torch.no_grad()
def _evaluate_predicted_mixture(
    cvae: VisualConditionedGeoMoCoCVAE,
    event_probe: EventModeProbeNet,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    *,
    cvae_event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_labels: dict[str, str],
    action_decoder: ActionDecoder | None,
    motion_mode: str,
    num_samples: int,
    top_m: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
) -> tuple[dict[str, float | None], dict[str, object]]:
    if loader is None:
        return {"mixture_prior_mse": None}, {}
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    total_count = 0
    event_counts = _new_event_counts()
    top1_counter: Counter[str] = Counter()
    true_counter: Counter[str] = Counter()
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        visual = _batch_visual(batch, device)
        base_conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
        source_probs = torch.softmax(
            event_probe(_batch_features(batch, probe_conditioner, device, probe_input_variant)),
            dim=-1,
        )
        event_probs = map_event_probabilities(source_probs, probe_class_names, cvae_event_classes)
        top_probs, top_indices = select_event_candidates(
            event_probs,
            cvae_event_classes,
            top_m=top_m,
            policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
        )
        _update_event_counts(
            event_counts,
            top1_counter,
            true_counter,
            batch["window_id"],
            top_indices,
            cvae_event_classes,
            event_labels,
        )

        rank_latent_means: list[Tensor] = []
        rank_logvars: list[Tensor] = []
        rank_motion_means: list[Tensor] = []
        rank_conditions: list[Tensor] = []
        for rank in range(top_m):
            event_one_hot = _event_one_hot(top_indices[:, rank], len(cvae_event_classes), device)
            conditioning = combine_conditioning(base_conditioning, event_one_hot)
            condition = cvae.condition(context, visual, conditioning)
            prior_mean, prior_logvar = cvae.encode_prior(condition)
            rank_latent_means.append(prior_mean)
            rank_logvars.append(prior_logvar)
            rank_motion_means.append(cvae.decode(condition, prior_mean))
            rank_conditions.append(condition)

        top1_prior = rank_motion_means[0]
        mixture_prior = sum(
            top_probs[:, rank].unsqueeze(-1) * rank_motion_means[rank]
            for rank in range(top_m)
        )
        samples = _sample_rank_mixture(
            cvae,
            rank_conditions,
            rank_latent_means,
            rank_logvars,
            num_samples=num_samples,
            top_m=top_m,
        )
        batch_metrics = _batch_metrics(
            top1_prior,
            mixture_prior,
            samples,
            motion,
            context,
            batch,
            action_decoder,
            device,
            motion_mode,
        )
        batch_size = int(context.shape[0])
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count), _event_report(
        event_counts,
        top1_counter,
        true_counter,
    )


def _batch_metrics(
    top1_prior: Tensor,
    mixture_prior: Tensor,
    samples: Tensor,
    motion: Tensor,
    context: Tensor,
    batch: dict[str, object],
    action_decoder: ActionDecoder | None,
    device: torch.device,
    motion_mode: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(_rename_prefix("top1_prior", _prediction_metrics(top1_prior, motion, motion_mode)))
    sample_metrics = _batch_sample_metrics(
        mixture_prior,
        samples,
        motion,
        context,
        batch,
        action_decoder,
        device,
        motion_mode,
    )
    for key, value in sample_metrics.items():
        if key.startswith("prior_mean_"):
            metrics[f"mixture_prior_{key.removeprefix('prior_mean_')}"] = value
        else:
            metrics[f"mixture_{key}"] = value
    if action_decoder is not None:
        actions = batch["actions"].to(device)
        top1_actions = action_decoder(context, top1_prior)
        metrics.update(_rename_prefix("top1_prior_action", action_metrics(top1_actions, actions)))
    return metrics


def _sample_rank_mixture(
    cvae: VisualConditionedGeoMoCoCVAE,
    rank_conditions: list[Tensor],
    rank_means: list[Tensor],
    rank_logvars: list[Tensor],
    *,
    num_samples: int,
    top_m: int,
) -> Tensor:
    samples: list[Tensor] = []
    for rank, count in enumerate(rank_uniform_counts(num_samples, top_m)):
        if count <= 0:
            continue
        samples.append(
            _sample_prior_motions(
                cvae,
                rank_conditions[rank],
                rank_means[rank],
                rank_logvars[rank],
                count,
            )
        )
    return torch.cat(samples, dim=0)


def _event_one_hot(indices: Tensor, num_classes: int, device: torch.device) -> Tensor:
    one_hot = torch.zeros((indices.shape[0], num_classes), dtype=torch.float32, device=device)
    one_hot[torch.arange(indices.shape[0], device=device), indices] = 1.0
    return one_hot


def _new_event_counts() -> dict[str, int]:
    return {
        "valid": 0,
        "top1_correct": 0,
        "top_m_correct": 0,
        "transition_tp": 0,
        "transition_fp": 0,
        "transition_fn": 0,
        "transition_timing_correct": 0,
        "transition_timing_total": 0,
    }


def _update_event_counts(
    counts: dict[str, int],
    top1_counter: Counter[str],
    true_counter: Counter[str],
    window_ids: object,
    top_indices: Tensor,
    event_classes: tuple[str, ...],
    event_labels: dict[str, str],
) -> None:
    cpu_indices = top_indices.detach().cpu()
    for row in range(cpu_indices.shape[0]):
        window_id = _batch_string_at(window_ids, row)
        true_label = event_labels.get(window_id)
        if true_label is None:
            continue
        pred_labels = [event_classes[int(index)] for index in cpu_indices[row].tolist()]
        pred_label = pred_labels[0]
        top1_counter[pred_label] += 1
        true_counter[true_label] += 1
        counts["valid"] += 1
        counts["top1_correct"] += int(pred_label == true_label)
        counts["top_m_correct"] += int(true_label in pred_labels)
        true_transition = event_label_is_transition(true_label)
        pred_transition = event_label_is_transition(pred_label)
        if pred_transition and true_transition:
            counts["transition_tp"] += 1
            counts["transition_timing_total"] += 1
            counts["transition_timing_correct"] += int(
                event_timing_bin(pred_label) == event_timing_bin(true_label)
            )
        elif pred_transition and not true_transition:
            counts["transition_fp"] += 1
        elif true_transition and not pred_transition:
            counts["transition_fn"] += 1


def _event_report(
    counts: dict[str, int],
    top1_counter: Counter[str],
    true_counter: Counter[str],
) -> dict[str, object]:
    valid = counts["valid"]
    tp = counts["transition_tp"]
    fp = counts["transition_fp"]
    fn = counts["transition_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    transition_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing_total = counts["transition_timing_total"]
    return {
        "num_valid": valid,
        "top1_accuracy": counts["top1_correct"] / valid if valid else 0.0,
        "top_m_coverage": counts["top_m_correct"] / valid if valid else 0.0,
        "transition_binary_f1": transition_f1,
        "transition_precision": precision,
        "transition_recall": recall,
        "transition_timing_accuracy": counts["transition_timing_correct"] / timing_total
        if timing_total
        else 0.0,
        "counts": counts,
        "top1_event_mode_counts": {key: int(value) for key, value in sorted(top1_counter.items())},
        "true_event_mode_counts": {key: int(value) for key, value in sorted(true_counter.items())},
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_m <= 0:
        raise ValueError("--top-m must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.top_m > args.num_samples:
        raise ValueError("--top-m cannot exceed --num-samples")
    if args.transition_reserve_threshold < 0.0:
        raise ValueError("--transition-reserve-threshold must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")


def _load_event_probe(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[EventModeProbeNet, dict[str, Any], CategoricalConditioner]:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    probe = metrics["probe"]
    model = EventModeProbeNet(
        input_dim=int(probe["input_dim"]),
        num_classes=int(probe["num_classes"]),
        hidden_dims=tuple(int(value) for value in metrics["hidden_dims"]),
        dropout=float(metrics["dropout"]),
        layer_norm=bool(metrics["layer_norm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metrics, _conditioner_from_metrics(metrics["conditioning"])


def _conditioner_from_metrics(conditioning: dict[str, Any]) -> CategoricalConditioner:
    condition_on = str(conditioning["condition_on"])
    vocab = tuple(str(value) for value in conditioning.get("vocab", []))
    return CategoricalConditioner(
        condition_on=condition_on,
        vocab=vocab,
        index_by_label={label: index for index, label in enumerate(vocab)},
    )


def _checkpoint_event_classes(metrics: dict[str, Any]) -> tuple[str, ...]:
    event_conditioning = metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        raise ValueError("cVAE checkpoint must include event_conditioning")
    class_names = tuple(str(value) for value in event_conditioning.get("class_names", []))
    if not class_names:
        raise ValueError("cVAE checkpoint event_conditioning has no class_names")
    return class_names


def _checkpoint_event_audit_json(metrics: dict[str, Any]) -> str | None:
    event_conditioning = metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        return None
    value = event_conditioning.get("event_mode_audit_json")
    return str(value) if value else None


def _load_event_labels(event_mode_audit_json: str | Path) -> dict[str, str]:
    report = json.loads(Path(event_mode_audit_json).expanduser().read_text(encoding="utf-8"))
    labels = report.get("window_labels")
    if not isinstance(labels, list):
        raise ValueError("event-mode audit JSON must include window_labels")
    return {str(item["window_id"]): str(item["event_mode"]) for item in labels}


def _probe_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe": metrics["probe"],
        "conditioning": metrics["conditioning"],
        "best_epoch": metrics.get("best_epoch"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
        "final": metrics.get("final"),
    }


def _rename_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
