#!/usr/bin/env python3
"""Evaluate prior samples and best-of-K coverage for visual cVAE priors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.event_conditioning import (  # noqa: E402
    EventModeConditioner,
    batch_event_mode_conditioning,
    combine_conditioning,
    load_event_mode_conditioner,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _parse_hidden_dims,
    _prediction_metrics,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Gate 2.4c cVAE prior-sample coverage."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=None,
        help="Defaults to the checkpoint dataset windows_jsonl.",
    )
    parser.add_argument(
        "--visual-feature-cache",
        default=None,
        help="Defaults to the checkpoint visual feature cache.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument(
        "--event-conditioning-mode",
        default=None,
        choices=["none", "oracle", "shuffled"],
        help="Defaults to checkpoint event-conditioning mode.",
    )
    parser.add_argument("--event-class-set", default=None, choices=["stable8", "all_observed"])
    parser.add_argument("--event-shuffle-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--action-decoder-checkpoint", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_metrics = checkpoint["metrics"]
    windows_jsonl = args.windows_jsonl or checkpoint_metrics["dataset"]["windows_jsonl"]
    visual_feature_cache = args.visual_feature_cache or checkpoint_metrics["visual_feature_cache"]
    split_by = args.split_by or checkpoint_metrics.get("split_by", "episode")
    condition_on = args.condition_on or checkpoint_metrics["conditioning"]["condition_on"]
    event_conditioning_cfg = checkpoint_metrics.get("event_conditioning", {})
    event_conditioning_mode = args.event_conditioning_mode or str(
        event_conditioning_cfg.get("mode", "none")
    )
    event_class_set = args.event_class_set or _event_class_set_from_checkpoint(event_conditioning_cfg)
    motion_mode = str(checkpoint_metrics.get("motion_mode", checkpoint_metrics["dataset"].get("motion_mode")))

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_checkpoint_conditioner(checkpoint_metrics)
    event_conditioner = load_event_mode_conditioner(
        args.event_mode_audit_json or _checkpoint_event_audit_json(checkpoint_metrics),
        mode=event_conditioning_mode,
        class_set=event_class_set,
        shuffle_seed=args.event_shuffle_seed,
    )
    if condition_on != conditioner.condition_on:
        raise ValueError(
            "condition_on must match checkpoint conditioning for cVAE eval: "
            f"{condition_on} vs {conditioner.condition_on}"
        )
    visual_token_config = _resolve_visual_token_config(
        dataset,
        checkpoint_metrics["visual_token_config"]["visual_token_count"],
        checkpoint_metrics["visual_token_config"]["visual_token_dim"],
    )
    model = _load_model(
        checkpoint,
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim + event_conditioner.dim,
        device=device,
    )
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    action_decoder_checkpoint = args.action_decoder_checkpoint or checkpoint_metrics.get(
        "action_decoder_checkpoint"
    )
    action_decoder = None
    action_decoder_config = None
    if action_decoder_checkpoint:
        action_decoder, action_decoder_config = _load_action_decoder(
            action_decoder_checkpoint,
            device,
        )
        if action_decoder_config["motion_mode"] != motion_mode:
            raise ValueError(
                "action decoder motion mode must match checkpoint motion mode: "
                f"{action_decoder_config['motion_mode']} vs {motion_mode}"
            )
        _freeze_action_decoder(action_decoder)

    metrics = _evaluate_samples(
        model,
        action_decoder,
        val_loader,
        device,
        conditioner,
        event_conditioner,
        motion_mode=motion_mode,
        num_samples=args.num_samples,
    )
    output = {
        "checkpoint": str(checkpoint_path),
        "dataset": spec.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "split_by": split_by,
        "motion_mode": motion_mode,
        "conditioning": conditioner.to_dict(),
        "event_conditioning": event_conditioner.to_dict(),
        "combined_conditioning_dim": conditioner.dim + event_conditioner.dim,
        "visual_feature_cache": str(visual_feature_cache),
        "visual_token_config": visual_token_config,
        "num_samples": args.num_samples,
        "action_decoder_checkpoint": str(action_decoder_checkpoint)
        if action_decoder_checkpoint
        else None,
        "action_decoder_config": action_decoder_config,
        "metrics": metrics,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "metrics": metrics}, indent=2))


def _load_model(
    checkpoint: dict[str, Any],
    *,
    context_dim: int,
    motion_dim: int,
    visual_token_config: dict[str, int],
    conditioner_dim: int,
    device: torch.device,
) -> VisualConditionedGeoMoCoCVAE:
    metrics = checkpoint["metrics"]
    model = VisualConditionedGeoMoCoCVAE(
        context_dim=context_dim,
        motion_dim=motion_dim,
        visual_token_dim=int(visual_token_config["visual_token_dim"]),
        visual_token_count=int(visual_token_config["visual_token_count"]),
        conditioning_dim=conditioner_dim,
        latent_dim=int(metrics["latent_dim"]),
        hidden_dims=_parse_hidden_dims(",".join(str(value) for value in metrics["hidden_dims"])),
        query_dim=int(metrics["visual_query_dim"]),
        num_heads=int(metrics["visual_num_heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _evaluate_samples(
    model: VisualConditionedGeoMoCoCVAE,
    action_decoder: ActionDecoder | None,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    *,
    motion_mode: str,
    num_samples: int,
) -> dict[str, float | None]:
    if loader is None:
        return {"prior_mean_mse": None}
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        visual = _batch_visual(batch, device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        condition = model.condition(context, visual, conditioning)
        prior_mean, prior_logvar = model.encode_prior(condition)
        prior_mean_motion = model.decode(condition, prior_mean)
        samples = _sample_prior_motions(model, condition, prior_mean, prior_logvar, num_samples)
        batch_metrics = _batch_sample_metrics(
            prior_mean_motion,
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
    return _average_metrics(totals, total_count)


def _sample_prior_motions(
    model: VisualConditionedGeoMoCoCVAE,
    condition: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    std = torch.exp(0.5 * prior_logvar)
    samples: list[torch.Tensor] = []
    for _ in range(num_samples):
        latent = prior_mean + std * torch.randn_like(std)
        samples.append(model.decode(condition, latent))
    return torch.stack(samples, dim=0)


def _batch_sample_metrics(
    prior_mean_motion: torch.Tensor,
    samples: torch.Tensor,
    motion: torch.Tensor,
    context: torch.Tensor,
    batch: dict[str, object],
    action_decoder: ActionDecoder | None,
    device: torch.device,
    motion_mode: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(_prefix("prior_mean", _prediction_metrics(prior_mean_motion, motion, motion_mode)))
    flat_samples = samples.reshape(-1, samples.shape[-1])
    repeated_motion = motion.unsqueeze(0).expand(samples.shape[0], -1, -1).reshape_as(flat_samples)
    metrics.update(_prefix("sample_mean", _prediction_metrics(flat_samples, repeated_motion, motion_mode)))
    best_motion = _select_best(samples, motion, target_kind="motion")
    metrics.update(_prefix("best_of_k_motion", _prediction_metrics(best_motion, motion, motion_mode)))
    metrics.update(_diversity_metrics(samples, prior_mean_motion))

    if action_decoder is not None:
        actions = batch["actions"].to(device)
        prior_mean_actions = action_decoder(context, prior_mean_motion)
        metrics.update(_prefix("prior_mean_action", action_metrics(prior_mean_actions, actions)))
        flat_context = context.unsqueeze(0).expand(samples.shape[0], -1, -1).reshape(-1, context.shape[-1])
        flat_actions = action_decoder(flat_context, flat_samples)
        repeated_actions = actions.unsqueeze(0).expand(samples.shape[0], -1, -1, -1)
        metrics.update(
            _prefix(
                "sample_mean_action",
                action_metrics(
                    flat_actions,
                    repeated_actions.reshape_as(flat_actions),
                ),
            )
        )
        sample_actions = flat_actions.reshape(
            samples.shape[0],
            context.shape[0],
            actions.shape[1],
            actions.shape[2],
        )
        best_action_motion = _select_best_action_motion(
            samples,
            sample_actions,
            actions,
        )
        best_action_actions = action_decoder(context, best_action_motion)
        metrics.update(_prefix("best_of_k_action", action_metrics(best_action_actions, actions)))
    return metrics


def _select_best(samples: torch.Tensor, target: torch.Tensor, *, target_kind: str) -> torch.Tensor:
    del target_kind
    per_sample_error = (samples - target.unsqueeze(0)).pow(2).mean(dim=-1)
    best_indices = per_sample_error.argmin(dim=0)
    batch_indices = torch.arange(target.shape[0], device=target.device)
    return samples[best_indices, batch_indices]


def _select_best_action_motion(
    samples: torch.Tensor,
    sample_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    action_errors = (sample_actions - target_actions.unsqueeze(0)).pow(2).mean(dim=(2, 3))
    best_indices = action_errors.argmin(dim=0)
    batch_indices = torch.arange(target_actions.shape[0], device=target_actions.device)
    return samples[best_indices, batch_indices]


def _diversity_metrics(samples: torch.Tensor, prior_mean_motion: torch.Tensor) -> dict[str, float]:
    distances_to_mean = torch.linalg.vector_norm(samples - prior_mean_motion.unsqueeze(0), dim=-1)
    metrics = {
        "sample_motion_variance": float(samples.var(dim=0, unbiased=False).mean().cpu()),
        "sample_to_prior_mean_l2": float(distances_to_mean.mean().cpu()),
    }
    if samples.shape[0] > 1:
        pairwise_values: list[torch.Tensor] = []
        for left in range(samples.shape[0]):
            for right in range(left + 1, samples.shape[0]):
                pairwise_values.append(
                    torch.linalg.vector_norm(samples[left] - samples[right], dim=-1)
                )
        metrics["sample_pair_l2"] = float(torch.cat(pairwise_values).mean().cpu())
    else:
        metrics["sample_pair_l2"] = 0.0
    return metrics


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _build_checkpoint_conditioner(metrics: dict[str, Any]) -> CategoricalConditioner:
    conditioning = metrics["conditioning"]
    condition_on = str(conditioning["condition_on"])
    vocab = tuple(str(value) for value in conditioning.get("vocab", []))
    return CategoricalConditioner(
        condition_on=condition_on,
        vocab=vocab,
        index_by_label={label: index for index, label in enumerate(vocab)},
    )


def _combined_batch_conditioning(
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    device: torch.device,
) -> torch.Tensor | None:
    base_conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    event_conditioning = batch_event_mode_conditioning(batch, event_conditioner, device)
    return combine_conditioning(base_conditioning, event_conditioning)


def _checkpoint_event_audit_json(metrics: dict[str, Any]) -> str | None:
    event_conditioning = metrics.get("event_conditioning")
    if not isinstance(event_conditioning, dict):
        return None
    value = event_conditioning.get("event_mode_audit_json")
    return str(value) if value else None


def _event_class_set_from_checkpoint(event_conditioning: object) -> str:
    if not isinstance(event_conditioning, dict):
        return "stable8"
    class_names = tuple(str(value) for value in event_conditioning.get("class_names", []))
    stable = (
        "sustain_open::none",
        "sustain_close::none",
        "transition_close::early",
        "transition_close::middle",
        "transition_close::late",
        "transition_open::early",
        "transition_open::middle",
        "transition_open::late",
    )
    return "stable8" if class_names == stable else "all_observed"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
