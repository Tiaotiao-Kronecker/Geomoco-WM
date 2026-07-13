#!/usr/bin/env python3
"""Train a visual-conditioned cVAE future EEF-motion prior."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
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
from geomoco_wm.data.window_dataset import MOTION_MODES, OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402
from geomoco_wm.models.geomoco_cvae import (  # noqa: E402
    VisualConditionedGeoMoCoCVAE,
    gaussian_kl_divergence,
)
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _average_metrics,
    _batch_conditioning,
    _build_conditioner,
    _freeze_action_decoder,
    _load_action_decoder,
    _make_loader,
    _parse_hidden_dims,
    _prediction_metrics,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gate 2.4b visual-conditioned cVAE future-motion prior."
    )
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default="outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl",
    )
    parser.add_argument(
        "--visual-feature-cache",
        required=True,
        help="HDF5 visual feature cache aligned to windows.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visual_cvae_future_motion/gate2_4b_smoke",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument(
        "--beta-kl-start",
        type=float,
        default=None,
        help="Optional KL beta warmup start. Defaults to --beta-kl.",
    )
    parser.add_argument(
        "--beta-kl-warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs used to linearly warm beta from start to target.",
    )
    parser.add_argument(
        "--free-bits",
        type=float,
        default=0.0,
        help="Per-latent-dimension free bits / free nats threshold for KL.",
    )
    parser.add_argument("--prior-recon-weight", type=float, default=1.0)
    parser.add_argument("--action-aware-loss-weight", type=float, default=0.03)
    parser.add_argument(
        "--motion-mode",
        default="future_delta",
        choices=MOTION_MODES,
        help="Prediction target stored in the exported window dataset.",
    )
    parser.add_argument(
        "--action-decoder-checkpoint",
        default=None,
        help="Frozen oracle action decoder used for prior-mean action loss and metrics.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split-by", default="episode", choices=["window", "episode"])
    parser.add_argument(
        "--condition-on",
        default="suite_task",
        choices=["none", "suite", "task", "suite_task"],
    )
    parser.add_argument(
        "--event-mode-audit-json",
        default=None,
        help="Gate 3.1a event-mode labels used for event-conditioned cVAE branches.",
    )
    parser.add_argument(
        "--event-conditioning-mode",
        default="none",
        choices=["none", "oracle", "shuffled"],
        help="Append oracle or shuffled event-mode one-hot conditioning.",
    )
    parser.add_argument(
        "--event-class-set",
        default="stable8",
        choices=["stable8", "all_observed"],
        help="Event-mode class set used for event conditioning.",
    )
    parser.add_argument("--event-shuffle-seed", type=int, default=0)
    parser.add_argument("--visual-token-count", type=int, default=None)
    parser.add_argument("--visual-token-dim", type=int, default=None)
    parser.add_argument("--visual-query-dim", type=int, default=384)
    parser.add_argument("--visual-num-heads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.beta_kl < 0.0:
        raise ValueError("--beta-kl must be non-negative")
    beta_kl_start = args.beta_kl if args.beta_kl_start is None else args.beta_kl_start
    if beta_kl_start < 0.0:
        raise ValueError("--beta-kl-start must be non-negative")
    if args.beta_kl_warmup_epochs < 0:
        raise ValueError("--beta-kl-warmup-epochs must be non-negative")
    if args.free_bits < 0.0:
        raise ValueError("--free-bits must be non-negative")
    if args.prior_recon_weight < 0.0:
        raise ValueError("--prior-recon-weight must be non-negative")
    if args.action_aware_loss_weight < 0.0:
        raise ValueError("--action-aware-loss-weight must be non-negative")
    _seed_everything(args.seed)
    dataset = OracleActionWindowDataset(
        args.windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=args.motion_mode,
        visual_feature_cache_path=args.visual_feature_cache,
    )
    spec = dataset.spec()
    conditioner = _build_conditioner(dataset, args.condition_on)
    event_conditioner = load_event_mode_conditioner(
        args.event_mode_audit_json,
        mode=args.event_conditioning_mode,
        class_set=args.event_class_set,
        shuffle_seed=args.event_shuffle_seed,
    )
    visual_token_config = _resolve_visual_token_config(
        dataset,
        args.visual_token_count,
        args.visual_token_dim,
    )
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "dataset": spec.to_dict(),
                    "conditioning": conditioner.to_dict(),
                    "event_conditioning": event_conditioner.to_dict(),
                    "combined_conditioning_dim": conditioner.dim + event_conditioner.dim,
                    "visual_token_config": visual_token_config,
                    "latent_dim": args.latent_dim,
                    "beta_kl": args.beta_kl,
                    "beta_kl_start": beta_kl_start,
                    "beta_kl_warmup_epochs": args.beta_kl_warmup_epochs,
                    "free_bits": args.free_bits,
                    "prior_recon_weight": args.prior_recon_weight,
                    "action_aware_loss_weight": args.action_aware_loss_weight,
                    "motion_mode": args.motion_mode,
                },
                indent=2,
            )
        )
        return

    device = _resolve_device(args.device)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, args.split_by)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    model = VisualConditionedGeoMoCoCVAE(
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_dim=int(visual_token_config["visual_token_dim"]),
        visual_token_count=int(visual_token_config["visual_token_count"]),
        conditioning_dim=conditioner.dim + event_conditioner.dim,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        query_dim=args.visual_query_dim,
        num_heads=args.visual_num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    action_decoder = None
    action_decoder_config = None
    if args.action_aware_loss_weight > 0.0 or args.action_decoder_checkpoint:
        if not args.action_decoder_checkpoint:
            raise ValueError(
                "--action-decoder-checkpoint is required when action-aware loss is positive"
            )
        action_decoder, action_decoder_config = _load_action_decoder(
            args.action_decoder_checkpoint,
            device,
        )
        if action_decoder_config["motion_mode"] != args.motion_mode:
            raise ValueError(
                "action decoder motion mode must match --motion-mode: "
                f"{action_decoder_config['motion_mode']} vs {args.motion_mode}"
            )
        _freeze_action_decoder(action_decoder)

    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_beta_kl = _beta_for_epoch(
            epoch,
            target_beta=args.beta_kl,
            start_beta=beta_kl_start,
            warmup_epochs=args.beta_kl_warmup_epochs,
        )
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer,
            conditioner,
            event_conditioner,
            action_decoder,
            epoch_beta_kl,
            args.free_bits,
            args.prior_recon_weight,
            args.action_aware_loss_weight,
            args.motion_mode,
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            loss_fn,
            conditioner,
            event_conditioner,
            action_decoder,
            epoch_beta_kl,
            args.free_bits,
            args.prior_recon_weight,
            args.action_aware_loss_weight,
            args.motion_mode,
        )
        row = {"epoch": epoch, "beta_kl": epoch_beta_kl}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    prior_mean_action_metrics = None
    posterior_action_metrics = None
    if action_decoder is not None:
        prior_mean_action_metrics = _evaluate_actions(
            model,
            action_decoder,
            val_loader,
            device,
            conditioner,
            event_conditioner,
            mode="prior_mean",
        )
        posterior_action_metrics = _evaluate_actions(
            model,
            action_decoder,
            val_loader,
            device,
            conditioner,
            event_conditioner,
            mode="posterior",
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": spec.to_dict(),
        "device": str(device),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dims": list(hidden_dims),
        "latent_dim": args.latent_dim,
        "beta_kl": args.beta_kl,
        "beta_kl_start": beta_kl_start,
        "beta_kl_warmup_epochs": args.beta_kl_warmup_epochs,
        "free_bits": args.free_bits,
        "prior_recon_weight": args.prior_recon_weight,
        "action_aware_loss_weight": args.action_aware_loss_weight,
        "motion_mode": args.motion_mode,
        "seed": args.seed,
        "split_by": args.split_by,
        "conditioning": conditioner.to_dict(),
        "event_conditioning": event_conditioner.to_dict(),
        "combined_conditioning_dim": conditioner.dim + event_conditioner.dim,
        "visual_feature_cache": str(Path(args.visual_feature_cache).expanduser()),
        "visual_token_config": visual_token_config,
        "visual_query_dim": args.visual_query_dim,
        "visual_num_heads": args.visual_num_heads,
        "history": history,
        "final": history[-1] if history else {},
        "action_decoder_checkpoint": str(Path(args.action_decoder_checkpoint).expanduser())
        if args.action_decoder_checkpoint
        else None,
        "action_decoder_config": action_decoder_config,
        "prior_mean_action_metrics": prior_mean_action_metrics,
        "posterior_action_metrics": posterior_action_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "final": metrics["final"],
                    "prior_mean_action_metrics": prior_mean_action_metrics,
                    "posterior_action_metrics": posterior_action_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _run_epoch(
    model: VisualConditionedGeoMoCoCVAE,
    loader: DataLoader | None,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    action_decoder: ActionDecoder | None,
    beta_kl: float,
    free_bits: float,
    prior_recon_weight: float,
    action_aware_loss_weight: float,
    motion_mode: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    model.train()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        visual = _batch_visual(batch, device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(context, visual, motion, conditioning)
        losses = _losses(
            output,
            motion,
            context,
            batch,
            device,
            loss_fn,
            action_decoder,
            beta_kl,
            free_bits,
            prior_recon_weight,
            action_aware_loss_weight,
        )
        losses["loss"].backward()
        optimizer.step()
        batch_metrics = _batch_metrics(output, motion, motion_mode)
        batch_metrics.update({key: float(value.detach().cpu()) for key, value in losses.items()})
        batch_size = int(context.shape[0])
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate(
    model: VisualConditionedGeoMoCoCVAE,
    loader: DataLoader | None,
    device: torch.device,
    loss_fn: nn.Module,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    action_decoder: ActionDecoder | None,
    beta_kl: float,
    free_bits: float,
    prior_recon_weight: float,
    action_aware_loss_weight: float,
    motion_mode: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None}
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        visual = _batch_visual(batch, device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        output = model(context, visual, motion, conditioning)
        losses = _losses(
            output,
            motion,
            context,
            batch,
            device,
            loss_fn,
            action_decoder,
            beta_kl,
            free_bits,
            prior_recon_weight,
            action_aware_loss_weight,
        )
        batch_metrics = _batch_metrics(output, motion, motion_mode)
        batch_metrics.update({key: float(value.cpu()) for key, value in losses.items()})
        batch_size = int(context.shape[0])
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


def _losses(
    output: object,
    motion: torch.Tensor,
    context: torch.Tensor,
    batch: dict[str, object],
    device: torch.device,
    loss_fn: nn.Module,
    action_decoder: ActionDecoder | None,
    beta_kl: float,
    free_bits: float,
    prior_recon_weight: float,
    action_aware_loss_weight: float,
) -> dict[str, torch.Tensor]:
    posterior_recon_loss = loss_fn(output.posterior_reconstruction, motion)
    prior_recon_loss = loss_fn(output.prior_mean_reconstruction, motion)
    kl_loss = gaussian_kl_divergence(
        output.posterior_mean,
        output.posterior_logvar,
        output.prior_mean,
        output.prior_logvar,
        free_bits=free_bits,
    )
    action_loss = torch.zeros((), dtype=motion.dtype, device=motion.device)
    if action_decoder is not None and action_aware_loss_weight > 0.0:
        actions = batch["actions"].to(device)
        pred_actions = action_decoder(context, output.prior_mean_reconstruction)
        action_loss = loss_fn(pred_actions, actions)
    loss = (
        posterior_recon_loss
        + prior_recon_weight * prior_recon_loss
        + beta_kl * kl_loss
        + action_aware_loss_weight * action_loss
    )
    return {
        "posterior_recon_loss": posterior_recon_loss,
        "prior_recon_loss": prior_recon_loss,
        "kl_loss": kl_loss,
        "raw_kl_loss": gaussian_kl_divergence(
            output.posterior_mean,
            output.posterior_logvar,
            output.prior_mean,
            output.prior_logvar,
            free_bits=0.0,
        ),
        "prior_action_loss": action_loss,
        "loss": loss,
    }


def _beta_for_epoch(
    epoch: int,
    *,
    target_beta: float,
    start_beta: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0:
        return target_beta
    progress = min(1.0, max(0.0, epoch / warmup_epochs))
    return start_beta + (target_beta - start_beta) * progress


def _batch_metrics(output: object, motion: torch.Tensor, motion_mode: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(
        _prefix_metrics(
            "posterior",
            _prediction_metrics(output.posterior_reconstruction, motion, motion_mode),
        )
    )
    metrics.update(
        _prefix_metrics(
            "prior",
            _prediction_metrics(output.prior_mean_reconstruction, motion, motion_mode),
        )
    )
    return metrics


@torch.no_grad()
def _evaluate_actions(
    model: VisualConditionedGeoMoCoCVAE,
    action_decoder: ActionDecoder,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    *,
    mode: str,
) -> dict[str, float | None]:
    if loader is None:
        return {"mse": None, "mae": None}
    model.eval()
    action_decoder.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        actions = batch["actions"].to(device)
        visual = _batch_visual(batch, device)
        conditioning = _combined_batch_conditioning(batch, conditioner, event_conditioner, device)
        if mode == "prior_mean":
            pred_motion = model.prior_mean_prediction(context, visual, conditioning)
        elif mode == "posterior":
            pred_motion = model(context, visual, motion, conditioning).posterior_reconstruction
        else:
            raise ValueError("mode must be one of: prior_mean, posterior")
        pred_actions = action_decoder(context, pred_motion)
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions, actions)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


def _batch_visual(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    visual = batch.get("visual")
    if not isinstance(visual, torch.Tensor):
        raise ValueError("visual-conditioned cVAE requires batch['visual']")
    return visual.to(device=device, dtype=torch.float32)


def _combined_batch_conditioning(
    batch: dict[str, object],
    conditioner: CategoricalConditioner,
    event_conditioner: EventModeConditioner,
    device: torch.device,
) -> torch.Tensor | None:
    base_conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    event_conditioning = batch_event_mode_conditioning(batch, event_conditioner, device)
    return combine_conditioning(base_conditioning, event_conditioning)


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
