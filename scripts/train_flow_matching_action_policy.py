#!/usr/bin/env python3
"""Train a conditional flow-matching action policy for Gate 3.9a."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.metrics.window_metrics import (  # noqa: E402
    merge_window_metric_records,
    per_window_action_metrics,
    window_metadata_records,
)
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _checkpoint_event_classes,
    _conditioner_from_metrics,
    _load_event_probe,
)
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    _batch_conditioning,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _resolve_visual_token_config,
    _seed_everything,
    _split_indices,
)
from train_predicted_event_mixture_action_head import (  # noqa: E402
    _add_metric_values,
    _apply_future_input_control,
    _cvae_config,
    _finalize_metric_values,
    _load_event_label_records,
    _predicted_event_future_inputs,
    _probe_summary,
    _sample_feature_dim,
    _weighted_action_loss,
)
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


class FlowMatchingActionPolicy(nn.Module):
    """Small conditional rectified-flow model over action chunks."""

    def __init__(
        self,
        *,
        context_dim: int,
        visual_dim: int,
        action_dim: int,
        horizon: int,
        conditioning_dim: int,
        condition_mode: str,
        motion_dim: int,
        sample_feature_dim: int,
        hidden_dims: tuple[int, ...],
        token_dim: int,
        num_heads: int,
        temporal_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if condition_mode not in {"direct_visual", "geomoco"}:
            raise ValueError("condition_mode must be one of: direct_visual, geomoco")
        if context_dim <= 0 or visual_dim <= 0 or action_dim <= 0 or horizon <= 0:
            raise ValueError("context/visual/action dimensions and horizon must be positive")
        if conditioning_dim < 0 or motion_dim < 0 or sample_feature_dim < 0:
            raise ValueError("conditioning, motion, and sample feature dims must be non-negative")
        if token_dim <= 0 or num_heads <= 0:
            raise ValueError("token_dim and num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.context_dim = context_dim
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.conditioning_dim = conditioning_dim
        self.condition_mode = condition_mode
        self.motion_dim = motion_dim
        self.sample_feature_dim = sample_feature_dim
        self.token_dim = token_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim + conditioning_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.visual_encoder = _norm_mlp(visual_dim, (token_dim,), token_dim, dropout=dropout)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.step_position = nn.Parameter(torch.zeros(horizon, token_dim))

        self.sample_encoder = (
            _norm_mlp(
                motion_dim + sample_feature_dim,
                (token_dim,),
                token_dim,
                dropout=dropout,
            )
            if condition_mode == "geomoco"
            else None
        )
        self.sample_attention = (
            nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
            if condition_mode == "geomoco"
            else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.output = _norm_mlp(token_dim, hidden_dims, action_dim, dropout=dropout)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        visual: torch.Tensor,
        conditioning: torch.Tensor | None,
        future_inputs: torch.Tensor | None = None,
        sample_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_actions.ndim != 3:
            raise ValueError(f"noisy_actions must be [B,H,A], got {noisy_actions.shape}")
        batch_size = int(noisy_actions.shape[0])
        if noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                "noisy_actions must be [B,H,A] with "
                f"H={self.horizon}, A={self.action_dim}; got {noisy_actions.shape}"
            )
        condition_token = self._condition_token(
            context,
            visual,
            conditioning,
            future_inputs,
            sample_features,
        )
        if time.ndim == 0:
            time = time.reshape(1).expand(batch_size)
        if time.ndim == 2 and time.shape[-1] == 1:
            time = time.squeeze(-1)
        if time.shape != (batch_size,):
            raise ValueError(f"time must be [B], got {time.shape}")

        action_tokens = self.action_encoder(noisy_actions.to(dtype=context.dtype))
        time_token = self.time_encoder(time.to(dtype=context.dtype).reshape(batch_size, 1))
        step_position = self.step_position.to(dtype=context.dtype).unsqueeze(0)
        tokens = action_tokens + condition_token.unsqueeze(1) + time_token.unsqueeze(1) + step_position
        tokens = self.temporal_encoder(tokens)
        return self.output(tokens.reshape(batch_size * self.horizon, -1)).reshape(
            batch_size,
            self.horizon,
            self.action_dim,
        )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        visual: torch.Tensor,
        conditioning: torch.Tensor | None,
        future_inputs: torch.Tensor | None,
        sample_features: torch.Tensor | None,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch_size = int(context.shape[0])
        actions = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            dtype=context.dtype,
            device=context.device,
        )
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            time = torch.full((batch_size,), float(step) * dt, dtype=context.dtype, device=context.device)
            velocity = self(
                actions,
                time,
                context,
                visual,
                conditioning,
                future_inputs,
                sample_features,
            )
            actions = actions + dt * velocity
        return actions

    def _condition_token(
        self,
        context: torch.Tensor,
        visual: torch.Tensor,
        conditioning: torch.Tensor | None,
        future_inputs: torch.Tensor | None,
        sample_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"context must be [B,{self.context_dim}], got {context.shape}")
        if visual.ndim != 2 or visual.shape[-1] != self.visual_dim:
            raise ValueError(f"visual must be [B,{self.visual_dim}], got {visual.shape}")
        if conditioning is None:
            if self.conditioning_dim != 0:
                raise ValueError("conditioning is required")
            base = context
        else:
            if conditioning.shape != (context.shape[0], self.conditioning_dim):
                raise ValueError(
                    "conditioning must be [B,C] with "
                    f"C={self.conditioning_dim}; got {conditioning.shape}"
                )
            base = torch.cat([context, conditioning.to(dtype=context.dtype)], dim=-1)
        token = self.context_encoder(base) + self.visual_encoder(visual.to(dtype=context.dtype))
        if self.condition_mode == "direct_visual":
            return token
        if future_inputs is None:
            raise ValueError("geomoco condition mode requires future_inputs")
        if future_inputs.ndim != 3 or future_inputs.shape[-1] != self.motion_dim:
            raise ValueError(
                "future_inputs must be [B,K,M] with "
                f"M={self.motion_dim}; got {future_inputs.shape}"
            )
        sample_parts = [future_inputs.to(dtype=context.dtype)]
        if self.sample_feature_dim > 0:
            if sample_features is None:
                raise ValueError("geomoco condition mode requires sample_features")
            if sample_features.shape[:2] != future_inputs.shape[:2]:
                raise ValueError(
                    "sample_features must match future_inputs [B,K]: "
                    f"{sample_features.shape[:2]} vs {future_inputs.shape[:2]}"
                )
            if sample_features.shape[-1] != self.sample_feature_dim:
                raise ValueError(
                    "sample_features dim must be "
                    f"{self.sample_feature_dim}; got {sample_features.shape[-1]}"
                )
            sample_parts.append(sample_features.to(dtype=context.dtype))
        if self.sample_encoder is None or self.sample_attention is None:
            raise RuntimeError("geomoco sample modules are missing")
        sample_tokens = self.sample_encoder(torch.cat(sample_parts, dim=-1))
        attended, _ = self.sample_attention(token.unsqueeze(1), sample_tokens, sample_tokens)
        return token + attended.squeeze(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Gate 3.9a flow-matching action policy."
    )
    parser.add_argument("--checkpoint", required=True, help="Event-conditioned cVAE model.pt.")
    parser.add_argument("--event-probe-checkpoint", required=True, help="Gate 3.1b probe model.pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--condition-mode",
        required=True,
        choices=["direct_visual", "geomoco"],
        help="direct_visual uses DINO/context only; geomoco adds predicted event-mixture samples.",
    )
    parser.add_argument("--windows-jsonl", nargs="+", default=None)
    parser.add_argument("--visual-feature-cache", default=None)
    parser.add_argument("--event-mode-audit-json", default=None)
    parser.add_argument("--event-top-m", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-feature-mode",
        default="event_rank_prob",
        choices=["none", "event_only", "rank_prob_only", "event_rank_prob", "shuffled_event_rank_prob"],
    )
    parser.add_argument(
        "--future-input-control",
        default="real",
        choices=["real", "mean_repeated", "context_only"],
    )
    parser.add_argument(
        "--event-candidate-policy",
        default="topk",
        choices=["topk", "transition_reserve"],
    )
    parser.add_argument("--transition-reserve-threshold", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="512,512")
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--num-eval-passes", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        default="mse",
        choices=["mse", "transition_mse", "gripper_mse", "flow_mse"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Load a trained flow policy model.pt and only run repeated validation.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON for --eval-only-checkpoint metrics.",
    )
    parser.add_argument(
        "--per-window-output-jsonl",
        default=None,
        help="Optional eval-only JSONL with one validation row per window for bootstrap CI.",
    )
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

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=motion_mode,
        visual_feature_cache_path=visual_feature_cache,
    )
    spec = dataset.spec()
    if spec.visual_dim <= 0:
        raise ValueError("flow policy requires a visual feature cache")
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
    sample_feature_dim = (
        _sample_feature_dim(args.sample_feature_mode, event_classes)
        if args.condition_mode == "geomoco"
        else 0
    )
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, split_by)
    event_labels = (
        _load_event_label_records(event_audit_json)
        if event_audit_json is not None
        else None
    )

    if args.eval_only_checkpoint is not None:
        _run_eval_only(
            args,
            cvae_checkpoint,
            cvae_metrics,
            dataset,
            spec,
            conditioner,
            event_classes,
            visual_token_config,
            event_probe,
            probe_metrics,
            probe_conditioner,
            val_indices,
            event_labels,
            device,
        )
        return

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_summary(
                    args,
                    cvae_path,
                    spec,
                    motion_mode,
                    conditioner,
                    event_classes,
                    probe_metrics,
                    train_indices,
                    val_indices,
                    visual_token_config,
                    sample_feature_dim,
                    event_audit_json,
                ),
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
    for module in (cvae, event_probe):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    model = FlowMatchingActionPolicy(
        context_dim=spec.context_dim,
        visual_dim=spec.visual_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
        conditioning_dim=conditioner.dim,
        condition_mode=args.condition_mode,
        motion_dim=spec.motion_dim,
        sample_feature_dim=sample_feature_dim,
        hidden_dims=hidden_dims,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_metric = float("inf")
    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            cvae,
            event_probe,
            train_loader,
            optimizer,
            device,
            conditioner,
            probe_conditioner,
            event_labels,
            event_classes=event_classes,
            probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
            probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
            args=args,
        )
        _seed_everything(args.seed + 10_000 + epoch)
        val_metrics = _evaluate(
            model,
            cvae,
            event_probe,
            val_loader,
            device,
            conditioner,
            probe_conditioner,
            event_labels,
            event_classes=event_classes,
            probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
            probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
            args=args,
        )
        row: dict[str, float | int | None] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate = val_metrics.get(args.selection_metric)
        if candidate is not None and float(candidate) < best_metric:
            best_metric = float(candidate)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    pass_metrics: list[dict[str, float | None]] = []
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(args.seed + 20_000 + eval_pass)
        pass_metrics.append(
            _evaluate(
                model,
                cvae,
                event_probe,
                val_loader,
                device,
                conditioner,
                probe_conditioner,
                event_labels,
                event_classes=event_classes,
                probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
                probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
                args=args,
            )
        )
    final_metrics = _mean_metrics(pass_metrics)

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
        "seed": args.seed,
        "split_by": split_by,
        "motion_mode": motion_mode,
        "input_mode": "flow_matching_action_policy",
        "condition_mode": args.condition_mode,
        "event_top_m": args.event_top_m,
        "num_samples": args.num_samples,
        "event_candidate_policy": args.event_candidate_policy,
        "transition_reserve_threshold": args.transition_reserve_threshold,
        "sample_feature_mode": args.sample_feature_mode,
        "future_input_control": args.future_input_control,
        "sample_feature_dim": sample_feature_dim,
        "eval_steps": args.eval_steps,
        "num_eval_passes": args.num_eval_passes,
        "selection_metric": args.selection_metric,
        "conditioning": conditioner.to_dict(),
        "cvae_event_classes": list(event_classes),
        "checkpoint": str(cvae_path),
        "visual_feature_cache": str(Path(visual_feature_cache).expanduser()),
        "visual_token_config": visual_token_config,
        "event_probe_checkpoint": str(Path(args.event_probe_checkpoint).expanduser().resolve()),
        "event_mode_audit_json": str(Path(event_audit_json).expanduser().resolve())
        if event_audit_json is not None
        else None,
        "event_probe": _probe_summary(probe_metrics),
        "cvae_config": _cvae_config(cvae_metrics),
        "model_config": _model_config(args, hidden_dims, spec, conditioner, sample_feature_dim),
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_metric": args.selection_metric,
        "best_selection_value": best_metric if best_state is not None else None,
        "pass_metrics": pass_metrics,
        "final_action_metrics": final_metrics,
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
                    "output_dir": str(output_dir),
                    "best_epoch": best_epoch,
                    "best_selection_value": best_metric if best_state is not None else None,
                    "final_action_metrics": final_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _run_epoch(
    model: FlowMatchingActionPolicy,
    cvae: nn.Module,
    event_probe: nn.Module,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    conditioner: Any,
    probe_conditioner: Any,
    event_labels: dict[str, Any] | None,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    if loader is None:
        return {"flow_mse": None}
    model.train()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        context, visual, conditioning, actions, future_inputs, sample_features = _prepare_batch(
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            event_classes=event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            args=args,
        )
        noise = torch.randn_like(actions)
        time = torch.rand(actions.shape[0], dtype=actions.dtype, device=actions.device)
        interp = time.reshape(-1, 1, 1)
        noisy_actions = (1.0 - interp) * noise + interp * actions
        target_velocity = actions - noise

        optimizer.zero_grad(set_to_none=True)
        velocity = model(
            noisy_actions,
            time,
            context,
            visual,
            conditioning,
            future_inputs,
            sample_features,
        )
        loss = F.mse_loss(velocity, target_velocity)
        loss.backward()
        optimizer.step()

        metrics = {"flow_mse": float(loss.detach().cpu())}
        _add_metric_values(totals, counts, metrics, int(actions.shape[0]))
    return _finalize_metric_values(totals, counts)


@torch.no_grad()
def _evaluate(
    model: FlowMatchingActionPolicy,
    cvae: nn.Module,
    event_probe: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: Any,
    probe_conditioner: Any,
    event_labels: dict[str, Any] | None,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    args: argparse.Namespace,
    per_window_records: list[dict[str, Any]] | None = None,
) -> dict[str, float | None]:
    if loader is None:
        return {"mse": None}
    model.eval()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        context, visual, conditioning, actions, future_inputs, sample_features = _prepare_batch(
            cvae,
            event_probe,
            batch,
            device,
            conditioner,
            probe_conditioner,
            event_classes=event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            args=args,
        )
        pred_actions = model.sample(
            context,
            visual,
            conditioning,
            future_inputs,
            sample_features,
            num_steps=args.eval_steps,
        )
        action_metric_values = action_metrics(pred_actions, actions)
        if per_window_records is not None:
            per_window_records.extend(
                merge_window_metric_records(
                    window_metadata_records(batch, event_labels),
                    per_window_action_metrics(pred_actions, actions),
                )
            )
        _, group_metrics, group_counts = _weighted_action_loss(
            pred_actions,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=1.0,
        )
        flow_mse = _flow_mse_for_eval(
            model,
            context,
            visual,
            conditioning,
            future_inputs,
            sample_features,
            actions,
        )
        _add_metric_values(totals, counts, action_metric_values, int(actions.shape[0]))
        _add_metric_values(totals, counts, group_metrics, group_counts)
        _add_metric_values(totals, counts, {"flow_mse": flow_mse}, int(actions.shape[0]))
    return _finalize_metric_values(totals, counts)


def _prepare_batch(
    cvae: nn.Module,
    event_probe: nn.Module,
    batch: dict[str, object],
    device: torch.device,
    conditioner: Any,
    probe_conditioner: Any,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    args: argparse.Namespace,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    context = batch["context"].to(device)
    visual = _batch_visual(batch, device)
    actions = batch["actions"].to(device)
    conditioning = _batch_conditioning(batch, conditioner, device, include_visual=False)
    future_inputs: torch.Tensor | None = None
    sample_features: torch.Tensor | None = None
    if args.condition_mode == "geomoco":
        future_inputs, sample_features = _predicted_event_future_inputs(
            cvae,
            event_probe,
            batch,
            context,
            conditioning,
            device,
            probe_conditioner,
            event_classes=event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=args.event_top_m,
            num_samples=args.num_samples,
            event_candidate_policy=args.event_candidate_policy,
            transition_reserve_threshold=args.transition_reserve_threshold,
            sample_feature_mode=args.sample_feature_mode,
        )
        future_inputs, sample_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            args.future_input_control,
        )
    return context, visual, conditioning, actions, future_inputs, sample_features


def _flow_mse_for_eval(
    model: FlowMatchingActionPolicy,
    context: torch.Tensor,
    visual: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor | None,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
) -> float:
    noise = torch.randn_like(actions)
    time = torch.rand(actions.shape[0], dtype=actions.dtype, device=actions.device)
    interp = time.reshape(-1, 1, 1)
    noisy_actions = (1.0 - interp) * noise + interp * actions
    target_velocity = actions - noise
    velocity = model(
        noisy_actions,
        time,
        context,
        visual,
        conditioning,
        future_inputs,
        sample_features,
    )
    return float(F.mse_loss(velocity, target_velocity).detach().cpu())


def _validate_args(args: argparse.Namespace) -> None:
    if args.event_top_m <= 0:
        raise ValueError("--event-top-m must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.event_top_m > args.num_samples:
        raise ValueError("--event-top-m cannot exceed --num-samples")
    if args.transition_reserve_threshold < 0.0:
        raise ValueError("--transition-reserve-threshold must be non-negative")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.token_dim <= 0:
        raise ValueError("--token-dim must be positive")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be positive")
    if args.token_dim % args.num_heads != 0:
        raise ValueError("--token-dim must be divisible by --num-heads")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be positive")
    if args.dropout < 0.0:
        raise ValueError("--dropout must be non-negative")
    if args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive")
    if args.num_eval_passes <= 0:
        raise ValueError("--num-eval-passes must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.condition_mode == "direct_visual" and args.future_input_control != "real":
        raise ValueError("direct_visual condition mode does not use future-input controls")
    if args.condition_mode == "geomoco" and args.future_input_control == "context_only":
        raise ValueError("use condition-mode direct_visual for no-prior flow")
    if args.output_json is not None and args.eval_only_checkpoint is None:
        raise ValueError("--output-json is only used with --eval-only-checkpoint")
    if args.per_window_output_jsonl is not None and args.eval_only_checkpoint is None:
        raise ValueError("--per-window-output-jsonl is only used with --eval-only-checkpoint")
    if args.per_window_output_jsonl is not None and args.num_eval_passes != 1:
        raise ValueError("--per-window-output-jsonl requires --num-eval-passes 1")


def _run_eval_only(
    args: argparse.Namespace,
    cvae_checkpoint: dict[str, Any],
    cvae_metrics: dict[str, Any],
    dataset: OracleActionWindowDataset,
    spec: Any,
    conditioner: Any,
    event_classes: tuple[str, ...],
    visual_token_config: dict[str, int],
    event_probe: nn.Module,
    probe_metrics: dict[str, Any],
    probe_conditioner: Any,
    val_indices: list[int],
    event_labels: dict[str, Any] | None,
    device: torch.device,
) -> None:
    checkpoint_path = Path(args.eval_only_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metrics = checkpoint["metrics"]
    config = metrics["model_config"]
    if str(config["condition_mode"]) != args.condition_mode:
        raise ValueError(
            "--condition-mode must match checkpoint: "
            f"{args.condition_mode} vs {config['condition_mode']}"
        )
    cvae = _load_model(
        cvae_checkpoint,
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        visual_token_config=visual_token_config,
        conditioner_dim=conditioner.dim + len(event_classes),
        device=device,
    )
    model = FlowMatchingActionPolicy(
        context_dim=int(config["context_dim"]),
        visual_dim=int(config["visual_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        conditioning_dim=int(config["conditioning_dim"]),
        condition_mode=str(config["condition_mode"]),
        motion_dim=int(config["motion_dim"]),
        sample_feature_dim=int(config["sample_feature_dim"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        token_dim=int(config["token_dim"]),
        num_heads=int(config["num_heads"]),
        temporal_layers=int(config["temporal_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cvae.eval()
    event_probe.eval()
    for module in (cvae, event_probe):
        for param in module.parameters():
            param.requires_grad_(False)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    pass_metrics: list[dict[str, float | None]] = []
    per_window_records: list[dict[str, Any]] | None = (
        [] if args.per_window_output_jsonl is not None else None
    )
    for eval_pass in range(args.num_eval_passes):
        _seed_everything(args.seed + 30_000 + eval_pass)
        pass_metrics.append(
            _evaluate(
                model,
                cvae,
                event_probe,
                val_loader,
                device,
                conditioner,
                probe_conditioner,
                event_labels,
                event_classes=event_classes,
                probe_class_names=tuple(str(name) for name in probe_metrics["probe"]["class_names"]),
                probe_input_variant=str(probe_metrics["probe"]["input_variant"]),
                args=args,
                per_window_records=per_window_records if eval_pass == 0 else None,
            )
        )
    mean_metrics = _mean_metrics(pass_metrics)
    output = {
        "checkpoint": str(checkpoint_path),
        "condition_mode": args.condition_mode,
        "eval_steps": args.eval_steps,
        "num_eval_passes": args.num_eval_passes,
        "seed": args.seed,
        "pass_metrics": pass_metrics,
        "mean_metrics": mean_metrics,
    }
    if args.per_window_output_jsonl is not None:
        per_window_path = Path(args.per_window_output_jsonl).expanduser().resolve()
        per_window_path.parent.mkdir(parents=True, exist_ok=True)
        assert per_window_records is not None
        per_window_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_window_records),
            encoding="utf-8",
        )
        output["per_window_output_jsonl"] = str(per_window_path)
    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(output, indent=2, ensure_ascii=False))


def _dry_run_summary(
    args: argparse.Namespace,
    cvae_path: Path,
    spec: Any,
    motion_mode: str,
    conditioner: Any,
    event_classes: tuple[str, ...],
    probe_metrics: dict[str, Any],
    train_indices: list[int],
    val_indices: list[int],
    visual_token_config: dict[str, int],
    sample_feature_dim: int,
    event_audit_json: str | Path | None,
) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "checkpoint": str(cvae_path),
        "event_probe_checkpoint": str(Path(args.event_probe_checkpoint).expanduser()),
        "dataset": spec.to_dict(),
        "motion_mode": motion_mode,
        "condition_mode": args.condition_mode,
        "conditioning": conditioner.to_dict(),
        "cvae_event_classes": list(event_classes),
        "probe_event_classes": list(probe_metrics["probe"]["class_names"]),
        "event_top_m": args.event_top_m,
        "num_samples": args.num_samples,
        "sample_feature_mode": args.sample_feature_mode,
        "sample_feature_dim": sample_feature_dim,
        "future_input_control": args.future_input_control,
        "event_candidate_policy": args.event_candidate_policy,
        "transition_reserve_threshold": args.transition_reserve_threshold,
        "eval_steps": args.eval_steps,
        "num_eval_passes": args.num_eval_passes,
        "event_mode_audit_json": str(Path(event_audit_json).expanduser().resolve())
        if event_audit_json is not None
        else None,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "visual_token_config": visual_token_config,
        "model_config": _model_config(
            args,
            _parse_hidden_dims(args.hidden_dims),
            spec,
            conditioner,
            sample_feature_dim,
        ),
    }


def _model_config(
    args: argparse.Namespace,
    hidden_dims: tuple[int, ...],
    spec: Any,
    conditioner: Any,
    sample_feature_dim: int,
) -> dict[str, Any]:
    return {
        "context_dim": int(spec.context_dim),
        "visual_dim": int(spec.visual_dim),
        "action_dim": int(spec.action_dim),
        "horizon": int(spec.horizon),
        "conditioning_dim": int(conditioner.dim),
        "condition_mode": args.condition_mode,
        "motion_dim": int(spec.motion_dim),
        "sample_feature_dim": sample_feature_dim,
        "hidden_dims": list(hidden_dims),
        "token_dim": args.token_dim,
        "num_heads": args.num_heads,
        "temporal_layers": args.temporal_layers,
        "dropout": args.dropout,
    }


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    output: dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = sum(float(value) for value in values) / len(values) if values else None
    return output


def _norm_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("hidden_dims must be positive")
        layers.extend(
            [
                nn.Linear(previous, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = hidden_dim
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


if __name__ == "__main__":
    main()
