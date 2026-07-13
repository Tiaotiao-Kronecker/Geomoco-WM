#!/usr/bin/env python3
"""Train action heads over predicted event-mixture cVAE samples."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

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
from geomoco_wm.metrics.window_metrics import (  # noqa: E402
    merge_window_metric_records,
    per_window_action_metrics,
    window_metadata_records,
)
from geomoco_wm.models.geomoco_cvae import VisualConditionedGeoMoCoCVAE  # noqa: E402
from geomoco_wm.models.motion_prior_action_head import MotionPriorActionHead  # noqa: E402
from evaluate_predicted_event_cvae_mixture import (  # noqa: E402
    _checkpoint_event_audit_json,
    _checkpoint_event_classes,
    _conditioner_from_metrics,
    _event_one_hot,
    _load_event_probe,
    _sample_rank_mixture,
)
from evaluate_visual_cvae_samples import _load_model  # noqa: E402
from train_event_mode_probe import EventModeProbeNet, _batch_features  # noqa: E402
from train_future_motion_predictor import (  # noqa: E402
    CategoricalConditioner,
    _batch_conditioning,
    _make_loader,
    _parse_hidden_dims,
    _resolve_device,
    _resolve_visual_token_config,
    _split_indices,
)
from train_motion_prior_action_head import _freeze_module  # noqa: E402
from train_visual_cvae_future_motion import _batch_visual  # noqa: E402


GRIPPER_ROUTE_FAMILIES = ("sustain", "transition_close", "transition_open")
GRIPPER_STEP_CLASSES = ("sustain", "close", "open")
GRIPPER_BOUNDARY_STEP_CLASSES = ("no_boundary", "close_start", "open_start")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Gate 3.1e action heads over predicted event-mixture samples."
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-top-m", type=int, default=4)
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
    parser.add_argument(
        "--sample-feature-mode",
        default="none",
        choices=[
            "none",
            "event_only",
            "rank_prob_only",
            "event_rank_prob",
            "shuffled_event_rank_prob",
        ],
        help="Optional per-sample metadata passed to the action head.",
    )
    parser.add_argument(
        "--future-input-control",
        default="real",
        choices=["real", "mean_repeated", "context_only"],
        help=(
            "Control how predicted event-mixture future samples are exposed to "
            "the action head. real uses the sampled set, mean_repeated repeats "
            "the per-window sample mean, and context_only removes motion-prior "
            "inputs while preserving the decoder architecture."
        ),
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="512,512")
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument(
        "--set-aggregator",
        default="context_attention",
        choices=["mean_pool", "context_attention", "multi_query_attention"],
    )
    parser.add_argument("--set-query-count", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--event-mode-audit-json",
        default=None,
        help="Defaults to the cVAE checkpoint event-mode audit JSON.",
    )
    parser.add_argument(
        "--loss-weight-mode",
        default="none",
        choices=["none", "transition"],
        help="Optionally upweight true transition event windows in the action loss.",
    )
    parser.add_argument("--transition-loss-weight", type=float, default=4.0)
    parser.add_argument(
        "--aux-gripper-loss-weight",
        type=float,
        default=0.0,
        help="Enable an auxiliary future-gripper head when positive.",
    )
    parser.add_argument(
        "--gripper-residual-mode",
        default="none",
        choices=["none", "event_family"],
        help="Optional event-routed gripper residual branch.",
    )
    parser.add_argument(
        "--gripper-residual-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the routed action loss when gripper residual routing is enabled.",
    )
    parser.add_argument(
        "--gripper-route-loss-weight",
        type=float,
        default=0.0,
        help="Weight for event-family route cross-entropy supervision.",
    )
    parser.add_argument(
        "--gripper-step-residual-mode",
        default="none",
        choices=["none", "event_step"],
        help="Optional per-step gripper residual branch.",
    )
    parser.add_argument(
        "--gripper-step-residual-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the step-routed action loss.",
    )
    parser.add_argument(
        "--gripper-step-loss-weight",
        type=float,
        default=0.0,
        help="Weight for per-step gripper command-state cross entropy.",
    )
    parser.add_argument(
        "--gripper-step-target-mode",
        default="command_state",
        choices=["command_state", "boundary_start"],
        help=(
            "Supervision target for the per-step gripper head. command_state "
            "uses action command signs; boundary_start uses close_step/open_step "
            "from the event-mode audit JSON."
        ),
    )
    parser.add_argument(
        "--gripper-step-positive-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Class weight for nonzero per-step gripper labels in the CE term. "
            "For boundary_start this upweights close_start/open_start steps."
        ),
    )
    parser.add_argument(
        "--gripper-step-residual-blend",
        default="all_classes",
        choices=["all_classes", "positive_only"],
        help=(
            "How to blend per-step gripper residuals. positive_only ignores "
            "class 0 residuals, which makes boundary_start residuals local to "
            "close/open positive probability."
        ),
    )
    parser.add_argument(
        "--gripper-step-oracle-boundary-residual-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an oracle-boundary diagnostic residual loss. When "
            "positive with boundary_start targets, the model residual is applied "
            "only at oracle close_step/open_step locations."
        ),
    )
    parser.add_argument(
        "--gripper-boundary-index-mode",
        default="none",
        choices=["none", "boundary_index"],
        help=(
            "Optional window-level close/open step-index localizer. It predicts "
            "one close index and one open index with an extra no-event class."
        ),
    )
    parser.add_argument(
        "--gripper-boundary-index-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the close/open boundary step-index CE objective.",
    )
    parser.add_argument(
        "--gripper-trajectory-residual-mode",
        default="none",
        choices=["none", "temporal_mlp"],
        help=(
            "Optional temporal gripper residual branch. temporal_mlp predicts "
            "one gripper residual per action step without replacing the rest "
            "of the action policy."
        ),
    )
    parser.add_argument(
        "--gripper-trajectory-residual-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the trajectory-routed action loss.",
    )
    parser.add_argument(
        "--event-time-conditioning-mode",
        default="none",
        choices=["none", "soft_boundary"],
        help=(
            "Optional soft close/open event-time latent for the temporal action "
            "decoder. soft_boundary predicts close/open distributions over "
            "H steps plus no-event and feeds them as decoder conditioning."
        ),
    )
    parser.add_argument(
        "--event-time-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the close/open event-time CE auxiliary loss.",
    )
    parser.add_argument(
        "--temporal-action-decoder-mode",
        default="none",
        choices=["none", "sequence_mlp", "temporal_transformer"],
        help=(
            "Optional joint temporal action-sequence decoder branch. "
            "sequence_mlp predicts each step from a context token plus step query; "
            "temporal_transformer adds a small step-token TransformerEncoder "
            "before predicting the full [H,A] action chunk. Both preserve the "
            "base action head output for attribution."
        ),
    )
    parser.add_argument(
        "--temporal-action-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the temporal action-sequence decoder loss.",
    )
    parser.add_argument(
        "--flow-action-decoder-mode",
        default="none",
        choices=["none", "rectified_mlp"],
        help=(
            "Optional small residual flow decoder after temporal_actions. "
            "rectified_mlp predicts a full action residual sequence conditioned "
            "on the Gate 3.4 temporal action output."
        ),
    )
    parser.add_argument(
        "--flow-action-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the deployable flow_actions weighted action loss.",
    )
    parser.add_argument(
        "--flow-matching-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the rectified-flow velocity matching auxiliary loss.",
    )
    parser.add_argument(
        "--sample-score-mode",
        default="none",
        choices=["none", "action_regret"],
        help=(
            "Optional supervised set-wise sample scorer. action_regret adds a "
            "candidate-score head whose softmax weights aggregate the sample set."
        ),
    )
    parser.add_argument(
        "--sample-score-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the sample-score candidate-comparison auxiliary loss.",
    )
    parser.add_argument(
        "--sample-score-target",
        default="motion_regret",
        choices=["motion_regret", "temporal_action_regret"],
        help=(
            "Target used to supervise sample scores. motion_regret compares "
            "samples to oracle future motion; temporal_action_regret compares "
            "single-sample action predictions to the action target."
        ),
    )
    parser.add_argument(
        "--sample-score-loss-type",
        default="soft_ce",
        choices=["soft_ce", "hard_ce", "combined"],
        help="Candidate-comparison loss over sample scores.",
    )
    parser.add_argument(
        "--sample-score-temperature",
        type=float,
        default=0.05,
        help="Temperature for converting candidate regrets into soft targets.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mse",
        choices=[
            "mse",
            "weighted_loss",
            "aux_replaced_mse",
            "routed_mse",
            "routed_weighted_loss",
            "step_routed_mse",
            "step_routed_weighted_loss",
            "oracle_step_routed_mse",
            "oracle_step_routed_weighted_loss",
            "boundary_index_pred_mse",
            "boundary_index_pred_weighted_loss",
            "trajectory_routed_mse",
            "trajectory_routed_weighted_loss",
            "trajectory_routed_transition_mse",
            "temporal_action_mse",
            "temporal_action_weighted_loss",
            "temporal_action_transition_mse",
            "flow_action_mse",
            "flow_action_weighted_loss",
            "flow_action_transition_mse",
            "flow_action_flow_matching_loss",
            "sample_score_loss",
            "sample_score_expected_regret",
            "sample_score_expected_vs_best_gap",
        ],
        help="Validation metric used for best-epoch selection.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument(
        "--train-sampling-mode",
        default="natural",
        choices=["natural", "transition_balanced"],
        help=(
            "Training sampler only. Validation remains the natural split. "
            "transition_balanced oversamples transition windows to the requested "
            "draw fraction."
        ),
    )
    parser.add_argument(
        "--transition-sampling-fraction",
        type=float,
        default=0.5,
        help="Target transition draw probability for transition_balanced training.",
    )
    parser.add_argument("--condition-on", default=None, choices=["none", "suite", "task", "suite_task"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
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
    event_labels = _load_training_event_labels(
        event_audit_json,
        loss_weight_mode=args.loss_weight_mode,
        gripper_residual_mode=args.gripper_residual_mode,
        gripper_route_loss_weight=args.gripper_route_loss_weight,
        gripper_step_residual_mode=args.gripper_step_residual_mode,
        gripper_boundary_index_mode=args.gripper_boundary_index_mode,
        gripper_trajectory_residual_mode=args.gripper_trajectory_residual_mode,
        event_time_conditioning_mode=args.event_time_conditioning_mode,
        temporal_action_decoder_mode=args.temporal_action_decoder_mode,
        flow_action_decoder_mode=args.flow_action_decoder_mode,
    )

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
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    sample_feature_dim = _sample_feature_dim(args.sample_feature_mode, event_classes)
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
                    "action_head_conditioning": conditioner.to_dict(),
                    "cvae_event_classes": list(event_classes),
                    "probe_event_classes": probe_metrics["probe"]["class_names"],
                    "event_top_m": args.event_top_m,
                    "num_samples": args.num_samples,
                    "event_candidate_policy": args.event_candidate_policy,
                    "transition_reserve_threshold": args.transition_reserve_threshold,
                    "sample_feature_mode": args.sample_feature_mode,
                    "future_input_control": args.future_input_control,
                    "sample_feature_dim": sample_feature_dim,
                    "loss_weight_mode": args.loss_weight_mode,
                    "transition_loss_weight": args.transition_loss_weight,
                    "aux_gripper_loss_weight": args.aux_gripper_loss_weight,
                    "gripper_residual_mode": args.gripper_residual_mode,
                    "gripper_residual_loss_weight": args.gripper_residual_loss_weight,
                    "gripper_route_loss_weight": args.gripper_route_loss_weight,
                    "gripper_step_residual_mode": args.gripper_step_residual_mode,
                    "gripper_step_residual_loss_weight": args.gripper_step_residual_loss_weight,
                    "gripper_step_loss_weight": args.gripper_step_loss_weight,
                    "gripper_step_target_mode": args.gripper_step_target_mode,
                    "gripper_step_positive_loss_weight": args.gripper_step_positive_loss_weight,
                    "gripper_step_residual_blend": args.gripper_step_residual_blend,
                    "gripper_step_oracle_boundary_residual_loss_weight": (
                        args.gripper_step_oracle_boundary_residual_loss_weight
                    ),
                    "gripper_boundary_index_mode": args.gripper_boundary_index_mode,
                    "gripper_boundary_index_loss_weight": (
                        args.gripper_boundary_index_loss_weight
                    ),
                    "gripper_trajectory_residual_mode": args.gripper_trajectory_residual_mode,
                    "gripper_trajectory_residual_loss_weight": (
                        args.gripper_trajectory_residual_loss_weight
                    ),
                    "event_time_conditioning_mode": args.event_time_conditioning_mode,
                    "event_time_loss_weight": args.event_time_loss_weight,
                    "temporal_action_decoder_mode": args.temporal_action_decoder_mode,
                    "temporal_action_loss_weight": args.temporal_action_loss_weight,
                    "flow_action_decoder_mode": args.flow_action_decoder_mode,
                    "flow_action_loss_weight": args.flow_action_loss_weight,
                    "flow_matching_loss_weight": args.flow_matching_loss_weight,
                    "sample_score_mode": args.sample_score_mode,
                    "sample_score_loss_weight": args.sample_score_loss_weight,
                    "sample_score_target": args.sample_score_target,
                    "sample_score_loss_type": args.sample_score_loss_type,
                    "sample_score_temperature": args.sample_score_temperature,
                    "selection_metric": args.selection_metric,
                    "train_sampling_mode": args.train_sampling_mode,
                    "transition_sampling_fraction": args.transition_sampling_fraction,
                    "train_sampling_summary": _train_sampling_summary(
                        dataset,
                        train_indices,
                        event_labels,
                        sampling_mode=args.train_sampling_mode,
                        transition_sampling_fraction=args.transition_sampling_fraction,
                    ),
                    "event_mode_audit_json": str(event_audit_json)
                    if event_audit_json is not None
                    else None,
                    "rank_sample_counts": list(
                        rank_uniform_counts(args.num_samples, args.event_top_m)
                    ),
                    "split_by": split_by,
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                    "visual_feature_cache": str(visual_feature_cache),
                    "visual_token_config": visual_token_config,
                    "model_config": _model_config(
                        args,
                        hidden_dims,
                        spec,
                        conditioner,
                        sample_feature_dim,
                    ),
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
    _freeze_module(cvae)
    _freeze_module(event_probe)
    model = MotionPriorActionHead(
        context_dim=spec.context_dim,
        motion_dim=spec.motion_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
        conditioning_dim=conditioner.dim,
        hidden_dims=hidden_dims,
        token_dim=args.token_dim,
        num_heads=args.num_heads,
        temporal_layers=args.temporal_layers,
        set_aggregator=args.set_aggregator,
        set_query_count=args.set_query_count,
        sample_feature_dim=sample_feature_dim,
        aux_gripper_head=args.aux_gripper_loss_weight > 0.0,
        gripper_residual_mode=args.gripper_residual_mode,
        gripper_route_count=len(GRIPPER_ROUTE_FAMILIES),
        gripper_step_residual_mode=args.gripper_step_residual_mode,
        gripper_step_class_count=len(_gripper_step_classes(args.gripper_step_target_mode)),
        gripper_step_residual_blend=args.gripper_step_residual_blend,
        gripper_boundary_index_mode=args.gripper_boundary_index_mode,
        gripper_trajectory_residual_mode=args.gripper_trajectory_residual_mode,
        event_time_conditioning_mode=args.event_time_conditioning_mode,
        temporal_action_decoder_mode=args.temporal_action_decoder_mode,
        flow_action_decoder_mode=args.flow_action_decoder_mode,
        sample_score_mode=args.sample_score_mode,
        dropout=args.dropout,
    ).to(device)

    train_loader = _make_train_loader(
        dataset,
        train_indices,
        args.batch_size,
        event_labels,
        sampling_mode=args.train_sampling_mode,
        transition_sampling_fraction=args.transition_sampling_fraction,
    )
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            event_top_m=args.event_top_m,
            num_samples=args.num_samples,
            event_candidate_policy=args.event_candidate_policy,
            transition_reserve_threshold=args.transition_reserve_threshold,
            sample_feature_mode=args.sample_feature_mode,
            future_input_control=args.future_input_control,
            loss_weight_mode=args.loss_weight_mode,
            transition_loss_weight=args.transition_loss_weight,
            aux_gripper_loss_weight=args.aux_gripper_loss_weight,
            gripper_residual_mode=args.gripper_residual_mode,
            gripper_residual_loss_weight=args.gripper_residual_loss_weight,
            gripper_route_loss_weight=args.gripper_route_loss_weight,
            gripper_step_residual_mode=args.gripper_step_residual_mode,
            gripper_step_residual_loss_weight=args.gripper_step_residual_loss_weight,
            gripper_step_loss_weight=args.gripper_step_loss_weight,
            gripper_step_target_mode=args.gripper_step_target_mode,
            gripper_step_positive_loss_weight=args.gripper_step_positive_loss_weight,
            gripper_step_oracle_boundary_residual_loss_weight=(
                args.gripper_step_oracle_boundary_residual_loss_weight
            ),
            gripper_boundary_index_mode=args.gripper_boundary_index_mode,
            gripper_boundary_index_loss_weight=args.gripper_boundary_index_loss_weight,
            gripper_trajectory_residual_mode=args.gripper_trajectory_residual_mode,
            gripper_trajectory_residual_loss_weight=(
                args.gripper_trajectory_residual_loss_weight
            ),
            event_time_conditioning_mode=args.event_time_conditioning_mode,
            event_time_loss_weight=args.event_time_loss_weight,
            temporal_action_decoder_mode=args.temporal_action_decoder_mode,
            temporal_action_loss_weight=args.temporal_action_loss_weight,
            flow_action_decoder_mode=args.flow_action_decoder_mode,
            flow_action_loss_weight=args.flow_action_loss_weight,
            flow_matching_loss_weight=args.flow_matching_loss_weight,
            sample_score_mode=args.sample_score_mode,
            sample_score_loss_weight=args.sample_score_loss_weight,
            sample_score_target=args.sample_score_target,
            sample_score_loss_type=args.sample_score_loss_type,
            sample_score_temperature=args.sample_score_temperature,
        )
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
            event_top_m=args.event_top_m,
            num_samples=args.num_samples,
            event_candidate_policy=args.event_candidate_policy,
            transition_reserve_threshold=args.transition_reserve_threshold,
            sample_feature_mode=args.sample_feature_mode,
            future_input_control=args.future_input_control,
            loss_weight_mode=args.loss_weight_mode,
            transition_loss_weight=args.transition_loss_weight,
            aux_gripper_loss_weight=args.aux_gripper_loss_weight,
            gripper_residual_mode=args.gripper_residual_mode,
            gripper_residual_loss_weight=args.gripper_residual_loss_weight,
            gripper_route_loss_weight=args.gripper_route_loss_weight,
            gripper_step_residual_mode=args.gripper_step_residual_mode,
            gripper_step_residual_loss_weight=args.gripper_step_residual_loss_weight,
            gripper_step_loss_weight=args.gripper_step_loss_weight,
            gripper_step_target_mode=args.gripper_step_target_mode,
            gripper_step_positive_loss_weight=args.gripper_step_positive_loss_weight,
            gripper_step_oracle_boundary_residual_loss_weight=(
                args.gripper_step_oracle_boundary_residual_loss_weight
            ),
            gripper_boundary_index_mode=args.gripper_boundary_index_mode,
            gripper_boundary_index_loss_weight=args.gripper_boundary_index_loss_weight,
            gripper_trajectory_residual_mode=args.gripper_trajectory_residual_mode,
            gripper_trajectory_residual_loss_weight=(
                args.gripper_trajectory_residual_loss_weight
            ),
            event_time_conditioning_mode=args.event_time_conditioning_mode,
            event_time_loss_weight=args.event_time_loss_weight,
            temporal_action_decoder_mode=args.temporal_action_decoder_mode,
            temporal_action_loss_weight=args.temporal_action_loss_weight,
            flow_action_decoder_mode=args.flow_action_decoder_mode,
            flow_action_loss_weight=args.flow_action_loss_weight,
            flow_matching_loss_weight=args.flow_matching_loss_weight,
            sample_score_mode=args.sample_score_mode,
            sample_score_loss_weight=args.sample_score_loss_weight,
            sample_score_target=args.sample_score_target,
            sample_score_loss_type=args.sample_score_loss_type,
            sample_score_temperature=args.sample_score_temperature,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        candidate_metric = val_metrics.get(args.selection_metric)
        if candidate_metric is not None and float(candidate_metric) < best_metric:
            best_metric = float(candidate_metric)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    if best_state is not None:
        model.load_state_dict(best_state)
    final_action_metrics = _evaluate(
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
        event_top_m=args.event_top_m,
        num_samples=args.num_samples,
        event_candidate_policy=args.event_candidate_policy,
        transition_reserve_threshold=args.transition_reserve_threshold,
        sample_feature_mode=args.sample_feature_mode,
        future_input_control=args.future_input_control,
        loss_weight_mode=args.loss_weight_mode,
        transition_loss_weight=args.transition_loss_weight,
        aux_gripper_loss_weight=args.aux_gripper_loss_weight,
        gripper_residual_mode=args.gripper_residual_mode,
        gripper_residual_loss_weight=args.gripper_residual_loss_weight,
        gripper_route_loss_weight=args.gripper_route_loss_weight,
        gripper_step_residual_mode=args.gripper_step_residual_mode,
        gripper_step_residual_loss_weight=args.gripper_step_residual_loss_weight,
        gripper_step_loss_weight=args.gripper_step_loss_weight,
        gripper_step_target_mode=args.gripper_step_target_mode,
        gripper_step_positive_loss_weight=args.gripper_step_positive_loss_weight,
        gripper_step_oracle_boundary_residual_loss_weight=(
            args.gripper_step_oracle_boundary_residual_loss_weight
        ),
        gripper_boundary_index_mode=args.gripper_boundary_index_mode,
        gripper_boundary_index_loss_weight=args.gripper_boundary_index_loss_weight,
        gripper_trajectory_residual_mode=args.gripper_trajectory_residual_mode,
        gripper_trajectory_residual_loss_weight=args.gripper_trajectory_residual_loss_weight,
        event_time_conditioning_mode=args.event_time_conditioning_mode,
        event_time_loss_weight=args.event_time_loss_weight,
        temporal_action_decoder_mode=args.temporal_action_decoder_mode,
        temporal_action_loss_weight=args.temporal_action_loss_weight,
        flow_action_decoder_mode=args.flow_action_decoder_mode,
        flow_action_loss_weight=args.flow_action_loss_weight,
        flow_matching_loss_weight=args.flow_matching_loss_weight,
        sample_score_mode=args.sample_score_mode,
        sample_score_loss_weight=args.sample_score_loss_weight,
        sample_score_target=args.sample_score_target,
        sample_score_loss_type=args.sample_score_loss_type,
        sample_score_temperature=args.sample_score_temperature,
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
        "seed": args.seed,
        "split_by": split_by,
        "motion_mode": motion_mode,
        "input_mode": "predicted_event_mixture_sample_set",
        "event_top_m": args.event_top_m,
        "num_samples": args.num_samples,
        "event_candidate_policy": args.event_candidate_policy,
        "transition_reserve_threshold": args.transition_reserve_threshold,
        "sample_feature_mode": args.sample_feature_mode,
        "future_input_control": args.future_input_control,
        "sample_feature_dim": sample_feature_dim,
        "loss_weight_mode": args.loss_weight_mode,
        "transition_loss_weight": args.transition_loss_weight,
        "aux_gripper_loss_weight": args.aux_gripper_loss_weight,
        "gripper_residual_mode": args.gripper_residual_mode,
        "gripper_residual_loss_weight": args.gripper_residual_loss_weight,
        "gripper_route_loss_weight": args.gripper_route_loss_weight,
        "gripper_route_families": list(GRIPPER_ROUTE_FAMILIES),
        "gripper_step_residual_mode": args.gripper_step_residual_mode,
        "gripper_step_residual_loss_weight": args.gripper_step_residual_loss_weight,
        "gripper_step_loss_weight": args.gripper_step_loss_weight,
        "gripper_step_target_mode": args.gripper_step_target_mode,
        "gripper_step_positive_loss_weight": args.gripper_step_positive_loss_weight,
        "gripper_step_residual_blend": args.gripper_step_residual_blend,
        "gripper_step_oracle_boundary_residual_loss_weight": (
            args.gripper_step_oracle_boundary_residual_loss_weight
        ),
        "gripper_boundary_index_mode": args.gripper_boundary_index_mode,
        "gripper_boundary_index_loss_weight": args.gripper_boundary_index_loss_weight,
        "gripper_trajectory_residual_mode": args.gripper_trajectory_residual_mode,
        "gripper_trajectory_residual_loss_weight": (
            args.gripper_trajectory_residual_loss_weight
        ),
        "event_time_conditioning_mode": args.event_time_conditioning_mode,
        "event_time_loss_weight": args.event_time_loss_weight,
        "temporal_action_decoder_mode": args.temporal_action_decoder_mode,
        "temporal_action_loss_weight": args.temporal_action_loss_weight,
        "flow_action_decoder_mode": args.flow_action_decoder_mode,
        "flow_action_loss_weight": args.flow_action_loss_weight,
        "flow_matching_loss_weight": args.flow_matching_loss_weight,
        "sample_score_mode": args.sample_score_mode,
        "sample_score_loss_weight": args.sample_score_loss_weight,
        "sample_score_target": args.sample_score_target,
        "sample_score_loss_type": args.sample_score_loss_type,
        "sample_score_temperature": args.sample_score_temperature,
        "gripper_step_classes": list(_gripper_step_classes(args.gripper_step_target_mode)),
        "selection_metric": args.selection_metric,
        "train_sampling_mode": args.train_sampling_mode,
        "transition_sampling_fraction": args.transition_sampling_fraction,
        "train_sampling_summary": _train_sampling_summary(
            dataset,
            train_indices,
            event_labels,
            sampling_mode=args.train_sampling_mode,
            transition_sampling_fraction=args.transition_sampling_fraction,
        ),
        "rank_sample_counts": list(rank_uniform_counts(args.num_samples, args.event_top_m)),
        "conditioning": conditioner.to_dict(),
        "action_head_conditioning_dim": conditioner.dim,
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
        "best_val_mse": best_metric if args.selection_metric == "mse" and best_state is not None else None,
        "best_selection_metric": args.selection_metric,
        "best_selection_value": best_metric if best_state is not None else None,
        "final_action_metrics": final_action_metrics,
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
                    "best_epoch": best_epoch,
                    "final_action_metrics": final_action_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.event_top_m <= 0:
        raise ValueError("--event-top-m must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.event_top_m > args.num_samples:
        raise ValueError("--event-top-m cannot exceed --num-samples")
    if args.transition_reserve_threshold < 0.0:
        raise ValueError("--transition-reserve-threshold must be non-negative")
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
    if args.temporal_layers < 0:
        raise ValueError("--temporal-layers must be non-negative")
    if args.set_query_count <= 0:
        raise ValueError("--set-query-count must be positive")
    if args.dropout < 0.0:
        raise ValueError("--dropout must be non-negative")
    if args.transition_loss_weight <= 0.0:
        raise ValueError("--transition-loss-weight must be positive")
    if not 0.0 < args.transition_sampling_fraction < 1.0:
        raise ValueError("--transition-sampling-fraction must be between 0 and 1")
    if args.aux_gripper_loss_weight < 0.0:
        raise ValueError("--aux-gripper-loss-weight must be non-negative")
    if args.gripper_residual_loss_weight < 0.0:
        raise ValueError("--gripper-residual-loss-weight must be non-negative")
    if args.gripper_route_loss_weight < 0.0:
        raise ValueError("--gripper-route-loss-weight must be non-negative")
    if args.gripper_residual_mode == "none":
        if args.gripper_residual_loss_weight > 0.0:
            raise ValueError(
                "--gripper-residual-loss-weight requires --gripper-residual-mode event_family"
            )
        if args.gripper_route_loss_weight > 0.0:
            raise ValueError(
                "--gripper-route-loss-weight requires --gripper-residual-mode event_family"
            )
    if args.gripper_residual_mode == "none" and args.selection_metric.startswith("routed_"):
        raise ValueError("routed selection metrics require --gripper-residual-mode event_family")
    if args.gripper_step_residual_loss_weight < 0.0:
        raise ValueError("--gripper-step-residual-loss-weight must be non-negative")
    if args.gripper_step_loss_weight < 0.0:
        raise ValueError("--gripper-step-loss-weight must be non-negative")
    if args.gripper_step_positive_loss_weight <= 0.0:
        raise ValueError("--gripper-step-positive-loss-weight must be positive")
    if args.gripper_step_oracle_boundary_residual_loss_weight < 0.0:
        raise ValueError(
            "--gripper-step-oracle-boundary-residual-loss-weight must be non-negative"
        )
    if args.gripper_boundary_index_loss_weight < 0.0:
        raise ValueError("--gripper-boundary-index-loss-weight must be non-negative")
    if args.gripper_trajectory_residual_loss_weight < 0.0:
        raise ValueError("--gripper-trajectory-residual-loss-weight must be non-negative")
    if args.event_time_loss_weight < 0.0:
        raise ValueError("--event-time-loss-weight must be non-negative")
    if args.temporal_action_loss_weight < 0.0:
        raise ValueError("--temporal-action-loss-weight must be non-negative")
    if args.flow_action_loss_weight < 0.0:
        raise ValueError("--flow-action-loss-weight must be non-negative")
    if args.flow_matching_loss_weight < 0.0:
        raise ValueError("--flow-matching-loss-weight must be non-negative")
    if args.sample_score_loss_weight < 0.0:
        raise ValueError("--sample-score-loss-weight must be non-negative")
    if args.sample_score_temperature <= 0.0:
        raise ValueError("--sample-score-temperature must be positive")
    if args.gripper_boundary_index_mode == "none":
        if args.gripper_boundary_index_loss_weight > 0.0:
            raise ValueError(
                "--gripper-boundary-index-loss-weight requires "
                "--gripper-boundary-index-mode boundary_index"
            )
        if args.selection_metric.startswith("boundary_index_"):
            raise ValueError(
                "boundary-index selection metrics require "
                "--gripper-boundary-index-mode boundary_index"
            )
    if (
        args.gripper_boundary_index_mode == "boundary_index"
        and args.gripper_step_residual_mode != "event_step"
    ):
        raise ValueError(
            "--gripper-boundary-index-mode boundary_index requires "
            "--gripper-step-residual-mode event_step"
        )
    if (
        args.gripper_boundary_index_mode == "boundary_index"
        and args.gripper_step_target_mode != "boundary_start"
    ):
        raise ValueError(
            "--gripper-boundary-index-mode boundary_index requires "
            "--gripper-step-target-mode boundary_start"
        )
    if args.gripper_step_residual_mode == "none":
        if args.gripper_step_residual_loss_weight > 0.0:
            raise ValueError(
                "--gripper-step-residual-loss-weight requires "
                "--gripper-step-residual-mode event_step"
            )
        if args.gripper_step_loss_weight > 0.0:
            raise ValueError(
                "--gripper-step-loss-weight requires --gripper-step-residual-mode event_step"
            )
        if args.gripper_step_oracle_boundary_residual_loss_weight > 0.0:
            raise ValueError(
                "--gripper-step-oracle-boundary-residual-loss-weight requires "
                "--gripper-step-residual-mode event_step"
            )
    if args.gripper_step_residual_mode == "none" and args.selection_metric.startswith("step_"):
        raise ValueError(
            "step-routed selection metrics require --gripper-step-residual-mode event_step"
        )
    if (
        args.gripper_step_residual_mode == "none"
        and args.selection_metric.startswith("oracle_step_")
    ):
        raise ValueError(
            "oracle-step-routed selection metrics require "
            "--gripper-step-residual-mode event_step"
        )
    if args.gripper_step_target_mode == "boundary_start" and args.gripper_step_residual_mode == "none":
        raise ValueError(
            "--gripper-step-target-mode boundary_start requires "
            "--gripper-step-residual-mode event_step"
        )
    if (
        args.gripper_step_oracle_boundary_residual_loss_weight > 0.0
        and args.gripper_step_target_mode != "boundary_start"
    ):
        raise ValueError(
            "--gripper-step-oracle-boundary-residual-loss-weight requires "
            "--gripper-step-target-mode boundary_start"
        )
    if (
        args.selection_metric.startswith("oracle_step_")
        and args.gripper_step_target_mode != "boundary_start"
    ):
        raise ValueError(
            "oracle-step-routed selection metrics require "
            "--gripper-step-target-mode boundary_start"
        )
    if args.gripper_trajectory_residual_mode == "none":
        if args.gripper_trajectory_residual_loss_weight > 0.0:
            raise ValueError(
                "--gripper-trajectory-residual-loss-weight requires "
                "--gripper-trajectory-residual-mode temporal_mlp"
            )
        if args.selection_metric.startswith("trajectory_routed_"):
            raise ValueError(
                "trajectory-routed selection metrics require "
                "--gripper-trajectory-residual-mode temporal_mlp"
            )
    if args.temporal_action_decoder_mode == "none":
        if args.temporal_action_loss_weight > 0.0:
            raise ValueError(
                "--temporal-action-loss-weight requires "
                "--temporal-action-decoder-mode"
            )
        if args.selection_metric.startswith("temporal_action_"):
            raise ValueError(
                "temporal-action selection metrics require "
                "--temporal-action-decoder-mode"
            )
    if args.event_time_conditioning_mode == "none":
        if args.event_time_loss_weight > 0.0:
            raise ValueError(
                "--event-time-loss-weight requires "
                "--event-time-conditioning-mode soft_boundary"
            )
    if (
        args.event_time_conditioning_mode != "none"
        and args.temporal_action_decoder_mode == "none"
    ):
        raise ValueError(
            "--event-time-conditioning-mode requires "
            "--temporal-action-decoder-mode"
        )
    if args.flow_action_decoder_mode == "none":
        if args.flow_action_loss_weight > 0.0:
            raise ValueError(
                "--flow-action-loss-weight requires "
                "--flow-action-decoder-mode rectified_mlp"
            )
        if args.flow_matching_loss_weight > 0.0:
            raise ValueError(
                "--flow-matching-loss-weight requires "
                "--flow-action-decoder-mode rectified_mlp"
            )
        if args.selection_metric.startswith("flow_action_"):
            raise ValueError("flow-action selection metrics require --flow-action-decoder-mode")
    if (
        args.flow_action_decoder_mode != "none"
        and args.temporal_action_decoder_mode == "none"
    ):
        raise ValueError(
            "--flow-action-decoder-mode requires "
            "--temporal-action-decoder-mode"
        )
    if args.sample_score_mode == "none":
        if args.sample_score_loss_weight > 0.0:
            raise ValueError(
                "--sample-score-loss-weight requires --sample-score-mode action_regret"
            )
        if args.selection_metric.startswith("sample_score_"):
            raise ValueError("sample-score selection metrics require --sample-score-mode")
    if (
        args.sample_score_mode != "none"
        and args.future_input_control == "context_only"
        and args.sample_score_loss_weight > 0.0
    ):
        raise ValueError(
            "sample-score supervision requires future inputs; use loss weight 0 "
            "for context-only capacity controls"
        )
    if (
        args.sample_score_target == "temporal_action_regret"
        and args.temporal_action_decoder_mode == "none"
    ):
        raise ValueError(
            "--sample-score-target temporal_action_regret requires "
            "--temporal-action-decoder-mode"
        )
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")


def _run_epoch(
    model: MotionPriorActionHead,
    cvae: VisualConditionedGeoMoCoCVAE,
    event_probe: EventModeProbeNet,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    event_labels: dict[str, Any] | None,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    future_input_control: str,
    loss_weight_mode: str,
    transition_loss_weight: float,
    aux_gripper_loss_weight: float,
    gripper_residual_mode: str,
    gripper_residual_loss_weight: float,
    gripper_route_loss_weight: float,
    gripper_step_residual_mode: str,
    gripper_step_residual_loss_weight: float,
    gripper_step_loss_weight: float,
    gripper_step_target_mode: str,
    gripper_step_positive_loss_weight: float,
    gripper_step_oracle_boundary_residual_loss_weight: float,
    gripper_boundary_index_mode: str,
    gripper_boundary_index_loss_weight: float,
    gripper_trajectory_residual_mode: str,
    gripper_trajectory_residual_loss_weight: float,
    event_time_conditioning_mode: str,
    event_time_loss_weight: float,
    temporal_action_decoder_mode: str,
    temporal_action_loss_weight: float,
    flow_action_decoder_mode: str,
    flow_action_loss_weight: float,
    flow_matching_loss_weight: float,
    sample_score_mode: str,
    sample_score_loss_weight: float,
    sample_score_target: str,
    sample_score_loss_type: str,
    sample_score_temperature: float,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None, "mse": None}
    model.train()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        action_conditioning = _batch_conditioning(
            batch,
            conditioner,
            device,
            include_visual=False,
        )
        future_inputs, sample_features = _predicted_event_future_inputs(
            cvae,
            event_probe,
            batch,
            context,
            action_conditioning,
            device,
            probe_conditioner,
            event_classes=event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=event_top_m,
            num_samples=num_samples,
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
        )
        future_inputs, sample_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            future_input_control,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(
            context,
            future_inputs,
            action_conditioning,
            sample_features,
        )
        pred_actions = _output_actions(output)
        loss, loss_metrics, loss_counts = _weighted_action_loss(
            pred_actions,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
        )
        loss, aux_metrics, aux_counts = _add_aux_gripper_loss(
            loss,
            output["aux_gripper"],
            pred_actions,
            actions,
            aux_gripper_loss_weight=aux_gripper_loss_weight,
        )
        loss, routed_metrics, routed_counts = _add_gripper_residual_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_residual_mode=gripper_residual_mode,
            gripper_residual_loss_weight=gripper_residual_loss_weight,
            gripper_route_loss_weight=gripper_route_loss_weight,
        )
        loss, step_metrics, step_counts = _add_gripper_step_residual_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_step_residual_mode=gripper_step_residual_mode,
            gripper_step_residual_loss_weight=gripper_step_residual_loss_weight,
            gripper_step_loss_weight=gripper_step_loss_weight,
            gripper_step_target_mode=gripper_step_target_mode,
            gripper_step_positive_loss_weight=gripper_step_positive_loss_weight,
            gripper_step_oracle_boundary_residual_loss_weight=(
                gripper_step_oracle_boundary_residual_loss_weight
            ),
        )
        loss, boundary_index_metrics, boundary_index_counts = _add_gripper_boundary_index_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_boundary_index_mode=gripper_boundary_index_mode,
            gripper_boundary_index_loss_weight=gripper_boundary_index_loss_weight,
        )
        loss, trajectory_metrics, trajectory_counts = _add_gripper_trajectory_residual_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_trajectory_residual_mode=gripper_trajectory_residual_mode,
            gripper_trajectory_residual_loss_weight=(
                gripper_trajectory_residual_loss_weight
            ),
        )
        loss, event_time_metrics, event_time_counts = _add_event_time_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            event_time_conditioning_mode=event_time_conditioning_mode,
            event_time_loss_weight=event_time_loss_weight,
        )
        loss, temporal_metrics, temporal_counts = _add_temporal_action_losses(
            loss,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            temporal_action_decoder_mode=temporal_action_decoder_mode,
            temporal_action_loss_weight=temporal_action_loss_weight,
        )
        loss, flow_metrics, flow_counts = _add_flow_action_losses(
            loss,
            model,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            flow_action_decoder_mode=flow_action_decoder_mode,
            flow_action_loss_weight=flow_action_loss_weight,
            flow_matching_loss_weight=flow_matching_loss_weight,
        )
        loss, sample_score_metrics, sample_score_counts = _add_sample_score_losses(
            loss,
            output,
            model,
            context,
            action_conditioning,
            future_inputs,
            sample_features,
            actions,
            batch,
            sample_score_mode=sample_score_mode,
            sample_score_loss_weight=sample_score_loss_weight,
            sample_score_target=sample_score_target,
            sample_score_loss_type=sample_score_loss_type,
            sample_score_temperature=sample_score_temperature,
            temporal_action_decoder_mode=temporal_action_decoder_mode,
        )
        loss.backward()
        optimizer.step()
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions.detach(), actions)
        _add_metric_values(totals, counts, batch_metrics, batch_size)
        _add_metric_values(totals, counts, loss_metrics, loss_counts)
        _add_metric_values(totals, counts, aux_metrics, aux_counts)
        _add_metric_values(totals, counts, routed_metrics, routed_counts)
        _add_metric_values(totals, counts, step_metrics, step_counts)
        _add_metric_values(totals, counts, boundary_index_metrics, boundary_index_counts)
        _add_metric_values(totals, counts, trajectory_metrics, trajectory_counts)
        _add_metric_values(totals, counts, event_time_metrics, event_time_counts)
        _add_metric_values(totals, counts, temporal_metrics, temporal_counts)
        _add_metric_values(totals, counts, flow_metrics, flow_counts)
        _add_metric_values(totals, counts, sample_score_metrics, sample_score_counts)
    return _finalize_metric_values(totals, counts)


@torch.no_grad()
def _evaluate(
    model: MotionPriorActionHead,
    cvae: VisualConditionedGeoMoCoCVAE,
    event_probe: EventModeProbeNet,
    loader: DataLoader | None,
    device: torch.device,
    conditioner: CategoricalConditioner,
    probe_conditioner: CategoricalConditioner,
    event_labels: dict[str, Any] | None,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    event_candidate_policy: str,
    transition_reserve_threshold: float,
    sample_feature_mode: str,
    future_input_control: str,
    loss_weight_mode: str,
    transition_loss_weight: float,
    aux_gripper_loss_weight: float,
    gripper_residual_mode: str,
    gripper_residual_loss_weight: float,
    gripper_route_loss_weight: float,
    gripper_step_residual_mode: str,
    gripper_step_residual_loss_weight: float,
    gripper_step_loss_weight: float,
    gripper_step_target_mode: str,
    gripper_step_positive_loss_weight: float,
    gripper_step_oracle_boundary_residual_loss_weight: float,
    gripper_boundary_index_mode: str,
    gripper_boundary_index_loss_weight: float,
    gripper_trajectory_residual_mode: str,
    gripper_trajectory_residual_loss_weight: float,
    event_time_conditioning_mode: str,
    event_time_loss_weight: float,
    temporal_action_decoder_mode: str,
    temporal_action_loss_weight: float,
    flow_action_decoder_mode: str,
    flow_action_loss_weight: float,
    flow_matching_loss_weight: float,
    sample_score_mode: str,
    sample_score_loss_weight: float,
    sample_score_target: str,
    sample_score_loss_type: str,
    sample_score_temperature: float,
    per_window_records: list[dict[str, Any]] | None = None,
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None, "mse": None}
    model.eval()
    cvae.eval()
    event_probe.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        context = batch["context"].to(device)
        actions = batch["actions"].to(device)
        action_conditioning = _batch_conditioning(
            batch,
            conditioner,
            device,
            include_visual=False,
        )
        future_inputs, sample_features = _predicted_event_future_inputs(
            cvae,
            event_probe,
            batch,
            context,
            action_conditioning,
            device,
            probe_conditioner,
            event_classes=event_classes,
            probe_class_names=probe_class_names,
            probe_input_variant=probe_input_variant,
            event_top_m=event_top_m,
            num_samples=num_samples,
            event_candidate_policy=event_candidate_policy,
            transition_reserve_threshold=transition_reserve_threshold,
            sample_feature_mode=sample_feature_mode,
        )
        future_inputs, sample_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            future_input_control,
        )
        output = model.forward_with_aux(
            context,
            future_inputs,
            action_conditioning,
            sample_features,
        )
        pred_actions = _output_actions(output)
        _, loss_metrics, loss_counts = _weighted_action_loss(
            pred_actions,
            actions,
            batch,
            event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
        )
        _, aux_metrics, aux_counts = _add_aux_gripper_loss(
            loss=None,
            aux_gripper=output["aux_gripper"],
            pred_actions=pred_actions,
            actions=actions,
            aux_gripper_loss_weight=aux_gripper_loss_weight,
        )
        _, routed_metrics, routed_counts = _add_gripper_residual_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_residual_mode=gripper_residual_mode,
            gripper_residual_loss_weight=gripper_residual_loss_weight,
            gripper_route_loss_weight=gripper_route_loss_weight,
        )
        _, step_metrics, step_counts = _add_gripper_step_residual_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_step_residual_mode=gripper_step_residual_mode,
            gripper_step_residual_loss_weight=gripper_step_residual_loss_weight,
            gripper_step_loss_weight=gripper_step_loss_weight,
            gripper_step_target_mode=gripper_step_target_mode,
            gripper_step_positive_loss_weight=gripper_step_positive_loss_weight,
            gripper_step_oracle_boundary_residual_loss_weight=(
                gripper_step_oracle_boundary_residual_loss_weight
            ),
        )
        _, boundary_index_metrics, boundary_index_counts = _add_gripper_boundary_index_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_boundary_index_mode=gripper_boundary_index_mode,
            gripper_boundary_index_loss_weight=gripper_boundary_index_loss_weight,
        )
        _, trajectory_metrics, trajectory_counts = _add_gripper_trajectory_residual_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            gripper_trajectory_residual_mode=gripper_trajectory_residual_mode,
            gripper_trajectory_residual_loss_weight=(
                gripper_trajectory_residual_loss_weight
            ),
        )
        _, event_time_metrics, event_time_counts = _add_event_time_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            event_time_conditioning_mode=event_time_conditioning_mode,
            event_time_loss_weight=event_time_loss_weight,
        )
        _, temporal_metrics, temporal_counts = _add_temporal_action_losses(
            loss=None,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            temporal_action_decoder_mode=temporal_action_decoder_mode,
            temporal_action_loss_weight=temporal_action_loss_weight,
        )
        _, flow_metrics, flow_counts = _add_flow_action_losses(
            loss=None,
            model=model,
            output=output,
            actions=actions,
            batch=batch,
            event_labels=event_labels,
            loss_weight_mode=loss_weight_mode,
            transition_loss_weight=transition_loss_weight,
            flow_action_decoder_mode=flow_action_decoder_mode,
            flow_action_loss_weight=flow_action_loss_weight,
            flow_matching_loss_weight=flow_matching_loss_weight,
        )
        _, sample_score_metrics, sample_score_counts = _add_sample_score_losses(
            loss=None,
            output=output,
            model=model,
            context=context,
            conditioning=action_conditioning,
            future_inputs=future_inputs,
            sample_features=sample_features,
            actions=actions,
            batch=batch,
            sample_score_mode=sample_score_mode,
            sample_score_loss_weight=sample_score_loss_weight,
            sample_score_target=sample_score_target,
            sample_score_loss_type=sample_score_loss_type,
            sample_score_temperature=sample_score_temperature,
            temporal_action_decoder_mode=temporal_action_decoder_mode,
        )
        batch_size = int(context.shape[0])
        batch_metrics = action_metrics(pred_actions, actions)
        if per_window_records is not None:
            metadata = window_metadata_records(batch, event_labels)
            per_window_records.extend(
                merge_window_metric_records(
                    metadata,
                    per_window_action_metrics(pred_actions, actions),
                )
            )
            temporal_actions = output.get("temporal_actions")
            if temporal_actions is not None:
                temporal_records = merge_window_metric_records(
                    metadata,
                    per_window_action_metrics(temporal_actions, actions),
                    prefix="temporal_action",
                )
                for target, source in zip(per_window_records[-batch_size:], temporal_records, strict=True):
                    target.update(
                        {
                            key: value
                            for key, value in source.items()
                            if key.startswith("temporal_action_")
                        }
                    )
            flow_actions = output.get("flow_actions")
            if flow_actions is not None:
                flow_records = merge_window_metric_records(
                    metadata,
                    per_window_action_metrics(flow_actions, actions),
                    prefix="flow_action",
                )
                for target, source in zip(per_window_records[-batch_size:], flow_records, strict=True):
                    target.update(
                        {
                            key: value
                            for key, value in source.items()
                            if key.startswith("flow_action_")
                        }
                    )
        _add_metric_values(totals, counts, batch_metrics, batch_size)
        _add_metric_values(totals, counts, loss_metrics, loss_counts)
        _add_metric_values(totals, counts, aux_metrics, aux_counts)
        _add_metric_values(totals, counts, routed_metrics, routed_counts)
        _add_metric_values(totals, counts, step_metrics, step_counts)
        _add_metric_values(totals, counts, boundary_index_metrics, boundary_index_counts)
        _add_metric_values(totals, counts, trajectory_metrics, trajectory_counts)
        _add_metric_values(totals, counts, event_time_metrics, event_time_counts)
        _add_metric_values(totals, counts, temporal_metrics, temporal_counts)
        _add_metric_values(totals, counts, flow_metrics, flow_counts)
        _add_metric_values(totals, counts, sample_score_metrics, sample_score_counts)
    return _finalize_metric_values(totals, counts)


@torch.no_grad()
def _predicted_event_future_inputs(
    cvae: VisualConditionedGeoMoCoCVAE,
    event_probe: EventModeProbeNet,
    batch: dict[str, object],
    context: torch.Tensor,
    base_conditioning: torch.Tensor | None,
    device: torch.device,
    probe_conditioner: CategoricalConditioner,
    *,
    event_classes: tuple[str, ...],
    probe_class_names: tuple[str, ...],
    probe_input_variant: str,
    event_top_m: int,
    num_samples: int,
    sample_feature_mode: str,
    event_candidate_policy: str = "topk",
    transition_reserve_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    return samples.permute(1, 0, 2).contiguous(), sample_features


def _model_config(
    args: argparse.Namespace,
    hidden_dims: tuple[int, ...],
    spec: object,
    conditioner: CategoricalConditioner,
    sample_feature_dim: int,
) -> dict[str, object]:
    return {
        "context_dim": int(spec.context_dim),
        "motion_dim": int(spec.motion_dim),
        "action_dim": int(spec.action_dim),
        "horizon": int(spec.horizon),
        "conditioning_dim": conditioner.dim,
        "base_conditioning_dim": conditioner.dim,
        "event_conditioning_dim": 0,
        "sample_feature_mode": args.sample_feature_mode,
        "future_input_control": args.future_input_control,
        "sample_feature_dim": sample_feature_dim,
        "aux_gripper_head": args.aux_gripper_loss_weight > 0.0,
        "aux_gripper_loss_weight": args.aux_gripper_loss_weight,
        "gripper_residual_mode": args.gripper_residual_mode,
        "gripper_residual_loss_weight": args.gripper_residual_loss_weight,
        "gripper_route_loss_weight": args.gripper_route_loss_weight,
        "gripper_route_count": len(GRIPPER_ROUTE_FAMILIES),
        "gripper_route_families": list(GRIPPER_ROUTE_FAMILIES),
        "gripper_step_residual_mode": args.gripper_step_residual_mode,
        "gripper_step_residual_loss_weight": args.gripper_step_residual_loss_weight,
        "gripper_step_loss_weight": args.gripper_step_loss_weight,
        "gripper_step_target_mode": args.gripper_step_target_mode,
        "gripper_step_positive_loss_weight": args.gripper_step_positive_loss_weight,
        "gripper_step_residual_blend": args.gripper_step_residual_blend,
        "gripper_step_oracle_boundary_residual_loss_weight": (
            args.gripper_step_oracle_boundary_residual_loss_weight
        ),
        "gripper_boundary_index_mode": args.gripper_boundary_index_mode,
        "gripper_boundary_index_loss_weight": args.gripper_boundary_index_loss_weight,
        "gripper_trajectory_residual_mode": args.gripper_trajectory_residual_mode,
        "gripper_trajectory_residual_loss_weight": (
            args.gripper_trajectory_residual_loss_weight
        ),
        "event_time_conditioning_mode": args.event_time_conditioning_mode,
        "event_time_loss_weight": args.event_time_loss_weight,
        "temporal_action_decoder_mode": args.temporal_action_decoder_mode,
        "temporal_action_loss_weight": args.temporal_action_loss_weight,
        "flow_action_decoder_mode": args.flow_action_decoder_mode,
        "flow_action_loss_weight": args.flow_action_loss_weight,
        "flow_matching_loss_weight": args.flow_matching_loss_weight,
        "sample_score_mode": args.sample_score_mode,
        "sample_score_loss_weight": args.sample_score_loss_weight,
        "sample_score_target": args.sample_score_target,
        "sample_score_loss_type": args.sample_score_loss_type,
        "sample_score_temperature": args.sample_score_temperature,
        "gripper_step_class_count": len(_gripper_step_classes(args.gripper_step_target_mode)),
        "gripper_step_classes": list(_gripper_step_classes(args.gripper_step_target_mode)),
        "hidden_dims": list(hidden_dims),
        "token_dim": args.token_dim,
        "num_heads": args.num_heads,
        "temporal_layers": args.temporal_layers,
        "set_aggregator": args.set_aggregator,
        "set_query_count": args.set_query_count,
        "dropout": args.dropout,
        "aggregation": args.set_aggregator,
    }


def _sample_feature_dim(sample_feature_mode: str, event_classes: tuple[str, ...]) -> int:
    if sample_feature_mode == "none":
        return 0
    if sample_feature_mode == "event_only":
        return len(event_classes)
    if sample_feature_mode == "rank_prob_only":
        return 2
    if sample_feature_mode in {"event_rank_prob", "shuffled_event_rank_prob"}:
        return len(event_classes) + 2
    raise ValueError(f"unsupported sample feature mode {sample_feature_mode!r}")


def _rank_sample_feature(
    event_one_hot: torch.Tensor,
    event_probability: torch.Tensor,
    *,
    rank: int,
    event_top_m: int,
    sample_feature_mode: str,
) -> torch.Tensor:
    rank_value = 0.0 if event_top_m <= 1 else float(rank) / float(event_top_m - 1)
    rank_feature = event_one_hot.new_full((event_one_hot.shape[0], 1), rank_value)
    prob_feature = event_probability.to(dtype=event_one_hot.dtype).unsqueeze(-1)
    rank_prob = torch.cat([rank_feature, prob_feature], dim=-1)
    if sample_feature_mode == "event_only":
        return event_one_hot
    if sample_feature_mode == "rank_prob_only":
        return rank_prob
    if sample_feature_mode == "event_rank_prob":
        return torch.cat([event_one_hot, rank_prob], dim=-1)
    if sample_feature_mode == "shuffled_event_rank_prob":
        shuffled_event = torch.roll(event_one_hot, shifts=1, dims=0)
        return torch.cat([shuffled_event, rank_prob], dim=-1)
    raise ValueError(f"unsupported sample feature mode {sample_feature_mode!r}")


def _sample_features_for_ranks(
    rank_sample_features: list[torch.Tensor],
    *,
    num_samples: int,
    event_top_m: int,
) -> torch.Tensor | None:
    if not rank_sample_features:
        return None
    chunks: list[torch.Tensor] = []
    for feature, count in zip(
        rank_sample_features,
        rank_uniform_counts(num_samples, event_top_m),
        strict=True,
    ):
        chunks.append(feature.unsqueeze(1).expand(-1, count, -1))
    return torch.cat(chunks, dim=1).contiguous()


def _apply_future_input_control(
    future_inputs: torch.Tensor | None,
    sample_features: torch.Tensor | None,
    future_input_control: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if future_input_control == "real":
        return future_inputs, sample_features
    if future_input_control == "context_only":
        return None, None
    if future_input_control == "mean_repeated":
        if future_inputs is None:
            raise ValueError("mean_repeated future input control requires future inputs")
        if future_inputs.ndim != 3:
            raise ValueError(
                "mean_repeated future input control expects [B,K,M] future inputs, "
                f"got {future_inputs.shape}"
            )
        mean = future_inputs.mean(dim=1, keepdim=True)
        return mean.expand(-1, future_inputs.shape[1], -1).contiguous(), sample_features
    raise ValueError(f"unsupported future_input_control {future_input_control!r}")


def _load_training_event_labels(
    event_mode_audit_json: str | Path | None,
    *,
    loss_weight_mode: str,
    gripper_residual_mode: str = "none",
    gripper_route_loss_weight: float = 0.0,
    gripper_step_residual_mode: str = "none",
    gripper_boundary_index_mode: str = "none",
    gripper_trajectory_residual_mode: str = "none",
    event_time_conditioning_mode: str = "none",
    temporal_action_decoder_mode: str = "none",
    flow_action_decoder_mode: str = "none",
) -> dict[str, Any] | None:
    needs_event_labels = (
        loss_weight_mode != "none"
        or gripper_residual_mode != "none"
        or gripper_route_loss_weight > 0.0
        or gripper_step_residual_mode != "none"
        or gripper_boundary_index_mode != "none"
        or gripper_trajectory_residual_mode != "none"
        or event_time_conditioning_mode != "none"
        or temporal_action_decoder_mode != "none"
        or flow_action_decoder_mode != "none"
    )
    if not needs_event_labels:
        return None
    if event_mode_audit_json is None:
        raise ValueError("--event-mode-audit-json is required for event-aware losses")
    return _load_event_label_records(event_mode_audit_json)


def _make_train_loader(
    dataset: OracleActionWindowDataset,
    indices: list[int],
    batch_size: int,
    event_labels: dict[str, Any] | None,
    *,
    sampling_mode: str,
    transition_sampling_fraction: float,
) -> DataLoader | None:
    if not indices:
        return None
    if sampling_mode == "natural":
        return _make_loader(dataset, indices, batch_size, shuffle=True)
    if sampling_mode != "transition_balanced":
        raise ValueError(f"unsupported train sampling mode {sampling_mode!r}")
    weights = _transition_balanced_sample_weights(
        dataset,
        indices,
        event_labels,
        transition_sampling_fraction=transition_sampling_fraction,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, sampler=sampler)


def _transition_balanced_sample_weights(
    dataset: OracleActionWindowDataset,
    indices: list[int],
    event_labels: dict[str, Any] | None,
    *,
    transition_sampling_fraction: float,
) -> list[float]:
    if event_labels is None:
        raise ValueError("transition-balanced sampling requires event labels")
    if not 0.0 < transition_sampling_fraction < 1.0:
        raise ValueError("transition_sampling_fraction must be between 0 and 1")
    transition_flags = [
        event_label_is_transition(
            _event_mode_for_record(event_labels.get(str(dataset.windows[index].window_id)))
        )
        for index in indices
    ]
    transition_count = sum(transition_flags)
    sustain_count = len(transition_flags) - transition_count
    if transition_count == 0 or sustain_count == 0:
        raise ValueError(
            "transition-balanced sampling requires both transition and sustain windows; "
            f"got transition={transition_count}, sustain={sustain_count}"
        )
    transition_weight = transition_sampling_fraction / float(transition_count)
    sustain_weight = (1.0 - transition_sampling_fraction) / float(sustain_count)
    return [transition_weight if flag else sustain_weight for flag in transition_flags]


def _train_sampling_summary(
    dataset: OracleActionWindowDataset,
    indices: list[int],
    event_labels: dict[str, Any] | None,
    *,
    sampling_mode: str,
    transition_sampling_fraction: float,
) -> dict[str, Any]:
    if not indices:
        return {
            "sampling_mode": sampling_mode,
            "num_train_indices": 0,
            "transition_count": 0,
            "sustain_count": 0,
            "natural_transition_fraction": 0.0,
            "target_transition_fraction": transition_sampling_fraction,
        }
    if event_labels is None:
        return {
            "sampling_mode": sampling_mode,
            "num_train_indices": len(indices),
            "event_labels": "unavailable",
        }
    transition_flags = [
        event_label_is_transition(
            _event_mode_for_record(event_labels.get(str(dataset.windows[index].window_id)))
        )
        for index in indices
    ]
    transition_count = sum(transition_flags)
    sustain_count = len(transition_flags) - transition_count
    summary: dict[str, Any] = {
        "sampling_mode": sampling_mode,
        "num_train_indices": len(indices),
        "transition_count": transition_count,
        "sustain_count": sustain_count,
        "natural_transition_fraction": transition_count / len(indices),
        "target_transition_fraction": transition_sampling_fraction,
    }
    if sampling_mode == "transition_balanced":
        weights = _transition_balanced_sample_weights(
            dataset,
            indices,
            event_labels,
            transition_sampling_fraction=transition_sampling_fraction,
        )
        transition_mass = sum(
            weight for weight, is_transition in zip(weights, transition_flags, strict=True)
            if is_transition
        )
        summary["sampled_transition_probability"] = transition_mass / sum(weights)
        summary["transition_weight_per_window"] = weights[transition_flags.index(True)]
        summary["sustain_weight_per_window"] = weights[transition_flags.index(False)]
    return summary


def _weighted_action_loss(
    pred_actions: torch.Tensor,
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float], dict[str, int]]:
    per_item_mse = (pred_actions - actions).square().mean(dim=(1, 2))
    transition_mask = _transition_mask(
        batch,
        event_labels,
        batch_size=int(actions.shape[0]),
        device=actions.device,
    )
    weights = torch.ones_like(per_item_mse)
    if loss_weight_mode == "transition":
        weights = torch.where(
            transition_mask,
            weights.new_full(weights.shape, transition_loss_weight),
            weights,
        )
    elif loss_weight_mode != "none":
        raise ValueError(f"unsupported loss_weight_mode {loss_weight_mode!r}")
    weighted_loss = (per_item_mse * weights).sum() / weights.sum().clamp_min(1e-12)
    metrics: dict[str, float] = {
        "loss": float(weighted_loss.detach().cpu()),
        "weighted_loss": float(weighted_loss.detach().cpu()),
        "transition_fraction": float(transition_mask.to(dtype=torch.float32).mean().detach().cpu()),
    }
    counts: dict[str, int] = {
        "loss": int(actions.shape[0]),
        "weighted_loss": int(actions.shape[0]),
        "transition_fraction": int(actions.shape[0]),
    }
    _add_group_loss_metrics(metrics, counts, per_item_mse, transition_mask)
    return weighted_loss, metrics, counts


def _add_aux_gripper_loss(
    loss: torch.Tensor | None,
    aux_gripper: torch.Tensor | None,
    pred_actions: torch.Tensor,
    actions: torch.Tensor,
    *,
    aux_gripper_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if aux_gripper_loss_weight <= 0.0:
        return loss, {}, {}
    if aux_gripper is None:
        raise ValueError("aux_gripper head is required when aux_gripper_loss_weight is positive")
    target_gripper = actions[..., -1]
    aux_loss = (aux_gripper - target_gripper).square().mean()
    total_loss = loss + aux_gripper_loss_weight * aux_loss if loss is not None else None
    aux_actions = _replace_action_gripper(pred_actions, aux_gripper)
    aux_metrics = {
        f"aux_replaced_{key}": value
        for key, value in action_metrics(aux_actions, actions).items()
    }
    aux_metrics["aux_gripper_loss"] = float(aux_loss.detach().cpu())
    aux_metrics["aux_gripper_mse"] = float(aux_loss.detach().cpu())
    counts = {key: int(actions.shape[0]) for key in aux_metrics}
    return total_loss, aux_metrics, counts


def _add_gripper_residual_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    gripper_residual_mode: str,
    gripper_residual_loss_weight: float,
    gripper_route_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if gripper_residual_mode == "none":
        return loss, {}, {}
    if gripper_residual_mode != "event_family":
        raise ValueError(f"unsupported gripper_residual_mode {gripper_residual_mode!r}")
    routed_actions = output.get("routed_actions")
    route_logits = output.get("gripper_route_logits")
    if routed_actions is None or route_logits is None:
        raise ValueError("routed gripper outputs are required for gripper residual routing")
    routed_loss, routed_loss_metrics, routed_loss_counts = _weighted_action_loss(
        routed_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    total_loss = (
        loss + gripper_residual_loss_weight * routed_loss
        if loss is not None
        else None
    )
    route_targets = _gripper_route_targets(
        batch,
        event_labels,
        batch_size=int(actions.shape[0]),
        device=actions.device,
    )
    valid_route_mask = route_targets >= 0
    route_metrics: dict[str, float] = {}
    route_counts: dict[str, int] = {}
    valid_count = int(valid_route_mask.sum().detach().cpu())
    if valid_count > 0:
        valid_logits = route_logits[valid_route_mask]
        valid_targets = route_targets[valid_route_mask]
        route_loss = F.cross_entropy(valid_logits, valid_targets)
        total_loss = (
            total_loss + gripper_route_loss_weight * route_loss
            if total_loss is not None
            else None
        )
        route_pred = valid_logits.argmax(dim=-1)
        route_metrics["gripper_route_ce"] = float(route_loss.detach().cpu())
        route_metrics["gripper_route_accuracy"] = float(
            (route_pred == valid_targets).to(dtype=torch.float32).mean().detach().cpu()
        )
        route_counts["gripper_route_ce"] = valid_count
        route_counts["gripper_route_accuracy"] = valid_count
    route_metrics["gripper_route_valid_fraction"] = float(
        valid_route_mask.to(dtype=torch.float32).mean().detach().cpu()
    )
    route_counts["gripper_route_valid_fraction"] = int(actions.shape[0])

    routed_metrics = {
        f"routed_{key}": value
        for key, value in action_metrics(routed_actions, actions).items()
    }
    routed_counts = {key: int(actions.shape[0]) for key in routed_metrics}
    for key, value in routed_loss_metrics.items():
        routed_metrics[f"routed_{key}"] = value
        routed_counts[f"routed_{key}"] = routed_loss_counts[key]
    routed_metrics.update(route_metrics)
    routed_counts.update(route_counts)
    if total_loss is not None:
        routed_metrics["objective_loss"] = float(total_loss.detach().cpu())
        routed_counts["objective_loss"] = int(actions.shape[0])
    return total_loss, routed_metrics, routed_counts


def _add_gripper_trajectory_residual_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    gripper_trajectory_residual_mode: str,
    gripper_trajectory_residual_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if gripper_trajectory_residual_mode == "none":
        return loss, {}, {}
    if gripper_trajectory_residual_mode != "temporal_mlp":
        raise ValueError(
            "unsupported gripper_trajectory_residual_mode "
            f"{gripper_trajectory_residual_mode!r}"
        )
    trajectory_routed_actions = output.get("trajectory_routed_actions")
    trajectory_residuals = output.get("gripper_trajectory_residuals")
    if trajectory_routed_actions is None or trajectory_residuals is None:
        raise ValueError(
            "trajectory-routed gripper outputs are required for trajectory residual routing"
        )
    trajectory_loss, trajectory_loss_metrics, trajectory_loss_counts = _weighted_action_loss(
        trajectory_routed_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    total_loss = (
        loss + gripper_trajectory_residual_loss_weight * trajectory_loss
        if loss is not None
        else None
    )
    target_residuals = actions[..., -1] - _output_actions(output)[..., -1]
    residual_mse = (trajectory_residuals - target_residuals).square().mean()
    metrics = {
        f"trajectory_routed_{key}": value
        for key, value in action_metrics(trajectory_routed_actions, actions).items()
    }
    counts = {key: int(actions.shape[0]) for key in metrics}
    for key, value in trajectory_loss_metrics.items():
        metrics[f"trajectory_routed_{key}"] = value
        counts[f"trajectory_routed_{key}"] = trajectory_loss_counts[key]
    metrics["gripper_trajectory_residual_mse"] = float(residual_mse.detach().cpu())
    counts["gripper_trajectory_residual_mse"] = int(actions.shape[0])
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = int(actions.shape[0])
    return total_loss, metrics, counts


def _add_temporal_action_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    temporal_action_decoder_mode: str,
    temporal_action_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if temporal_action_decoder_mode == "none":
        return loss, {}, {}
    if temporal_action_decoder_mode not in {"sequence_mlp", "temporal_transformer"}:
        raise ValueError(
            f"unsupported temporal_action_decoder_mode {temporal_action_decoder_mode!r}"
        )
    temporal_actions = output.get("temporal_actions")
    if temporal_actions is None:
        raise ValueError("temporal action outputs are required for temporal action loss")
    temporal_loss, temporal_loss_metrics, temporal_loss_counts = _weighted_action_loss(
        temporal_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    total_loss = (
        loss + temporal_action_loss_weight * temporal_loss
        if loss is not None
        else None
    )
    metrics = {
        f"temporal_action_{key}": value
        for key, value in action_metrics(temporal_actions, actions).items()
    }
    counts = {key: int(actions.shape[0]) for key in metrics}
    for key, value in temporal_loss_metrics.items():
        metrics[f"temporal_action_{key}"] = value
        counts[f"temporal_action_{key}"] = temporal_loss_counts[key]
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = int(actions.shape[0])
    return total_loss, metrics, counts


def _add_flow_action_losses(
    loss: torch.Tensor | None,
    model: MotionPriorActionHead,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    flow_action_decoder_mode: str,
    flow_action_loss_weight: float,
    flow_matching_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if flow_action_decoder_mode == "none":
        return loss, {}, {}
    if flow_action_decoder_mode != "rectified_mlp":
        raise ValueError(f"unsupported flow_action_decoder_mode {flow_action_decoder_mode!r}")
    features = output.get("features")
    temporal_actions = output.get("temporal_actions")
    flow_actions = output.get("flow_actions")
    flow_velocity = output.get("flow_action_velocity")
    if features is None or temporal_actions is None or flow_actions is None or flow_velocity is None:
        raise ValueError("flow action decoder requires features, temporal_actions, and flow_actions")
    flow_loss, flow_loss_metrics, flow_loss_counts = _weighted_action_loss(
        flow_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    total_loss = (
        loss + flow_action_loss_weight * flow_loss
        if loss is not None
        else None
    )
    residual_target = actions - temporal_actions.detach()
    deploy_residual_mse = (flow_velocity - residual_target).square().mean()
    flow_matching_loss = flow_velocity.new_tensor(0.0)
    if flow_matching_loss_weight > 0.0:
        noise = torch.randn_like(residual_target)
        flow_time = torch.rand(residual_target.shape[0], device=residual_target.device)
        interp = (1.0 - flow_time).reshape(-1, 1, 1) * noise + flow_time.reshape(
            -1,
            1,
            1,
        ) * residual_target
        velocity_target = residual_target - noise
        flow_match_output = model.flow_action_outputs(
            features,
            temporal_actions.detach(),
            flow_noise=interp,
            flow_time=flow_time,
        )
        flow_match_velocity = flow_match_output["flow_action_velocity"]
        if flow_match_velocity is None:
            raise ValueError("flow action decoder did not return velocity")
        flow_matching_loss = (flow_match_velocity - velocity_target).square().mean()
        total_loss = (
            total_loss + flow_matching_loss_weight * flow_matching_loss
            if total_loss is not None
            else None
        )
    metrics = {
        f"flow_action_{key}": value
        for key, value in action_metrics(flow_actions, actions).items()
    }
    counts = {key: int(actions.shape[0]) for key in metrics}
    for key, value in flow_loss_metrics.items():
        metrics[f"flow_action_{key}"] = value
        counts[f"flow_action_{key}"] = flow_loss_counts[key]
    metrics["flow_action_residual_mse"] = float(deploy_residual_mse.detach().cpu())
    metrics["flow_action_flow_matching_loss"] = float(flow_matching_loss.detach().cpu())
    counts["flow_action_residual_mse"] = int(actions.shape[0])
    counts["flow_action_flow_matching_loss"] = int(actions.shape[0])
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = int(actions.shape[0])
    return total_loss, metrics, counts


def _add_sample_score_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    model: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor | None,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
    batch: dict[str, object],
    *,
    sample_score_mode: str,
    sample_score_loss_weight: float,
    sample_score_target: str,
    sample_score_loss_type: str,
    sample_score_temperature: float,
    temporal_action_decoder_mode: str,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if sample_score_mode == "none":
        return loss, {}, {}
    if sample_score_mode != "action_regret":
        raise ValueError(f"unsupported sample_score_mode {sample_score_mode!r}")
    sample_scores = output.get("sample_scores")
    sample_score_probs = output.get("sample_score_probs")
    if sample_scores is None or sample_score_probs is None:
        raise ValueError("sample score outputs are required when sample_score_mode is enabled")
    if future_inputs is None:
        if sample_score_loss_weight > 0.0:
            raise ValueError("sample score supervision requires future inputs")
        return loss, {}, {}
    if future_inputs.ndim != 3:
        raise ValueError(f"future_inputs must be [B,K,M], got {future_inputs.shape}")
    if sample_scores.shape != future_inputs.shape[:2]:
        raise ValueError(
            "sample_scores shape must match future_inputs [B,K]: "
            f"{sample_scores.shape} vs {future_inputs.shape[:2]}"
        )
    regrets = _sample_score_regrets(
        model,
        context,
        conditioning,
        future_inputs,
        sample_features,
        actions,
        batch,
        sample_score_target=sample_score_target,
        temporal_action_decoder_mode=temporal_action_decoder_mode,
    ).detach()
    targets = torch.softmax(-regrets / sample_score_temperature, dim=-1)
    log_probs = torch.log_softmax(sample_scores, dim=-1)
    soft_ce = -(targets * log_probs).sum(dim=-1).mean()
    best_indices = regrets.argmin(dim=-1)
    hard_ce = F.cross_entropy(sample_scores, best_indices)
    if sample_score_loss_type == "soft_ce":
        score_loss = soft_ce
    elif sample_score_loss_type == "hard_ce":
        score_loss = hard_ce
    elif sample_score_loss_type == "combined":
        score_loss = 0.5 * (soft_ce + hard_ce)
    else:
        raise ValueError(f"unsupported sample_score_loss_type {sample_score_loss_type!r}")
    total_loss = loss + sample_score_loss_weight * score_loss if loss is not None else None
    expected_regret = (sample_score_probs * regrets).sum(dim=-1)
    mean_regret = regrets.mean(dim=-1)
    best_regret = regrets.min(dim=-1).values
    pred_indices = sample_scores.argmax(dim=-1)
    metrics = {
        "sample_score_loss": float(score_loss.detach().cpu()),
        "sample_score_soft_ce": float(soft_ce.detach().cpu()),
        "sample_score_hard_ce": float(hard_ce.detach().cpu()),
        "sample_score_top1_accuracy": float(
            (pred_indices == best_indices).to(dtype=torch.float32).mean().detach().cpu()
        ),
        "sample_score_expected_regret": float(expected_regret.mean().detach().cpu()),
        "sample_score_mean_regret": float(mean_regret.mean().detach().cpu()),
        "sample_score_best_regret": float(best_regret.mean().detach().cpu()),
        "sample_score_best_vs_mean_gap": float((mean_regret - best_regret).mean().detach().cpu()),
        "sample_score_expected_vs_best_gap": float(
            (expected_regret - best_regret).mean().detach().cpu()
        ),
        "sample_score_entropy": float(
            (-(sample_score_probs * sample_score_probs.clamp_min(1e-12).log()).sum(dim=-1))
            .mean()
            .detach()
            .cpu()
        ),
    }
    counts = {key: int(actions.shape[0]) for key in metrics}
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = int(actions.shape[0])
    return total_loss, metrics, counts


def _sample_score_regrets(
    model: MotionPriorActionHead,
    context: torch.Tensor,
    conditioning: torch.Tensor | None,
    future_inputs: torch.Tensor,
    sample_features: torch.Tensor | None,
    actions: torch.Tensor,
    batch: dict[str, object],
    *,
    sample_score_target: str,
    temporal_action_decoder_mode: str,
) -> torch.Tensor:
    if sample_score_target == "motion_regret":
        motion_target = batch["motion"].to(device=future_inputs.device, dtype=future_inputs.dtype)
        if motion_target.shape != (future_inputs.shape[0], future_inputs.shape[-1]):
            raise ValueError(
                "motion target shape must match future_inputs [B,M]: "
                f"{motion_target.shape} vs {(future_inputs.shape[0], future_inputs.shape[-1])}"
            )
        return (future_inputs - motion_target.unsqueeze(1)).square().mean(dim=-1)
    if sample_score_target == "temporal_action_regret":
        if temporal_action_decoder_mode == "none":
            raise ValueError(
                "temporal_action_regret requires --temporal-action-decoder-mode"
            )
        predictions: list[torch.Tensor] = []
        for sample_index in range(future_inputs.shape[1]):
            single_output = model.forward_with_aux(
                context,
                future_inputs[:, sample_index : sample_index + 1, :],
                conditioning,
                sample_features[:, sample_index : sample_index + 1, :]
                if sample_features is not None
                else None,
            )
            temporal_actions = single_output.get("temporal_actions")
            if temporal_actions is None:
                raise ValueError("temporal actions are required for temporal_action_regret")
            predictions.append(temporal_actions)
        stacked = torch.stack(predictions, dim=1)
        return (stacked - actions.unsqueeze(1)).square().mean(dim=(2, 3))
    raise ValueError(f"unsupported sample_score_target {sample_score_target!r}")


def _add_gripper_step_residual_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    gripper_step_residual_mode: str,
    gripper_step_residual_loss_weight: float,
    gripper_step_loss_weight: float,
    gripper_step_target_mode: str = "command_state",
    gripper_step_positive_loss_weight: float = 1.0,
    gripper_step_oracle_boundary_residual_loss_weight: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if gripper_step_residual_mode == "none":
        return loss, {}, {}
    if gripper_step_residual_mode != "event_step":
        raise ValueError(f"unsupported gripper_step_residual_mode {gripper_step_residual_mode!r}")
    step_routed_actions = output.get("step_routed_actions")
    step_logits = output.get("gripper_step_logits")
    if step_routed_actions is None or step_logits is None:
        raise ValueError("step-routed gripper outputs are required for step residual routing")
    step_routed_loss, step_loss_metrics, step_loss_counts = _weighted_action_loss(
        step_routed_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    total_loss = (
        loss + gripper_step_residual_loss_weight * step_routed_loss
        if loss is not None
        else None
    )
    step_classes = _gripper_step_classes(gripper_step_target_mode)
    if step_logits.shape[-1] != len(step_classes):
        raise ValueError(
            "gripper step logits class dimension does not match target mode: "
            f"{step_logits.shape[-1]} vs {len(step_classes)}"
        )
    step_targets = _gripper_step_targets_for_mode(
        actions,
        batch,
        event_labels,
        target_mode=gripper_step_target_mode,
    )
    class_weight = None
    if gripper_step_positive_loss_weight != 1.0:
        class_weight = step_logits.new_ones(len(step_classes))
        class_weight[1:] = gripper_step_positive_loss_weight
    step_ce = F.cross_entropy(
        step_logits.reshape(-1, len(step_classes)),
        step_targets.reshape(-1),
        weight=class_weight,
    )
    total_loss = (
        total_loss + gripper_step_loss_weight * step_ce
        if total_loss is not None
        else None
    )
    oracle_step_metrics: dict[str, float] = {}
    oracle_step_counts: dict[str, int] = {}
    if gripper_step_target_mode == "boundary_start":
        base_actions = output.get("actions")
        step_residuals = output.get("gripper_step_residuals")
        if (
            (base_actions is None or step_residuals is None)
            and gripper_step_oracle_boundary_residual_loss_weight > 0.0
        ):
            raise ValueError(
                "oracle boundary step residual metrics require actions and gripper_step_residuals"
            )
        if base_actions is not None and step_residuals is not None:
            oracle_step_actions = _oracle_boundary_step_routed_actions(
                base_actions,
                step_targets,
                step_residuals,
            )
            oracle_step_loss, oracle_loss_metrics, oracle_loss_counts = _weighted_action_loss(
                oracle_step_actions,
                actions,
                batch,
                event_labels,
                loss_weight_mode=loss_weight_mode,
                transition_loss_weight=transition_loss_weight,
            )
            total_loss = (
                total_loss
                + gripper_step_oracle_boundary_residual_loss_weight * oracle_step_loss
                if total_loss is not None
                else None
            )
            oracle_step_metrics = {
                f"oracle_step_routed_{key}": value
                for key, value in action_metrics(oracle_step_actions, actions).items()
            }
            oracle_step_counts = {
                key: int(actions.shape[0])
                for key in oracle_step_metrics
            }
            for key, value in oracle_loss_metrics.items():
                oracle_step_metrics[f"oracle_step_routed_{key}"] = value
                oracle_step_counts[f"oracle_step_routed_{key}"] = oracle_loss_counts[key]

    step_pred = step_logits.argmax(dim=-1)
    step_metrics = {
        f"step_routed_{key}": value
        for key, value in action_metrics(step_routed_actions, actions).items()
    }
    step_counts = {key: int(actions.shape[0]) for key in step_metrics}
    for key, value in step_loss_metrics.items():
        step_metrics[f"step_routed_{key}"] = value
        step_counts[f"step_routed_{key}"] = step_loss_counts[key]
    step_metrics.update(oracle_step_metrics)
    step_counts.update(oracle_step_counts)
    step_metrics["gripper_step_ce"] = float(step_ce.detach().cpu())
    step_metrics["gripper_step_accuracy"] = float(
        (step_pred == step_targets).to(dtype=torch.float32).mean().detach().cpu()
    )
    positive_fraction = float(
        (step_targets != 0)
        .to(dtype=torch.float32)
        .mean()
        .detach()
        .cpu()
    )
    step_metrics["gripper_step_positive_fraction"] = positive_fraction
    fraction_key = (
        "gripper_step_boundary_fraction"
        if gripper_step_target_mode == "boundary_start"
        else "gripper_step_command_fraction"
    )
    step_metrics[fraction_key] = positive_fraction
    step_counts["gripper_step_ce"] = int(actions.shape[0])
    step_counts["gripper_step_accuracy"] = int(actions.shape[0])
    step_counts["gripper_step_positive_fraction"] = int(actions.shape[0])
    step_counts[fraction_key] = int(actions.shape[0])
    if total_loss is not None:
        step_metrics["objective_loss"] = float(total_loss.detach().cpu())
        step_counts["objective_loss"] = int(actions.shape[0])
    return total_loss, step_metrics, step_counts


def _oracle_boundary_step_routed_actions(
    actions: torch.Tensor,
    step_targets: torch.Tensor,
    step_residuals: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(f"actions must be [B,H,A], got {actions.shape}")
    if actions.shape[-1] <= 6:
        raise ValueError("oracle boundary step routing requires a gripper action channel")
    if step_targets.shape != actions.shape[:2]:
        raise ValueError(
            "step_targets shape must match actions [B,H]: "
            f"{step_targets.shape} vs {actions.shape[:2]}"
        )
    if step_residuals.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "step_residuals shape must match actions [B,H,*]: "
            f"{step_residuals.shape[:2]} vs {actions.shape[:2]}"
        )
    if step_residuals.shape[-1] <= 1:
        raise ValueError("oracle boundary step routing requires positive residual classes")
    step_targets = step_targets.to(device=step_residuals.device, dtype=torch.long)
    if step_targets.min() < 0 or step_targets.max() >= step_residuals.shape[-1]:
        raise ValueError("step_targets contain an out-of-range class index")
    target_residual = torch.gather(
        step_residuals,
        dim=-1,
        index=step_targets.unsqueeze(-1),
    ).squeeze(-1)
    target_residual = torch.where(
        step_targets > 0,
        target_residual,
        torch.zeros_like(target_residual),
    )
    output = actions.clone()
    output[..., -1] = output[..., -1] + target_residual
    return output


def _predicted_boundary_step_routed_actions(
    actions: torch.Tensor,
    step_logits: torch.Tensor,
    step_residuals: torch.Tensor,
    *,
    threshold: float | None,
) -> torch.Tensor:
    if step_logits.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "step_logits shape must match actions [B,H,*]: "
            f"{step_logits.shape[:2]} vs {actions.shape[:2]}"
        )
    if step_logits.shape != step_residuals.shape:
        raise ValueError(
            "step_logits shape must match step_residuals shape: "
            f"{step_logits.shape} vs {step_residuals.shape}"
        )
    if step_logits.shape[-1] <= 1:
        raise ValueError("predicted boundary routing requires positive classes")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if threshold is None:
        step_targets = step_logits.argmax(dim=-1)
    else:
        positive_scores, positive_offsets = torch.softmax(step_logits, dim=-1)[
            ..., 1:
        ].max(dim=-1)
        step_targets = positive_offsets + 1
        step_targets = torch.where(
            positive_scores >= threshold,
            step_targets,
            torch.zeros_like(step_targets),
        )
    return _oracle_boundary_step_routed_actions(actions, step_targets, step_residuals)


def _add_gripper_boundary_index_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    loss_weight_mode: str,
    transition_loss_weight: float,
    gripper_boundary_index_mode: str,
    gripper_boundary_index_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if gripper_boundary_index_mode == "none":
        return loss, {}, {}
    if gripper_boundary_index_mode != "boundary_index":
        raise ValueError(f"unsupported gripper_boundary_index_mode {gripper_boundary_index_mode!r}")
    boundary_logits = output.get("gripper_boundary_index_logits")
    base_actions = output.get("actions")
    step_residuals = output.get("gripper_step_residuals")
    if boundary_logits is None or base_actions is None or step_residuals is None:
        raise ValueError(
            "boundary-index localizer requires boundary logits, actions, and step residuals"
        )
    targets = _gripper_boundary_index_targets(
        batch,
        event_labels,
        horizon=int(actions.shape[1]),
        device=actions.device,
    )
    if boundary_logits.shape != (actions.shape[0], 2, actions.shape[1] + 1):
        raise ValueError(
            "boundary index logits must be [B,2,H+1], got "
            f"{boundary_logits.shape} for actions {actions.shape}"
        )
    class_count = int(boundary_logits.shape[-1])
    index_ce = F.cross_entropy(
        boundary_logits.reshape(-1, class_count),
        targets.reshape(-1),
    )
    total_loss = (
        loss + gripper_boundary_index_loss_weight * index_ce
        if loss is not None
        else None
    )
    pred_actions = _boundary_index_predicted_actions(
        base_actions,
        boundary_logits,
        step_residuals,
    )
    pred_loss, pred_loss_metrics, pred_loss_counts = _weighted_action_loss(
        pred_actions,
        actions,
        batch,
        event_labels,
        loss_weight_mode=loss_weight_mode,
        transition_loss_weight=transition_loss_weight,
    )
    metrics = {
        f"boundary_index_pred_{key}": value
        for key, value in action_metrics(pred_actions, actions).items()
    }
    counts = {key: int(actions.shape[0]) for key in metrics}
    for key, value in pred_loss_metrics.items():
        metrics[f"boundary_index_pred_{key}"] = value
        counts[f"boundary_index_pred_{key}"] = pred_loss_counts[key]
    pred = boundary_logits.argmax(dim=-1)
    valid_mask = targets < (class_count - 1)
    valid_close = valid_mask[:, 0]
    valid_open = valid_mask[:, 1]
    metrics["gripper_boundary_index_ce"] = float(index_ce.detach().cpu())
    metrics["gripper_boundary_index_accuracy"] = float(
        (pred == targets).to(dtype=torch.float32).mean().detach().cpu()
    )
    metrics["gripper_boundary_index_event_fraction"] = float(
        valid_mask.to(dtype=torch.float32).mean().detach().cpu()
    )
    metrics["gripper_boundary_index_close_accuracy"] = _masked_accuracy(
        pred[:, 0],
        targets[:, 0],
        valid_close,
    )
    metrics["gripper_boundary_index_open_accuracy"] = _masked_accuracy(
        pred[:, 1],
        targets[:, 1],
        valid_open,
    )
    metrics["gripper_boundary_index_close_within1"] = _masked_within_one(
        pred[:, 0],
        targets[:, 0],
        valid_close,
    )
    metrics["gripper_boundary_index_open_within1"] = _masked_within_one(
        pred[:, 1],
        targets[:, 1],
        valid_open,
    )
    metric_count = int(actions.shape[0])
    counts.update({key: metric_count for key in metrics if key not in counts})
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = metric_count
    return total_loss, metrics, counts


def _add_event_time_losses(
    loss: torch.Tensor | None,
    output: dict[str, torch.Tensor | None],
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    event_time_conditioning_mode: str,
    event_time_loss_weight: float,
) -> tuple[torch.Tensor | None, dict[str, float], dict[str, int]]:
    if event_time_conditioning_mode == "none":
        return loss, {}, {}
    if event_time_conditioning_mode != "soft_boundary":
        raise ValueError(
            f"unsupported event_time_conditioning_mode {event_time_conditioning_mode!r}"
        )
    logits = output.get("event_time_logits")
    probs = output.get("event_time_probs")
    if logits is None or probs is None:
        raise ValueError("event-time conditioning requires event_time logits/probs")
    if logits.shape != (actions.shape[0], 2, actions.shape[1] + 1):
        raise ValueError(
            "event_time_logits must be [B,2,H+1], got "
            f"{logits.shape} for actions {actions.shape}"
        )
    targets = _gripper_boundary_index_targets(
        batch,
        event_labels,
        horizon=int(actions.shape[1]),
        device=actions.device,
    )
    class_count = int(logits.shape[-1])
    event_time_ce = F.cross_entropy(
        logits.reshape(-1, class_count),
        targets.reshape(-1),
    )
    total_loss = (
        loss + event_time_loss_weight * event_time_ce
        if loss is not None
        else None
    )
    pred = logits.argmax(dim=-1)
    valid_mask = targets < (class_count - 1)
    valid_close = valid_mask[:, 0]
    valid_open = valid_mask[:, 1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    metrics = {
        "event_time_ce": float(event_time_ce.detach().cpu()),
        "event_time_accuracy": float(
            (pred == targets).to(dtype=torch.float32).mean().detach().cpu()
        ),
        "event_time_event_fraction": float(
            valid_mask.to(dtype=torch.float32).mean().detach().cpu()
        ),
        "event_time_close_accuracy": _masked_accuracy(pred[:, 0], targets[:, 0], valid_close),
        "event_time_open_accuracy": _masked_accuracy(pred[:, 1], targets[:, 1], valid_open),
        "event_time_close_within1": _masked_within_one(pred[:, 0], targets[:, 0], valid_close),
        "event_time_open_within1": _masked_within_one(pred[:, 1], targets[:, 1], valid_open),
        "event_time_entropy": float(entropy.mean().detach().cpu()),
    }
    count = int(actions.shape[0])
    counts = {key: count for key in metrics}
    if total_loss is not None:
        metrics["objective_loss"] = float(total_loss.detach().cpu())
        counts["objective_loss"] = count
    return total_loss, metrics, counts


def _gripper_boundary_index_targets(
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if event_labels is None:
        raise ValueError("event labels are required for boundary-index targets")
    batch_size = _batch_size_from_window_ids(batch["window_id"])
    no_event = horizon
    targets = torch.full((batch_size, 2), no_event, dtype=torch.long, device=device)
    for row in range(batch_size):
        record = event_labels.get(_batch_string_at(batch["window_id"], row))
        close_step = _event_step_value(record, "close_step")
        open_step = _event_step_value(record, "open_step")
        if close_step is not None and 0 <= close_step < horizon:
            targets[row, 0] = close_step
        if open_step is not None and 0 <= open_step < horizon:
            targets[row, 1] = open_step
    return targets


def _boundary_index_predicted_actions(
    actions: torch.Tensor,
    boundary_logits: torch.Tensor,
    step_residuals: torch.Tensor,
) -> torch.Tensor:
    if boundary_logits.ndim != 3 or boundary_logits.shape[1] != 2:
        raise ValueError(f"boundary_logits must be [B,2,H+1], got {boundary_logits.shape}")
    if step_residuals.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "step_residuals shape must match actions [B,H,*]: "
            f"{step_residuals.shape[:2]} vs {actions.shape[:2]}"
        )
    if boundary_logits.shape[0] != actions.shape[0] or boundary_logits.shape[-1] != actions.shape[1] + 1:
        raise ValueError(
            "boundary_logits shape must match actions batch/horizon: "
            f"{boundary_logits.shape} vs {actions.shape}"
        )
    if step_residuals.shape[-1] < 3:
        raise ValueError("boundary-index routing requires close/open residual classes")
    horizon = int(actions.shape[1])
    no_event = horizon
    pred_indices = boundary_logits.argmax(dim=-1)
    targets = torch.full(actions.shape[:2], no_event, dtype=torch.long, device=actions.device)
    close_indices = pred_indices[:, 0]
    open_indices = pred_indices[:, 1]
    row_indices = torch.arange(actions.shape[0], device=actions.device)
    close_valid = close_indices < horizon
    open_valid = open_indices < horizon
    targets[row_indices[close_valid], close_indices[close_valid]] = 1
    targets[row_indices[open_valid], open_indices[open_valid]] = 2
    targets = torch.where(
        targets == no_event,
        torch.zeros_like(targets),
        targets,
    )
    return _oracle_boundary_step_routed_actions(actions, targets, step_residuals)


def _masked_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().detach().cpu())
    if count == 0:
        return 0.0
    return float((pred[mask] == target[mask]).to(dtype=torch.float32).mean().detach().cpu())


def _masked_within_one(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().detach().cpu())
    if count == 0:
        return 0.0
    return float((pred[mask] - target[mask]).abs().le(1).to(dtype=torch.float32).mean().detach().cpu())


def _gripper_step_classes(target_mode: str) -> tuple[str, ...]:
    if target_mode == "command_state":
        return GRIPPER_STEP_CLASSES
    if target_mode == "boundary_start":
        return GRIPPER_BOUNDARY_STEP_CLASSES
    raise ValueError(f"unsupported gripper step target mode {target_mode!r}")


def _gripper_step_targets_for_mode(
    actions: torch.Tensor,
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "command_state":
        return _gripper_step_targets(actions)
    if target_mode == "boundary_start":
        return _gripper_boundary_step_targets(
            batch,
            event_labels,
            horizon=int(actions.shape[1]),
            device=actions.device,
        )
    raise ValueError(f"unsupported gripper step target mode {target_mode!r}")


def _gripper_step_targets(
    actions: torch.Tensor,
    *,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    if command_threshold <= 0.0:
        raise ValueError("command_threshold must be positive")
    gripper = actions[..., -1]
    targets = torch.full_like(
        gripper,
        fill_value=GRIPPER_STEP_CLASSES.index("sustain"),
        dtype=torch.long,
    )
    targets = torch.where(
        gripper >= command_threshold,
        torch.full_like(targets, GRIPPER_STEP_CLASSES.index("close")),
        targets,
    )
    targets = torch.where(
        gripper <= -command_threshold,
        torch.full_like(targets, GRIPPER_STEP_CLASSES.index("open")),
        targets,
    )
    return targets


def _gripper_boundary_step_targets(
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if event_labels is None:
        raise ValueError("event labels are required for boundary-start step targets")
    batch_size = _batch_size_from_window_ids(batch["window_id"])
    targets = torch.full(
        (batch_size, horizon),
        fill_value=GRIPPER_BOUNDARY_STEP_CLASSES.index("no_boundary"),
        dtype=torch.long,
        device=device,
    )
    close_index = GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")
    open_index = GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")
    for row in range(batch_size):
        record = event_labels.get(_batch_string_at(batch["window_id"], row))
        close_step = _event_step_value(record, "close_step")
        open_step = _event_step_value(record, "open_step")
        if close_step is not None and 0 <= close_step < horizon:
            targets[row, close_step] = close_index
        if open_step is not None and 0 <= open_step < horizon:
            targets[row, open_step] = open_index
    return targets


def _gripper_route_targets(
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if event_labels is None:
        raise ValueError("event labels are required for gripper residual routing")
    values = [
        _gripper_route_target_for_label(
            _event_mode_for_record(event_labels.get(_batch_string_at(batch["window_id"], row)))
        )
        for row in range(batch_size)
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def _gripper_route_target_for_label(label: str) -> int:
    if label.startswith("transition_close"):
        return GRIPPER_ROUTE_FAMILIES.index("transition_close")
    if label.startswith("transition_open"):
        return GRIPPER_ROUTE_FAMILIES.index("transition_open")
    if label.startswith("sustain_") or label.startswith("hold"):
        return GRIPPER_ROUTE_FAMILIES.index("sustain")
    return -1


def _replace_action_gripper(pred_actions: torch.Tensor, aux_gripper: torch.Tensor) -> torch.Tensor:
    if aux_gripper.shape != pred_actions[..., -1].shape:
        raise ValueError(
            "aux_gripper shape must match action gripper shape: "
            f"{aux_gripper.shape} vs {pred_actions[..., -1].shape}"
        )
    output = pred_actions.clone()
    output[..., -1] = aux_gripper.to(dtype=pred_actions.dtype, device=pred_actions.device)
    return output


def _output_actions(output: dict[str, torch.Tensor | None]) -> torch.Tensor:
    actions = output.get("actions")
    if actions is None:
        raise ValueError("model output is missing actions")
    return actions


def _transition_mask(
    batch: dict[str, object],
    event_labels: dict[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if event_labels is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    values = [
        event_label_is_transition(
            _event_mode_for_record(event_labels.get(_batch_string_at(batch["window_id"], row)))
        )
        for row in range(batch_size)
    ]
    return torch.tensor(values, dtype=torch.bool, device=device)


def _load_event_label_records(event_mode_audit_json: str | Path) -> dict[str, dict[str, Any]]:
    report = json.loads(Path(event_mode_audit_json).expanduser().read_text(encoding="utf-8"))
    labels = report.get("window_labels")
    if not isinstance(labels, list):
        raise ValueError("event-mode audit JSON must include window_labels")
    records: dict[str, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("event-mode audit window_labels must contain objects")
        records[str(item["window_id"])] = dict(item)
    return records


def _event_mode_for_record(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return str(record.get("event_mode", ""))
    return ""


def _event_step_value(record: Any, field: str) -> int | None:
    if not isinstance(record, dict):
        return None
    value = record.get(field)
    if value is None:
        return None
    return int(value)


def _add_group_loss_metrics(
    metrics: dict[str, float],
    counts: dict[str, int],
    per_item_mse: torch.Tensor,
    transition_mask: torch.Tensor,
) -> None:
    transition_count = int(transition_mask.sum().detach().cpu())
    sustain_count = int((~transition_mask).sum().detach().cpu())
    if transition_count > 0:
        metrics["transition_mse"] = float(per_item_mse[transition_mask].mean().detach().cpu())
        counts["transition_mse"] = transition_count
    if sustain_count > 0:
        metrics["sustain_mse"] = float(per_item_mse[~transition_mask].mean().detach().cpu())
        counts["sustain_mse"] = sustain_count


def _add_metric_values(
    totals: dict[str, float],
    counts: dict[str, int],
    metrics: dict[str, float],
    metric_counts: int | dict[str, int],
) -> None:
    for key, value in metrics.items():
        count = metric_counts if isinstance(metric_counts, int) else metric_counts[key]
        totals[key] = totals.get(key, 0.0) + float(value) * int(count)
        counts[key] = counts.get(key, 0) + int(count)


def _finalize_metric_values(
    totals: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float | None]:
    if not counts:
        return {"loss": None, "mse": None}
    return {
        key: totals[key] / counts[key] if counts[key] else None
        for key in sorted(totals)
    }


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]


def _batch_size_from_window_ids(values: object) -> int:
    if isinstance(values, (list, tuple)):
        return len(values)
    return len(values)  # type: ignore[arg-type]


def _cvae_config(metrics: dict[str, Any]) -> dict[str, object]:
    return {
        "hidden_dims": metrics.get("hidden_dims"),
        "latent_dim": metrics.get("latent_dim"),
        "free_bits": metrics.get("free_bits"),
        "beta_kl": metrics.get("beta_kl"),
        "prior_recon_weight": metrics.get("prior_recon_weight"),
        "action_aware_loss_weight": metrics.get("action_aware_loss_weight"),
        "event_conditioning": metrics.get("event_conditioning"),
    }


def _probe_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe": metrics["probe"],
        "conditioning": metrics["conditioning"],
        "best_epoch": metrics.get("best_epoch"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
        "final": metrics.get("final"),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
