"""Action heads that consume GeoMoCo-WM future-motion priors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MotionPriorActionHead(nn.Module):
    """Decode actions from context plus an optional set of future-motion samples."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        action_dim: int,
        horizon: int,
        conditioning_dim: int = 0,
        hidden_dims: tuple[int, ...] = (512, 512),
        token_dim: int = 256,
        num_heads: int = 4,
        temporal_layers: int = 1,
        set_aggregator: str = "context_attention",
        set_query_count: int = 4,
        sample_feature_dim: int = 0,
        aux_gripper_head: bool = False,
        gripper_residual_mode: str = "none",
        gripper_route_count: int = 3,
        gripper_step_residual_mode: str = "none",
        gripper_step_class_count: int = 3,
        gripper_step_residual_blend: str = "all_classes",
        gripper_boundary_index_mode: str = "none",
        gripper_trajectory_residual_mode: str = "none",
        event_time_conditioning_mode: str = "none",
        temporal_action_decoder_mode: str = "none",
        flow_action_decoder_mode: str = "none",
        sample_score_mode: str = "none",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if motion_dim <= 0:
            raise ValueError("motion_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if motion_dim % horizon != 0:
            raise ValueError("motion_dim must be divisible by horizon")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if temporal_layers < 0:
            raise ValueError("temporal_layers must be non-negative")
        if set_aggregator not in {"mean_pool", "context_attention", "multi_query_attention"}:
            raise ValueError(
                "set_aggregator must be one of: "
                "mean_pool, context_attention, multi_query_attention"
            )
        if set_query_count <= 0:
            raise ValueError("set_query_count must be positive")
        if sample_feature_dim < 0:
            raise ValueError("sample_feature_dim must be non-negative")
        if not isinstance(aux_gripper_head, bool):
            raise ValueError("aux_gripper_head must be a bool")
        if gripper_residual_mode not in {"none", "event_family"}:
            raise ValueError("gripper_residual_mode must be one of: none, event_family")
        if gripper_route_count <= 0:
            raise ValueError("gripper_route_count must be positive")
        if gripper_step_residual_mode not in {"none", "event_step"}:
            raise ValueError("gripper_step_residual_mode must be one of: none, event_step")
        if gripper_step_class_count <= 0:
            raise ValueError("gripper_step_class_count must be positive")
        if gripper_step_residual_blend not in {"all_classes", "positive_only"}:
            raise ValueError(
                "gripper_step_residual_blend must be one of: all_classes, positive_only"
            )
        if gripper_boundary_index_mode not in {"none", "boundary_index"}:
            raise ValueError(
                "gripper_boundary_index_mode must be one of: none, boundary_index"
            )
        if gripper_trajectory_residual_mode not in {"none", "temporal_mlp"}:
            raise ValueError(
                "gripper_trajectory_residual_mode must be one of: none, temporal_mlp"
            )
        if event_time_conditioning_mode not in {"none", "soft_boundary"}:
            raise ValueError(
                "event_time_conditioning_mode must be one of: none, soft_boundary"
            )
        if temporal_action_decoder_mode not in {"none", "sequence_mlp", "temporal_transformer"}:
            raise ValueError(
                "temporal_action_decoder_mode must be one of: "
                "none, sequence_mlp, temporal_transformer"
            )
        if event_time_conditioning_mode != "none" and temporal_action_decoder_mode == "none":
            raise ValueError(
                "event time conditioning requires temporal_action_decoder_mode"
            )
        if flow_action_decoder_mode not in {"none", "rectified_mlp"}:
            raise ValueError("flow_action_decoder_mode must be one of: none, rectified_mlp")
        if flow_action_decoder_mode != "none" and temporal_action_decoder_mode == "none":
            raise ValueError(
                "flow action decoder requires temporal_action_decoder_mode"
            )
        if sample_score_mode not in {"none", "action_regret"}:
            raise ValueError("sample_score_mode must be one of: none, action_regret")
        if (
            gripper_residual_mode != "none"
            or gripper_step_residual_mode != "none"
            or gripper_trajectory_residual_mode != "none"
        ) and action_dim <= 6:
            raise ValueError("gripper residual routing requires a gripper action channel")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")

        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.motion_step_dim = motion_dim // horizon
        self.action_dim = action_dim
        self.horizon = horizon
        self.conditioning_dim = conditioning_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.temporal_layers = temporal_layers
        self.set_aggregator = set_aggregator
        self.set_query_count = set_query_count
        self.sample_feature_dim = sample_feature_dim
        self.aux_gripper_head = aux_gripper_head
        self.gripper_residual_mode = gripper_residual_mode
        self.gripper_route_count = gripper_route_count
        self.gripper_step_residual_mode = gripper_step_residual_mode
        self.gripper_step_class_count = gripper_step_class_count
        self.gripper_step_residual_blend = gripper_step_residual_blend
        self.gripper_boundary_index_mode = gripper_boundary_index_mode
        self.gripper_boundary_index_class_count = horizon + 1
        self.gripper_trajectory_residual_mode = gripper_trajectory_residual_mode
        self.event_time_conditioning_mode = event_time_conditioning_mode
        self.event_time_class_count = horizon + 1
        self.temporal_action_decoder_mode = temporal_action_decoder_mode
        self.flow_action_decoder_mode = flow_action_decoder_mode
        self.sample_score_mode = sample_score_mode
        base_dim = context_dim + conditioning_dim
        feature_dim = token_dim * 3 + 3

        self.context_encoder = _norm_mlp(base_dim, (token_dim,), token_dim, dropout=0.0)
        self.step_encoder = nn.Sequential(
            nn.Linear(self.motion_step_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(horizon, token_dim))
        if temporal_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=token_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal_encoder: nn.Module = nn.TransformerEncoder(
                layer,
                num_layers=temporal_layers,
            )
        else:
            self.temporal_encoder = nn.Identity()
        if sample_feature_dim > 0:
            self.sample_feature_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(sample_feature_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )
            sample_pool_dim = token_dim * 3 + 4
        else:
            self.sample_feature_encoder = None
            sample_pool_dim = token_dim * 2 + 4
        self.sample_pool = nn.Sequential(
            nn.Linear(sample_pool_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
        )
        self.set_attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        if set_aggregator == "multi_query_attention":
            self.set_query_offsets = nn.Parameter(torch.zeros(set_query_count, token_dim))
        else:
            self.register_parameter("set_query_offsets", None)
        self.empty_future_token = nn.Parameter(torch.zeros(token_dim))
        self.head = _norm_mlp(
            feature_dim,
            hidden_dims,
            action_dim * horizon,
            dropout=dropout,
        )
        self.aux_gripper = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                horizon,
                dropout=dropout,
            )
            if aux_gripper_head
            else None
        )
        self.gripper_route_logits = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                gripper_route_count,
                dropout=dropout,
            )
            if gripper_residual_mode == "event_family"
            else None
        )
        self.gripper_residual = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                horizon * gripper_route_count,
                dropout=dropout,
            )
            if gripper_residual_mode == "event_family"
            else None
        )
        self.gripper_step_logits = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                horizon * gripper_step_class_count,
                dropout=dropout,
            )
            if gripper_step_residual_mode == "event_step"
            else None
        )
        self.gripper_step_residual = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                horizon * gripper_step_class_count,
                dropout=dropout,
            )
            if gripper_step_residual_mode == "event_step"
            else None
        )
        self.gripper_boundary_index_logits = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                2 * self.gripper_boundary_index_class_count,
                dropout=dropout,
            )
            if gripper_boundary_index_mode == "boundary_index"
            else None
        )
        self.gripper_trajectory_residual = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                horizon,
                dropout=dropout,
            )
            if gripper_trajectory_residual_mode == "temporal_mlp"
            else None
        )
        self.event_time_logits = (
            _norm_mlp(
                feature_dim,
                hidden_dims,
                2 * self.event_time_class_count,
                dropout=dropout,
            )
            if event_time_conditioning_mode == "soft_boundary"
            else None
        )
        self.event_time_feature = (
            nn.Sequential(
                nn.Linear(2 * self.event_time_class_count, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )
            if event_time_conditioning_mode == "soft_boundary"
            else None
        )
        self.temporal_action_feature = (
            nn.Sequential(
                nn.Linear(feature_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )
            if temporal_action_decoder_mode != "none"
            else None
        )
        if temporal_action_decoder_mode != "none":
            self.temporal_action_queries = nn.Parameter(torch.zeros(horizon, token_dim))
        else:
            self.register_parameter("temporal_action_queries", None)
        if temporal_action_decoder_mode == "temporal_transformer":
            temporal_decoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=token_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal_action_sequence_encoder: nn.Module | None = nn.TransformerEncoder(
                temporal_decoder_layer,
                num_layers=max(1, temporal_layers),
            )
        else:
            self.temporal_action_sequence_encoder = None
        self.temporal_action_decoder = (
            _norm_mlp(
                token_dim * 2
                if temporal_action_decoder_mode == "sequence_mlp"
                else token_dim,
                hidden_dims,
                action_dim,
                dropout=dropout,
            )
            if temporal_action_decoder_mode != "none"
            else None
        )
        self.flow_action_time = (
            nn.Sequential(
                nn.Linear(1, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )
            if flow_action_decoder_mode == "rectified_mlp"
            else None
        )
        self.flow_action_feature = (
            nn.Sequential(
                nn.Linear(feature_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )
            if flow_action_decoder_mode == "rectified_mlp"
            else None
        )
        self.flow_action_decoder = (
            _norm_mlp(
                token_dim * 2 + action_dim * 2,
                hidden_dims,
                action_dim,
                dropout=dropout,
            )
            if flow_action_decoder_mode == "rectified_mlp"
            else None
        )
        self.sample_score = (
            _norm_mlp(
                token_dim * 2,
                (token_dim,),
                1,
                dropout=dropout,
            )
            if sample_score_mode == "action_regret"
            else None
        )

    def forward(
        self,
        context: Tensor,
        future_motions: Tensor | None = None,
        conditioning: Tensor | None = None,
        sample_features: Tensor | None = None,
    ) -> Tensor:
        """Return an action chunk of shape ``[B, horizon, action_dim]``."""

        features, _ = self._features_with_aux(context, future_motions, conditioning, sample_features)
        flat_actions = self.head(features)
        return flat_actions.reshape(context.shape[0], self.horizon, self.action_dim)

    def forward_with_aux(
        self,
        context: Tensor,
        future_motions: Tensor | None = None,
        conditioning: Tensor | None = None,
        sample_features: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Return actions and optional auxiliary predictions."""

        features, feature_aux = self._features_with_aux(
            context,
            future_motions,
            conditioning,
            sample_features,
        )
        flat_actions = self.head(features)
        actions = flat_actions.reshape(context.shape[0], self.horizon, self.action_dim)
        aux_gripper = self.aux_gripper(features) if self.aux_gripper is not None else None
        routed = self._routed_gripper_actions(features, actions)
        step_routed = self._step_routed_gripper_actions(features, actions)
        boundary_index = self._boundary_index_outputs(features, actions.shape[0])
        trajectory_routed = self._trajectory_routed_gripper_actions(features, actions)
        event_time = self._event_time_outputs(features, actions.shape[0])
        temporal_actions = self._temporal_action_outputs(
            features,
            actions.shape[0],
            event_time.get("event_time_probs"),
        )
        flow_actions = self._flow_action_outputs(
            features,
            temporal_actions.get("temporal_actions"),
            flow_noise=None,
            flow_time=None,
        )
        return {
            "features": features,
            "actions": actions,
            "aux_gripper": aux_gripper,
            **routed,
            **step_routed,
            **boundary_index,
            **trajectory_routed,
            **event_time,
            **temporal_actions,
            **flow_actions,
            **feature_aux,
        }

    def flow_action_outputs(
        self,
        features: Tensor,
        temporal_actions: Tensor,
        flow_noise: Tensor | None = None,
        flow_time: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Return residual-flow predictions from already computed features."""

        return self._flow_action_outputs(
            features,
            temporal_actions,
            flow_noise=flow_noise,
            flow_time=flow_time,
        )

    def _routed_gripper_actions(self, features: Tensor, actions: Tensor) -> dict[str, Tensor | None]:
        if self.gripper_residual_mode == "none":
            return {
                "routed_actions": None,
                "gripper_route_logits": None,
                "gripper_route_probs": None,
                "gripper_residuals": None,
            }
        if self.gripper_route_logits is None or self.gripper_residual is None:
            raise RuntimeError("gripper residual routing modules are missing")
        route_logits = self.gripper_route_logits(features)
        route_probs = torch.softmax(route_logits, dim=-1)
        residuals = self.gripper_residual(features).reshape(
            actions.shape[0],
            self.horizon,
            self.gripper_route_count,
        )
        blended_residual = torch.einsum("br,bhr->bh", route_probs, residuals)
        routed_actions = actions.clone()
        routed_actions[..., -1] = routed_actions[..., -1] + blended_residual
        return {
            "routed_actions": routed_actions,
            "gripper_route_logits": route_logits,
            "gripper_route_probs": route_probs,
            "gripper_residuals": residuals,
        }

    def _step_routed_gripper_actions(
        self,
        features: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor | None]:
        if self.gripper_step_residual_mode == "none":
            return {
                "step_routed_actions": None,
                "gripper_step_logits": None,
                "gripper_step_probs": None,
                "gripper_step_residuals": None,
            }
        if self.gripper_step_logits is None or self.gripper_step_residual is None:
            raise RuntimeError("gripper step residual modules are missing")
        step_logits = self.gripper_step_logits(features).reshape(
            actions.shape[0],
            self.horizon,
            self.gripper_step_class_count,
        )
        step_probs = torch.softmax(step_logits, dim=-1)
        step_residuals = self.gripper_step_residual(features).reshape(
            actions.shape[0],
            self.horizon,
            self.gripper_step_class_count,
        )
        return self._step_routed_gripper_actions_from_tensors(
            actions,
            step_logits,
            step_probs,
            step_residuals,
        )

    def _step_routed_gripper_actions_from_tensors(
        self,
        actions: Tensor,
        step_logits: Tensor,
        step_probs: Tensor,
        step_residuals: Tensor,
    ) -> dict[str, Tensor | None]:
        if self.gripper_step_residual_blend == "all_classes":
            blended_residual = torch.sum(step_probs * step_residuals, dim=-1)
        elif self.gripper_step_residual_blend == "positive_only":
            if self.gripper_step_class_count <= 1:
                blended_residual = torch.zeros_like(actions[..., -1])
            else:
                blended_residual = torch.sum(
                    step_probs[..., 1:] * step_residuals[..., 1:],
                    dim=-1,
                )
        else:
            raise ValueError(
                f"unsupported gripper_step_residual_blend {self.gripper_step_residual_blend!r}"
            )
        step_routed_actions = actions.clone()
        step_routed_actions[..., -1] = step_routed_actions[..., -1] + blended_residual
        return {
            "step_routed_actions": step_routed_actions,
            "gripper_step_logits": step_logits,
            "gripper_step_probs": step_probs,
            "gripper_step_residuals": step_residuals,
        }

    def _boundary_index_outputs(
        self,
        features: Tensor,
        batch_size: int,
    ) -> dict[str, Tensor | None]:
        if self.gripper_boundary_index_mode == "none":
            return {
                "gripper_boundary_index_logits": None,
                "gripper_boundary_index_probs": None,
            }
        if self.gripper_boundary_index_logits is None:
            raise RuntimeError("gripper boundary index module is missing")
        logits = self.gripper_boundary_index_logits(features).reshape(
            batch_size,
            2,
            self.gripper_boundary_index_class_count,
        )
        probs = torch.softmax(logits, dim=-1)
        return {
            "gripper_boundary_index_logits": logits,
            "gripper_boundary_index_probs": probs,
        }

    def _trajectory_routed_gripper_actions(
        self,
        features: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor | None]:
        if self.gripper_trajectory_residual_mode == "none":
            return {
                "trajectory_routed_actions": None,
                "gripper_trajectory_residuals": None,
            }
        if self.gripper_trajectory_residual_mode != "temporal_mlp":
            raise ValueError(
                "unsupported gripper_trajectory_residual_mode "
                f"{self.gripper_trajectory_residual_mode!r}"
            )
        if self.gripper_trajectory_residual is None:
            raise RuntimeError("gripper trajectory residual module is missing")
        residuals = self.gripper_trajectory_residual(features).reshape(
            actions.shape[0],
            self.horizon,
        )
        trajectory_routed_actions = actions.clone()
        trajectory_routed_actions[..., -1] = (
            trajectory_routed_actions[..., -1] + residuals
        )
        return {
            "trajectory_routed_actions": trajectory_routed_actions,
            "gripper_trajectory_residuals": residuals,
        }

    def _temporal_action_outputs(
        self,
        features: Tensor,
        batch_size: int,
        event_time_probs: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        if self.temporal_action_decoder_mode == "none":
            return {"temporal_actions": None}
        if self.temporal_action_decoder_mode not in {"sequence_mlp", "temporal_transformer"}:
            raise ValueError(
                f"unsupported temporal_action_decoder_mode {self.temporal_action_decoder_mode!r}"
            )
        if (
            self.temporal_action_feature is None
            or self.temporal_action_queries is None
            or self.temporal_action_decoder is None
        ):
            raise RuntimeError("temporal action decoder modules are missing")
        context_token = self.temporal_action_feature(features)
        if self.event_time_conditioning_mode == "soft_boundary":
            if self.event_time_feature is None:
                raise RuntimeError("event time feature module is missing")
            if event_time_probs is None:
                raise ValueError("event_time_probs are required for event time conditioning")
            if event_time_probs.shape != (batch_size, 2, self.event_time_class_count):
                raise ValueError(
                    "event_time_probs must be [B,2,H+1], "
                    f"got {event_time_probs.shape}"
                )
            event_time_token = self.event_time_feature(
                event_time_probs.reshape(batch_size, 2 * self.event_time_class_count)
            )
            context_token = context_token + event_time_token
        step_queries = self.temporal_action_queries.to(dtype=features.dtype).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        context_tokens = context_token.unsqueeze(1).expand(-1, self.horizon, -1)
        if self.temporal_action_decoder_mode == "sequence_mlp":
            decoder_input = torch.cat([context_tokens, step_queries], dim=-1)
        else:
            if self.temporal_action_sequence_encoder is None:
                raise RuntimeError("temporal action sequence encoder is missing")
            decoder_input = self.temporal_action_sequence_encoder(
                context_tokens + step_queries
            )
        flat_actions = self.temporal_action_decoder(
            decoder_input.reshape(batch_size * self.horizon, -1)
        )
        return {
            "temporal_actions": flat_actions.reshape(
                batch_size,
                self.horizon,
                self.action_dim,
            )
        }

    def _event_time_outputs(
        self,
        features: Tensor,
        batch_size: int,
    ) -> dict[str, Tensor | None]:
        if self.event_time_conditioning_mode == "none":
            return {
                "event_time_logits": None,
                "event_time_probs": None,
            }
        if self.event_time_conditioning_mode != "soft_boundary":
            raise ValueError(
                f"unsupported event_time_conditioning_mode {self.event_time_conditioning_mode!r}"
            )
        if self.event_time_logits is None:
            raise RuntimeError("event time logits module is missing")
        logits = self.event_time_logits(features).reshape(
            batch_size,
            2,
            self.event_time_class_count,
        )
        probs = torch.softmax(logits, dim=-1)
        return {
            "event_time_logits": logits,
            "event_time_probs": probs,
        }

    def _flow_action_outputs(
        self,
        features: Tensor,
        temporal_actions: Tensor | None,
        *,
        flow_noise: Tensor | None,
        flow_time: Tensor | None,
    ) -> dict[str, Tensor | None]:
        if self.flow_action_decoder_mode == "none":
            return {
                "flow_actions": None,
                "flow_action_velocity": None,
                "flow_action_residual": None,
                "flow_action_time": None,
            }
        if self.flow_action_decoder_mode != "rectified_mlp":
            raise ValueError(
                f"unsupported flow_action_decoder_mode {self.flow_action_decoder_mode!r}"
            )
        if temporal_actions is None:
            raise ValueError("flow action decoder requires temporal actions")
        if (
            self.flow_action_feature is None
            or self.flow_action_time is None
            or self.flow_action_decoder is None
        ):
            raise RuntimeError("flow action decoder modules are missing")
        if temporal_actions.shape != (
            temporal_actions.shape[0],
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                "temporal_actions must be [B,H,A], "
                f"got {temporal_actions.shape}"
            )
        batch_size = int(temporal_actions.shape[0])
        if flow_noise is None:
            flow_noise = temporal_actions.new_zeros(temporal_actions.shape)
        if flow_noise.shape != temporal_actions.shape:
            raise ValueError(
                "flow_noise shape must match temporal_actions: "
                f"{flow_noise.shape} vs {temporal_actions.shape}"
            )
        if flow_time is None:
            flow_time = temporal_actions.new_zeros((batch_size,))
        if flow_time.ndim == 0:
            flow_time = flow_time.reshape(1).expand(batch_size)
        if flow_time.ndim == 2 and flow_time.shape[-1] == 1:
            flow_time = flow_time.squeeze(-1)
        if flow_time.shape != (batch_size,):
            raise ValueError(f"flow_time must be [B], got {flow_time.shape}")
        context_token = self.flow_action_feature(features)
        context_tokens = context_token.unsqueeze(1).expand(-1, self.horizon, -1)
        time_token = self.flow_action_time(
            flow_time.to(dtype=features.dtype).reshape(batch_size, 1)
        )
        time_tokens = time_token.unsqueeze(1).expand(-1, self.horizon, -1)
        decoder_input = torch.cat(
            [
                context_tokens,
                time_tokens,
                temporal_actions.to(dtype=features.dtype),
                flow_noise.to(dtype=features.dtype),
            ],
            dim=-1,
        )
        velocity = self.flow_action_decoder(
            decoder_input.reshape(batch_size * self.horizon, -1)
        ).reshape(batch_size, self.horizon, self.action_dim)
        flow_actions = temporal_actions + velocity
        return {
            "flow_actions": flow_actions,
            "flow_action_velocity": velocity,
            "flow_action_residual": velocity,
            "flow_action_time": flow_time,
        }

    def oracle_step_routed_gripper_actions_from_targets(
        self,
        actions: Tensor,
        step_targets: Tensor,
        step_residuals: Tensor,
    ) -> Tensor:
        """Apply step residuals only at oracle positive boundary targets."""

        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,H,A], got {actions.shape}")
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
        if step_residuals.shape[-1] != self.gripper_step_class_count:
            raise ValueError(
                "step_residuals class dimension must match gripper_step_class_count: "
                f"{step_residuals.shape[-1]} vs {self.gripper_step_class_count}"
            )
        if actions.shape[-1] <= 6:
            raise ValueError("oracle step routing requires a gripper action channel")
        step_targets = step_targets.to(device=step_residuals.device, dtype=torch.long)
        if step_targets.min() < 0 or step_targets.max() >= self.gripper_step_class_count:
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
        oracle_actions = actions.clone()
        oracle_actions[..., -1] = oracle_actions[..., -1] + target_residual
        return oracle_actions

    def _features(
        self,
        context: Tensor,
        future_motions: Tensor | None,
        conditioning: Tensor | None,
        sample_features: Tensor | None,
    ) -> Tensor:
        features, _ = self._features_with_aux(context, future_motions, conditioning, sample_features)
        return features

    def _features_with_aux(
        self,
        context: Tensor,
        future_motions: Tensor | None,
        conditioning: Tensor | None,
        sample_features: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        base = self._base_features(context, conditioning)
        context_token = self.context_encoder(base)
        future_tokens, summary = self._future_tokens(context, future_motions, sample_features)
        mean_token = future_tokens.mean(dim=1)
        sample_scores, sample_score_probs = self._sample_score_outputs(
            context_token,
            future_tokens,
        )
        if sample_score_probs is not None:
            attended_token = torch.bmm(sample_score_probs.unsqueeze(1), future_tokens).squeeze(1)
        else:
            attended_token = self._aggregate_set(context_token, future_tokens, mean_token)
        features = torch.cat([context_token, attended_token, mean_token, summary], dim=-1)
        return features, {
            "sample_scores": sample_scores,
            "sample_score_probs": sample_score_probs,
        }

    def _sample_score_outputs(
        self,
        context_token: Tensor,
        future_tokens: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.sample_score is None:
            return None, None
        batch_size, samples, _ = future_tokens.shape
        context_tokens = context_token.unsqueeze(1).expand(-1, samples, -1)
        score_input = torch.cat([context_tokens, future_tokens], dim=-1)
        scores = self.sample_score(score_input.reshape(batch_size * samples, -1)).reshape(
            batch_size,
            samples,
        )
        return scores, torch.softmax(scores, dim=-1)

    def _aggregate_set(
        self,
        context_token: Tensor,
        future_tokens: Tensor,
        mean_token: Tensor,
    ) -> Tensor:
        if self.set_aggregator == "mean_pool":
            return mean_token
        if self.set_aggregator == "context_attention":
            attended, _ = self.set_attention(
                context_token.unsqueeze(1),
                future_tokens,
                future_tokens,
                need_weights=False,
            )
            return attended.squeeze(1)
        if self.set_aggregator == "multi_query_attention":
            if self.set_query_offsets is None:
                raise RuntimeError("multi_query_attention missing set_query_offsets")
            queries = context_token.unsqueeze(1) + self.set_query_offsets.unsqueeze(0)
            attended, _ = self.set_attention(
                queries,
                future_tokens,
                future_tokens,
                need_weights=False,
            )
            return attended.mean(dim=1)
        raise ValueError(f"unsupported set_aggregator {self.set_aggregator!r}")

    def _base_features(self, context: Tensor, conditioning: Tensor | None) -> Tensor:
        if context.ndim != 2:
            raise ValueError(f"context must be rank-2 [B,C], got {context.shape}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"context dim must be {self.context_dim}, got {context.shape[-1]}")
        if self.conditioning_dim == 0:
            if conditioning is not None and conditioning.shape[-1] != 0:
                raise ValueError("conditioning was provided but conditioning_dim is 0")
            return context
        if conditioning is None:
            raise ValueError("conditioning is required when conditioning_dim is positive")
        if conditioning.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                "conditioning batch shape must match context batch shape: "
                f"{conditioning.shape[:-1]} vs {context.shape[:-1]}"
            )
        if conditioning.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"conditioning dim must be {self.conditioning_dim}, got {conditioning.shape[-1]}"
            )
        return torch.cat([context, conditioning.to(dtype=context.dtype)], dim=-1)

    def _future_tokens(
        self,
        context: Tensor,
        future_motions: Tensor | None,
        sample_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = int(context.shape[0])
        if future_motions is None:
            if sample_features is not None:
                raise ValueError("sample_features cannot be provided without future_motions")
            token = self.empty_future_token.to(dtype=context.dtype).expand(batch_size, 1, -1)
            summary = token.new_zeros((batch_size, 3))
            return token, summary
        if future_motions.ndim == 2:
            future_motions = future_motions.unsqueeze(1)
        if future_motions.ndim != 3:
            raise ValueError(
                "future_motions must be rank-3 [B,K,M] or rank-2 [B,M], "
                f"got {future_motions.shape}"
            )
        if future_motions.shape[0] != batch_size:
            raise ValueError(
                "future_motions batch size must match context: "
                f"{future_motions.shape[0]} vs {batch_size}"
            )
        if future_motions.shape[1] <= 0:
            raise ValueError("future_motions must contain at least one sample")
        if future_motions.shape[-1] != self.motion_dim:
            raise ValueError(
                f"future_motions dim must be {self.motion_dim}, got {future_motions.shape[-1]}"
            )
        sample_feature_tokens = self._sample_feature_tokens(
            sample_features,
            batch_size=batch_size,
            samples=int(future_motions.shape[1]),
            context=context,
        )
        future_motions = future_motions.to(device=context.device, dtype=context.dtype)
        batch, samples, _ = future_motions.shape
        flat_steps = future_motions.reshape(batch * samples, self.horizon, self.motion_step_dim)
        encoded = self.step_encoder(flat_steps) + self.position.to(dtype=context.dtype).unsqueeze(0)
        encoded = self.temporal_encoder(encoded)
        pooled = encoded.mean(dim=1)
        last = encoded[:, -1]
        summary_per_sample = self._motion_summary(flat_steps)
        sample_parts = [pooled, last, summary_per_sample]
        if sample_feature_tokens is not None:
            sample_parts.append(sample_feature_tokens)
        sample_tokens = self.sample_pool(torch.cat(sample_parts, dim=-1))
        sample_tokens = sample_tokens.reshape(batch, samples, self.token_dim)
        summary = torch.cat(
            [
                future_motions.mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
                future_motions.std(dim=1, unbiased=False).mean(dim=-1, keepdim=True),
                future_motions.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
            ],
            dim=-1,
        )
        return sample_tokens, summary

    def _sample_feature_tokens(
        self,
        sample_features: Tensor | None,
        *,
        batch_size: int,
        samples: int,
        context: Tensor,
    ) -> Tensor | None:
        if self.sample_feature_dim == 0:
            if sample_features is not None and sample_features.shape[-1] != 0:
                raise ValueError("sample_features were provided but sample_feature_dim is 0")
            return None
        if sample_features is None:
            raise ValueError("sample_features are required when sample_feature_dim is positive")
        if sample_features.ndim != 3:
            raise ValueError(
                "sample_features must be rank-3 [B,K,F], "
                f"got {sample_features.shape}"
            )
        if sample_features.shape[0] != batch_size or sample_features.shape[1] != samples:
            raise ValueError(
                "sample_features batch/sample shape must match future_motions: "
                f"{sample_features.shape[:2]} vs {(batch_size, samples)}"
            )
        if sample_features.shape[-1] != self.sample_feature_dim:
            raise ValueError(
                f"sample_features dim must be {self.sample_feature_dim}, "
                f"got {sample_features.shape[-1]}"
            )
        if self.sample_feature_encoder is None:
            raise RuntimeError("sample_feature_encoder is missing")
        flat_features = sample_features.to(device=context.device, dtype=context.dtype).reshape(
            batch_size * samples,
            self.sample_feature_dim,
        )
        return self.sample_feature_encoder(flat_features)

    def _motion_summary(self, motion_steps: Tensor) -> Tensor:
        abs_mean = motion_steps.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        final_abs = motion_steps[:, -1].abs().mean(dim=-1, keepdim=True)
        if self.horizon > 1:
            smooth = motion_steps.diff(dim=1).abs().mean(dim=(1, 2), keepdim=True)
        else:
            smooth = abs_mean.new_zeros((abs_mean.shape[0], 1))
        gripper_abs = motion_steps[:, :, -1].abs().mean(dim=1, keepdim=True)
        return torch.cat([abs_mean, final_abs, smooth.reshape(-1, 1), gripper_abs], dim=-1)


class PostHocActionResidualAdapter(nn.Module):
    """Small residual action-sequence adapter over frozen action-head features."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        step_dim: int = 32,
        dropout: float = 0.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if step_dim <= 0:
            raise ValueError("step_dim must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dims = tuple(hidden_dims)
        self.step_dim = step_dim
        self.dropout = dropout
        self.step_embedding = nn.Parameter(torch.zeros(horizon, step_dim))
        self.residual = _norm_mlp(
            feature_dim + action_dim + step_dim,
            hidden_dims,
            action_dim,
            dropout=dropout,
        )
        if zero_init_output:
            last = self.residual[-1]
            if not isinstance(last, nn.Linear):
                raise RuntimeError("residual adapter final layer must be linear")
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, features: Tensor, temporal_actions: Tensor) -> dict[str, Tensor]:
        """Return adapted actions and residuals for frozen temporal actions."""

        if features.ndim != 2:
            raise ValueError(f"features must be rank-2 [B,F], got {features.shape}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"features dim must be {self.feature_dim}, got {features.shape[-1]}")
        if temporal_actions.ndim != 3:
            raise ValueError(
                "temporal_actions must be rank-3 [B,H,A], "
                f"got {temporal_actions.shape}"
            )
        if temporal_actions.shape[0] != features.shape[0]:
            raise ValueError(
                "temporal_actions batch size must match features: "
                f"{temporal_actions.shape[0]} vs {features.shape[0]}"
            )
        if temporal_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                "temporal_actions shape must be [B,H,A] with "
                f"H={self.horizon}, A={self.action_dim}; got {temporal_actions.shape}"
            )
        batch_size = int(features.shape[0])
        feature_tokens = features.unsqueeze(1).expand(-1, self.horizon, -1)
        step_tokens = self.step_embedding.to(dtype=features.dtype).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        decoder_input = torch.cat(
            [
                feature_tokens,
                temporal_actions.to(dtype=features.dtype),
                step_tokens,
            ],
            dim=-1,
        )
        residual = self.residual(decoder_input.reshape(batch_size * self.horizon, -1)).reshape(
            batch_size,
            self.horizon,
            self.action_dim,
        )
        return {
            "adapter_actions": temporal_actions + residual,
            "adapter_residual": residual,
        }


def _norm_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("hidden_dims must be positive")
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
