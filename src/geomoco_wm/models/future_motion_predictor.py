"""Deterministic future EEF-motion predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from geomoco_wm.models.common import MLP


class FutureMotionPredictor(nn.Module):
    """Predict a flattened future EEF-delta chunk from current context."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.conditioning_dim = conditioning_dim
        self.net = MLP(context_dim + conditioning_dim, hidden_dims, motion_dim)

    def forward(self, context: Tensor, conditioning: Tensor | None = None) -> Tensor:
        if self.conditioning_dim == 0:
            if conditioning is not None and conditioning.shape[-1] != 0:
                raise ValueError("conditioning was provided but conditioning_dim is 0")
            return self.net(context)

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
        return self.net(torch.cat([context, conditioning.to(dtype=context.dtype)], dim=-1))


class VisualCrossAttentionFutureMotionPredictor(nn.Module):
    """Predict future EEF deltas by attending a proprio/task query to visual tokens."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        visual_token_dim: int,
        visual_token_count: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        conditioning_dim: int = 0,
        query_dim: int = 384,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if visual_token_dim <= 0:
            raise ValueError("visual_token_dim must be positive")
        if visual_token_count <= 0:
            raise ValueError("visual_token_count must be positive")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.conditioning_dim = conditioning_dim
        self.visual_token_dim = visual_token_dim
        self.visual_token_count = visual_token_count
        self.query_dim = query_dim
        self.num_heads = num_heads
        base_dim = context_dim + conditioning_dim
        self.query_net = MLP(base_dim, (query_dim,), query_dim)
        self.visual_proj = (
            nn.Identity()
            if visual_token_dim == query_dim
            else nn.Linear(visual_token_dim, query_dim)
        )
        self.attention = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.net = MLP(base_dim + query_dim, hidden_dims, motion_dim)

    def forward(
        self,
        context: Tensor,
        visual_features: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        base = self._base_features(context, conditioning)
        visual_tokens = self._visual_tokens(visual_features)
        query = self.query_net(base).unsqueeze(1)
        keys_values = self.visual_proj(visual_tokens.to(dtype=context.dtype))
        grounded, _ = self.attention(query, keys_values, keys_values, need_weights=False)
        return self.net(torch.cat([base, grounded.squeeze(1)], dim=-1))

    def _base_features(self, context: Tensor, conditioning: Tensor | None) -> Tensor:
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

    def _visual_tokens(self, visual_features: Tensor) -> Tensor:
        expected_dim = self.visual_token_count * self.visual_token_dim
        if visual_features.shape[-1] != expected_dim:
            raise ValueError(
                f"visual feature dim must be {expected_dim}, got {visual_features.shape[-1]}"
            )
        return visual_features.reshape(*visual_features.shape[:-1], self.visual_token_count, self.visual_token_dim)


class StepwiseVisualCrossAttentionFutureMotionPredictor(nn.Module):
    """Predict each future EEF-delta step with its own visual-attention query."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        visual_token_dim: int,
        visual_token_count: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        conditioning_dim: int = 0,
        query_dim: int = 384,
        num_heads: int = 4,
        future_step_dim: int = 6,
    ) -> None:
        super().__init__()
        if conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative")
        if visual_token_dim <= 0:
            raise ValueError("visual_token_dim must be positive")
        if visual_token_count <= 0:
            raise ValueError("visual_token_count must be positive")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        if future_step_dim <= 0:
            raise ValueError("future_step_dim must be positive")
        if motion_dim % future_step_dim != 0:
            raise ValueError(
                f"motion_dim must be divisible by future_step_dim: "
                f"{motion_dim} vs {future_step_dim}"
            )
        self.context_dim = context_dim
        self.motion_dim = motion_dim
        self.conditioning_dim = conditioning_dim
        self.visual_token_dim = visual_token_dim
        self.visual_token_count = visual_token_count
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.future_step_dim = future_step_dim
        self.horizon = motion_dim // future_step_dim
        base_dim = context_dim + conditioning_dim
        self.base_query_net = MLP(base_dim, (query_dim,), query_dim)
        self.step_embeddings = nn.Parameter(torch.empty(self.horizon, query_dim))
        nn.init.normal_(self.step_embeddings, std=0.02)
        self.visual_proj = (
            nn.Identity()
            if visual_token_dim == query_dim
            else nn.Linear(visual_token_dim, query_dim)
        )
        self.attention = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.step_net = MLP(base_dim + query_dim + query_dim, hidden_dims, future_step_dim)

    def forward(
        self,
        context: Tensor,
        visual_features: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        base = self._base_features(context, conditioning)
        visual_tokens = self._visual_tokens(visual_features)
        base_query = self.base_query_net(base).unsqueeze(1)
        step_embeddings = self.step_embeddings.unsqueeze(0).to(dtype=context.dtype)
        queries = base_query + step_embeddings
        keys_values = self.visual_proj(visual_tokens.to(dtype=context.dtype))
        grounded, _ = self.attention(queries, keys_values, keys_values, need_weights=False)

        base_per_step = base.unsqueeze(1).expand(-1, self.horizon, -1)
        step_per_batch = step_embeddings.expand(base.shape[0], -1, -1)
        step_inputs = torch.cat([base_per_step, grounded, step_per_batch], dim=-1)
        step_outputs = self.step_net(step_inputs.reshape(-1, step_inputs.shape[-1]))
        return step_outputs.reshape(base.shape[0], self.motion_dim)

    def _base_features(self, context: Tensor, conditioning: Tensor | None) -> Tensor:
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

    def _visual_tokens(self, visual_features: Tensor) -> Tensor:
        expected_dim = self.visual_token_count * self.visual_token_dim
        if visual_features.shape[-1] != expected_dim:
            raise ValueError(
                f"visual feature dim must be {expected_dim}, got {visual_features.shape[-1]}"
            )
        return visual_features.reshape(
            *visual_features.shape[:-1],
            self.visual_token_count,
            self.visual_token_dim,
        )
