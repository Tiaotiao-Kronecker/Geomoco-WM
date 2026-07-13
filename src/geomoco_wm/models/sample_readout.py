"""Readout heads for selecting sampled future-motion candidates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SampleScoreNet(nn.Module):
    """Score future-motion/action candidates conditioned on visual state."""

    def __init__(
        self,
        condition_dim: int,
        motion_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if motion_dim <= 0:
            raise ValueError("motion_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.condition_dim = condition_dim
        self.motion_dim = motion_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.input_dim = condition_dim + motion_dim + action_dim * horizon
        layers: list[nn.Module] = []
        prev_dim = self.input_dim
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
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, condition: Tensor, motion: Tensor, action_chunk: Tensor) -> Tensor:
        if condition.shape[:-1] != motion.shape[:-1]:
            raise ValueError(
                "condition and motion batch shapes must match: "
                f"{condition.shape[:-1]} vs {motion.shape[:-1]}"
            )
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition dim must be {self.condition_dim}, got {condition.shape[-1]}")
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion dim must be {self.motion_dim}, got {motion.shape[-1]}")
        if action_chunk.shape[:-2] != motion.shape[:-1]:
            raise ValueError(
                "action_chunk batch shape must match motion batch shape: "
                f"{action_chunk.shape[:-2]} vs {motion.shape[:-1]}"
            )
        if action_chunk.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                "action_chunk shape must end with "
                f"({self.horizon}, {self.action_dim}), got {action_chunk.shape[-2:]}"
            )
        flat_action = action_chunk.reshape(*action_chunk.shape[:-2], self.horizon * self.action_dim)
        features = torch.cat([condition, motion, flat_action], dim=-1)
        return self.net(features).squeeze(-1)


class TemporalSampleScoreNet(nn.Module):
    """Temporal scorer for future-motion/action chunks."""

    def __init__(
        self,
        condition_dim: int,
        motion_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        temporal_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if motion_dim <= 0:
            raise ValueError("motion_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if motion_dim % horizon != 0:
            raise ValueError("motion_dim must be divisible by horizon")
        if temporal_dim <= 0:
            raise ValueError("temporal_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if temporal_dim % num_heads != 0:
            raise ValueError("temporal_dim must be divisible by num_heads")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.condition_dim = condition_dim
        self.motion_dim = motion_dim
        self.motion_step_dim = motion_dim // horizon
        self.action_dim = action_dim
        self.horizon = horizon
        self.temporal_dim = temporal_dim
        self.summary_dim = 7
        step_feature_dim = self.motion_step_dim + action_dim
        self.step_embed = nn.Sequential(
            nn.Linear(step_feature_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.SiLU(),
        )
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(horizon, temporal_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=num_heads,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_input_dim = temporal_dim * 3 + self.summary_dim
        layers: list[nn.Module] = []
        prev_dim = head_input_dim
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
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, condition: Tensor, motion: Tensor, action_chunk: Tensor) -> Tensor:
        if condition.shape[:-1] != motion.shape[:-1]:
            raise ValueError(
                "condition and motion batch shapes must match: "
                f"{condition.shape[:-1]} vs {motion.shape[:-1]}"
            )
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition dim must be {self.condition_dim}, got {condition.shape[-1]}")
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion dim must be {self.motion_dim}, got {motion.shape[-1]}")
        if action_chunk.shape[:-2] != motion.shape[:-1]:
            raise ValueError(
                "action_chunk batch shape must match motion batch shape: "
                f"{action_chunk.shape[:-2]} vs {motion.shape[:-1]}"
            )
        if action_chunk.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                "action_chunk shape must end with "
                f"({self.horizon}, {self.action_dim}), got {action_chunk.shape[-2:]}"
            )
        batch_shape = motion.shape[:-1]
        flat_condition = condition.reshape(-1, self.condition_dim)
        flat_motion = motion.reshape(-1, self.horizon, self.motion_step_dim)
        flat_action = action_chunk.reshape(-1, self.horizon, self.action_dim)
        step_features = torch.cat([flat_motion, flat_action], dim=-1)
        tokens = self.step_embed(step_features) + self.position.unsqueeze(0)
        encoded = self.temporal_encoder(tokens)
        pooled = encoded.mean(dim=1)
        last = encoded[:, -1]
        condition_features = self.condition_embed(flat_condition)
        summary = self._summary_features(flat_motion, flat_action)
        features = torch.cat([pooled, last, condition_features, summary], dim=-1)
        scores = self.head(features).squeeze(-1)
        return scores.reshape(batch_shape)

    def _summary_features(self, motion_steps: Tensor, action_chunk: Tensor) -> Tensor:
        motion_abs_mean = motion_steps.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        motion_final_abs = motion_steps[:, -1].abs().mean(dim=-1, keepdim=True)
        action_abs_mean = action_chunk.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        if self.horizon > 1:
            motion_smooth = motion_steps.diff(dim=1).abs().mean(dim=(1, 2), keepdim=True)
            action_smooth = action_chunk.diff(dim=1).abs().mean(dim=(1, 2), keepdim=True)
            gripper_smooth = action_chunk[..., -1].diff(dim=1).abs().mean(dim=1, keepdim=True)
        else:
            motion_smooth = motion_abs_mean.new_zeros((motion_abs_mean.shape[0], 1))
            action_smooth = motion_abs_mean.new_zeros((motion_abs_mean.shape[0], 1))
            gripper_smooth = motion_abs_mean.new_zeros((motion_abs_mean.shape[0], 1))
        gripper_abs_mean = action_chunk[..., -1].abs().mean(dim=1, keepdim=True)
        return torch.cat(
            [
                motion_abs_mean,
                motion_final_abs,
                action_abs_mean,
                motion_smooth.reshape(-1, 1),
                action_smooth.reshape(-1, 1),
                gripper_abs_mean,
                gripper_smooth.reshape(-1, 1),
            ],
            dim=-1,
        )
