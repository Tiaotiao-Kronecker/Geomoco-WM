"""Lightweight inverse-dynamics action decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from geomoco_wm.models.common import MLP


class ActionDecoder(nn.Module):
    """Decode observation context and motion representation into action chunks."""

    def __init__(
        self,
        context_dim: int,
        motion_rep_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.net = MLP(context_dim + motion_rep_dim, hidden_dims, action_dim * horizon)

    def forward(self, context: Tensor, motion_representation: Tensor) -> Tensor:
        flat_actions = self.net(torch.cat([context, motion_representation], dim=-1))
        return flat_actions.reshape(*flat_actions.shape[:-1], self.horizon, self.action_dim)
