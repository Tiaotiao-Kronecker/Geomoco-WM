"""Torch datasets for exported GeoMoCo-WM LIBERO windows."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, read_window_jsonl
from geomoco_wm.data.visual_feature_cache import VisualFeatureCache


MOTION_MODES = (
    "future_delta",
    "future_gripper",
    "future_delta_gripper",
    "none",
)


@dataclass(frozen=True)
class OracleActionWindowSpec:
    windows_jsonl: list[str]
    num_windows: int
    context_dim: int
    motion_dim: int
    action_dim: int
    horizon: int
    motion_mode: str
    suite_counts: dict[str, int]
    task_counts: dict[str, int]
    visual_dim: int = 0
    visual_feature_cache: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OracleActionWindowDataset(Dataset):
    """Dataset for the oracle diagnostic: GT future motion -> action chunk."""

    def __init__(
        self,
        windows_jsonl: str | Path | Sequence[str | Path],
        max_windows: int | None = None,
        motion_mode: str = "future_delta",
        visual_feature_cache_path: str | Path | None = None,
    ) -> None:
        self.windows_jsonl_paths = _normalize_windows_jsonl_paths(windows_jsonl)
        if motion_mode not in MOTION_MODES:
            joined_modes = ", ".join(MOTION_MODES)
            raise ValueError(f"motion_mode must be one of: {joined_modes}")
        self.motion_mode = motion_mode
        self.visual_feature_cache = (
            VisualFeatureCache(visual_feature_cache_path)
            if visual_feature_cache_path is not None
            else None
        )
        self.visual_feature_cache_path = (
            str(Path(visual_feature_cache_path).expanduser().resolve())
            if visual_feature_cache_path is not None
            else None
        )
        windows: list[LiberoWindowRecord] = []
        for path in self.windows_jsonl_paths:
            windows.extend(read_window_jsonl(path))
        if max_windows is not None:
            if max_windows <= 0:
                raise ValueError("max_windows must be positive when provided")
            windows = windows[:max_windows]
        if not windows:
            joined_paths = ", ".join(str(path) for path in self.windows_jsonl_paths)
            raise ValueError(f"no windows found in {joined_paths}")
        self.windows = windows
        first = windows[0]
        self.horizon = len(first.action_chunk)
        self.action_dim = len(first.action_chunk[0])
        self.context_dim = len(_context_features(first))
        self.motion_dim = len(_motion_features(first, self.motion_mode))
        self.visual_dim = (
            int(self.visual_feature_cache.feature_dim)
            if self.visual_feature_cache is not None
            else 0
        )
        for window in windows:
            self._validate_window_shape(window)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        window = self.windows[index]
        item: dict[str, Tensor | str] = {
            "context": torch.tensor(_context_features(window), dtype=torch.float32),
            "motion": torch.tensor(_motion_features(window, self.motion_mode), dtype=torch.float32),
            "actions": torch.tensor(window.action_chunk, dtype=torch.float32),
            "window_id": window.window_id,
            "episode_id": window.episode_id,
            "task_id": window.task_id,
            "suite_name": window.suite_name,
        }
        if self.visual_feature_cache is not None:
            item["visual"] = torch.tensor(
                self.visual_feature_cache.get(window.window_id),
                dtype=torch.float32,
            )
        return item

    def spec(self) -> OracleActionWindowSpec:
        return OracleActionWindowSpec(
            windows_jsonl=[str(path) for path in self.windows_jsonl_paths],
            num_windows=len(self.windows),
            context_dim=self.context_dim,
            motion_dim=self.motion_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            motion_mode=self.motion_mode,
            suite_counts=_count_by(self.windows, "suite_name"),
            task_counts=_count_by(self.windows, "task_id"),
            visual_dim=self.visual_dim,
            visual_feature_cache=self.visual_feature_cache_path,
        )

    def _validate_window_shape(self, window: LiberoWindowRecord) -> None:
        if len(window.action_chunk) != self.horizon:
            raise ValueError(f"{window.window_id} has a different action horizon")
        if len(window.action_chunk[0]) != self.action_dim:
            raise ValueError(f"{window.window_id} has a different action dimension")
        if len(_context_features(window)) != self.context_dim:
            raise ValueError(f"{window.window_id} has a different context dimension")
        if len(_motion_features(window, self.motion_mode)) != self.motion_dim:
            raise ValueError(f"{window.window_id} has a different motion dimension")
        if (
            self.visual_feature_cache is not None
            and window.window_id not in self.visual_feature_cache.index_by_window_id
        ):
            raise ValueError(f"{window.window_id} is missing from the visual feature cache")


def _context_features(window: LiberoWindowRecord) -> list[float]:
    return (
        list(window.anchor_ee_state)
        + list(window.current_gripper_state)
        + list(window.current_joint_state)
    )


def _motion_features(window: LiberoWindowRecord, motion_mode: str) -> list[float]:
    if motion_mode == "none":
        return []
    future_delta = [value for row in window.future_delta_ee_states for value in row]
    future_gripper = [float(row[-1]) for row in window.action_chunk]
    if motion_mode == "future_delta":
        return future_delta
    if motion_mode == "future_gripper":
        return future_gripper
    if motion_mode == "future_delta_gripper":
        return future_delta + future_gripper
    joined_modes = ", ".join(MOTION_MODES)
    raise ValueError(f"motion_mode must be one of: {joined_modes}")


def _normalize_windows_jsonl_paths(
    windows_jsonl: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(windows_jsonl, (str, Path)):
        paths = [windows_jsonl]
    else:
        paths = list(windows_jsonl)
    if not paths:
        raise ValueError("windows_jsonl must contain at least one path")
    return [Path(path).expanduser().resolve() for path in paths]


def _count_by(windows: Sequence[LiberoWindowRecord], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(getattr(window, field_name)) for window in windows)
    return {str(key): int(value) for key, value in sorted(counts.items())}
