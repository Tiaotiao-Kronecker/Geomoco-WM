"""Visual feature-cache helpers for LIBERO window datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional at import time
    h5py = None


SCHEMA_VERSION = "geomoco_wm_visual_feature_cache_v0"


class VisualFeatureCacheError(ValueError):
    """Raised when a visual feature cache is missing or malformed."""


@dataclass(frozen=True)
class VisualFeatureCacheMetadata:
    schema_version: str
    source_windows_jsonl: str
    model_name: str
    feature_mode: str
    camera_keys: list[str]
    image_size: int
    num_windows: int
    feature_dim: int
    part_count: int
    visual_token_count: int | None = None
    visual_token_dim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisualFeatureCache:
    """In-memory reader for window-aligned visual features."""

    def __init__(self, path: str | Path) -> None:
        if h5py is None:
            raise VisualFeatureCacheError("h5py is required to read visual feature caches")
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise VisualFeatureCacheError(f"visual feature cache does not exist: {self.path}")

        with h5py.File(self.path, "r") as handle:
            schema_version = str(handle.attrs.get("schema_version", ""))
            if schema_version != SCHEMA_VERSION:
                raise VisualFeatureCacheError(
                    f"unsupported visual feature cache schema: {schema_version!r}"
                )
            if "window_ids" not in handle or "features" not in handle:
                raise VisualFeatureCacheError("cache must contain window_ids and features datasets")
            window_ids = [_decode_string(value) for value in handle["window_ids"][()]]
            features = np.asarray(handle["features"][()], dtype=np.float32)
            metadata_json = str(handle.attrs.get("metadata_json", "{}"))

        if features.ndim != 2:
            raise VisualFeatureCacheError("features must be a rank-2 array")
        if len(window_ids) != int(features.shape[0]):
            raise VisualFeatureCacheError("window_ids length must match features rows")
        if len(set(window_ids)) != len(window_ids):
            raise VisualFeatureCacheError("window_ids must be unique")

        self.window_ids = window_ids
        self.features = features
        self.metadata = json.loads(metadata_json)
        self.index_by_window_id = {window_id: index for index, window_id in enumerate(window_ids)}
        self.feature_dim = int(features.shape[1])

    def get(self, window_id: str) -> np.ndarray:
        try:
            index = self.index_by_window_id[window_id]
        except KeyError as exc:
            raise VisualFeatureCacheError(f"missing visual feature for window_id={window_id!r}") from exc
        return self.features[index]


def write_visual_feature_cache(
    path: str | Path,
    *,
    window_ids: Sequence[str],
    features: np.ndarray,
    metadata: VisualFeatureCacheMetadata,
) -> None:
    if h5py is None:
        raise VisualFeatureCacheError("h5py is required to write visual feature caches")
    output_path = Path(path).expanduser().resolve()
    if features.ndim != 2:
        raise VisualFeatureCacheError("features must be a rank-2 array")
    if len(window_ids) != int(features.shape[0]):
        raise VisualFeatureCacheError("window_ids length must match features rows")
    if metadata.num_windows != len(window_ids):
        raise VisualFeatureCacheError("metadata.num_windows must match window_ids length")
    if metadata.feature_dim != int(features.shape[1]):
        raise VisualFeatureCacheError("metadata.feature_dim must match feature dimension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["metadata_json"] = json.dumps(metadata.to_dict(), ensure_ascii=False)
        handle.create_dataset("window_ids", data=np.asarray(list(window_ids), dtype=object), dtype=string_dtype)
        handle.create_dataset("features", data=np.asarray(features, dtype=np.float32), compression="gzip")


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
