#!/usr/bin/env python3
"""Create a shuffled visual feature cache while preserving window-id order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.visual_feature_cache import (  # noqa: E402
    VisualFeatureCache,
    VisualFeatureCacheMetadata,
    write_visual_feature_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derange visual feature rows while keeping the original window_ids. "
            "This is a visual-grounding control: every window receives another "
            "window's visual feature."
        )
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = shuffle_visual_feature_cache(
        input_path=args.input_path,
        output_path=args.output_path,
        seed=args.seed,
        summary_json=args.summary_json,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def shuffle_visual_feature_cache(
    *,
    input_path: str | Path,
    output_path: str | Path,
    seed: int,
    summary_json: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    source_path = Path(input_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if source_path == target_path:
        raise ValueError("input-path and output-path must be different")

    cache = VisualFeatureCache(source_path)
    permutation = deranged_permutation(len(cache.window_ids), seed=seed)
    metadata = _shuffled_metadata(cache, seed)
    summary_path = _summary_path(summary_json, target_path)
    summary = {
        "input_path": str(source_path),
        "output_path": str(target_path),
        "summary_json": str(summary_path),
        "seed": seed,
        "num_windows": len(cache.window_ids),
        "feature_shape": [int(dim) for dim in cache.features.shape],
        "metadata": metadata.to_dict(),
        "permutation_preview": [int(value) for value in permutation[:10]],
        "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return summary

    shuffled_features = np.asarray(cache.features[permutation], dtype=np.float32)
    write_visual_feature_cache(
        target_path,
        window_ids=cache.window_ids,
        features=shuffled_features,
        metadata=metadata,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def deranged_permutation(num_items: int, *, seed: int) -> np.ndarray:
    if num_items < 2:
        raise ValueError("at least two feature rows are required for a shuffled control")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    permutation = np.empty(num_items, dtype=np.int64)
    permutation[order] = np.roll(order, -1)
    return permutation


def _shuffled_metadata(cache: VisualFeatureCache, seed: int) -> VisualFeatureCacheMetadata:
    metadata = cache.metadata
    feature_mode = str(metadata.get("feature_mode", "unknown"))
    return VisualFeatureCacheMetadata(
        schema_version=str(metadata.get("schema_version", "geomoco_wm_visual_feature_cache_v0")),
        source_windows_jsonl=str(metadata.get("source_windows_jsonl", "")),
        model_name=str(metadata.get("model_name", "unknown")),
        feature_mode=f"{feature_mode}_shuffled_seed{seed}",
        camera_keys=[str(value) for value in metadata.get("camera_keys", [])],
        image_size=int(metadata.get("image_size", 0)),
        num_windows=len(cache.window_ids),
        feature_dim=int(cache.features.shape[1]),
        part_count=int(metadata.get("part_count", 0)),
        visual_token_count=_optional_int(metadata.get("visual_token_count")),
        visual_token_dim=_optional_int(metadata.get("visual_token_dim")),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _summary_path(summary_json: str | Path | None, output_path: Path) -> Path:
    if summary_json:
        return Path(summary_json).expanduser().resolve()
    return output_path.with_suffix(".summary.json")


if __name__ == "__main__":
    main()
