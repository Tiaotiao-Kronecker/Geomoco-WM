#!/usr/bin/env python3
"""Cache frozen DINOv2 global features for exported LIBERO windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, read_window_jsonl  # noqa: E402
from geomoco_wm.data.visual_feature_cache import (  # noqa: E402
    VisualFeatureCacheMetadata,
    write_visual_feature_cache,
)


DEFAULT_TORCHHUB_DIR = "/home/user/.cache/torch/hub/facebookresearch_dinov2_main"
DEFAULT_WINDOWS_JSONL = "outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl"
DEFAULT_OUTPUT_PATH = (
    "outputs/visual_features/gate2_2a_dinov2_vits14_reg_global_2files_h8.h5"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache DINOv2 global tokens for each exported LIBERO window."
    )
    parser.add_argument("--windows-jsonl", default=DEFAULT_WINDOWS_JSONL)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--model-name", default="dinov2_vits14_reg")
    parser.add_argument("--torchhub-dir", default=DEFAULT_TORCHHUB_DIR)
    parser.add_argument("--camera-keys", nargs="+", default=None)
    parser.add_argument(
        "--feature-mode",
        default="global",
        choices=["global", "patch_pool"],
        help="Cache DINO global tokens or spatially pooled patch tokens.",
    )
    parser.add_argument(
        "--patch-pool-grid",
        type=int,
        default=4,
        help="Spatial grid size for patch_pool mode. 4 gives 16 tokens per image.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    windows_path = Path(args.windows_jsonl).expanduser().resolve()
    windows = read_window_jsonl(windows_path)
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        windows = windows[: args.max_windows]
    if not windows:
        raise ValueError(f"no windows found in {windows_path}")

    camera_keys = _resolve_camera_keys(windows, args.camera_keys)
    context_len = len(windows[0].context_frame_indices)
    part_count = context_len * len(camera_keys)
    for window in windows:
        if len(window.context_frame_indices) != context_len:
            raise ValueError("all windows must have the same context length")
        missing = [key for key in camera_keys if key not in window.camera_keys]
        if missing:
            raise ValueError(f"{window.window_id} is missing cameras: {missing}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "windows_jsonl": str(windows_path),
                    "num_windows": len(windows),
                    "camera_keys": camera_keys,
                    "context_len": context_len,
                    "part_count": part_count,
                    "model_name": args.model_name,
                    "feature_mode": args.feature_mode,
                    "patch_pool_grid": args.patch_pool_grid,
                    "output_path": str(Path(args.output_path).expanduser().resolve()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    device = _resolve_device(args.device)
    model = _load_dino_model(args.torchhub_dir, args.model_name, device)
    feature_chunks: list[np.ndarray] = []
    window_ids: list[str] = []
    handles: dict[str, h5py.File] = {}

    try:
        for start in tqdm(range(0, len(windows), args.batch_size), desc="cache dino"):
            batch_windows = windows[start : start + args.batch_size]
            image_batch = _load_window_images(batch_windows, camera_keys, handles)
            image_tensor = _preprocess_images(image_batch, device, args.image_size)
            with torch.no_grad():
                token_features = _extract_features(model, image_tensor, args.feature_mode, args.patch_pool_grid)
            if int(token_features.shape[0]) != len(batch_windows) * part_count:
                raise RuntimeError("DINO feature rows do not match loaded images")
            token_features = token_features.reshape(len(batch_windows), part_count, *token_features.shape[1:])
            feature_chunks.append(token_features.flatten(start_dim=1).cpu().numpy().astype(np.float32))
            window_ids.extend(window.window_id for window in batch_windows)
    finally:
        for handle in handles.values():
            handle.close()

    features = np.concatenate(feature_chunks, axis=0)
    output_path = Path(args.output_path).expanduser().resolve()
    metadata = VisualFeatureCacheMetadata(
        schema_version="geomoco_wm_visual_feature_cache_v0",
        source_windows_jsonl=str(windows_path),
        model_name=args.model_name,
        feature_mode=_cache_feature_mode(args.feature_mode, args.patch_pool_grid),
        camera_keys=camera_keys,
        image_size=args.image_size,
        num_windows=len(window_ids),
        feature_dim=int(features.shape[1]),
        part_count=part_count,
        visual_token_count=_visual_token_count(args.feature_mode, part_count, args.patch_pool_grid),
        visual_token_dim=int(_visual_token_dim(features, args.feature_mode, part_count, args.patch_pool_grid)),
    )
    write_visual_feature_cache(
        output_path,
        window_ids=window_ids,
        features=features,
        metadata=metadata,
    )
    summary = metadata.to_dict()
    summary.update(
        {
            "output_path": str(output_path),
            "summary_json": str(_summary_path(args.summary_json, output_path)),
            "device": str(device),
            "dtype": "float32",
            "feature_shape": [int(dim) for dim in features.shape],
            "dino_token_dim": int(_visual_token_dim(features, args.feature_mode, part_count, args.patch_pool_grid)),
            "visual_token_count": _visual_token_count(args.feature_mode, part_count, args.patch_pool_grid),
            "visual_token_dim": int(_visual_token_dim(features, args.feature_mode, part_count, args.patch_pool_grid)),
        }
    )
    summary_path = _summary_path(args.summary_json, output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _resolve_camera_keys(
    windows: list[LiberoWindowRecord],
    requested_camera_keys: list[str] | None,
) -> list[str]:
    if requested_camera_keys:
        return [str(key) for key in requested_camera_keys]
    return list(windows[0].camera_keys)


def _load_dino_model(torchhub_dir: str, model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(str(Path(torchhub_dir).expanduser().resolve()), model_name, source="local")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _extract_features(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    feature_mode: str,
    patch_pool_grid: int,
) -> torch.Tensor:
    if feature_mode == "global":
        return _as_global_features(model(image_tensor))
    if feature_mode == "patch_pool":
        if patch_pool_grid <= 0:
            raise ValueError("patch_pool_grid must be positive")
        features = model.forward_features(image_tensor)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise TypeError("DINO model does not expose x_norm_patchtokens")
        patch_tokens = features["x_norm_patchtokens"]
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch tokens must be rank 3, got {patch_tokens.shape}")
        num_patches = int(patch_tokens.shape[1])
        grid_size = int(round(num_patches**0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"patch token count must be a square grid, got {num_patches}")
        token_dim = int(patch_tokens.shape[2])
        patch_grid = patch_tokens.reshape(-1, grid_size, grid_size, token_dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patch_grid, (patch_pool_grid, patch_pool_grid))
        return pooled.permute(0, 2, 3, 1).reshape(-1, patch_pool_grid * patch_pool_grid, token_dim)
    raise ValueError("feature_mode must be one of: global, patch_pool")


def _load_window_images(
    windows: list[LiberoWindowRecord],
    camera_keys: list[str],
    handles: dict[str, h5py.File],
) -> np.ndarray:
    images: list[np.ndarray] = []
    for window in windows:
        source_file = str(Path(window.source_file).expanduser().resolve())
        handle = handles.get(source_file)
        if handle is None:
            handle = h5py.File(source_file, "r")
            handles[source_file] = handle
        for frame_index in window.context_frame_indices:
            for camera_key in camera_keys:
                dataset = handle[f"data/{window.demo_name}/obs/{camera_key}"]
                images.append(np.asarray(dataset[int(frame_index)], dtype=np.uint8))
    return np.stack(images, axis=0)


def _preprocess_images(images: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    if images.ndim != 4 or int(images.shape[-1]) != 3:
        raise ValueError(f"expected NHWC RGB uint8 images, got shape {images.shape}")
    tensor = torch.from_numpy(images).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    if int(tensor.shape[-1]) != image_size or int(tensor.shape[-2]) != image_size:
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _as_global_features(raw_features: object) -> torch.Tensor:
    if isinstance(raw_features, torch.Tensor):
        return raw_features.detach().to(dtype=torch.float32)
    if isinstance(raw_features, dict):
        for key in ("x_norm_clstoken", "x_prenorm", "cls_token"):
            value = raw_features.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return value.detach().to(dtype=torch.float32)
    raise TypeError(f"unsupported DINO output type: {type(raw_features)!r}")


def _cache_feature_mode(feature_mode: str, patch_pool_grid: int) -> str:
    if feature_mode == "global":
        return "global_context_camera_concat"
    if feature_mode == "patch_pool":
        return f"patch_pool_{patch_pool_grid}x{patch_pool_grid}_context_camera_concat"
    raise ValueError("feature_mode must be one of: global, patch_pool")


def _visual_token_count(feature_mode: str, part_count: int, patch_pool_grid: int) -> int:
    if feature_mode == "global":
        return part_count
    if feature_mode == "patch_pool":
        return part_count * patch_pool_grid * patch_pool_grid
    raise ValueError("feature_mode must be one of: global, patch_pool")


def _visual_token_dim(
    features: np.ndarray,
    feature_mode: str,
    part_count: int,
    patch_pool_grid: int,
) -> int:
    token_count = _visual_token_count(feature_mode, part_count, patch_pool_grid)
    if int(features.shape[1]) % token_count != 0:
        raise ValueError("feature dimension is not divisible by visual token count")
    return int(features.shape[1] // token_count)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _summary_path(summary_json: str | None, output_path: Path) -> Path:
    if summary_json:
        return Path(summary_json).expanduser().resolve()
    return output_path.with_suffix(".summary.json")


if __name__ == "__main__":
    main()
