"""Read-only Gate 0 inspection for official LIBERO HDF5 demonstrations.

This module intentionally does not export images or build training windows.
Its job is to answer the first engineering question: do the local HDF5 files
contain the visual, proprioceptive, EEF, and action fields needed by the first
GeoMoCo-WM dataset exporter?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional at import time
    h5py = None


DEFAULT_REQUIRED_DEMO_KEYS = ("actions", "obs")
DEFAULT_REQUIRED_OBS_KEYS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "ee_pos",
    "ee_ori",
    "gripper_states",
    "joint_states",
)
DEFAULT_CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
OBJECT_STATE_KEYS = ("object-state", "object_state", "object_states")
MAX_EXAMPLES = 50


class LiberoHdf5InspectionError(ValueError):
    """Raised when the LIBERO HDF5 inspection cannot be completed."""


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    total = float(sum(values))
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": total / len(values),
    }


def _parse_demo_index(demo_name: str) -> tuple[int, str]:
    try:
        return int(demo_name.split("_")[-1]), demo_name
    except ValueError:
        return 10**9, demo_name


def _derive_task_id_from_hdf5_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        return stem[: -len("_demo")]
    return stem


def _shape_key(shape: Sequence[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def _dataset_shape(dataset: Any) -> list[int]:
    return [int(dim) for dim in dataset.shape]


def _first_dim(dataset: Any) -> int | None:
    if not hasattr(dataset, "shape") or len(dataset.shape) == 0:
        return None
    return int(dataset.shape[0])


def _add_example(examples: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    if len(examples) < MAX_EXAMPLES:
        examples.append(payload)


def _counter_sample(counter: Counter[str], limit: int = 20) -> Counter[str]:
    result: Counter[str] = Counter()
    for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
        result[value] = count
        if len(result) >= limit:
            break
    return result


def _json_counter_preview(mapping: dict[str, int], limit: int = 12) -> str:
    items = list(mapping.items())
    preview = dict(items[:limit])
    suffix = "" if len(items) <= limit else f" ... (+{len(items) - limit} more)"
    return json.dumps(preview, ensure_ascii=False) + suffix


def discover_libero_hdf5_files(input_path: str | Path) -> list[Path]:
    """Return sorted `.hdf5` files from a LIBERO file or suite directory."""

    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.glob("*.hdf5"))
    else:
        raise LiberoHdf5InspectionError(f"input path does not exist: {root}")

    if not files:
        raise LiberoHdf5InspectionError(f"no .hdf5 files found under {root}")
    return files


def inspect_libero_hdf5_suite(
    input_path: str | Path,
    *,
    suite_name: str | None = None,
    max_files: int | None = None,
    max_demos_per_file: int | None = None,
    required_demo_keys: Sequence[str] = DEFAULT_REQUIRED_DEMO_KEYS,
    required_obs_keys: Sequence[str] = DEFAULT_REQUIRED_OBS_KEYS,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> dict[str, Any]:
    """Inspect LIBERO HDF5 files without loading image tensors into memory."""

    if h5py is None:
        raise LiberoHdf5InspectionError("h5py is required to inspect LIBERO HDF5 files")
    if max_files is not None and max_files <= 0:
        raise LiberoHdf5InspectionError("max_files must be positive when provided")
    if max_demos_per_file is not None and max_demos_per_file <= 0:
        raise LiberoHdf5InspectionError("max_demos_per_file must be positive when provided")

    input_root = Path(input_path).expanduser().resolve()
    files = discover_libero_hdf5_files(input_root)
    if max_files is not None:
        files = files[:max_files]
    effective_suite_name = suite_name or (
        input_root.name if input_root.is_dir() else input_root.parent.name
    )

    task_counts: Counter[str] = Counter()
    demo_key_counts: Counter[str] = Counter()
    obs_key_counts: Counter[str] = Counter()
    action_dim_counts: Counter[int | str] = Counter()
    ee_state_dim_counts: Counter[int | str] = Counter()
    gripper_dim_counts: Counter[int | str] = Counter()
    joint_dim_counts: Counter[int | str] = Counter()
    camera_shape_counts: Counter[str] = Counter()
    obs_shape_counts: Counter[str] = Counter()
    demo_lengths: list[float] = []
    missing_required_examples: list[dict[str, Any]] = []
    length_mismatch_examples: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []

    demos_with_actions = 0
    demos_with_7d_actions = 0
    demos_with_agentview = 0
    demos_with_all_cameras = 0
    demos_with_eef_pose = 0
    demos_with_gripper = 0
    demos_with_joint = 0
    demos_with_object_state = 0
    demos_with_aligned_lengths = 0
    num_demos = 0
    num_frames = 0

    for source_file in files:
        task_id = _derive_task_id_from_hdf5_path(source_file)
        file_num_demos = 0
        file_demo_lengths: list[int] = []
        file_missing_examples: list[dict[str, Any]] = []
        file_length_mismatch_examples: list[dict[str, Any]] = []

        with h5py.File(source_file, "r") as handle:
            if "data" not in handle:
                raise LiberoHdf5InspectionError(f"{source_file} is missing top-level `data` group")

            data_group = handle["data"]
            demo_names = sorted(data_group.keys(), key=_parse_demo_index)
            if max_demos_per_file is not None:
                demo_names = demo_names[:max_demos_per_file]

            for demo_name in demo_names:
                file_num_demos += 1
                demo_group = data_group[demo_name]
                demo_keys = set(demo_group.keys())
                obs_group = demo_group["obs"] if "obs" in demo_group else None
                obs_keys = set(obs_group.keys()) if obs_group is not None else set()

                task_counts[task_id] += 1
                num_demos += 1
                for key in demo_keys:
                    demo_key_counts[key] += 1
                for key in obs_keys:
                    obs_key_counts[key] += 1

                missing_demo = sorted(set(required_demo_keys) - demo_keys)
                missing_obs = sorted(set(required_obs_keys) - obs_keys)
                if missing_demo or missing_obs:
                    example = {
                        "file": str(source_file),
                        "demo": demo_name,
                        "missing_demo_keys": missing_demo,
                        "missing_obs_keys": missing_obs,
                    }
                    _add_example(missing_required_examples, example)
                    _add_example(file_missing_examples, example)

                lengths: dict[str, int] = {}
                if "actions" in demo_group:
                    actions = demo_group["actions"]
                    demos_with_actions += 1
                    action_shape = _dataset_shape(actions)
                    action_dim = action_shape[-1] if len(action_shape) >= 2 else "scalar_or_empty"
                    action_dim_counts[action_dim] += 1
                    if action_dim == 7:
                        demos_with_7d_actions += 1
                    action_length = _first_dim(actions)
                    if action_length is not None:
                        lengths["actions"] = action_length

                if obs_group is not None:
                    if "ee_states" in obs_group:
                        ee_shape = _dataset_shape(obs_group["ee_states"])
                        ee_dim = ee_shape[-1] if len(ee_shape) >= 2 else "scalar_or_empty"
                        ee_state_dim_counts[ee_dim] += 1
                    elif "ee_pos" in obs_group and "ee_ori" in obs_group:
                        pos_shape = _dataset_shape(obs_group["ee_pos"])
                        ori_shape = _dataset_shape(obs_group["ee_ori"])
                        pos_dim = pos_shape[-1] if len(pos_shape) >= 2 else "scalar_or_empty"
                        ori_dim = ori_shape[-1] if len(ori_shape) >= 2 else "scalar_or_empty"
                        ee_state_dim_counts[f"pos{pos_dim}+ori{ori_dim}"] += 1

                    if (
                        "ee_states" in obs_group
                        and len(obs_group["ee_states"].shape) >= 2
                        and obs_group["ee_states"].shape[-1] == 6
                    ) or (
                        "ee_pos" in obs_group
                        and "ee_ori" in obs_group
                        and len(obs_group["ee_pos"].shape) >= 2
                        and len(obs_group["ee_ori"].shape) >= 2
                        and obs_group["ee_pos"].shape[-1] == 3
                        and obs_group["ee_ori"].shape[-1] == 3
                    ):
                        demos_with_eef_pose += 1

                    if "gripper_states" in obs_group:
                        gripper_shape = _dataset_shape(obs_group["gripper_states"])
                        gripper_dim = gripper_shape[-1] if len(gripper_shape) >= 2 else "scalar"
                        gripper_dim_counts[gripper_dim] += 1
                        demos_with_gripper += 1

                    if "joint_states" in obs_group:
                        joint_shape = _dataset_shape(obs_group["joint_states"])
                        joint_dim = joint_shape[-1] if len(joint_shape) >= 2 else "scalar"
                        joint_dim_counts[joint_dim] += 1
                        demos_with_joint += 1

                    if any(key in obs_group for key in OBJECT_STATE_KEYS):
                        demos_with_object_state += 1

                    if "agentview_rgb" in obs_group:
                        demos_with_agentview += 1
                    if all(key in obs_group for key in camera_keys):
                        demos_with_all_cameras += 1

                    for key in obs_keys:
                        dataset = obs_group[key]
                        shape = _dataset_shape(dataset)
                        obs_shape_counts[f"{key}:{_shape_key(shape)}"] += 1
                        length = _first_dim(dataset)
                        if length is not None:
                            lengths[f"obs/{key}"] = length
                        if key in camera_keys:
                            camera_shape_counts[f"{key}:{_shape_key(shape)}"] += 1

                for key in ("dones", "rewards", "robot_states", "states"):
                    if key in demo_group:
                        length = _first_dim(demo_group[key])
                        if length is not None:
                            lengths[key] = length

                demo_length = lengths.get("actions")
                if demo_length is None and lengths:
                    demo_length = next(iter(lengths.values()))
                if demo_length is not None:
                    file_demo_lengths.append(demo_length)
                    demo_lengths.append(float(demo_length))
                    num_frames += demo_length

                if lengths and len(set(lengths.values())) == 1:
                    demos_with_aligned_lengths += 1
                else:
                    example = {
                        "file": str(source_file),
                        "demo": demo_name,
                        "lengths": lengths,
                    }
                    _add_example(length_mismatch_examples, example)
                    _add_example(file_length_mismatch_examples, example)

        file_reports.append(
            {
                "path": str(source_file),
                "task_id": task_id,
                "num_demos_scanned": file_num_demos,
                "num_frames_scanned": int(sum(file_demo_lengths)),
                "demo_length": _summarize_numeric([float(length) for length in file_demo_lengths]),
                "missing_required_examples": file_missing_examples,
                "length_mismatch_examples": file_length_mismatch_examples,
            }
        )

    all_required_fields_present = not missing_required_examples and num_demos > 0
    all_lengths_aligned = demos_with_aligned_lengths == num_demos and num_demos > 0
    supports_visual_grounding_export = demos_with_agentview == num_demos and num_demos > 0
    supports_dual_camera_export = demos_with_all_cameras == num_demos and num_demos > 0
    supports_eef_motion_targets = demos_with_eef_pose == num_demos and num_demos > 0
    supports_action_chunks = demos_with_7d_actions == num_demos and num_demos > 0
    supports_proprio_context = (
        demos_with_gripper == num_demos and demos_with_joint == num_demos and num_demos > 0
    )
    supports_object_state_teacher = demos_with_object_state == num_demos and num_demos > 0
    supports_gate0_dataset_export = (
        all_required_fields_present
        and all_lengths_aligned
        and supports_visual_grounding_export
        and supports_eef_motion_targets
        and supports_action_chunks
    )

    warnings: list[str] = []
    if not all_required_fields_present:
        warnings.append("Some demos are missing required demo or observation keys.")
    if not all_lengths_aligned:
        warnings.append(
            "Some demos have inconsistent sequence lengths across actions/obs/state fields."
        )
    if not supports_dual_camera_export:
        warnings.append(
            "Not all demos contain both default cameras; start with available cameras only."
        )
    if not supports_proprio_context:
        warnings.append(
            "Gripper or joint states are incomplete; proprio-conditioned probes may be limited."
        )
    if not supports_object_state_teacher:
        warnings.append(
            "Object-state teacher fields are not universally available; keep them diagnostic-only."
        )
    if not supports_gate0_dataset_export:
        warnings.append(
            "Do not start dataset export until the missing fields or alignment "
            "issues are understood."
        )

    return {
        "audit_type": "libero_hdf5_gate0_inspection",
        "input_path": str(input_root),
        "suite_name": effective_suite_name,
        "limits": {
            "max_files": max_files,
            "max_demos_per_file": max_demos_per_file,
        },
        "required": {
            "demo_keys": list(required_demo_keys),
            "obs_keys": list(required_obs_keys),
            "camera_keys": list(camera_keys),
            "missing_required_examples": missing_required_examples,
        },
        "summary": {
            "num_files": len(files),
            "num_demos": num_demos,
            "num_frames": num_frames,
            "tasks": _counter_dict(task_counts),
            "demo_length": _summarize_numeric(demo_lengths),
            "demo_keys": _counter_dict(demo_key_counts),
            "obs_keys": _counter_dict(obs_key_counts),
        },
        "field_shapes": {
            "action_dim_counts": _counter_dict(action_dim_counts),
            "ee_state_dim_counts": _counter_dict(ee_state_dim_counts),
            "gripper_dim_counts": _counter_dict(gripper_dim_counts),
            "joint_dim_counts": _counter_dict(joint_dim_counts),
            "camera_shape_counts": _counter_dict(camera_shape_counts),
            "obs_shape_counts_sample": _counter_dict(_counter_sample(obs_shape_counts)),
        },
        "coverage": {
            "actions_present_ratio": _ratio(demos_with_actions, num_demos),
            "actions_7d_ratio": _ratio(demos_with_7d_actions, num_demos),
            "agentview_rgb_ratio": _ratio(demos_with_agentview, num_demos),
            "all_default_cameras_ratio": _ratio(demos_with_all_cameras, num_demos),
            "eef_pose_ratio": _ratio(demos_with_eef_pose, num_demos),
            "gripper_ratio": _ratio(demos_with_gripper, num_demos),
            "joint_ratio": _ratio(demos_with_joint, num_demos),
            "object_state_teacher_ratio": _ratio(demos_with_object_state, num_demos),
            "aligned_lengths_ratio": _ratio(demos_with_aligned_lengths, num_demos),
        },
        "readiness": {
            "supports_gate0_dataset_export": supports_gate0_dataset_export,
            "supports_visual_grounding_export": supports_visual_grounding_export,
            "supports_dual_camera_export": supports_dual_camera_export,
            "supports_eef_motion_targets": supports_eef_motion_targets,
            "supports_action_chunks": supports_action_chunks,
            "supports_proprio_context": supports_proprio_context,
            "supports_object_state_teacher": supports_object_state_teacher,
        },
        "length_mismatch_examples": length_mismatch_examples,
        "files": file_reports,
        "warnings": warnings,
    }


def render_libero_hdf5_inspection_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for human review."""

    summary = report["summary"]
    coverage = report["coverage"]
    readiness = report["readiness"]
    warnings = report["warnings"]

    lines = [
        "# Gate 0 LIBERO HDF5 Inspection",
        "",
        "## Purpose",
        "",
        "This read-only check verifies whether the local LIBERO HDF5 files expose "
        "the fields required for the first visual-grounded GeoMoCo-WM dataset exporter.",
        "",
        "## Scope",
        "",
        f"- Input path: `{report['input_path']}`",
        f"- Suite name: `{report['suite_name']}`",
        f"- Files scanned: `{summary['num_files']}`",
        f"- Demos scanned: `{summary['num_demos']}`",
        f"- Frames scanned: `{summary['num_frames']}`",
        f"- Limits: `{json.dumps(report['limits'], ensure_ascii=False)}`",
        "",
        "## Readiness",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in readiness.items():
        lines.append(f"| {key} | `{value}` |")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Field | Ratio |",
            "| --- | ---: |",
        ]
    )
    for key, value in coverage.items():
        lines.append(f"| {key} | `{value:.6f}` |")

    lines.extend(
        [
            "",
            "## Shapes",
            "",
            "- Action dim counts: "
            f"`{_json_counter_preview(report['field_shapes']['action_dim_counts'])}`",
            "- EEF state dim counts: "
            f"`{_json_counter_preview(report['field_shapes']['ee_state_dim_counts'])}`",
            "- Gripper dim counts: "
            f"`{_json_counter_preview(report['field_shapes']['gripper_dim_counts'])}`",
            "- Joint dim counts: "
            f"`{_json_counter_preview(report['field_shapes']['joint_dim_counts'])}`",
            "- Camera shape counts preview: "
            f"`{_json_counter_preview(report['field_shapes']['camera_shape_counts'])}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Review Rule",
            "",
            "Only run the real image/export step after `supports_gate0_dataset_export` "
            "is true, or after we explicitly decide that a missing optional field "
            "should remain diagnostic-only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_libero_hdf5_inspection_report(
    report: dict[str, Any],
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> None:
    """Write JSON and optional Markdown inspection artifacts."""

    output_json_path = Path(output_json).expanduser()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if output_md is not None:
        output_md_path = Path(output_md).expanduser()
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(
            render_libero_hdf5_inspection_markdown(report) + "\n",
            encoding="utf-8",
        )
