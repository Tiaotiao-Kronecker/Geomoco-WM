"""LIBERO action-semantics audit utilities.

The official LIBERO demonstrations store normalized robosuite actions.  This
module records the controller contract we rely on before reporting
SE(3)-aware / geodesic action metrics.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional at import time
    h5py = None

from geomoco_wm.data.libero_hdf5_inspect import (
    LiberoHdf5InspectionError,
    discover_libero_hdf5_files,
)


MAX_AUDIT_EXAMPLES = 20


@dataclass(frozen=True)
class LiberoOscPoseActionSemantics:
    """The action convention used by LIBERO's OSC_POSE demonstrations."""

    name: str = "libero_osc_pose_normalized_delta"
    controller_type: str = "OSC_POSE"
    control_delta: bool = True
    action_dim: int = 7
    translation_indices: tuple[int, int, int] = (0, 1, 2)
    rotation_indices: tuple[int, int, int] = (3, 4, 5)
    gripper_indices: tuple[int] = (6,)
    normalized_input_min: float = -1.0
    normalized_input_max: float = 1.0
    translation_scale_m: tuple[float, float, float] = (0.05, 0.05, 0.05)
    rotation_scale_rad: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rotation_representation: str = "axis_angle_rotvec_after_controller_scaling"
    rotation_composition: str = "R_goal = Exp(rotvec_scaled) @ R_current"
    metric_rule: str = (
        "compare translation after meter scaling and rotation by SO(3) geodesic "
        "between Exp(pred_rotvec_scaled) and Exp(target_rotvec_scaled)"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_libero_osc_pose_action_semantics() -> LiberoOscPoseActionSemantics:
    """Return the canonical action semantics used by this project."""

    return LiberoOscPoseActionSemantics()


def default_libero_osc_pose_action_semantics_dict() -> dict[str, Any]:
    """Return a JSON-serializable copy of the canonical action semantics."""

    return default_libero_osc_pose_action_semantics().to_dict()


def audit_libero_action_semantics_suite(
    input_path: str | Path,
    *,
    suite_name: str | None = None,
    max_files: int | None = None,
    max_demos_per_file: int | None = None,
) -> dict[str, Any]:
    """Audit one LIBERO suite or one HDF5 file for action-controller semantics."""

    if h5py is None:
        raise LiberoHdf5InspectionError("h5py is required to audit LIBERO action semantics")
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

    action_dim_counts: Counter[str] = Counter()
    controller_type_counts: Counter[str] = Counter()
    control_delta_counts: Counter[str] = Counter()
    input_min_counts: Counter[str] = Counter()
    input_max_counts: Counter[str] = Counter()
    output_min_counts: Counter[str] = Counter()
    output_max_counts: Counter[str] = Counter()
    warnings: list[str] = []
    file_reports: list[dict[str, Any]] = []

    num_demos = 0
    num_demos_with_actions = 0
    num_files_with_env_args = 0
    num_files_with_controller = 0
    output_scale_matches = 0
    input_range_matches = 0

    for source_file in files:
        with h5py.File(source_file, "r") as handle:
            if "data" not in handle:
                raise LiberoHdf5InspectionError(f"{source_file} is missing top-level `data` group")

            data_group = handle["data"]
            env_args_raw = data_group.attrs.get("env_args")
            env_args = _decode_json_attr(env_args_raw)
            controller_config = _find_controller_config(env_args)
            controller_summary = _summarize_controller_config(controller_config)

            if env_args is not None:
                num_files_with_env_args += 1
            else:
                _add_warning_once(warnings, f"missing env_args attr: {source_file}")
            if controller_summary:
                num_files_with_controller += 1
                controller_type_counts[str(controller_summary.get("type"))] += 1
                control_delta_counts[str(controller_summary.get("control_delta"))] += 1
                input_min_counts[_json_key(controller_summary.get("input_min"))] += 1
                input_max_counts[_json_key(controller_summary.get("input_max"))] += 1
                output_min_counts[_json_key(controller_summary.get("output_min"))] += 1
                output_max_counts[_json_key(controller_summary.get("output_max"))] += 1
                if _matches_controller_output_scale(controller_summary):
                    output_scale_matches += 1
                if _matches_controller_input_range(controller_summary):
                    input_range_matches += 1
            else:
                _add_warning_once(warnings, f"missing controller config: {source_file}")

            demo_names = sorted(data_group.keys(), key=_parse_demo_index)
            if max_demos_per_file is not None:
                demo_names = demo_names[:max_demos_per_file]

            file_action_dim_counts: Counter[str] = Counter()
            for demo_name in demo_names:
                num_demos += 1
                demo_group = data_group[demo_name]
                if "actions" not in demo_group:
                    action_dim_counts["missing"] += 1
                    file_action_dim_counts["missing"] += 1
                    continue
                actions = demo_group["actions"]
                num_demos_with_actions += 1
                action_dim = actions.shape[-1] if len(actions.shape) >= 2 else "scalar_or_empty"
                action_dim_counts[str(action_dim)] += 1
                file_action_dim_counts[str(action_dim)] += 1

            file_reports.append(
                {
                    "file": str(source_file),
                    "has_env_args": env_args is not None,
                    "controller": controller_summary,
                    "action_dim_counts": _counter_dict(file_action_dim_counts),
                }
            )

    semantics = default_libero_osc_pose_action_semantics_dict()
    all_actions_7d = num_demos > 0 and action_dim_counts == Counter({"7": num_demos})
    all_controllers_osc_pose = (
        num_files_with_controller == len(files)
        and controller_type_counts == Counter({semantics["controller_type"]: len(files)})
    )
    all_control_delta = (
        num_files_with_controller == len(files)
        and control_delta_counts == Counter({str(semantics["control_delta"]): len(files)})
    )
    all_input_range_matches = input_range_matches == len(files)
    all_output_scale_matches = output_scale_matches == len(files)
    supports_geodesic = (
        all_actions_7d
        and all_controllers_osc_pose
        and all_control_delta
        and all_input_range_matches
        and all_output_scale_matches
    )

    if not supports_geodesic:
        _add_warning_once(
            warnings,
            "action semantics were not fully confirmed; keep geodesic metrics disabled or "
            "treat them as tentative for this slice",
        )

    return {
        "input_path": str(input_root),
        "suite_name": effective_suite_name,
        "summary": {
            "num_files": len(files),
            "num_files_with_env_args": num_files_with_env_args,
            "num_files_with_controller": num_files_with_controller,
            "num_demos": num_demos,
            "num_demos_with_actions": num_demos_with_actions,
            "action_dim_counts": _counter_dict(action_dim_counts),
            "controller_type_counts": _counter_dict(controller_type_counts),
            "control_delta_counts": _counter_dict(control_delta_counts),
            "input_min_counts": _counter_dict(input_min_counts),
            "input_max_counts": _counter_dict(input_max_counts),
            "output_min_counts": _counter_dict(output_min_counts),
            "output_max_counts": _counter_dict(output_max_counts),
        },
        "expected_semantics": semantics,
        "readiness": {
            "supports_geodesic_action_metrics": supports_geodesic,
            "all_actions_7d": all_actions_7d,
            "all_controllers_osc_pose": all_controllers_osc_pose,
            "all_control_delta": all_control_delta,
            "all_input_range_matches": all_input_range_matches,
            "all_output_scale_matches": all_output_scale_matches,
        },
        "source_evidence": _source_evidence(),
        "files": file_reports,
        "warnings": warnings,
    }


def combine_action_semantics_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-suite audit reports into one run-level report."""

    readiness_keys = (
        "supports_geodesic_action_metrics",
        "all_actions_7d",
        "all_controllers_osc_pose",
        "all_control_delta",
        "all_input_range_matches",
        "all_output_scale_matches",
    )
    summary = {
        "num_suites": len(reports),
        "num_files": sum(int(report["summary"]["num_files"]) for report in reports),
        "num_demos": sum(int(report["summary"]["num_demos"]) for report in reports),
    }
    readiness = {
        key: bool(reports) and all(bool(report["readiness"].get(key)) for report in reports)
        for key in readiness_keys
    }
    warnings = [
        f"{report['suite_name']}: {warning}"
        for report in reports
        for warning in report.get("warnings", [])
    ]
    return {
        "summary": summary,
        "expected_semantics": default_libero_osc_pose_action_semantics_dict(),
        "readiness": readiness,
        "source_evidence": _source_evidence(),
        "suite_reports": list(reports),
        "warnings": warnings,
    }


def write_action_semantics_audit_report(
    report: dict[str, Any],
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> None:
    """Write JSON and optional Markdown action-semantics audit reports."""

    output_json_path = Path(output_json).expanduser().resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if output_md is not None:
        output_md_path = Path(output_md).expanduser().resolve()
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(render_action_semantics_audit_markdown(report), "utf-8")


def render_action_semantics_audit_markdown(report: dict[str, Any]) -> str:
    """Render an action-semantics audit report as Markdown."""

    lines: list[str] = [
        "# LIBERO Action Semantics Audit",
        "",
        "## Summary",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Readiness"])
    for key, value in report.get("readiness", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    semantics = report.get("expected_semantics", {})
    lines.extend(
        [
            "",
            "## Canonical Semantics",
            f"- controller: `{semantics.get('controller_type')}`",
            f"- action dim: `{semantics.get('action_dim')}`",
            f"- translation scale: `{semantics.get('translation_scale_m')}` meters",
            f"- rotation scale: `{semantics.get('rotation_scale_rad')}` radians",
            f"- rotation representation: `{semantics.get('rotation_representation')}`",
            f"- rotation composition: `{semantics.get('rotation_composition')}`",
            f"- metric rule: `{semantics.get('metric_rule')}`",
        ]
    )

    lines.extend(["", "## Source Evidence"])
    for evidence in report.get("source_evidence", []):
        path = evidence.get("path")
        path_text = f" `{path}`" if path else ""
        lines.append(
            f"- `{evidence.get('kind')}`{path_text}: {evidence.get('finding')} "
            f"(exists: `{evidence.get('exists')}`)"
        )

    suite_reports = report.get("suite_reports")
    if suite_reports is not None:
        lines.extend(["", "## Suite Reports"])
        for suite_report in suite_reports:
            suite_summary = suite_report.get("summary", {})
            suite_readiness = suite_report.get("readiness", {})
            lines.append(
                f"- `{suite_report.get('suite_name')}`: "
                f"files `{suite_summary.get('num_files')}`, demos "
                f"`{suite_summary.get('num_demos')}`, geodesic ready "
                f"`{suite_readiness.get('supports_geodesic_action_metrics')}`"
            )
    else:
        lines.extend(["", "## Files"])
        for file_report in report.get("files", [])[:MAX_AUDIT_EXAMPLES]:
            controller = file_report.get("controller") or {}
            lines.append(
                f"- `{file_report.get('file')}`: controller `{controller.get('type')}`, "
                f"actions `{file_report.get('action_dim_counts')}`"
            )

    warnings = report.get("warnings", [])
    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _decode_json_attr(value: Any) -> Any:
    if value is None:
        return None
    try:
        value = value.item()
    except (AttributeError, ValueError):
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _find_controller_config(value: Any) -> Any:
    if isinstance(value, dict):
        if "controller_configs" in value:
            return _find_controller_config(value["controller_configs"])
        if _looks_like_controller_config(value):
            return value
        for child in value.values():
            found = _find_controller_config(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_controller_config(child)
            if found is not None:
                return found
    return None


def _looks_like_controller_config(value: dict[str, Any]) -> bool:
    return "type" in value and (
        "control_delta" in value or "output_max" in value or "output_min" in value
    )


def _summarize_controller_config(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "type": value.get("type"),
        "control_delta": value.get("control_delta"),
        "input_min": _float_list_or_scalar(value.get("input_min")),
        "input_max": _float_list_or_scalar(value.get("input_max")),
        "output_min": _float_list_or_scalar(value.get("output_min")),
        "output_max": _float_list_or_scalar(value.get("output_max")),
    }


def _float_list_or_scalar(value: Any) -> float | list[float] | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return None


def _matches_controller_input_range(controller: dict[str, Any]) -> bool:
    expected = default_libero_osc_pose_action_semantics()
    return _scalar_or_all_close(controller.get("input_min"), expected.normalized_input_min) and (
        _scalar_or_all_close(controller.get("input_max"), expected.normalized_input_max)
    )


def _matches_controller_output_scale(controller: dict[str, Any]) -> bool:
    expected = default_libero_osc_pose_action_semantics()
    output_min = _as_float_list(controller.get("output_min"))
    output_max = _as_float_list(controller.get("output_max"))
    if output_min is None or output_max is None or len(output_min) < 6 or len(output_max) < 6:
        return False
    expected_max = list(expected.translation_scale_m) + list(expected.rotation_scale_rad)
    expected_min = [-value for value in expected_max]
    return _all_close(output_max[:6], expected_max) and _all_close(output_min[:6], expected_min)


def _scalar_or_all_close(value: Any, expected: float) -> bool:
    values = _as_float_list(value)
    if values is None:
        return False
    return all(abs(item - expected) <= 1e-8 for item in values)


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, tuple):
        return [float(item) for item in value]
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return None


def _all_close(values: Sequence[float], expected: Sequence[float], atol: float = 1e-8) -> bool:
    return len(values) == len(expected) and all(
        abs(float(value) - float(target)) <= atol for value, target in zip(values, expected)
    )


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _parse_demo_index(demo_name: str) -> tuple[int, str]:
    try:
        return int(demo_name.split("_")[-1]), demo_name
    except ValueError:
        return 10**9, demo_name


def _add_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings and len(warnings) < MAX_AUDIT_EXAMPLES:
        warnings.append(warning)


def _source_evidence() -> list[dict[str, Any]]:
    return [
        {
            "kind": "hdf5_env_args",
            "path": None,
            "exists": True,
            "finding": (
                "`data.attrs['env_args']` stores robosuite `controller_configs`; "
                "this audit checks `type`, `control_delta`, normalized input range, "
                "and output scaling."
            ),
        },
        {
            "kind": "local_robosuite_source",
            "path": (
                "/home/user/projects/se3-group-motion-control/.venv/lib/python3.10/"
                "site-packages/robosuite/controllers/osc.py"
            ),
            "exists": Path(
                "/home/user/projects/se3-group-motion-control/.venv/lib/python3.10/"
                "site-packages/robosuite/controllers/osc.py"
            ).exists(),
            "finding": "`OSC_POSE.set_goal` scales normalized deltas before setting goals.",
        },
        {
            "kind": "local_robosuite_source",
            "path": (
                "/home/user/projects/se3-group-motion-control/.venv/lib/python3.10/"
                "site-packages/robosuite/utils/control_utils.py"
            ),
            "exists": Path(
                "/home/user/projects/se3-group-motion-control/.venv/lib/python3.10/"
                "site-packages/robosuite/utils/control_utils.py"
            ).exists(),
            "finding": (
                "`set_goal_orientation` interprets rotation delta as axis-angle and "
                "left-multiplies `Exp(delta)` with the current orientation."
            ),
        },
    ]
