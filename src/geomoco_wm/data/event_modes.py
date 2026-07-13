"""Event-mode targets built from weak gripper-transition labels."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from geomoco_wm.data.event_labels import (
    GripperEventConfig,
    GripperEventLabel,
    audit_gripper_events_from_windows,
    label_gripper_events_for_windows,
)
from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, read_window_jsonl


EVENT_TYPE_ALIASES = {
    "close_transition": "transition_close",
    "open_transition": "transition_open",
    "mixed_transition": "mixed_transition",
    "sustain_close": "sustain_close",
    "sustain_open": "sustain_open",
    "hold": "hold",
}


@dataclass(frozen=True)
class EventModeLabel:
    """Window-level event mode used by Gate 3.1."""

    window_id: str
    episode_id: str
    suite_name: str
    task_id: str
    raw_event_type: str
    event_type: str
    timing_bin: str
    event_mode: str
    event_step: int | None
    close_step: int | None
    open_step: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timing_bin_for_step(event_step: int | None, *, horizon: int) -> str:
    """Map an event step to early/middle/late/none bins."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if event_step is None:
        return "none"
    if event_step < 0 or event_step >= horizon:
        raise ValueError("event_step must be within the action horizon")
    early_end = math.ceil(horizon / 3)
    middle_end = math.ceil(2 * horizon / 3)
    if event_step < early_end:
        return "early"
    if event_step < middle_end:
        return "middle"
    return "late"


def normalize_event_type(raw_event_type: str) -> str:
    """Normalize historical transition label names into Gate 3.1 mode names."""

    try:
        return EVENT_TYPE_ALIASES[raw_event_type]
    except KeyError as exc:
        raise ValueError(f"unknown gripper event type: {raw_event_type}") from exc


def build_event_mode_label(
    window: LiberoWindowRecord,
    gripper_label: GripperEventLabel,
) -> EventModeLabel:
    """Combine gripper event type and timing into a single event-mode label."""

    horizon = len(window.action_chunk)
    event_type = normalize_event_type(gripper_label.event_type)
    timing_bin = timing_bin_for_step(gripper_label.event_step, horizon=horizon)
    return EventModeLabel(
        window_id=window.window_id,
        episode_id=window.episode_id,
        suite_name=window.suite_name,
        task_id=window.task_id,
        raw_event_type=gripper_label.event_type,
        event_type=event_type,
        timing_bin=timing_bin,
        event_mode=f"{event_type}::{timing_bin}",
        event_step=gripper_label.event_step,
        close_step=gripper_label.close_step,
        open_step=gripper_label.open_step,
    )


def materialize_event_modes_for_windows(
    windows: Sequence[LiberoWindowRecord],
    *,
    config: GripperEventConfig,
    label_mode: str = "transition",
) -> list[EventModeLabel]:
    """Return Gate 3.1 event-mode labels for exported LIBERO windows."""

    if label_mode != "transition":
        raise ValueError("Gate 3.1 event modes require label_mode='transition'")
    gripper_labels = label_gripper_events_for_windows(
        windows,
        config=config,
        label_mode=label_mode,
    )
    return [
        build_event_mode_label(window, gripper_labels[window.window_id])
        for window in windows
    ]


def audit_event_modes_from_windows(
    windows_jsonl: str | Path | Sequence[str | Path],
    *,
    max_windows: int | None = None,
    command_threshold: float = 0.5,
    close_sign: int | None = None,
    infer_close_sign: bool = True,
    max_sign_audit_windows: int = 5000,
    train_ratio: float = 0.8,
    split_by: str = "episode",
    seed: int = 7,
    min_class_count: int = 50,
    shortcut_step_fraction: float = 0.8,
    include_window_labels: bool = True,
) -> dict[str, Any]:
    """Materialize and audit Gate 3.1 event-mode targets."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if split_by not in ("episode", "window"):
        raise ValueError("split_by must be one of: episode, window")
    if min_class_count <= 0:
        raise ValueError("min_class_count must be positive")
    if not 0.0 < shortcut_step_fraction <= 1.0:
        raise ValueError("shortcut_step_fraction must be in (0, 1]")

    paths = _normalize_paths(windows_jsonl)
    windows: list[LiberoWindowRecord] = []
    for path in paths:
        windows.extend(read_window_jsonl(path))
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        windows = windows[:max_windows]
    if not windows:
        raise ValueError("no windows found for event-mode audit")

    gripper_audit = audit_gripper_events_from_windows(
        paths,
        max_windows=max_windows,
        command_threshold=command_threshold,
        close_sign=close_sign,
        infer_close_sign=infer_close_sign,
        label_mode="transition",
        max_sign_audit_windows=max_sign_audit_windows,
    )
    config = GripperEventConfig(
        command_threshold=command_threshold,
        close_sign=int(gripper_audit["config"]["close_sign"]),
    )
    labels = materialize_event_modes_for_windows(
        windows,
        config=config,
        label_mode="transition",
    )
    train_indices, val_indices = _split_indices(
        windows,
        train_ratio=train_ratio,
        split_by=split_by,
        seed=seed,
    )

    report = {
        "schema_version": "geomoco_wm_event_mode_audit_v0",
        "windows_jsonl": [str(path) for path in paths],
        "num_windows": len(windows),
        "horizon": len(windows[0].action_chunk),
        "config": config.to_dict(),
        "label_mode": "transition",
        "train_ratio": train_ratio,
        "split_by": split_by,
        "seed": seed,
        "min_class_count": min_class_count,
        "shortcut_step_fraction": shortcut_step_fraction,
        "gripper_event_audit": {
            "event_type_counts": gripper_audit["event_type_counts"],
            "event_step_counts": gripper_audit["event_step_counts"],
            "sign_audit": gripper_audit.get("sign_audit"),
            "warnings": gripper_audit["warnings"],
        },
        "event_type_counts": _counter_dict(_count_attr(labels, "event_type")),
        "event_type_fraction": _fraction_dict(_count_attr(labels, "event_type")),
        "timing_bin_counts": _counter_dict(_count_attr(labels, "timing_bin")),
        "timing_bin_fraction": _fraction_dict(_count_attr(labels, "timing_bin")),
        "event_mode_counts": _counter_dict(_count_attr(labels, "event_mode")),
        "event_mode_fraction": _fraction_dict(_count_attr(labels, "event_mode")),
        "event_step_counts": _counter_dict(_event_step_counts(labels)),
        "suite_event_mode_counts": _nested_counts(labels, "suite_name", "event_mode"),
        "task_event_mode_counts": _task_counts(labels),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_event_mode_counts": _counter_dict(
            _count_attr([labels[index] for index in train_indices], "event_mode")
        ),
        "val_event_mode_counts": _counter_dict(
            _count_attr([labels[index] for index in val_indices], "event_mode")
        ),
    }
    rare_modes = _rare_modes(report["event_mode_counts"], min_class_count=min_class_count)
    report["rare_event_modes"] = rare_modes
    report["missing_split_modes"] = _missing_split_modes(
        report["event_mode_counts"],
        report["train_event_mode_counts"],
        report["val_event_mode_counts"],
        min_class_count=min_class_count,
    )
    report["warnings"] = _event_mode_warnings(
        report,
        rare_modes=rare_modes,
        shortcut_step_fraction=shortcut_step_fraction,
    )
    if include_window_labels:
        report["window_labels"] = [label.to_dict() for label in labels]
    return report


def render_event_mode_audit_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown report for Gate 3.1a."""

    lines = [
        "# Gate 3.1a Event Mode Target Audit",
        "",
        f"- windows: `{report['num_windows']}`",
        f"- horizon: `{report['horizon']}`",
        f"- close sign: `{report['config']['close_sign']}`",
        f"- command threshold: `{report['config']['command_threshold']}`",
        f"- split: `{report['split_by']}`, train ratio `{report['train_ratio']}`, seed `{report['seed']}`",
        "",
        "## Event Modes",
        "",
        "| event mode | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    fractions = report["event_mode_fraction"]
    for mode, count in report["event_mode_counts"].items():
        lines.append(f"| `{mode}` | {count} | {fractions.get(mode, 0.0):.4f} |")

    lines.extend(["", "## Event Types", "", "| event type | count | fraction |", "| --- | ---: | ---: |"])
    type_fractions = report["event_type_fraction"]
    for event_type, count in report["event_type_counts"].items():
        lines.append(f"| `{event_type}` | {count} | {type_fractions.get(event_type, 0.0):.4f} |")

    lines.extend(["", "## Timing Bins", "", "| timing bin | count | fraction |", "| --- | ---: | ---: |"])
    timing_fractions = report["timing_bin_fraction"]
    for timing_bin, count in report["timing_bin_counts"].items():
        lines.append(f"| `{timing_bin}` | {count} | {timing_fractions.get(timing_bin, 0.0):.4f} |")

    lines.extend(["", "## Event Steps", "", "| step | count |", "| --- | ---: |"])
    for step, count in report["event_step_counts"].items():
        lines.append(f"| `{step}` | {count} |")

    lines.extend(["", "## Train/Val Balance", "", "| split | size | modes |", "| --- | ---: | ---: |"])
    lines.append(f"| train | {report['train_size']} | {len(report['train_event_mode_counts'])} |")
    lines.append(f"| val | {report['val_size']} | {len(report['val_event_mode_counts'])} |")

    missing = report.get("missing_split_modes", [])
    if missing:
        lines.extend(["", "## Missing Split Modes", ""])
        for item in missing:
            lines.append(
                f"- `{item['event_mode']}` count `{item['count']}` missing in `{item['missing_split']}`"
            )

    rare = report.get("rare_event_modes", [])
    if rare:
        lines.extend(["", "## Rare Modes", ""])
        for item in rare:
            lines.append(f"- `{item['event_mode']}` count `{item['count']}`")

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def write_event_mode_audit_report(
    report: dict[str, Any],
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> None:
    """Write Gate 3.1a event-mode audit artifacts."""

    json_path = Path(output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if output_md is not None:
        md_path = Path(output_md).expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_event_mode_audit_markdown(report), encoding="utf-8")


def _split_indices(
    windows: Sequence[LiberoWindowRecord],
    *,
    train_ratio: float,
    split_by: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    if split_by == "window":
        indices = list(range(len(windows)))
        rng.shuffle(indices)
        split = int(round(len(indices) * train_ratio))
        return sorted(indices[:split]), sorted(indices[split:])

    episode_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        episode_to_indices[window.episode_id].append(index)
    episodes = sorted(episode_to_indices)
    rng.shuffle(episodes)
    split = int(round(len(episodes) * train_ratio))
    train_episode_set = set(episodes[:split])
    train_indices = [
        index
        for episode_id, indices in episode_to_indices.items()
        if episode_id in train_episode_set
        for index in indices
    ]
    val_indices = [
        index
        for episode_id, indices in episode_to_indices.items()
        if episode_id not in train_episode_set
        for index in indices
    ]
    return sorted(train_indices), sorted(val_indices)


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        normalized = [paths]
    else:
        normalized = list(paths)
    if not normalized:
        raise ValueError("windows_jsonl must contain at least one path")
    return [Path(path).expanduser().resolve() for path in normalized]


def _count_attr(labels: Sequence[EventModeLabel], attr: str) -> Counter[str]:
    return Counter(str(getattr(label, attr)) for label in labels)


def _event_step_counts(labels: Sequence[EventModeLabel]) -> Counter[str]:
    return Counter(str(label.event_step) for label in labels if label.event_step is not None)


def _nested_counts(labels: Sequence[EventModeLabel], outer_attr: str, inner_attr: str) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        nested[str(getattr(label, outer_attr))][str(getattr(label, inner_attr))] += 1
    return {key: _counter_dict(value) for key, value in sorted(nested.items())}


def _task_counts(labels: Sequence[EventModeLabel]) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        task_key = f"{label.suite_name}/{label.task_id}"
        nested[task_key][label.event_mode] += 1
    return {key: _counter_dict(value) for key, value in sorted(nested.items())}


def _rare_modes(counts: dict[str, int], *, min_class_count: int) -> list[dict[str, Any]]:
    return [
        {"event_mode": mode, "count": count}
        for mode, count in counts.items()
        if count < min_class_count
    ]


def _missing_split_modes(
    all_counts: dict[str, int],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    *,
    min_class_count: int,
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for mode, count in all_counts.items():
        if count < min_class_count:
            continue
        if train_counts.get(mode, 0) == 0:
            missing.append({"event_mode": mode, "count": count, "missing_split": "train"})
        if val_counts.get(mode, 0) == 0:
            missing.append({"event_mode": mode, "count": count, "missing_split": "val"})
    return missing


def _event_mode_warnings(
    report: dict[str, Any],
    *,
    rare_modes: Sequence[dict[str, Any]],
    shortcut_step_fraction: float,
) -> list[str]:
    warnings = list(report["gripper_event_audit"].get("warnings", []))
    transition_count = sum(
        count
        for mode, count in report["event_mode_counts"].items()
        if mode.startswith("transition_") or mode.startswith("mixed_transition")
    )
    if transition_count == 0:
        warnings.append("No transition event modes found.")
    event_step_counts = report["event_step_counts"]
    if transition_count > 0 and event_step_counts:
        step0_count = event_step_counts.get("0", 0)
        step0_fraction = step0_count / transition_count
        if step0_fraction >= shortcut_step_fraction:
            warnings.append(
                f"Step-0 transition events dominate with fraction {step0_fraction:.3f}."
            )
    if rare_modes:
        warnings.append(
            f"{len(rare_modes)} event modes have fewer than {report['min_class_count']} windows."
        )
    if report["missing_split_modes"]:
        warnings.append("Some common event modes are missing from train or validation split.")
    return warnings


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _fraction_dict(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {str(key): float(value / total) for key, value in sorted(counter.items())}
