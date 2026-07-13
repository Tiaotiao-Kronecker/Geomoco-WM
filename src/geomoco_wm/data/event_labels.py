"""Weak manipulation event labels for exported LIBERO windows."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, read_window_jsonl

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional at import time
    h5py = None


@dataclass(frozen=True)
class GripperEventConfig:
    """Threshold/sign convention for deriving gripper events from action chunks."""

    command_threshold: float = 0.5
    close_sign: int = -1

    def __post_init__(self) -> None:
        if self.command_threshold <= 0.0:
            raise ValueError("command_threshold must be positive")
        if self.close_sign not in (-1, 1):
            raise ValueError("close_sign must be -1 or 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GripperEventLabel:
    """Window-level event label derived from the future action chunk."""

    event_type: str
    has_close: bool
    has_open: bool
    close_step: int | None
    open_step: int | None
    event_step: int | None
    event_strength: float
    close_fraction: float
    open_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label_gripper_events(
    action_chunk: Sequence[Sequence[float]],
    config: GripperEventConfig | None = None,
) -> GripperEventLabel:
    """Label open/close/hold events from a future action chunk."""

    cfg = config or GripperEventConfig()
    if not action_chunk:
        raise ValueError("action_chunk must be non-empty")
    gripper_values = []
    for row in action_chunk:
        if not row:
            raise ValueError("action rows must be non-empty")
        gripper_values.append(float(row[-1]))

    close_steps = [
        index
        for index, value in enumerate(gripper_values)
        if value * cfg.close_sign >= cfg.command_threshold
    ]
    open_steps = [
        index
        for index, value in enumerate(gripper_values)
        if value * cfg.close_sign <= -cfg.command_threshold
    ]
    close_step = close_steps[0] if close_steps else None
    open_step = open_steps[0] if open_steps else None
    has_close = close_step is not None
    has_open = open_step is not None
    if has_close and has_open:
        event_type = "mixed"
        event_step = min(close_step, open_step)
    elif has_close:
        event_type = "close"
        event_step = close_step
    elif has_open:
        event_type = "open"
        event_step = open_step
    else:
        event_type = "hold"
        event_step = None

    horizon = len(gripper_values)
    return GripperEventLabel(
        event_type=event_type,
        has_close=has_close,
        has_open=has_open,
        close_step=close_step,
        open_step=open_step,
        event_step=event_step,
        event_strength=max(abs(value) for value in gripper_values),
        close_fraction=len(close_steps) / horizon,
        open_fraction=len(open_steps) / horizon,
    )


def label_gripper_transition_events(
    action_chunk: Sequence[Sequence[float]],
    *,
    previous_gripper_command: float | None,
    config: GripperEventConfig | None = None,
) -> GripperEventLabel:
    """Label open/close transition events from gripper command sign changes."""

    cfg = config or GripperEventConfig()
    if not action_chunk:
        raise ValueError("action_chunk must be non-empty")
    gripper_values = [float(row[-1]) for row in action_chunk]
    states = [_command_state(value, cfg) for value in gripper_values]
    previous_state = (
        _command_state(previous_gripper_command, cfg)
        if previous_gripper_command is not None
        else "unknown"
    )
    prior_and_future_states = [previous_state] + states
    close_steps = [
        index
        for index, state in enumerate(states)
        if state == "close" and prior_and_future_states[index] != "close"
    ]
    open_steps = [
        index
        for index, state in enumerate(states)
        if state == "open" and prior_and_future_states[index] != "open"
    ]
    close_step = close_steps[0] if close_steps else None
    open_step = open_steps[0] if open_steps else None
    has_close = close_step is not None
    has_open = open_step is not None
    if has_close and has_open:
        event_type = "mixed_transition"
        event_step = min(close_step, open_step)
    elif has_close:
        event_type = "close_transition"
        event_step = close_step
    elif has_open:
        event_type = "open_transition"
        event_step = open_step
    elif all(state == "close" for state in states):
        event_type = "sustain_close"
        event_step = None
    elif all(state == "open" for state in states):
        event_type = "sustain_open"
        event_step = None
    else:
        event_type = "hold"
        event_step = None

    horizon = len(gripper_values)
    return GripperEventLabel(
        event_type=event_type,
        has_close=has_close,
        has_open=has_open,
        close_step=close_step,
        open_step=open_step,
        event_step=event_step,
        event_strength=max(abs(value) for value in gripper_values),
        close_fraction=sum(state == "close" for state in states) / horizon,
        open_fraction=sum(state == "open" for state in states) / horizon,
    )


def infer_close_sign_from_width_deltas(
    commands: Sequence[float],
    width_deltas: Sequence[float],
    *,
    command_threshold: float = 0.5,
) -> dict[str, Any]:
    """Infer which gripper command sign closes by correlating with width deltas."""

    if command_threshold <= 0.0:
        raise ValueError("command_threshold must be positive")
    if len(commands) != len(width_deltas):
        raise ValueError("commands and width_deltas must have the same length")

    positive = [float(delta) for cmd, delta in zip(commands, width_deltas) if cmd >= command_threshold]
    negative = [float(delta) for cmd, delta in zip(commands, width_deltas) if cmd <= -command_threshold]
    positive_mean = _mean_or_none(positive)
    negative_mean = _mean_or_none(negative)
    inferred_close_sign: int | None = None
    confidence = 0.0
    if positive and negative:
        assert positive_mean is not None
        assert negative_mean is not None
        inferred_close_sign = 1 if positive_mean < negative_mean else -1
        confidence = abs(positive_mean - negative_mean)

    return {
        "command_threshold": command_threshold,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_mean_width_delta": positive_mean,
        "negative_mean_width_delta": negative_mean,
        "inferred_close_sign": inferred_close_sign,
        "confidence_width_delta_gap": confidence,
    }


def audit_gripper_events_from_windows(
    windows_jsonl: str | Path | Sequence[str | Path],
    *,
    max_windows: int | None = None,
    command_threshold: float = 0.5,
    close_sign: int | None = None,
    infer_close_sign: bool = True,
    label_mode: str = "transition",
    max_sign_audit_windows: int = 5000,
) -> dict[str, Any]:
    """Audit gripper-transition event labels over exported window JSONL files."""

    paths = _normalize_paths(windows_jsonl)
    windows: list[LiberoWindowRecord] = []
    for path in paths:
        windows.extend(read_window_jsonl(path))
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        windows = windows[:max_windows]
    if not windows:
        raise ValueError("no windows found for gripper event audit")

    sign_audit = _audit_close_sign_from_hdf5(
        windows[:max_sign_audit_windows],
        command_threshold=command_threshold,
    ) if infer_close_sign else None
    effective_close_sign = close_sign
    if effective_close_sign is None and sign_audit is not None:
        inferred = sign_audit.get("inferred_close_sign")
        effective_close_sign = int(inferred) if inferred in (-1, 1) else None
    if effective_close_sign is None:
        effective_close_sign = -1
    config = GripperEventConfig(
        command_threshold=command_threshold,
        close_sign=effective_close_sign,
    )

    if label_mode not in ("command", "transition"):
        raise ValueError("label_mode must be one of: command, transition")
    previous_gripper_commands = (
        _load_previous_gripper_commands(windows) if label_mode == "transition" else {}
    )
    labels = [
        _label_window_gripper_event(
            window,
            config,
            label_mode=label_mode,
            previous_gripper_commands=previous_gripper_commands,
        )
        for window in windows
    ]
    suite_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    close_steps: Counter[str] = Counter()
    open_steps: Counter[str] = Counter()
    event_steps: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    command_bins: Counter[str] = Counter()
    motion_by_event: dict[str, list[dict[str, float]]] = defaultdict(list)
    close_step_by_task: dict[str, Counter[str]] = defaultdict(Counter)

    for window, label in zip(windows, labels):
        event_types[label.event_type] += 1
        suite_counts[window.suite_name][label.event_type] += 1
        task_key = f"{window.suite_name}/{window.task_id}"
        task_counts[task_key][label.event_type] += 1
        if label.close_step is not None:
            close_steps[str(label.close_step)] += 1
            close_step_by_task[task_key][str(label.close_step)] += 1
        if label.open_step is not None:
            open_steps[str(label.open_step)] += 1
        if label.event_step is not None:
            event_steps[str(label.event_step)] += 1
        for row in window.action_chunk:
            gripper_value = float(row[-1])
            if gripper_value * config.close_sign >= command_threshold:
                command_bins["close"] += 1
            elif gripper_value * config.close_sign <= -command_threshold:
                command_bins["open"] += 1
            else:
                command_bins["hold"] += 1
        motion_by_event[label.event_type].append(_motion_summary(window.future_delta_ee_states))

    task_shortcut_risks = _task_shortcut_risks(close_step_by_task)
    warnings = _audit_warnings(sign_audit, event_types, task_shortcut_risks)
    return {
        "schema_version": "geomoco_wm_gripper_event_audit_v0",
        "windows_jsonl": [str(path) for path in paths],
        "num_windows": len(windows),
        "horizon": len(windows[0].action_chunk),
        "config": config.to_dict(),
        "label_mode": label_mode,
        "sign_audit": sign_audit,
        "event_type_counts": _counter_dict(event_types),
        "event_type_fraction": _fraction_dict(event_types),
        "command_step_counts": _counter_dict(command_bins),
        "command_step_fraction": _fraction_dict(command_bins),
        "close_step_counts": _counter_dict(close_steps),
        "open_step_counts": _counter_dict(open_steps),
        "event_step_counts": _counter_dict(event_steps),
        "suite_event_type_counts": {
            key: _counter_dict(value) for key, value in sorted(suite_counts.items())
        },
        "task_event_type_counts": {
            key: _counter_dict(value) for key, value in sorted(task_counts.items())
        },
        "motion_by_event_type": {
            key: _summarize_motion_values(value) for key, value in sorted(motion_by_event.items())
        },
        "task_close_step_shortcut_risks": task_shortcut_risks,
        "warnings": warnings,
    }


def label_gripper_events_for_windows(
    windows: Sequence[LiberoWindowRecord],
    *,
    config: GripperEventConfig,
    label_mode: str = "transition",
) -> dict[str, GripperEventLabel]:
    """Return gripper event labels keyed by window id."""

    if label_mode not in ("command", "transition"):
        raise ValueError("label_mode must be one of: command, transition")
    previous_gripper_commands = (
        _load_previous_gripper_commands(windows) if label_mode == "transition" else {}
    )
    return {
        window.window_id: _label_window_gripper_event(
            window,
            config,
            label_mode=label_mode,
            previous_gripper_commands=previous_gripper_commands,
        )
        for window in windows
    }


def previous_gripper_commands_for_windows(
    windows: Sequence[LiberoWindowRecord],
) -> dict[str, float | None]:
    """Return previous gripper action commands keyed by window id."""

    return _load_previous_gripper_commands(windows)


def render_gripper_event_audit_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown report for a gripper-event audit."""

    lines = [
        "# Gate 2.4h-a Gripper Event Audit",
        "",
        f"- windows: `{report['num_windows']}`",
        f"- horizon: `{report['horizon']}`",
        f"- close sign: `{report['config']['close_sign']}`",
        f"- command threshold: `{report['config']['command_threshold']}`",
        f"- label mode: `{report.get('label_mode', 'command')}`",
        "",
        "## Event Types",
        "",
        "| event | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    fractions = report["event_type_fraction"]
    for event, count in report["event_type_counts"].items():
        lines.append(f"| `{event}` | {count} | {fractions.get(event, 0.0):.4f} |")

    lines.extend(["", "## Step Commands", "", "| command | count | fraction |", "| --- | ---: | ---: |"])
    command_fractions = report["command_step_fraction"]
    for command, count in report["command_step_counts"].items():
        lines.append(f"| `{command}` | {count} | {command_fractions.get(command, 0.0):.4f} |")

    lines.extend(["", "## First Close Step", "", "| step | count |", "| --- | ---: |"])
    for step, count in report["close_step_counts"].items():
        lines.append(f"| `{step}` | {count} |")

    lines.extend(["", "## Motion By Event Type", "", "| event | n | final trans L2 | path trans L2 |", "| --- | ---: | ---: | ---: |"])
    for event, summary in report["motion_by_event_type"].items():
        lines.append(
            f"| `{event}` | {summary['count']} | "
            f"{summary['final_translation_l2_mean']:.6f} | "
            f"{summary['translation_path_l2_mean']:.6f} |"
        )

    sign_audit = report.get("sign_audit")
    if sign_audit is not None:
        lines.extend(
            [
                "",
                "## Sign Audit",
                "",
                f"- inferred close sign: `{sign_audit.get('inferred_close_sign')}`",
                f"- positive count: `{sign_audit.get('positive_count')}`",
                f"- negative count: `{sign_audit.get('negative_count')}`",
                f"- positive mean width delta: `{sign_audit.get('positive_mean_width_delta')}`",
                f"- negative mean width delta: `{sign_audit.get('negative_mean_width_delta')}`",
            ]
        )

    risks = report.get("task_close_step_shortcut_risks", [])
    if risks:
        lines.extend(["", "## Potential Timing Shortcut Risks", ""])
        for risk in risks[:20]:
            lines.append(
                "- "
                f"`{risk['task']}`: top close step `{risk['top_step']}` "
                f"share `{risk['top_step_fraction']:.3f}` over `{risk['num_close_windows']}` close windows"
            )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def write_gripper_event_audit_report(
    report: dict[str, Any],
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
) -> None:
    """Write gripper-event audit artifacts."""

    json_path = Path(output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if output_md is not None:
        md_path = Path(output_md).expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_gripper_event_audit_markdown(report), encoding="utf-8")


def _label_window_gripper_event(
    window: LiberoWindowRecord,
    config: GripperEventConfig,
    *,
    label_mode: str,
    previous_gripper_commands: dict[str, float | None],
) -> GripperEventLabel:
    if label_mode == "command":
        return label_gripper_events(window.action_chunk, config)
    return label_gripper_transition_events(
        window.action_chunk,
        previous_gripper_command=previous_gripper_commands.get(window.window_id),
        config=config,
    )


def _command_state(value: float | None, config: GripperEventConfig) -> str:
    if value is None:
        return "unknown"
    if value * config.close_sign >= config.command_threshold:
        return "close"
    if value * config.close_sign <= -config.command_threshold:
        return "open"
    return "hold"


def _audit_close_sign_from_hdf5(
    windows: Sequence[LiberoWindowRecord],
    *,
    command_threshold: float,
) -> dict[str, Any] | None:
    if h5py is None:
        return None
    commands: list[float] = []
    width_deltas: list[float] = []
    for window in windows:
        source_file = Path(window.source_file)
        if not source_file.exists():
            continue
        with h5py.File(source_file, "r") as handle:
            gripper_path = f"data/{window.demo_name}/obs/gripper_states"
            if gripper_path not in handle:
                continue
            gripper_states = handle[gripper_path]
            action_end = min(window.action_end, gripper_states.shape[0] - 1)
            for local_index, action_index in enumerate(range(window.action_start, action_end)):
                if local_index >= len(window.action_chunk):
                    break
                width_now = _gripper_width(gripper_states[action_index])
                width_next = _gripper_width(gripper_states[action_index + 1])
                commands.append(float(window.action_chunk[local_index][-1]))
                width_deltas.append(width_next - width_now)
    if not commands:
        return None
    report = infer_close_sign_from_width_deltas(
        commands,
        width_deltas,
        command_threshold=command_threshold,
    )
    report["num_pairs"] = len(commands)
    return report


def _load_previous_gripper_commands(
    windows: Sequence[LiberoWindowRecord],
) -> dict[str, float | None]:
    previous: dict[str, float | None] = {}
    if h5py is None:
        return {window.window_id: None for window in windows}
    grouped: dict[tuple[str, str], list[LiberoWindowRecord]] = defaultdict(list)
    for window in windows:
        grouped[(window.source_file, window.demo_name)].append(window)
    for (source_file, demo_name), group_windows in grouped.items():
        path = Path(source_file)
        if not path.exists():
            for window in group_windows:
                previous[window.window_id] = None
            continue
        with h5py.File(path, "r") as handle:
            action_path = f"data/{demo_name}/actions"
            if action_path not in handle:
                for window in group_windows:
                    previous[window.window_id] = None
                continue
            actions = handle[action_path]
            for window in group_windows:
                previous_index = window.action_start - 1
                if previous_index < 0:
                    previous[window.window_id] = None
                else:
                    previous[window.window_id] = float(actions[previous_index][-1])
    return previous


def _gripper_width(gripper_state: Sequence[float]) -> float:
    values = [float(value) for value in gripper_state]
    if len(values) >= 2:
        return abs(values[0] - values[1])
    if values:
        return abs(values[0])
    return 0.0


def _motion_summary(future_delta_ee_states: Sequence[Sequence[float]]) -> dict[str, float]:
    if not future_delta_ee_states:
        return {"final_translation_l2": 0.0, "translation_path_l2": 0.0}
    translations = [[float(value) for value in row[:3]] for row in future_delta_ee_states]
    final_translation_l2 = _vector_l2(translations[-1])
    previous = [0.0, 0.0, 0.0]
    path = 0.0
    for translation in translations:
        path += _vector_l2([value - prev for value, prev in zip(translation, previous)])
        previous = translation
    return {
        "final_translation_l2": final_translation_l2,
        "translation_path_l2": path,
    }


def _summarize_motion_values(values: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        "count": len(values),
        "final_translation_l2_mean": _mean(item["final_translation_l2"] for item in values),
        "translation_path_l2_mean": _mean(item["translation_path_l2"] for item in values),
    }


def _task_shortcut_risks(close_step_by_task: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for task, counts in close_step_by_task.items():
        total = sum(counts.values())
        if total < 50:
            continue
        top_step, top_count = counts.most_common(1)[0]
        top_fraction = top_count / total
        if top_fraction >= 0.8:
            risks.append(
                {
                    "task": task,
                    "num_close_windows": total,
                    "top_step": top_step,
                    "top_step_fraction": top_fraction,
                }
            )
    return sorted(risks, key=lambda item: (-item["top_step_fraction"], item["task"]))


def _audit_warnings(
    sign_audit: dict[str, Any] | None,
    event_types: Counter[str],
    shortcut_risks: Sequence[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if sign_audit is None:
        warnings.append("Could not infer gripper close sign from HDF5 gripper-width deltas.")
    elif sign_audit.get("inferred_close_sign") not in (-1, 1):
        warnings.append("Gripper close sign inference was inconclusive.")
    has_close_events = any(
        "close" in event_type or "mixed" in event_type for event_type in event_types
    )
    if not has_close_events:
        warnings.append("No close events found under the current threshold/sign convention.")
    if shortcut_risks:
        warnings.append("Some tasks have highly concentrated close-step distributions.")
    return warnings


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        normalized = [paths]
    else:
        normalized = list(paths)
    if not normalized:
        raise ValueError("windows_jsonl must contain at least one path")
    return [Path(path).expanduser().resolve() for path in normalized]


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _vector_l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _fraction_dict(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {str(key): float(value / total) for key, value in sorted(counter.items())}
