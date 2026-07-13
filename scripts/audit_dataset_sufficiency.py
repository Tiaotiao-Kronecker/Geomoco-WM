#!/usr/bin/env python3
"""Audit whether the current GeoMoCo-WM slice has enough event coverage."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from geomoco_wm.data.predicted_event_mixture import event_label_is_transition  # noqa: E402
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402
from train_future_motion_predictor import _split_indices  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit GeoMoCo-WM dataset/event coverage for sufficiency."
    )
    parser.add_argument(
        "--event-mode-audit-json",
        required=True,
        help="Event-mode audit JSON with window_labels.",
    )
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default=None,
        help="Defaults to windows_jsonl recorded in the event-mode audit JSON.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--split-by", default=None, choices=["window", "episode"])
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    event_report = json.loads(
        Path(args.event_mode_audit_json).expanduser().read_text(encoding="utf-8")
    )
    windows_jsonl = args.windows_jsonl or event_report["windows_jsonl"]
    train_ratio = float(args.train_ratio or event_report.get("train_ratio", 0.8))
    split_by = str(args.split_by or event_report.get("split_by", "episode"))
    seed = int(args.seed if args.seed is not None else event_report.get("seed", 7))

    dataset = OracleActionWindowDataset(
        windows_jsonl,
        motion_mode="future_delta_gripper",
    )
    labels = _label_records(event_report)
    missing_labels = [
        str(window.window_id)
        for window in dataset.windows
        if str(window.window_id) not in labels
    ]
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} windows are missing event labels; "
            f"first missing: {missing_labels[0]}"
        )
    train_indices, val_indices = _split_indices(dataset, train_ratio, seed, split_by)
    all_indices = list(range(len(dataset.windows)))

    report = {
        "config": {
            "windows_jsonl": [str(path) for path in dataset.windows_jsonl_paths],
            "event_mode_audit_json": str(Path(args.event_mode_audit_json).expanduser().resolve()),
            "train_ratio": train_ratio,
            "split_by": split_by,
            "seed": seed,
        },
        "summary": _coverage_summary(dataset, labels, all_indices),
        "train": _coverage_summary(dataset, labels, train_indices),
        "val": _coverage_summary(dataset, labels, val_indices),
        "suite_breakdown": _group_breakdown(dataset, labels, all_indices, field="suite_name"),
        "task_breakdown": _group_breakdown(dataset, labels, all_indices, field="task_id"),
        "source_file_breakdown": _group_breakdown(dataset, labels, all_indices, field="source_file"),
        "warnings": _warnings(dataset, labels, all_indices, train_indices, val_indices),
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_md is not None:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "summary": report["summary"]}, indent=2))


def _label_records(event_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = event_report.get("window_labels")
    if not isinstance(labels, list):
        raise ValueError("event-mode audit JSON must include window_labels")
    records: dict[str, dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("window_labels must contain objects")
        records[str(item["window_id"])] = dict(item)
    return records


def _coverage_summary(
    dataset: OracleActionWindowDataset,
    labels: dict[str, dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    windows = [dataset.windows[index] for index in indices]
    records = [labels[str(window.window_id)] for window in windows]
    event_type_counts = Counter(str(record["event_type"]) for record in records)
    event_mode_counts = Counter(str(record["event_mode"]) for record in records)
    timing_bin_counts = Counter(str(record["timing_bin"]) for record in records)
    event_step_counts = Counter(
        int(record["event_step"])
        for record in records
        if record.get("event_step") is not None
    )
    close_step_counts = Counter(
        int(record["close_step"])
        for record in records
        if record.get("close_step") is not None
    )
    open_step_counts = Counter(
        int(record["open_step"])
        for record in records
        if record.get("open_step") is not None
    )
    transition_count = sum(
        1 for record in records if event_label_is_transition(str(record["event_type"]))
    )
    episode_counts = Counter(str(window.episode_id) for window in windows)
    suite_counts = Counter(str(window.suite_name) for window in windows)
    task_counts = Counter(str(window.task_id) for window in windows)
    source_counts = Counter(str(window.source_file) for window in windows)
    windows_per_episode = list(episode_counts.values())
    return {
        "num_windows": len(windows),
        "num_episodes": len(episode_counts),
        "num_suites": len(suite_counts),
        "num_tasks": len(task_counts),
        "num_source_files": len(source_counts),
        "windows_per_episode": _distribution(windows_per_episode),
        "suite_counts": _sorted_counter(suite_counts),
        "task_counts": _sorted_counter(task_counts),
        "source_file_counts": _sorted_counter(source_counts),
        "event_type_counts": _sorted_counter(event_type_counts),
        "event_type_fraction": _fractions(event_type_counts, len(windows)),
        "event_mode_counts": _sorted_counter(event_mode_counts),
        "timing_bin_counts": _sorted_counter(timing_bin_counts),
        "timing_bin_fraction": _fractions(timing_bin_counts, len(windows)),
        "event_step_counts": _sorted_counter(event_step_counts),
        "close_step_counts": _sorted_counter(close_step_counts),
        "open_step_counts": _sorted_counter(open_step_counts),
        "transition_count": transition_count,
        "transition_fraction": _safe_div(transition_count, len(windows)),
        "close_transition_count": event_type_counts.get("transition_close", 0),
        "open_transition_count": event_type_counts.get("transition_open", 0),
        "mixed_transition_count": event_type_counts.get("mixed_transition", 0),
    }


def _group_breakdown(
    dataset: OracleActionWindowDataset,
    labels: dict[str, dict[str, Any]],
    indices: list[int],
    *,
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        grouped[str(getattr(dataset.windows[index], field))].append(index)
    return {
        key: _small_group_summary(dataset, labels, group_indices)
        for key, group_indices in sorted(grouped.items())
    }


def _small_group_summary(
    dataset: OracleActionWindowDataset,
    labels: dict[str, dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    records = [labels[str(dataset.windows[index].window_id)] for index in indices]
    event_type_counts = Counter(str(record["event_type"]) for record in records)
    episode_count = len({str(dataset.windows[index].episode_id) for index in indices})
    transition_count = sum(
        1 for record in records if event_label_is_transition(str(record["event_type"]))
    )
    return {
        "num_windows": len(indices),
        "num_episodes": episode_count,
        "transition_count": transition_count,
        "transition_fraction": _safe_div(transition_count, len(indices)),
        "event_type_counts": _sorted_counter(event_type_counts),
    }


def _warnings(
    dataset: OracleActionWindowDataset,
    labels: dict[str, dict[str, Any]],
    all_indices: list[int],
    train_indices: list[int],
    val_indices: list[int],
) -> list[str]:
    warnings: list[str] = []
    summary = _coverage_summary(dataset, labels, all_indices)
    transition_fraction = float(summary["transition_fraction"])
    if transition_fraction < 0.15:
        warnings.append(
            f"transition windows are sparse: {transition_fraction:.4f} of all windows"
        )
    if int(summary["close_transition_count"]) < 1000:
        warnings.append(
            f"close transition count is below 1000: {summary['close_transition_count']}"
        )
    if int(summary["open_transition_count"]) < 1000:
        warnings.append(
            f"open transition count is below 1000: {summary['open_transition_count']}"
        )
    train_modes = set(_coverage_summary(dataset, labels, train_indices)["event_mode_counts"])
    val_modes = set(_coverage_summary(dataset, labels, val_indices)["event_mode_counts"])
    missing_train = sorted(val_modes - train_modes)
    missing_val = sorted(train_modes - val_modes)
    if missing_train:
        warnings.append(f"event modes missing from train: {missing_train}")
    if missing_val:
        warnings.append(f"event modes missing from val: {missing_val}")
    return warnings


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
    }


def _sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _fractions(counter: Counter[Any], total: int) -> dict[str, float]:
    return {str(key): _safe_div(value, total) for key, value in sorted(counter.items())}


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dataset Sufficiency Audit",
        "",
        "## Summary",
        "",
        f"- windows: `{summary['num_windows']}`",
        f"- episodes: `{summary['num_episodes']}`",
        f"- suites: `{summary['num_suites']}`",
        f"- tasks: `{summary['num_tasks']}`",
        f"- source files: `{summary['num_source_files']}`",
        f"- transition fraction: `{summary['transition_fraction']:.6f}`",
        f"- close transitions: `{summary['close_transition_count']}`",
        f"- open transitions: `{summary['open_transition_count']}`",
        f"- mixed transitions: `{summary['mixed_transition_count']}`",
        "",
        "## Event Types",
        "",
        "| event type | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    for key, count in summary["event_type_counts"].items():
        fraction = summary["event_type_fraction"].get(key, 0.0)
        lines.append(f"| {key} | {count} | {fraction:.6f} |")
    lines.extend(["", "## Timing Bins", "", "| timing bin | count | fraction |", "| --- | ---: | ---: |"])
    for key, count in summary["timing_bin_counts"].items():
        fraction = summary["timing_bin_fraction"].get(key, 0.0)
        lines.append(f"| {key} | {count} | {fraction:.6f} |")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Task Breakdown", "", "| task | windows | episodes | transition fraction |", "| --- | ---: | ---: | ---: |"])
    for key, value in report["task_breakdown"].items():
        lines.append(
            f"| {key} | {value['num_windows']} | {value['num_episodes']} | "
            f"{value['transition_fraction']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
