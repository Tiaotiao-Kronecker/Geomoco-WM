#!/usr/bin/env python3
"""Episode-bootstrap confidence intervals for paired per-window metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED_FIELDS = {
    "window_id",
    "episode_id",
    "task_id",
    "suite_name",
    "source_file",
    "demo_name",
    "event_type",
    "event_mode",
    "timing_bin",
    "split",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two aligned per-window metric files by resampling validation "
            "episodes with replacement."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline JSONL/JSON/CSV file.")
    parser.add_argument("--candidate", required=True, help="Candidate JSONL/JSON/CSV file.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to compare. Defaults to common numeric columns.",
    )
    parser.add_argument("--window-field", default="window_id")
    parser.add_argument("--episode-field", default="episode_id")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--direction",
        choices=["lower_is_better", "higher_is_better"],
        default="lower_is_better",
        help="For losses/MSE use lower_is_better, so gain = baseline - candidate.",
    )
    parser.add_argument(
        "--episode-weighting",
        choices=["window_weighted", "episode_mean"],
        default="window_weighted",
        help=(
            "window_weighted samples episodes then aggregates their windows; "
            "episode_mean bootstraps per-episode means equally."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Optional exact-match row filter. May be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = episode_bootstrap_report(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        metrics=args.metrics,
        window_field=args.window_field,
        episode_field=args.episode_field,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        ci=args.ci,
        direction=args.direction,
        episode_weighting=args.episode_weighting,
        filters=parse_filters(args.filter),
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_md is not None:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "metrics": report["metrics"]}, indent=2))


def episode_bootstrap_report(
    *,
    baseline_path: Path,
    candidate_path: Path,
    metrics: list[str] | None = None,
    window_field: str = "window_id",
    episode_field: str = "episode_id",
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    num_bootstrap: int = 2000,
    seed: int = 7,
    ci: float = 0.95,
    direction: str = "lower_is_better",
    episode_weighting: str = "window_weighted",
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be between 0 and 1")
    if direction not in {"lower_is_better", "higher_is_better"}:
        raise ValueError("direction must be lower_is_better or higher_is_better")
    if episode_weighting not in {"window_weighted", "episode_mean"}:
        raise ValueError("unsupported episode_weighting")

    baseline_rows = load_metric_rows(baseline_path)
    candidate_rows = load_metric_rows(candidate_path)
    paired_rows, resolved_metrics = align_rows(
        baseline_rows,
        candidate_rows,
        metrics=metrics,
        window_field=window_field,
        episode_field=episode_field,
        filters=filters or {},
    )
    if not paired_rows:
        raise ValueError("no aligned rows remain after filtering")
    episode_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        episode_to_rows[str(row["episode_id"])].append(row)
    episode_ids = sorted(episode_to_rows)
    if not episode_ids:
        raise ValueError("no episodes remain after filtering")

    observed = {
        metric: _gain_for_rows(
            paired_rows,
            metric,
            direction=direction,
            episode_weighting=episode_weighting,
        )
        for metric in resolved_metrics
    }
    bootstrap_values = bootstrap_gains(
        episode_to_rows,
        resolved_metrics,
        num_bootstrap=num_bootstrap,
        seed=seed,
        direction=direction,
        episode_weighting=episode_weighting,
    )
    alpha = 1.0 - ci
    metric_reports = {}
    for metric in resolved_metrics:
        samples = sorted(bootstrap_values[metric])
        ci_low = percentile(samples, alpha / 2.0)
        ci_high = percentile(samples, 1.0 - alpha / 2.0)
        mean_gain = statistics.fmean(samples)
        std_gain = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        half_width = (ci_high - ci_low) / 2.0
        crosses_zero = ci_low <= 0.0 <= ci_high
        metric_reports[metric] = {
            "observed_gain": observed[metric],
            "bootstrap_mean_gain": mean_gain,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_level": ci,
            "ci_half_width": half_width,
            "bootstrap_std": std_gain,
            "crosses_zero": crosses_zero,
            "reliable_positive": ci_low > 0.0,
            "reliable_negative": ci_high < 0.0,
            "effect_to_half_width": abs(observed[metric]) / half_width
            if half_width > 0.0
            else math.inf,
        }

    return {
        "config": {
            "baseline": str(baseline_path.expanduser().resolve()),
            "candidate": str(candidate_path.expanduser().resolve()),
            "baseline_name": baseline_name,
            "candidate_name": candidate_name,
            "window_field": window_field,
            "episode_field": episode_field,
            "metrics": resolved_metrics,
            "num_bootstrap": num_bootstrap,
            "seed": seed,
            "ci": ci,
            "direction": direction,
            "episode_weighting": episode_weighting,
            "filters": filters or {},
            "gain_definition": _gain_definition(direction, baseline_name, candidate_name),
        },
        "num_windows": len(paired_rows),
        "num_episodes": len(episode_ids),
        "episodes": {
            "min_windows": min(len(episode_to_rows[episode_id]) for episode_id in episode_ids),
            "max_windows": max(len(episode_to_rows[episode_id]) for episode_id in episode_ids),
            "mean_windows": statistics.fmean(
                len(episode_to_rows[episode_id]) for episode_id in episode_ids
            ),
        },
        "metrics": metric_reports,
    }


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("records", "rows", "window_metrics", "per_window"):
                if isinstance(payload.get(key), list):
                    return [dict(item) for item in payload[key]]
        raise ValueError(f"{resolved} JSON must be a list or contain records/rows/window_metrics")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"{resolved}:{line_number} is not a JSON object")
            rows.append(dict(item))
    return rows


def align_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    metrics: list[str] | None,
    window_field: str,
    episode_field: str,
    filters: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline_by_window = _index_by_field(baseline_rows, window_field, "baseline")
    candidate_by_window = _index_by_field(candidate_rows, window_field, "candidate")
    missing_candidate = sorted(set(baseline_by_window) - set(candidate_by_window))
    missing_baseline = sorted(set(candidate_by_window) - set(baseline_by_window))
    if missing_candidate or missing_baseline:
        raise ValueError(
            "baseline/candidate window_id mismatch: "
            f"{len(missing_candidate)} missing candidate, {len(missing_baseline)} missing baseline"
        )
    if metrics is None:
        metrics = infer_numeric_metrics(
            baseline_rows,
            candidate_rows,
            excluded_fields={window_field, episode_field, *DEFAULT_EXCLUDED_FIELDS},
        )
    if not metrics:
        raise ValueError("no metrics were specified or inferred")

    paired: list[dict[str, Any]] = []
    for window_id in sorted(baseline_by_window):
        baseline = baseline_by_window[window_id]
        candidate = candidate_by_window[window_id]
        episode_id = str(baseline.get(episode_field, ""))
        if not episode_id:
            raise ValueError(f"baseline row {window_id} is missing {episode_field}")
        candidate_episode_id = str(candidate.get(episode_field, ""))
        if candidate_episode_id != episode_id:
            raise ValueError(
                f"episode mismatch for {window_id}: {episode_id} vs {candidate_episode_id}"
            )
        if not _matches_filters(baseline, candidate, filters):
            continue
        row: dict[str, Any] = {"window_id": str(window_id), "episode_id": episode_id}
        for metric in metrics:
            row[f"baseline::{metric}"] = _as_finite_float(baseline.get(metric), metric, window_id)
            row[f"candidate::{metric}"] = _as_finite_float(candidate.get(metric), metric, window_id)
        paired.append(row)
    return paired, list(metrics)


def infer_numeric_metrics(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    excluded_fields: set[str],
) -> list[str]:
    if not baseline_rows or not candidate_rows:
        return []
    common_keys = set(baseline_rows[0]).intersection(candidate_rows[0]) - excluded_fields
    inferred: list[str] = []
    for key in sorted(common_keys):
        if all(_is_finite_float(row.get(key)) for row in baseline_rows) and all(
            _is_finite_float(row.get(key)) for row in candidate_rows
        ):
            inferred.append(key)
    return inferred


def bootstrap_gains(
    episode_to_rows: dict[str, list[dict[str, Any]]],
    metrics: list[str],
    *,
    num_bootstrap: int,
    seed: int,
    direction: str,
    episode_weighting: str,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    episode_ids = sorted(episode_to_rows)
    values = {metric: [] for metric in metrics}
    for _ in range(num_bootstrap):
        sampled_episode_ids = [rng.choice(episode_ids) for _ in episode_ids]
        sampled_rows: list[dict[str, Any]] = []
        for episode_id in sampled_episode_ids:
            sampled_rows.extend(episode_to_rows[episode_id])
        for metric in metrics:
            if episode_weighting == "episode_mean":
                per_episode = [
                    _gain_for_rows(
                        episode_to_rows[episode_id],
                        metric,
                        direction=direction,
                        episode_weighting="window_weighted",
                    )
                    for episode_id in sampled_episode_ids
                ]
                values[metric].append(statistics.fmean(per_episode))
            else:
                values[metric].append(
                    _gain_for_rows(
                        sampled_rows,
                        metric,
                        direction=direction,
                        episode_weighting="window_weighted",
                    )
                )
    return values


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires non-empty values")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def parse_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"filter must be FIELD=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"filter has empty field: {item!r}")
        filters[key] = value
    return filters


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "# Episode Bootstrap CI",
        "",
        "## Summary",
        "",
        f"- baseline: `{config['baseline_name']}`",
        f"- candidate: `{config['candidate_name']}`",
        f"- gain: `{config['gain_definition']}`",
        f"- windows: `{report['num_windows']}`",
        f"- episodes: `{report['num_episodes']}`",
        f"- bootstrap samples: `{config['num_bootstrap']}`",
        f"- CI: `{config['ci']:.2f}`",
        f"- episode weighting: `{config['episode_weighting']}`",
        "",
        "## Metrics",
        "",
        "| metric | observed gain | bootstrap mean | CI low | CI high | crosses 0 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric, values in report["metrics"].items():
        lines.append(
            f"| {metric} | {values['observed_gain']:.8f} | "
            f"{values['bootstrap_mean_gain']:.8f} | {values['ci_low']:.8f} | "
            f"{values['ci_high']:.8f} | {values['crosses_zero']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _index_by_field(
    rows: list[dict[str, Any]],
    field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        value = str(row.get(field, ""))
        if not value:
            raise ValueError(f"{label} row {row_number} is missing {field}")
        if value in indexed:
            raise ValueError(f"{label} has duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def _matches_filters(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    filters: dict[str, str],
) -> bool:
    for key, expected in filters.items():
        baseline_value = str(baseline.get(key, ""))
        candidate_value = str(candidate.get(key, baseline_value))
        if baseline_value != expected or candidate_value != expected:
            return False
    return True


def _gain_for_rows(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    direction: str,
    episode_weighting: str,
) -> float:
    if not rows:
        raise ValueError("cannot compute gain on zero rows")
    if episode_weighting == "episode_mean":
        episode_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            episode_to_rows[str(row["episode_id"])].append(row)
        return statistics.fmean(
            _gain_for_rows(
                episode_rows,
                metric,
                direction=direction,
                episode_weighting="window_weighted",
            )
            for episode_rows in episode_to_rows.values()
        )
    baseline_mean = statistics.fmean(float(row[f"baseline::{metric}"]) for row in rows)
    candidate_mean = statistics.fmean(float(row[f"candidate::{metric}"]) for row in rows)
    if direction == "higher_is_better":
        return candidate_mean - baseline_mean
    return baseline_mean - candidate_mean


def _gain_definition(direction: str, baseline_name: str, candidate_name: str) -> str:
    if direction == "higher_is_better":
        return f"{candidate_name} - {baseline_name}"
    return f"{baseline_name} - {candidate_name}"


def _as_finite_float(value: Any, metric: str, window_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metric} for window {window_id} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{metric} for window {window_id} is not finite: {value!r}")
    return parsed


def _is_finite_float(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - keeps CLI failures readable.
        print(f"error: {error}", file=sys.stderr)
        raise
