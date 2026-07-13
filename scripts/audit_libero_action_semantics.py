#!/usr/bin/env python3
"""Audit LIBERO action semantics before enabling geodesic action metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.action_semantics import (  # noqa: E402
    audit_libero_action_semantics_suite,
    combine_action_semantics_reports,
    write_action_semantics_audit_report,
)
from geomoco_wm.data.libero_hdf5_export import DEFAULT_LIBERO_SUITE_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit LIBERO HDF5 controller metadata for action metric semantics."
    )
    parser.add_argument(
        "--input-path",
        default="/home/user/dataset/libero_official/libero_goal",
        help="LIBERO HDF5 file or suite directory. Ignored when --all-libero-suites is set.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/user/dataset/libero_official",
        help="Root containing LIBERO suite directories for --all-libero-suites.",
    )
    parser.add_argument("--suite-name", default=None, help="Optional suite name override.")
    parser.add_argument(
        "--all-libero-suites",
        action="store_true",
        help="Audit all requested suite directories under --dataset-root.",
    )
    parser.add_argument(
        "--suite-names",
        nargs="+",
        default=list(DEFAULT_LIBERO_SUITE_NAMES),
        help="Suite names used with --all-libero-suites.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--max-files-per-suite",
        type=int,
        default=None,
        help="Per-suite file cap for --all-libero-suites. Defaults to --max-files.",
    )
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default="outputs/action_semantics/libero_action_semantics_audit.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/action_semantics/libero_action_semantics_audit.md",
        help="Markdown report path. Use an empty string to skip Markdown.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_libero_suites:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        max_files_per_suite = args.max_files_per_suite
        if max_files_per_suite is None:
            max_files_per_suite = args.max_files
        suite_reports = [
            audit_libero_action_semantics_suite(
                dataset_root / suite_name,
                suite_name=suite_name,
                max_files=max_files_per_suite,
                max_demos_per_file=args.max_demos_per_file,
            )
            for suite_name in args.suite_names
        ]
        report = combine_action_semantics_reports(suite_reports)
        report["dataset_root"] = str(dataset_root)
        report["suite_names"] = list(args.suite_names)
    else:
        report = audit_libero_action_semantics_suite(
            args.input_path,
            suite_name=args.suite_name,
            max_files=args.max_files,
            max_demos_per_file=args.max_demos_per_file,
        )

    output_md = args.output_md or None
    write_action_semantics_audit_report(report, args.output_json, output_md)
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).expanduser()),
                "output_md": str(Path(output_md).expanduser()) if output_md else None,
                "summary": report["summary"],
                "readiness": report["readiness"],
                "warnings": report["warnings"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
