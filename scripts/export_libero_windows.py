#!/usr/bin/env python3
"""Export LIBERO HDF5 demos into lightweight GeoMoCo-WM window records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.libero_hdf5_export import (  # noqa: E402
    DEFAULT_LIBERO_SUITE_NAMES,
    export_libero_hdf5_suite_collection,
    export_libero_hdf5_windows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export official LIBERO HDF5 demos into episode/window JSONL files."
    )
    parser.add_argument(
        "--input-path",
        default="/home/user/dataset/libero_official/libero_goal",
        help="Official LIBERO HDF5 file or suite directory.",
    )
    parser.add_argument(
        "--input-root",
        default="/home/user/dataset/libero_official",
        help="Root containing LIBERO suite directories for multi-suite export.",
    )
    parser.add_argument("--suite-name", default=None, help="Optional suite name override.")
    parser.add_argument(
        "--suite-names",
        nargs="+",
        default=None,
        help="Suite directory names for multi-suite export.",
    )
    parser.add_argument(
        "--all-libero-suites",
        action="store_true",
        help="Export the standard four LIBERO suites: spatial, object, goal, and 10.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for episodes.jsonl, windows.jsonl, and summary.json.",
    )
    parser.add_argument("--context-len", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--max-files-per-suite",
        type=int,
        default=None,
        help="Optional per-suite file cap for multi-suite export.",
    )
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--max-windows-per-suite",
        type=int,
        default=None,
        help="Optional per-suite window cap for multi-suite export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_libero_suites or args.suite_names:
        suite_names = tuple(args.suite_names) if args.suite_names else DEFAULT_LIBERO_SUITE_NAMES
        output_dir = args.output_dir or "outputs/libero_windows/libero_all_suites"
        summary = export_libero_hdf5_suite_collection(
            input_root=args.input_root,
            output_dir=output_dir,
            suite_names=suite_names,
            context_len=args.context_len,
            horizon=args.horizon,
            stride=args.stride,
            max_files_per_suite=args.max_files_per_suite or args.max_files,
            max_demos_per_file=args.max_demos_per_file,
            max_windows_per_suite=args.max_windows_per_suite or args.max_windows,
        )
    else:
        output_dir = args.output_dir or "outputs/libero_windows/libero_goal"
        summary = export_libero_hdf5_windows(
            input_path=args.input_path,
            output_dir=output_dir,
            suite_name=args.suite_name,
            context_len=args.context_len,
            horizon=args.horizon,
            stride=args.stride,
            max_files=args.max_files,
            max_demos_per_file=args.max_demos_per_file,
            max_windows=args.max_windows,
        )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
