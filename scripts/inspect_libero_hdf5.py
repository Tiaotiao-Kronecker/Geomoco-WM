#!/usr/bin/env python3
"""Inspect official LIBERO HDF5 files before building GeoMoCo-WM datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.libero_hdf5_inspect import (  # noqa: E402
    inspect_libero_hdf5_suite,
    write_libero_hdf5_inspection_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a read-only Gate 0 inspection on LIBERO HDF5 demonstrations."
    )
    parser.add_argument(
        "--input-path",
        default="/home/user/dataset/libero_official/libero_goal",
        help="Official LIBERO HDF5 file or suite directory.",
    )
    parser.add_argument("--suite-name", default=None, help="Optional suite name override.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file cap for smoke checks.",
    )
    parser.add_argument(
        "--max-demos-per-file",
        type=int,
        default=None,
        help="Optional per-file demo cap for smoke checks.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/gate0/libero_hdf5_inspection.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/gate0/libero_hdf5_inspection.md",
        help="Markdown report path. Use an empty string to skip Markdown.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = inspect_libero_hdf5_suite(
        input_path=args.input_path,
        suite_name=args.suite_name,
        max_files=args.max_files,
        max_demos_per_file=args.max_demos_per_file,
    )
    output_md = args.output_md or None
    write_libero_hdf5_inspection_report(report, args.output_json, output_md)
    console_payload = {
        "input_path": report["input_path"],
        "suite_name": report["suite_name"],
        "num_files": report["summary"]["num_files"],
        "num_demos": report["summary"]["num_demos"],
        "num_frames": report["summary"]["num_frames"],
        "output_json": str(Path(args.output_json).expanduser()),
        "output_md": str(Path(output_md).expanduser()) if output_md else None,
        "readiness": report["readiness"],
        "warnings": report["warnings"],
    }
    print(json.dumps(console_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
