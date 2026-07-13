#!/usr/bin/env python3
"""Audit weak gripper-transition event labels for exported LIBERO windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.event_labels import (  # noqa: E402
    audit_gripper_events_from_windows,
    write_gripper_event_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Gate 2.4h-a gripper events.")
    parser.add_argument("--windows-jsonl", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    parser.add_argument(
        "--label-mode",
        default="transition",
        choices=["transition", "command"],
        help="Transition mode marks sign changes; command mode marks command state.",
    )
    parser.add_argument(
        "--close-sign",
        default="auto",
        choices=["auto", "negative", "positive"],
        help="Use HDF5 width-delta sign audit by default.",
    )
    parser.add_argument("--max-sign-audit-windows", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    close_sign = _parse_close_sign(args.close_sign)
    report = audit_gripper_events_from_windows(
        args.windows_jsonl,
        max_windows=args.max_windows,
        command_threshold=args.command_threshold,
        close_sign=close_sign,
        infer_close_sign=args.close_sign == "auto",
        label_mode=args.label_mode,
        max_sign_audit_windows=args.max_sign_audit_windows,
    )
    write_gripper_event_audit_report(
        report,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).expanduser().resolve()),
                "output_md": str(Path(args.output_md).expanduser().resolve())
                if args.output_md
                else None,
                "num_windows": report["num_windows"],
                "event_type_counts": report["event_type_counts"],
                "warnings": report["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_close_sign(value: str) -> int | None:
    if value == "auto":
        return None
    if value == "negative":
        return -1
    if value == "positive":
        return 1
    raise ValueError("close-sign must be auto, negative, or positive")


if __name__ == "__main__":
    main()
