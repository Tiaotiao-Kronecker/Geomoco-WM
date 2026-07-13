#!/usr/bin/env python3
"""Train the first oracle future-motion -> action decoder diagnostic."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.window_dataset import MOTION_MODES, OracleActionWindowDataset  # noqa: E402
from geomoco_wm.data.action_semantics import (  # noqa: E402
    default_libero_osc_pose_action_semantics_dict,
)
from geomoco_wm.metrics.action_metrics import action_metrics  # noqa: E402
from geomoco_wm.models.action_decoder import ActionDecoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an oracle diagnostic decoder from GT future EEF motion to action chunks."
    )
    parser.add_argument(
        "--windows-jsonl",
        nargs="+",
        default="outputs/libero_windows/libero_goal_smoke/windows.jsonl",
        help="Input windows.jsonl produced by scripts/export_libero_windows.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oracle_action_decoder/libero_goal_smoke",
        help="Output directory for metrics.json and model.pt.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--split-by",
        default="window",
        choices=["window", "episode"],
        help="Use window-level smoke splits or episode-level splits for real comparisons.",
    )
    parser.add_argument(
        "--motion-mode",
        default="future_delta",
        choices=MOTION_MODES,
        help="Use GT future EEF deltas or no motion input for a direct-context baseline.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and print shapes only.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch JSON logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _seed_everything(args.seed)
    dataset = OracleActionWindowDataset(
        args.windows_jsonl,
        max_windows=args.max_windows,
        motion_mode=args.motion_mode,
    )
    spec = dataset.spec()
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "dataset": spec.to_dict()}, indent=2))
        return

    device = _resolve_device(args.device)
    train_indices, val_indices = _split_indices(dataset, args.train_ratio, args.seed, args.split_by)
    train_loader = _make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, args.batch_size, shuffle=False)

    model = ActionDecoder(
        context_dim=spec.context_dim,
        motion_rep_dim=spec.motion_dim,
        action_dim=spec.action_dim,
        horizon=spec.horizon,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = _evaluate(model, val_loader, device, loss_fn)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": spec.to_dict(),
        "device": str(device),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
        "motion_mode": args.motion_mode,
        "split_by": args.split_by,
        "seed": args.seed,
        "action_semantics": default_libero_osc_pose_action_semantics_dict(),
        "history": history,
        "final": history[-1] if history else {},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics},
        output_dir / "model.pt",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "metrics_json": str(output_dir / "metrics.json"),
                    "model_pt": str(output_dir / "model.pt"),
                    "final": metrics["final"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _run_epoch(
    model: ActionDecoder,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        actions = batch["actions"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(context, motion)
        loss = loss_fn(pred, actions)
        loss.backward()
        optimizer.step()
        batch_size = int(context.shape[0])
        batch_metrics = _action_metrics(pred.detach(), actions)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


@torch.no_grad()
def _evaluate(
    model: ActionDecoder,
    loader: DataLoader | None,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float | None]:
    if loader is None:
        return {"mse": None, "mae": None}
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0
    for batch in loader:
        context = batch["context"].to(device)
        motion = batch["motion"].to(device)
        actions = batch["actions"].to(device)
        pred = model(context, motion)
        loss = loss_fn(pred, actions)
        batch_size = int(context.shape[0])
        batch_metrics = _action_metrics(pred, actions)
        batch_metrics["mse"] = float(loss.cpu())
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        total_count += batch_size
    return _average_metrics(totals, total_count)


def _action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return normalized, physical-scale, and geodesic action metrics."""

    return action_metrics(pred, target)


def _average_metrics(totals: dict[str, float], total_count: int) -> dict[str, float | None]:
    if total_count == 0:
        return {"mse": None, "mae": None}
    return {key: value / total_count for key, value in totals.items()}


def _make_loader(
    dataset: OracleActionWindowDataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader | None:
    if not indices:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle)


def _split_indices(
    dataset: OracleActionWindowDataset,
    train_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[list[int], list[int]]:
    if split_by == "window":
        return _split_window_indices(len(dataset), train_ratio, seed)
    if split_by == "episode":
        return _split_episode_indices(dataset, train_ratio, seed)
    raise ValueError("split_by must be one of: window, episode")


def _split_window_indices(num_items: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError("train_ratio must be in (0, 1]")
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items == 1 or train_ratio == 1.0:
        return indices, []
    train_size = max(1, min(num_items - 1, int(num_items * train_ratio)))
    return indices[:train_size], indices[train_size:]


def _split_episode_indices(
    dataset: OracleActionWindowDataset,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError("train_ratio must be in (0, 1]")
    episode_to_indices: dict[str, list[int]] = {}
    for index, window in enumerate(dataset.windows):
        episode_to_indices.setdefault(window.episode_id, []).append(index)
    episode_ids = sorted(episode_to_indices)
    if len(episode_ids) <= 1:
        return _split_window_indices(len(dataset), train_ratio, seed)
    random.Random(seed).shuffle(episode_ids)
    if train_ratio == 1.0:
        return [index for episode_id in episode_ids for index in episode_to_indices[episode_id]], []
    train_episode_count = max(1, min(len(episode_ids) - 1, int(len(episode_ids) * train_ratio)))
    train_episode_ids = set(episode_ids[:train_episode_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for episode_id in episode_ids:
        target = train_indices if episode_id in train_episode_ids else val_indices
        target.extend(episode_to_indices[episode_id])
    random.Random(seed).shuffle(train_indices)
    random.Random(seed + 1).shuffle(val_indices)
    return train_indices, val_indices


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer")
    return dims


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
