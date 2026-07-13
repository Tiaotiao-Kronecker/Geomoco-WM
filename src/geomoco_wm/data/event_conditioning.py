"""Utilities for event-mode conditioning vectors."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor


STABLE_EVENT_MODE_CLASSES = (
    "sustain_open::none",
    "sustain_close::none",
    "transition_close::early",
    "transition_close::middle",
    "transition_close::late",
    "transition_open::early",
    "transition_open::middle",
    "transition_open::late",
)


@dataclass(frozen=True)
class EventModeConditioner:
    """One-hot event-mode conditioner keyed by window id."""

    mode: str
    class_names: tuple[str, ...]
    label_by_window_id: dict[str, str]
    event_mode_audit_json: str | None = None
    class_set: str = "stable8"
    shuffle_seed: int = 0

    @property
    def dim(self) -> int:
        return len(self.class_names) if self.mode != "none" else 0

    @property
    def index_by_label(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.class_names)}

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["class_names"] = list(self.class_names)
        payload["num_labels"] = len(self.label_by_window_id)
        payload.pop("label_by_window_id")
        return payload


def load_event_mode_conditioner(
    event_mode_audit_json: str | Path | None,
    *,
    mode: str,
    class_set: str = "stable8",
    shuffle_seed: int = 0,
) -> EventModeConditioner:
    """Load event-mode labels and optional shuffled controls."""

    if mode not in ("none", "oracle", "shuffled"):
        raise ValueError("event conditioning mode must be one of: none, oracle, shuffled")
    if class_set not in ("stable8", "all_observed"):
        raise ValueError("event class set must be one of: stable8, all_observed")
    if mode == "none":
        return EventModeConditioner(
            mode="none",
            class_names=(),
            label_by_window_id={},
            event_mode_audit_json=None,
            class_set=class_set,
            shuffle_seed=shuffle_seed,
        )
    if event_mode_audit_json is None:
        raise ValueError("event_mode_audit_json is required when event conditioning is enabled")
    resolved_path = Path(event_mode_audit_json).expanduser().resolve()
    report = json.loads(resolved_path.read_text(encoding="utf-8"))
    class_names = _class_names(report, class_set)
    class_set_lookup = set(class_names)
    labels = {
        str(item["window_id"]): str(item["event_mode"])
        for item in report.get("window_labels", [])
        if str(item["event_mode"]) in class_set_lookup
    }
    if not labels:
        raise ValueError("no event-mode labels remain after class filtering")
    if mode == "shuffled":
        labels = _shuffle_values(labels, shuffle_seed)
    return EventModeConditioner(
        mode=mode,
        class_names=class_names,
        label_by_window_id=labels,
        event_mode_audit_json=str(resolved_path),
        class_set=class_set,
        shuffle_seed=shuffle_seed,
    )


def batch_event_mode_conditioning(
    batch: dict[str, object],
    conditioner: EventModeConditioner,
    device: torch.device,
) -> Tensor | None:
    """Return one-hot event-mode conditioning for a batch."""

    if conditioner.dim == 0:
        return None
    window_ids = batch["window_id"]
    rows = []
    for row in range(_batch_size(window_ids)):
        window_id = _batch_string_at(window_ids, row)
        try:
            label = conditioner.label_by_window_id[window_id]
        except KeyError as exc:
            raise ValueError(f"{window_id} is missing from event-mode conditioner") from exc
        index = conditioner.index_by_label[label]
        rows.append(index)
    one_hot = torch.zeros((len(rows), conditioner.dim), dtype=torch.float32, device=device)
    one_hot[torch.arange(len(rows), device=device), torch.tensor(rows, device=device)] = 1.0
    return one_hot


def combine_conditioning(
    base_conditioning: Tensor | None,
    event_conditioning: Tensor | None,
) -> Tensor | None:
    """Concatenate optional suite/task and event-mode conditioning tensors."""

    if base_conditioning is None:
        return event_conditioning
    if event_conditioning is None:
        return base_conditioning
    return torch.cat([base_conditioning, event_conditioning.to(dtype=base_conditioning.dtype)], dim=-1)


def _class_names(report: dict[str, object], class_set: str) -> tuple[str, ...]:
    if class_set == "stable8":
        return STABLE_EVENT_MODE_CLASSES
    counts = report.get("event_mode_counts")
    if not isinstance(counts, dict):
        raise ValueError("event-mode audit JSON must include event_mode_counts")
    return tuple(sorted(str(label) for label in counts))


def _shuffle_values(labels: dict[str, str], seed: int) -> dict[str, str]:
    keys = sorted(labels)
    values = [labels[key] for key in keys]
    rng = random.Random(seed)
    rng.shuffle(values)
    return {key: value for key, value in zip(keys, values)}


def _batch_size(values: object) -> int:
    if isinstance(values, (list, tuple)):
        return len(values)
    return len(values)  # type: ignore[arg-type]


def _batch_string_at(values: object, row: int) -> str:
    if isinstance(values, (list, tuple)):
        return str(values[row])
    return str(values[row])  # type: ignore[index]
