"""Helpers for predicted event-mode mixture conditioning."""

from __future__ import annotations

from torch import Tensor


EVENT_CANDIDATE_POLICIES = ("topk", "transition_reserve")


def map_event_probabilities(
    source_probs: Tensor,
    source_class_names: tuple[str, ...],
    target_class_names: tuple[str, ...],
) -> Tensor:
    """Map event probabilities from a predictor class set to a target class set.

    Missing target classes receive zero probability. Rows are renormalized over
    the target classes so a stable8 predictor can drive an all-observed cVAE.
    """

    if source_probs.ndim != 2:
        raise ValueError(f"source_probs must be rank-2 [B,C], got {source_probs.shape}")
    if source_probs.shape[-1] != len(source_class_names):
        raise ValueError(
            "source_probs class dim must match source_class_names: "
            f"{source_probs.shape[-1]} vs {len(source_class_names)}"
        )
    if not target_class_names:
        raise ValueError("target_class_names must be non-empty")
    source_index = {name: index for index, name in enumerate(source_class_names)}
    mapped = source_probs.new_zeros((source_probs.shape[0], len(target_class_names)))
    for target_index, name in enumerate(target_class_names):
        source_index_value = source_index.get(name)
        if source_index_value is not None:
            mapped[:, target_index] = source_probs[:, source_index_value]
    row_sums = mapped.sum(dim=-1, keepdim=True)
    if bool((row_sums <= 0.0).any().item()):
        raise ValueError("mapped probabilities have an all-zero row")
    return mapped / row_sums


def rank_uniform_counts(num_samples: int, top_m: int) -> tuple[int, ...]:
    """Allocate a fixed K sample budget uniformly over top-M event ranks."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if top_m > num_samples:
        raise ValueError("top_m cannot exceed num_samples for rank-uniform allocation")
    base = num_samples // top_m
    remainder = num_samples % top_m
    return tuple(base + (1 if rank < remainder else 0) for rank in range(top_m))


def select_event_candidates(
    event_probs: Tensor,
    event_class_names: tuple[str, ...],
    *,
    top_m: int,
    policy: str = "topk",
    transition_reserve_threshold: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Select top-M event candidates for predicted-event mixture sampling."""

    if event_probs.ndim != 2:
        raise ValueError(f"event_probs must be rank-2 [B,C], got {event_probs.shape}")
    if event_probs.shape[-1] != len(event_class_names):
        raise ValueError(
            "event_probs class dim must match event_class_names: "
            f"{event_probs.shape[-1]} vs {len(event_class_names)}"
        )
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if top_m > event_probs.shape[-1]:
        raise ValueError("top_m cannot exceed number of event classes")
    if policy not in EVENT_CANDIDATE_POLICIES:
        raise ValueError(f"unsupported event candidate policy {policy!r}")
    if transition_reserve_threshold < 0.0:
        raise ValueError("transition_reserve_threshold must be non-negative")

    top_probs, top_indices = event_probs.topk(k=top_m, dim=-1)
    if policy == "topk":
        return _renormalize_top_probs(top_probs), top_indices

    transition_mask = event_probs.new_tensor(
        [event_label_is_transition(label) for label in event_class_names],
        dtype=event_probs.dtype,
    ).bool()
    if not bool(transition_mask.any().item()):
        return _renormalize_top_probs(top_probs), top_indices

    selected_transition = transition_mask[top_indices].any(dim=-1)
    transition_probs = event_probs.masked_fill(~transition_mask.unsqueeze(0), -1.0)
    best_transition_probs, best_transition_indices = transition_probs.max(dim=-1)
    should_reserve = (~selected_transition) & (best_transition_probs >= transition_reserve_threshold)
    if bool(should_reserve.any().item()):
        top_indices = top_indices.clone()
        top_probs = top_probs.clone()
        replace_slot = top_probs.argmin(dim=-1)
        row_indices = should_reserve.nonzero(as_tuple=False).flatten()
        top_indices[row_indices, replace_slot[row_indices]] = best_transition_indices[row_indices]
        top_probs[row_indices, replace_slot[row_indices]] = best_transition_probs[row_indices]
        top_probs, order = top_probs.sort(dim=-1, descending=True)
        top_indices = top_indices.gather(dim=-1, index=order)
    return _renormalize_top_probs(top_probs), top_indices


def event_label_is_transition(label: str) -> bool:
    """Return whether an event-mode label denotes a gripper transition."""

    return label.startswith("transition_") or label.startswith("mixed_transition")


def event_timing_bin(label: str) -> str:
    """Extract the timing-bin suffix from an event-mode label."""

    if "::" not in label:
        return "unknown"
    return label.rsplit("::", maxsplit=1)[-1]


def _renormalize_top_probs(top_probs: Tensor) -> Tensor:
    return top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
