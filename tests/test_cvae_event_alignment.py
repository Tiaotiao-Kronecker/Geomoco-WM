from __future__ import annotations

import unittest

import torch

from geomoco_wm.data.event_labels import GripperEventLabel
from scripts.evaluate_cvae_event_alignment import (
    CoverageAccumulator,
    EventReadoutAccumulator,
    event_alignment_error,
    event_oracle_indices_for_batch,
    select_labels_by_indices,
)


def make_label(event_type: str, event_step: int | None = None) -> GripperEventLabel:
    return GripperEventLabel(
        event_type=event_type,
        has_close=event_type == "close_transition",
        has_open=event_type == "open_transition",
        close_step=event_step if event_type == "close_transition" else None,
        open_step=event_step if event_type == "open_transition" else None,
        event_step=event_step,
        event_strength=1.0,
        close_fraction=1.0 if "close" in event_type else 0.0,
        open_fraction=1.0 if "open" in event_type else 0.0,
    )


class CvaeEventAlignmentTests(unittest.TestCase):
    def test_event_alignment_error_prioritizes_type_then_step(self) -> None:
        target = make_label("close_transition", 2)
        same_type_late = make_label("close_transition", 4)
        wrong_type_same_step = make_label("open_transition", 2)

        self.assertLess(
            event_alignment_error(same_type_late, target, horizon=8),
            event_alignment_error(wrong_type_same_step, target, horizon=8),
        )

    def test_event_oracle_indices_pick_best_event_match(self) -> None:
        target = [make_label("close_transition", 2), make_label("sustain_open")]
        sample_labels = [
            [make_label("open_transition", 2), make_label("sustain_close")],
            [make_label("close_transition", 3), make_label("sustain_open")],
        ]

        indices = event_oracle_indices_for_batch(sample_labels, target, horizon=8)
        selected = select_labels_by_indices(sample_labels, indices)

        self.assertTrue(torch.equal(indices, torch.tensor([1, 1])))
        self.assertEqual([label.event_type for label in selected], ["close_transition", "sustain_open"])

    def test_coverage_accumulator_reports_transition_step_coverage(self) -> None:
        target = make_label("close_transition", 2)
        samples = [
            make_label("open_transition", 2),
            make_label("close_transition", 3),
        ]
        accumulator = CoverageAccumulator(horizon=8)

        accumulator.add(samples, target)
        metrics = accumulator.metrics()

        self.assertEqual(metrics["any_event_type_match"], 1.0)
        self.assertEqual(metrics["any_transition_step_exact"], 0.0)
        self.assertEqual(metrics["any_transition_step_within_1"], 1.0)

    def test_readout_accumulator_tracks_hold_as_mismatch(self) -> None:
        accumulator = EventReadoutAccumulator()

        accumulator.add_many(
            [make_label("hold"), make_label("open_transition", 1)],
            [make_label("close_transition", 1), make_label("open_transition", 1)],
        )
        metrics = accumulator.metrics()

        self.assertEqual(metrics["event_type_accuracy"], 0.5)
        self.assertEqual(metrics["transition_type_accuracy"], 0.5)
        self.assertEqual(metrics["transition_step_exact"], 0.5)


if __name__ == "__main__":
    unittest.main()
