from __future__ import annotations

import unittest

import torch

from scripts.audit_gripper_boundary_timing_head import (
    average_precision,
    boundary_quality_report,
    localization_report,
    precision_recall_f1,
    threshold_reports,
)
from scripts.train_predicted_event_mixture_action_head import GRIPPER_BOUNDARY_STEP_CLASSES


class GripperBoundaryTimingAuditTests(unittest.TestCase):
    def test_precision_recall_f1_counts_binary_outcomes(self) -> None:
        report = precision_recall_f1(
            torch.tensor([True, True, False, False]),
            torch.tensor([True, False, True, False]),
        )

        self.assertEqual(report["tp"], 1)
        self.assertEqual(report["fp"], 1)
        self.assertEqual(report["fn"], 1)
        self.assertEqual(report["tn"], 1)
        self.assertAlmostEqual(report["precision"], 0.5)
        self.assertAlmostEqual(report["recall"], 0.5)
        self.assertAlmostEqual(report["f1"], 0.5)

    def test_average_precision_uses_positive_ranks(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.1])
        targets = torch.tensor([False, True, True])

        self.assertAlmostEqual(average_precision(scores, targets), (1 / 2 + 2 / 3) / 2)

    def test_threshold_reports_uses_scores(self) -> None:
        reports = threshold_reports(
            torch.tensor([0.9, 0.7, 0.2]),
            torch.tensor([True, False, True]),
            (0.5,),
        )

        self.assertAlmostEqual(reports["0.500"]["precision"], 0.5)
        self.assertAlmostEqual(reports["0.500"]["recall"], 0.5)

    def test_boundary_quality_report_includes_argmax_and_localization(self) -> None:
        close = GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")
        open_ = GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")
        probs = torch.zeros((2, 3, len(GRIPPER_BOUNDARY_STEP_CLASSES)))
        probs[..., 0] = 0.9
        probs[0, 1, :] = torch.tensor([0.1, 0.8, 0.1])
        probs[1, 2, :] = torch.tensor([0.2, 0.1, 0.7])
        targets = torch.zeros((2, 3), dtype=torch.long)
        targets[0, 1] = close
        targets[1, 2] = open_
        records = [
            {
                "window_index": 0,
                "close_start_step": 1,
                "open_start_step": None,
            },
            {
                "window_index": 1,
                "close_start_step": None,
                "open_start_step": 2,
            },
        ]

        report = boundary_quality_report(probs, targets, records, thresholds=(0.5,), horizon=3)

        self.assertAlmostEqual(report["overall"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["argmax"]["any_boundary"]["recall"], 1.0)
        self.assertAlmostEqual(report["classes"]["close_start"]["target_fraction"], 1 / 6)
        self.assertAlmostEqual(report["localization"]["close_start"]["top1_exact"], 1.0)
        self.assertAlmostEqual(report["thresholds"]["0.500"]["precision"], 1.0)

    def test_localization_report_handles_missing_class(self) -> None:
        probs = torch.zeros((1, 3, len(GRIPPER_BOUNDARY_STEP_CLASSES)))
        probs[..., 0] = 1.0
        records = [{"window_index": 0, "close_start_step": None, "open_start_step": None}]

        report = localization_report(probs, records)

        self.assertEqual(report["close_start"]["count"], 0)
        self.assertIsNone(report["close_start"]["top1_exact"])


if __name__ == "__main__":
    unittest.main()
