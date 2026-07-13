from __future__ import annotations

import unittest

import torch

from scripts.audit_predicted_event_mixture_action_head_groups import (
    _boundary_index_actions_from_output,
    _event_family,
    _finalize_report,
    _mean_reports,
    _per_item_action_metrics,
    _predicted_boundary_actions_from_output,
    _threshold_key,
)


class PredictedEventMixtureActionHeadGroupAuditTests(unittest.TestCase):
    def test_per_item_action_metrics_keeps_batch_axis(self) -> None:
        pred = torch.zeros((2, 3, 7))
        target = torch.ones((2, 3, 7))

        metrics = _per_item_action_metrics(pred, target)

        self.assertEqual(tuple(metrics["mse"].shape), (2,))
        self.assertEqual(tuple(metrics["gripper_mse"].shape), (2,))
        self.assertEqual(tuple(metrics["translation_m_mse"].shape), (2,))
        self.assertEqual(tuple(metrics["rotation_geodesic_deg"].shape), (2,))
        self.assertAlmostEqual(float(metrics["mse"][0]), 1.0)
        self.assertAlmostEqual(float(metrics["gripper_mse"][1]), 1.0)

    def test_finalize_report_sorts_worst_groups(self) -> None:
        report = _finalize_report(
            overall_totals={"mse": 30.0},
            overall_count=10,
            group_totals={
                "suite/easy": {"mse": 10.0},
                "suite/hard": {"mse": 50.0},
            },
            group_counts={"suite/easy": 25, "suite/hard": 25},
        )

        self.assertAlmostEqual(report["overall"]["mse"], 3.0)
        self.assertEqual(report["worst_groups"][0]["group"], "suite/hard")
        self.assertAlmostEqual(report["groups"]["suite/easy"]["mse"], 0.4)

    def test_mean_reports_averages_group_metrics(self) -> None:
        reports = [
            {
                "overall": {"mse": 1.0},
                "groups": {"all": {"count": 10, "mse": 1.0}},
                "worst_groups": [],
            },
            {
                "overall": {"mse": 3.0},
                "groups": {"all": {"count": 20, "mse": 5.0}},
                "worst_groups": [],
            },
        ]

        mean_report = _mean_reports(reports)

        self.assertAlmostEqual(mean_report["overall"]["mse"], 2.0)
        self.assertEqual(mean_report["groups"]["all"]["count"], 15)
        self.assertAlmostEqual(mean_report["groups"]["all"]["mse"], 3.0)

    def test_event_family_uses_prefix(self) -> None:
        self.assertEqual(_event_family("transition_open::late"), "transition_open")
        self.assertEqual(_event_family("unknown"), "unknown")

    def test_threshold_key_is_metric_name_safe(self) -> None:
        self.assertEqual(_threshold_key(0.05), "0p05")

    def test_predicted_boundary_actions_from_output_reports_variants(self) -> None:
        pred_actions = torch.zeros((1, 2, 7))
        output = {
            "gripper_step_logits": torch.tensor([[[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]]]),
            "gripper_step_residuals": torch.tensor(
                [[[100.0, 1.0, 2.0], [100.0, 3.0, 4.0]]]
            ),
        }

        actions = _predicted_boundary_actions_from_output(
            pred_actions,
            output,
            include=True,
            thresholds=(0.5,),
        )

        self.assertEqual(set(actions), {"pred_boundary_argmax", "pred_boundary_t0p50"})
        self.assertTrue(torch.equal(actions["pred_boundary_argmax"][..., -1], torch.tensor([[0.0, 3.0]])))

    def test_boundary_index_actions_from_output_reports_routed_actions(self) -> None:
        pred_actions = torch.zeros((1, 3, 7))
        output = {
            "gripper_boundary_index_logits": torch.tensor(
                [[[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]]]
            ),
            "gripper_step_residuals": torch.tensor(
                [[[100.0, 1.0, 2.0], [100.0, 3.0, 4.0], [100.0, 5.0, 6.0]]]
            ),
        }

        actions = _boundary_index_actions_from_output(pred_actions, output, include=True)

        self.assertIsNotNone(actions)
        assert actions is not None
        self.assertTrue(torch.equal(actions[..., -1], torch.tensor([[0.0, 3.0, 6.0]])))


if __name__ == "__main__":
    unittest.main()
