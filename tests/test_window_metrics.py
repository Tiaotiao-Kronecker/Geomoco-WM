from __future__ import annotations

import unittest

import torch

from geomoco_wm.metrics.window_metrics import (
    merge_window_metric_records,
    per_window_action_metrics,
    window_metadata_records,
)


class WindowMetricsTests(unittest.TestCase):
    def test_per_window_action_metrics(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]],
                [[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]],
            ]
        )
        target = torch.zeros_like(pred)

        rows = per_window_action_metrics(pred, target)

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["mse"], 5.0 / 7.0, delta=1e-6)
        self.assertAlmostEqual(rows[0]["translation_mse"], 1.0 / 3.0, delta=1e-6)
        self.assertAlmostEqual(rows[0]["gripper_mse"], 4.0, delta=1e-6)
        self.assertAlmostEqual(rows[1]["rotation_mse"], 3.0, delta=1e-6)
        self.assertAlmostEqual(rows[1]["se3_mse"], 9.0 / 6.0, delta=1e-6)

    def test_window_metadata_records_with_event_labels(self) -> None:
        batch = {
            "window_id": ["w0", "w1"],
            "episode_id": ["e0", "e1"],
            "task_id": ["task_a", "task_b"],
            "suite_name": ["suite", "suite"],
        }
        labels = {
            "w0": {
                "event_type": "transition_close",
                "event_mode": "transition_close::middle",
                "timing_bin": "middle",
                "event_step": 3,
                "close_step": 3,
                "open_step": None,
            }
        }

        rows = window_metadata_records(batch, labels)

        self.assertEqual(rows[0]["event_type"], "transition_close")
        self.assertEqual(rows[0]["close_step"], 3)
        self.assertNotIn("event_type", rows[1])

    def test_merge_window_metric_records(self) -> None:
        merged = merge_window_metric_records(
            [{"window_id": "w0"}],
            [{"mse": 0.5}],
            prefix="temporal_action",
        )

        self.assertEqual(merged, [{"window_id": "w0", "temporal_action_mse": 0.5}])


if __name__ == "__main__":
    unittest.main()
