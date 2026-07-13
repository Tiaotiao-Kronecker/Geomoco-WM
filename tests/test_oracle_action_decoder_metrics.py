from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_oracle_action_decoder import _action_metrics  # noqa: E402


class OracleActionDecoderMetricTests(unittest.TestCase):
    def test_action_metrics_split_7d_action(self) -> None:
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]])
        target = torch.zeros_like(pred)

        metrics = _action_metrics(pred, target)

        self.assertAlmostEqual(metrics["mse"], 20.0, delta=1e-5)
        self.assertAlmostEqual(metrics["mae"], 4.0, delta=1e-5)
        self.assertAlmostEqual(metrics["translation_mse"], 14.0 / 3.0, delta=1e-5)
        self.assertAlmostEqual(metrics["translation_mae"], 2.0, delta=1e-5)
        self.assertAlmostEqual(metrics["rotation_mse"], 77.0 / 3.0, delta=1e-5)
        self.assertAlmostEqual(metrics["rotation_mae"], 5.0, delta=1e-5)
        self.assertAlmostEqual(metrics["se3_mse"], 91.0 / 6.0, delta=1e-5)
        self.assertAlmostEqual(metrics["se3_mae"], 3.5, delta=1e-5)
        self.assertAlmostEqual(metrics["gripper_mse"], 49.0, delta=1e-5)
        self.assertAlmostEqual(metrics["gripper_mae"], 7.0, delta=1e-5)
        self.assertIn("translation_l2", metrics)
        self.assertIn("rotation_l2", metrics)
        self.assertIn("se3_l2", metrics)
        self.assertIn("translation_m_l2", metrics)
        self.assertIn("rotation_geodesic_rad", metrics)

    def test_action_metrics_uses_libero_translation_scale(self) -> None:
        pred = torch.zeros((1, 1, 7))
        target = torch.zeros_like(pred)
        pred[..., 0] = 1.0

        metrics = _action_metrics(pred, target)

        self.assertAlmostEqual(metrics["translation_m_l2"], 0.05, delta=1e-6)
        self.assertAlmostEqual(metrics["translation_m_mae"], 0.05 / 3.0, delta=1e-6)
        self.assertAlmostEqual(metrics["translation_m_mse"], (0.05**2) / 3.0, delta=1e-6)

    def test_action_metrics_uses_libero_rotation_geodesic(self) -> None:
        pred = torch.zeros((1, 1, 7))
        target = torch.zeros_like(pred)
        pred[..., 5] = 1.0

        metrics = _action_metrics(pred, target)

        self.assertAlmostEqual(metrics["rotation_geodesic_rad"], 0.5, delta=1e-5)
        self.assertAlmostEqual(metrics["rotation_geodesic_deg"], math.degrees(0.5), delta=1e-4)


if __name__ == "__main__":
    unittest.main()
