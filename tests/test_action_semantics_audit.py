from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from geomoco_wm.data.action_semantics import (
    audit_libero_action_semantics_suite,
    combine_action_semantics_reports,
)


class ActionSemanticsAuditTests(unittest.TestCase):
    def test_audit_confirms_libero_osc_pose_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "task_demo.hdf5"
            _write_synthetic_libero_hdf5(hdf5_path)

            report = audit_libero_action_semantics_suite(hdf5_path, suite_name="libero_test")

            self.assertTrue(report["readiness"]["supports_geodesic_action_metrics"])
            self.assertTrue(report["readiness"]["all_actions_7d"])
            self.assertTrue(report["readiness"]["all_output_scale_matches"])
            self.assertEqual(report["summary"]["action_dim_counts"], {"7": 1})

    def test_combine_requires_all_suites_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "task_demo.hdf5"
            _write_synthetic_libero_hdf5(hdf5_path)

            report = audit_libero_action_semantics_suite(hdf5_path, suite_name="libero_test")
            combined = combine_action_semantics_reports([report])

            self.assertTrue(combined["readiness"]["supports_geodesic_action_metrics"])
            self.assertEqual(combined["summary"]["num_suites"], 1)


def _write_synthetic_libero_hdf5(path: Path) -> None:
    env_args = {
        "env_kwargs": {
            "controller_configs": {
                "type": "OSC_POSE",
                "control_delta": True,
                "input_min": -1,
                "input_max": 1,
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            }
        }
    }
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
