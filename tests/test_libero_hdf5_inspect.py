from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from geomoco_wm.data.libero_hdf5_inspect import (  # noqa: E402
    discover_libero_hdf5_files,
    inspect_libero_hdf5_suite,
    render_libero_hdf5_inspection_markdown,
    write_libero_hdf5_inspection_report,
)


def make_fake_libero_hdf5(path: Path, *, include_wrist: bool = True) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
        demo.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.bool_))
        demo.create_dataset("rewards", data=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        demo.create_dataset("robot_states", data=np.zeros((3, 9), dtype=np.float32))
        demo.create_dataset("states", data=np.zeros((3, 16), dtype=np.float32))

        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        if include_wrist:
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("ee_pos", data=np.zeros((3, 3), dtype=np.float32))
        obs.create_dataset("ee_ori", data=np.zeros((3, 3), dtype=np.float32))
        obs.create_dataset("ee_states", data=np.zeros((3, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((3, 2), dtype=np.float32))
        obs.create_dataset("joint_states", data=np.zeros((3, 7), dtype=np.float32))


class LiberoHdf5InspectionTests(unittest.TestCase):
    def test_discovers_hdf5_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            make_fake_libero_hdf5(root / "task_a_demo.hdf5")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")

            files = discover_libero_hdf5_files(root)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "task_a_demo.hdf5")

    def test_inspects_complete_fake_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5")

            report = inspect_libero_hdf5_suite(root, suite_name="libero_goal")

        self.assertEqual(report["summary"]["num_files"], 1)
        self.assertEqual(report["summary"]["num_demos"], 1)
        self.assertEqual(report["summary"]["num_frames"], 3)
        self.assertTrue(report["readiness"]["supports_gate0_dataset_export"])
        self.assertTrue(report["readiness"]["supports_eef_motion_targets"])
        self.assertFalse(report["readiness"]["supports_object_state_teacher"])
        self.assertIn("7", report["field_shapes"]["action_dim_counts"])

    def test_missing_wrist_camera_blocks_dual_camera_but_not_agentview_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", include_wrist=False)

            report = inspect_libero_hdf5_suite(root, suite_name="libero_goal")

        self.assertFalse(report["readiness"]["supports_gate0_dataset_export"])
        self.assertTrue(report["readiness"]["supports_visual_grounding_export"])
        self.assertFalse(report["readiness"]["supports_dual_camera_export"])
        self.assertEqual(report["coverage"]["all_default_cameras_ratio"], 0.0)

    def test_writes_json_and_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5")
            report = inspect_libero_hdf5_suite(root, suite_name="libero_goal")
            output_json = Path(tmp_dir) / "report.json"
            output_md = Path(tmp_dir) / "report.md"

            write_libero_hdf5_inspection_report(report, output_json, output_md)
            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(loaded["audit_type"], "libero_hdf5_gate0_inspection")
        self.assertIn("Gate 0 LIBERO HDF5 Inspection", markdown)
        markdown = render_libero_hdf5_inspection_markdown(report)
        self.assertIn("supports_gate0_dataset_export", markdown)


if __name__ == "__main__":
    unittest.main()
