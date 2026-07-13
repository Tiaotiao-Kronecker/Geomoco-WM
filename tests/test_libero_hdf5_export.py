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

from geomoco_wm.data.libero_hdf5_export import (  # noqa: E402
    export_libero_hdf5_suite_collection,
    export_libero_hdf5_windows,
    read_episode_jsonl,
    read_window_jsonl,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset  # noqa: E402


def make_fake_libero_hdf5(path: Path, *, length: int = 8) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset(
            "actions",
            data=np.array([[float(step + dim) for dim in range(7)] for step in range(length)]),
        )
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset(
            "ee_states",
            data=np.array([[float(step + dim) for dim in range(6)] for step in range(length)]),
        )
        obs.create_dataset(
            "gripper_states",
            data=np.array([[float(step), float(-step)] for step in range(length)]),
        )
        obs.create_dataset(
            "joint_states",
            data=np.array([[float(step)] * 7 for step in range(length)]),
        )


class LiberoHdf5ExportTests(unittest.TestCase):
    def test_exports_episode_and_window_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=8)
            output_dir = Path(tmp_dir) / "export"

            summary = export_libero_hdf5_windows(
                root,
                output_dir,
                suite_name="libero_goal",
                context_len=2,
                horizon=3,
                stride=2,
            )
            episodes = read_episode_jsonl(output_dir / "episodes.jsonl")
            windows = read_window_jsonl(output_dir / "windows.jsonl")

        self.assertEqual(summary.num_episodes, 1)
        self.assertEqual(summary.num_windows, 2)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(windows), 2)
        first = windows[0]
        self.assertEqual(first.context_frame_indices, [0, 1])
        self.assertEqual(first.future_frame_indices, [2, 3, 4])
        self.assertEqual(first.anchor_index, 1)
        self.assertEqual(first.action_start, 1)
        self.assertEqual(first.action_end, 4)
        self.assertEqual(first.anchor_ee_state, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(first.future_delta_ee_states[0], [1.0] * 6)
        self.assertEqual(first.action_chunk[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(first.current_gripper_state, [1.0, -1.0])
        self.assertEqual(first.current_joint_state, [1.0] * 7)

    def test_max_windows_stops_export_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=8)

            summary = export_libero_hdf5_windows(
                root,
                Path(tmp_dir) / "export",
                suite_name="libero_goal",
                context_len=2,
                horizon=3,
                stride=1,
                max_windows=1,
            )

        self.assertEqual(summary.num_windows, 1)
        self.assertIn("max_windows", " ".join(summary.warnings))

    def test_short_episode_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=4)
            output_dir = Path(tmp_dir) / "export"

            summary = export_libero_hdf5_windows(
                root,
                output_dir,
                suite_name="libero_goal",
                context_len=2,
                horizon=4,
                stride=1,
            )
            windows = read_window_jsonl(output_dir / "windows.jsonl")
            summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary.num_windows, 0)
        self.assertEqual(summary.dropped_short_episodes, 1)
        self.assertEqual(len(windows), 0)
        self.assertEqual(summary_payload["dropped_short_episodes"], 1)

    def test_exports_suite_collection_with_combined_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_root = Path(tmp_dir) / "official"
            goal_root = input_root / "libero_goal"
            object_root = input_root / "libero_object"
            goal_root.mkdir(parents=True)
            object_root.mkdir(parents=True)
            make_fake_libero_hdf5(goal_root / "goal_task_demo.hdf5", length=8)
            make_fake_libero_hdf5(object_root / "object_task_demo.hdf5", length=8)
            output_dir = Path(tmp_dir) / "export"

            summary = export_libero_hdf5_suite_collection(
                input_root,
                output_dir,
                suite_names=("libero_goal", "libero_object"),
                context_len=2,
                horizon=3,
                stride=2,
            )
            combined_episodes = read_episode_jsonl(output_dir / "episodes.jsonl")
            combined_windows = read_window_jsonl(output_dir / "windows.jsonl")

        self.assertEqual(summary.num_suites, 2)
        self.assertEqual(summary.num_episodes, 2)
        self.assertEqual(summary.num_windows, 4)
        self.assertEqual(len(combined_episodes), 2)
        self.assertEqual(len(combined_windows), 4)
        self.assertEqual({window.suite_name for window in combined_windows}, {"libero_goal", "libero_object"})
        self.assertIn("libero_goal/goal_task", summary.tasks)
        self.assertIn("libero_object/object_task", summary.tasks)

    def test_oracle_dataset_reads_multiple_window_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_root = Path(tmp_dir) / "official"
            goal_root = input_root / "libero_goal"
            object_root = input_root / "libero_object"
            goal_root.mkdir(parents=True)
            object_root.mkdir(parents=True)
            make_fake_libero_hdf5(goal_root / "goal_task_demo.hdf5", length=8)
            make_fake_libero_hdf5(object_root / "object_task_demo.hdf5", length=8)
            output_dir = Path(tmp_dir) / "export"
            export_libero_hdf5_suite_collection(
                input_root,
                output_dir,
                suite_names=("libero_goal", "libero_object"),
                context_len=2,
                horizon=3,
                stride=2,
            )

            dataset = OracleActionWindowDataset(
                [
                    output_dir / "libero_goal" / "windows.jsonl",
                    output_dir / "libero_object" / "windows.jsonl",
                ],
                motion_mode="future_delta",
            )
            spec = dataset.spec()
            first_item = dataset[0]

        self.assertEqual(len(dataset), 4)
        self.assertEqual(spec.suite_counts, {"libero_goal": 2, "libero_object": 2})
        self.assertEqual(spec.motion_dim, 18)
        self.assertEqual(first_item["suite_name"], "libero_goal")
        self.assertEqual(tuple(first_item["actions"].shape), (3, 7))

    def test_oracle_dataset_future_gripper_motion_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=8)
            output_dir = Path(tmp_dir) / "export"
            export_libero_hdf5_windows(
                root,
                output_dir,
                suite_name="libero_goal",
                context_len=2,
                horizon=3,
                stride=2,
            )

            dataset = OracleActionWindowDataset(
                output_dir / "windows.jsonl",
                motion_mode="future_gripper",
            )
            first_item = dataset[0]

        self.assertEqual(dataset.spec().motion_dim, 3)
        self.assertEqual(first_item["motion"].tolist(), [7.0, 8.0, 9.0])

    def test_oracle_dataset_future_delta_gripper_motion_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=8)
            output_dir = Path(tmp_dir) / "export"
            export_libero_hdf5_windows(
                root,
                output_dir,
                suite_name="libero_goal",
                context_len=2,
                horizon=3,
                stride=2,
            )

            dataset = OracleActionWindowDataset(
                output_dir / "windows.jsonl",
                motion_mode="future_delta_gripper",
            )
            first_motion = dataset[0]["motion"].tolist()

        self.assertEqual(dataset.spec().motion_dim, 21)
        self.assertEqual(first_motion[:6], [1.0] * 6)
        self.assertEqual(first_motion[-3:], [7.0, 8.0, 9.0])


if __name__ == "__main__":
    unittest.main()
