from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from geomoco_wm.data.event_labels import (
    GripperEventConfig,
    audit_gripper_events_from_windows,
    infer_close_sign_from_width_deltas,
    label_gripper_events_for_windows,
    label_gripper_events,
    label_gripper_transition_events,
    previous_gripper_commands_for_windows,
)
from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, write_window_jsonl
from scripts.train_phase_event_probe import EventProbeNet, _classification_metrics


def make_window(
    window_id: str,
    action_gripper_values: list[float],
    *,
    suite_name: str = "libero_goal",
    task_id: str = "open_the_middle_drawer",
) -> LiberoWindowRecord:
    horizon = len(action_gripper_values)
    return LiberoWindowRecord(
        schema_version="libero_hdf5_window_v0",
        window_id=window_id,
        episode_id=f"{suite_name}__{task_id}__demo_000",
        task_id=task_id,
        suite_name=suite_name,
        source_file="/missing/source.hdf5",
        demo_name="demo_0",
        context_start=0,
        context_end=2,
        anchor_index=1,
        future_start=2,
        future_end=2 + horizon,
        action_start=1,
        action_end=1 + horizon,
        context_frame_indices=[0, 1],
        future_frame_indices=list(range(2, 2 + horizon)),
        camera_keys=["agentview_rgb"],
        anchor_ee_state=[0.0] * 6,
        future_ee_states=[[float(step)] * 6 for step in range(horizon)],
        future_delta_ee_states=[[0.01 * float(step + 1), 0.0, 0.0, 0.0, 0.0, 0.0] for step in range(horizon)],
        action_chunk=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value] for value in action_gripper_values],
        current_gripper_state=[0.04, -0.04],
        current_joint_state=[0.0] * 7,
    )


class EventLabelTests(unittest.TestCase):
    def test_label_gripper_events_detects_close_and_open_steps(self) -> None:
        label = label_gripper_events(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value] for value in [0.0, -1.0, -1.0, 1.0]],
            GripperEventConfig(command_threshold=0.5, close_sign=-1),
        )

        self.assertEqual(label.event_type, "mixed")
        self.assertTrue(label.has_close)
        self.assertTrue(label.has_open)
        self.assertEqual(label.close_step, 1)
        self.assertEqual(label.open_step, 3)

    def test_infer_close_sign_from_width_delta(self) -> None:
        report = infer_close_sign_from_width_deltas(
            commands=[-1.0, -1.0, 1.0, 1.0],
            width_deltas=[-0.01, -0.02, 0.02, 0.01],
            command_threshold=0.5,
        )

        self.assertEqual(report["inferred_close_sign"], -1)
        self.assertEqual(report["negative_count"], 2)
        self.assertEqual(report["positive_count"], 2)

    def test_label_gripper_transition_events_uses_previous_command(self) -> None:
        label = label_gripper_transition_events(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value] for value in [-1.0, 1.0, 1.0]],
            previous_gripper_command=-1.0,
            config=GripperEventConfig(command_threshold=0.5, close_sign=1),
        )

        self.assertEqual(label.event_type, "close_transition")
        self.assertEqual(label.close_step, 1)

    def test_label_gripper_transition_events_marks_sustained_state(self) -> None:
        label = label_gripper_transition_events(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value] for value in [1.0, 1.0, 1.0]],
            previous_gripper_command=1.0,
            config=GripperEventConfig(command_threshold=0.5, close_sign=1),
        )

        self.assertEqual(label.event_type, "sustain_close")
        self.assertIsNone(label.close_step)

    def test_audit_gripper_events_from_window_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            windows_path = Path(tmp_dir) / "windows.jsonl"
            write_window_jsonl(
                [
                    make_window("w0", [-1.0, -1.0, -1.0]),
                    make_window("w1", [1.0, 1.0, 1.0]),
                    make_window("w2", [0.0, 0.0, 0.0]),
                ],
                windows_path,
            )

            report = audit_gripper_events_from_windows(
                windows_path,
                command_threshold=0.5,
                close_sign=-1,
                infer_close_sign=False,
                label_mode="command",
            )

        self.assertEqual(report["num_windows"], 3)
        self.assertEqual(report["event_type_counts"], {"close": 1, "hold": 1, "open": 1})
        self.assertEqual(report["close_step_counts"], {"0": 1})
        self.assertEqual(report["command_step_counts"], {"close": 3, "hold": 3, "open": 3})

    def test_label_gripper_events_for_windows_returns_window_mapping(self) -> None:
        windows = [
            make_window("w0", [1.0, 1.0, 1.0]),
            make_window("w1", [-1.0, -1.0, -1.0]),
        ]

        labels = label_gripper_events_for_windows(
            windows,
            config=GripperEventConfig(command_threshold=0.5, close_sign=1),
            label_mode="command",
        )

        self.assertEqual(labels["w0"].event_type, "close")
        self.assertEqual(labels["w1"].event_type, "open")

    def test_previous_gripper_commands_for_missing_sources_returns_none(self) -> None:
        windows = [make_window("w0", [1.0, 1.0, 1.0])]

        previous = previous_gripper_commands_for_windows(windows)

        self.assertEqual(previous, {"w0": None})

    def test_event_probe_net_shape(self) -> None:
        model = EventProbeNet(
            input_dim=6,
            num_classes=5,
            hidden_dims=(4,),
            dropout=0.0,
            layer_norm=True,
        )

        logits = model(torch.zeros((3, 6)))

        self.assertEqual(tuple(logits.shape), (3, 5))

    def test_classification_metrics_reports_macro_f1(self) -> None:
        confusion = torch.tensor(
            [
                [2, 0],
                [1, 1],
            ],
            dtype=torch.long,
        )

        metrics = _classification_metrics(confusion)

        self.assertAlmostEqual(float(metrics["accuracy"]), 0.75, delta=1e-6)
        self.assertIn("macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
