from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from geomoco_wm.data.event_labels import GripperEventConfig, label_gripper_transition_events
from geomoco_wm.data.event_conditioning import (
    batch_event_mode_conditioning,
    combine_conditioning,
    load_event_mode_conditioner,
)
from geomoco_wm.data.event_modes import (
    audit_event_modes_from_windows,
    build_event_mode_label,
    normalize_event_type,
    timing_bin_for_step,
)
from geomoco_wm.data.predicted_event_mixture import (
    event_label_is_transition,
    event_timing_bin,
    map_event_probabilities,
    rank_uniform_counts,
)
from geomoco_wm.data.libero_hdf5_export import LiberoWindowRecord, write_window_jsonl
from scripts.train_event_mode_probe import EventModeProbeNet, _classification_metrics


def make_window(
    window_id: str,
    action_gripper_values: list[float],
    *,
    suite_name: str = "libero_goal",
    task_id: str = "open_the_middle_drawer",
    demo_index: int = 0,
) -> LiberoWindowRecord:
    horizon = len(action_gripper_values)
    return LiberoWindowRecord(
        schema_version="libero_hdf5_window_v0",
        window_id=window_id,
        episode_id=f"{suite_name}__{task_id}__demo_{demo_index:03d}",
        task_id=task_id,
        suite_name=suite_name,
        source_file="/missing/source.hdf5",
        demo_name=f"demo_{demo_index}",
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
        future_delta_ee_states=[
            [0.01 * float(step + 1), 0.0, 0.0, 0.0, 0.0, 0.0]
            for step in range(horizon)
        ],
        action_chunk=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value]
            for value in action_gripper_values
        ],
        current_gripper_state=[0.04, -0.04],
        current_joint_state=[0.0] * 7,
    )


class EventModeTests(unittest.TestCase):
    def test_timing_bin_for_horizon_eight(self) -> None:
        self.assertEqual(timing_bin_for_step(None, horizon=8), "none")
        self.assertEqual(timing_bin_for_step(0, horizon=8), "early")
        self.assertEqual(timing_bin_for_step(2, horizon=8), "early")
        self.assertEqual(timing_bin_for_step(3, horizon=8), "middle")
        self.assertEqual(timing_bin_for_step(5, horizon=8), "middle")
        self.assertEqual(timing_bin_for_step(6, horizon=8), "late")
        self.assertEqual(timing_bin_for_step(7, horizon=8), "late")

    def test_normalizes_historical_transition_names(self) -> None:
        self.assertEqual(normalize_event_type("close_transition"), "transition_close")
        self.assertEqual(normalize_event_type("open_transition"), "transition_open")
        self.assertEqual(normalize_event_type("mixed_transition"), "mixed_transition")

    def test_build_event_mode_label_combines_type_and_timing(self) -> None:
        window = make_window("w0", [0.0, 0.0, 1.0, 1.0])
        gripper_label = label_gripper_transition_events(
            window.action_chunk,
            previous_gripper_command=0.0,
            config=GripperEventConfig(command_threshold=0.5, close_sign=1),
        )

        mode_label = build_event_mode_label(window, gripper_label)

        self.assertEqual(mode_label.event_type, "transition_close")
        self.assertEqual(mode_label.timing_bin, "middle")
        self.assertEqual(mode_label.event_mode, "transition_close::middle")

    def test_audit_event_modes_reports_split_balance(self) -> None:
        windows = [
            make_window("w0", [0.0, 0.0, 1.0, 1.0], demo_index=0),
            make_window("w1", [0.0, 0.0, -1.0, -1.0], demo_index=1),
            make_window("w2", [0.0, 0.0, 0.0, 0.0], demo_index=2),
            make_window("w3", [0.0, 0.0, 1.0, -1.0], demo_index=3),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            windows_path = Path(tmp_dir) / "windows.jsonl"
            write_window_jsonl(windows, windows_path)
            report = audit_event_modes_from_windows(
                windows_path,
                command_threshold=0.5,
                close_sign=1,
                infer_close_sign=False,
                train_ratio=0.5,
                split_by="episode",
                min_class_count=1,
            )
            output_json = Path(tmp_dir) / "event_modes.json"
            output_json.write_text(json.dumps(report), encoding="utf-8")
            conditioner = load_event_mode_conditioner(
                output_json,
                mode="oracle",
                class_set="all_observed",
            )
            batch = {"window_id": ["w0", "w1"]}
            event_conditioning = batch_event_mode_conditioning(
                batch,
                conditioner,
                torch.device("cpu"),
            )

        self.assertEqual(report["num_windows"], 4)
        self.assertIn("transition_close::middle", report["event_mode_counts"])
        self.assertIn("transition_open::middle", report["event_mode_counts"])
        self.assertEqual(report["train_size"] + report["val_size"], 4)
        self.assertEqual(len(report["window_labels"]), 4)
        self.assertIsNotNone(event_conditioning)
        assert event_conditioning is not None
        self.assertEqual(tuple(event_conditioning.shape), (2, conditioner.dim))

    def test_combine_conditioning_handles_optional_parts(self) -> None:
        base = torch.zeros((2, 3))
        event = torch.ones((2, 2))

        combined = combine_conditioning(base, event)

        self.assertEqual(tuple(combined.shape), (2, 5))
        self.assertIs(combine_conditioning(base, None), base)

    def test_event_mode_probe_net_shape(self) -> None:
        model = EventModeProbeNet(
            input_dim=6,
            num_classes=8,
            hidden_dims=(4,),
            dropout=0.0,
            layer_norm=True,
        )

        logits = model(torch.zeros((3, 6)))

        self.assertEqual(tuple(logits.shape), (3, 8))

    def test_event_mode_metrics_include_transition_diagnostics(self) -> None:
        class_names = (
            "sustain_open::none",
            "sustain_close::none",
            "transition_close::early",
            "transition_close::middle",
        )
        confusion = torch.tensor(
            [
                [2, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 2, 0],
                [0, 0, 1, 1],
            ],
            dtype=torch.long,
        )

        metrics = _classification_metrics(confusion, class_names)

        self.assertAlmostEqual(float(metrics["accuracy"]), 0.75, delta=1e-6)
        self.assertIn("transition_binary_f1", metrics)
        self.assertIn("transition_timing_accuracy", metrics)

    def test_maps_stable_event_probs_to_all_observed_event_classes(self) -> None:
        source_classes = (
            "sustain_open::none",
            "sustain_close::none",
            "transition_close::early",
        )
        target_classes = (
            "mixed_transition::early",
            "sustain_close::none",
            "sustain_open::none",
            "transition_close::early",
        )
        source_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)

        mapped = map_event_probabilities(source_probs, source_classes, target_classes)

        self.assertEqual(tuple(mapped.shape), (1, 4))
        self.assertAlmostEqual(float(mapped[0, 0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(mapped[0, 1]), 0.3, delta=1e-6)
        self.assertAlmostEqual(float(mapped[0, 2]), 0.2, delta=1e-6)
        self.assertAlmostEqual(float(mapped[0, 3]), 0.5, delta=1e-6)
        self.assertAlmostEqual(float(mapped.sum()), 1.0, delta=1e-6)

    def test_rank_uniform_sample_counts_preserve_budget(self) -> None:
        self.assertEqual(rank_uniform_counts(16, 1), (16,))
        self.assertEqual(rank_uniform_counts(16, 2), (8, 8))
        self.assertEqual(rank_uniform_counts(16, 3), (6, 5, 5))
        self.assertEqual(sum(rank_uniform_counts(17, 4)), 17)

    def test_event_label_helpers_parse_transition_and_timing(self) -> None:
        self.assertTrue(event_label_is_transition("transition_close::middle"))
        self.assertTrue(event_label_is_transition("mixed_transition::early"))
        self.assertFalse(event_label_is_transition("sustain_open::none"))
        self.assertEqual(event_timing_bin("transition_open::late"), "late")


if __name__ == "__main__":
    unittest.main()
