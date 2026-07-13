from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.shuffle_visual_feature_cache import shuffle_visual_feature_cache
from geomoco_wm.data.libero_hdf5_export import export_libero_hdf5_windows, read_window_jsonl
from geomoco_wm.data.visual_feature_cache import (
    VisualFeatureCache,
    VisualFeatureCacheMetadata,
    write_visual_feature_cache,
)
from geomoco_wm.data.window_dataset import OracleActionWindowDataset


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


class VisualFeatureCacheTests(unittest.TestCase):
    def test_cache_reader_and_oracle_dataset_visual_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "libero_goal"
            root.mkdir()
            make_fake_libero_hdf5(root / "task_a_demo.hdf5", length=8)
            export_dir = Path(tmp_dir) / "export"
            export_libero_hdf5_windows(
                root,
                export_dir,
                suite_name="libero_goal",
                context_len=2,
                horizon=3,
                stride=2,
            )
            windows = read_window_jsonl(export_dir / "windows.jsonl")
            features = np.arange(len(windows) * 5, dtype=np.float32).reshape(len(windows), 5)
            cache_path = Path(tmp_dir) / "visual_features.h5"
            write_visual_feature_cache(
                cache_path,
                window_ids=[window.window_id for window in windows],
                features=features,
                metadata=VisualFeatureCacheMetadata(
                    schema_version="geomoco_wm_visual_feature_cache_v0",
                    source_windows_jsonl=str(export_dir / "windows.jsonl"),
                    model_name="synthetic",
                    feature_mode="test",
                    camera_keys=["agentview_rgb", "eye_in_hand_rgb"],
                    image_size=8,
                    num_windows=len(windows),
                    feature_dim=5,
                    part_count=1,
                ),
            )

            cache = VisualFeatureCache(cache_path)
            dataset = OracleActionWindowDataset(
                export_dir / "windows.jsonl",
                motion_mode="future_delta",
                visual_feature_cache_path=cache_path,
            )
            item = dataset[0]
            spec = dataset.spec()

        self.assertEqual(cache.feature_dim, 5)
        self.assertEqual(spec.visual_dim, 5)
        self.assertEqual(tuple(item["visual"].shape), (5,))
        np.testing.assert_allclose(item["visual"].numpy(), features[0])

    def test_shuffle_visual_feature_cache_preserves_ids_and_deranges_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            window_ids = [f"window_{index}" for index in range(5)]
            features = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
            input_path = Path(tmp_dir) / "visual_features.h5"
            output_path = Path(tmp_dir) / "visual_features_shuffled.h5"
            write_visual_feature_cache(
                input_path,
                window_ids=window_ids,
                features=features,
                metadata=VisualFeatureCacheMetadata(
                    schema_version="geomoco_wm_visual_feature_cache_v0",
                    source_windows_jsonl="windows.jsonl",
                    model_name="synthetic",
                    feature_mode="patch_pool_4x4_context_camera_concat",
                    camera_keys=["agentview_rgb", "eye_in_hand_rgb"],
                    image_size=8,
                    num_windows=len(window_ids),
                    feature_dim=4,
                    part_count=4,
                    visual_token_count=4,
                    visual_token_dim=1,
                ),
            )

            summary = shuffle_visual_feature_cache(
                input_path=input_path,
                output_path=output_path,
                seed=11,
            )
            shuffled = VisualFeatureCache(output_path)

        self.assertEqual(shuffled.window_ids, window_ids)
        self.assertEqual(summary["fixed_points"], 0)
        self.assertEqual(shuffled.metadata["visual_token_count"], 4)
        self.assertEqual(shuffled.metadata["visual_token_dim"], 1)
        for row, original_feature in enumerate(features):
            self.assertFalse(np.array_equal(shuffled.features[row], original_feature))
        self.assertEqual(
            sorted(tuple(row) for row in shuffled.features),
            sorted(tuple(row) for row in features),
        )


if __name__ == "__main__":
    unittest.main()
