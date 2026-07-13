from __future__ import annotations

import unittest

import torch

from scripts.audit_motion_prior_action_head_usage import (
    _build_variants,
    _has_gripper_transition,
)


class MotionPriorActionHeadAuditTests(unittest.TestCase):
    def test_build_variants_shapes(self) -> None:
        samples = torch.zeros((3, 5, 14))

        variants = _build_variants(samples, subset_samples=2)

        self.assertEqual(tuple(variants["original"].shape), (3, 5, 14))
        self.assertEqual(tuple(variants["permuted"].shape), (3, 5, 14))
        self.assertEqual(tuple(variants["mean_repeated"].shape), (3, 5, 14))
        self.assertEqual(tuple(variants["mean_single"].shape), (3, 14))
        self.assertEqual(tuple(variants["first_single"].shape), (3, 14))
        self.assertEqual(tuple(variants["subset"].shape), (3, 2, 14))
        self.assertEqual(tuple(variants["batch_mismatch"].shape), (3, 5, 14))

    def test_has_gripper_transition(self) -> None:
        actions = torch.tensor(
            [
                [[0, 0, 0, 0, 0, 0, -1.0], [0, 0, 0, 0, 0, 0, -1.0]],
                [[0, 0, 0, 0, 0, 0, -1.0], [0, 0, 0, 0, 0, 0, 1.0]],
                [[0, 0, 0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0, 0, 0.2]],
            ],
            dtype=torch.float32,
        )

        flags = _has_gripper_transition(actions, threshold=0.5)

        self.assertEqual(flags.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
