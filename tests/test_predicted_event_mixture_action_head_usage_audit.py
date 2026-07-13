from __future__ import annotations

import math
import unittest

import torch

from scripts.audit_predicted_event_mixture_action_head_usage import (
    _build_usage_variants,
    _event_entropy,
    _tertile_labels,
    _transition_rank_repeated_variant,
)


class PredictedEventMixtureActionHeadUsageAuditTests(unittest.TestCase):
    def test_build_usage_variants_collapses_and_masks_expected_slots(self) -> None:
        torch.manual_seed(3)
        future_inputs = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        sample_features = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)
        rank_slots = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        top_indices = torch.tensor([[0, 1, 2, 3], [0, 3, 0, 3]], dtype=torch.long)
        event_classes = (
            "sustain_open::none",
            "transition_close::early",
            "transition_open::late",
            "sustain_closed::none",
        )

        variants = _build_usage_variants(
            future_inputs,
            sample_features,
            rank_slots,
            top_indices,
            event_classes,
            subset_samples=3,
        )

        self.assertEqual(
            set(variants),
            {
                "original",
                "permuted",
                "mean_repeated",
                "subset_k4",
                "rank1_only",
                "rank1_repeated",
                "drop_rank1",
                "transition_rank_repeated",
                "batch_mismatch",
            },
        )
        mean = future_inputs.mean(dim=1, keepdim=True).expand_as(future_inputs)
        self.assertTrue(torch.equal(variants["mean_repeated"].future_inputs, mean))
        self.assertTrue(torch.equal(variants["mean_repeated"].sample_features, sample_features))
        self.assertTrue(torch.equal(variants["rank1_only"].future_inputs, future_inputs[:, :2, :]))
        self.assertTrue(torch.equal(variants["rank1_only"].sample_features, sample_features[:, :2, :]))
        self.assertEqual(tuple(variants["subset_k4"].future_inputs.shape), (2, 3, 3))
        self.assertTrue(torch.equal(variants["drop_rank1"].future_inputs, future_inputs[:, 2:, :]))
        self.assertTrue(torch.equal(variants["batch_mismatch"].future_inputs[0], future_inputs[1]))
        self.assertTrue(torch.equal(variants["batch_mismatch"].sample_features[1], sample_features[0]))

        rank1_mean = future_inputs[:, :2, :].mean(dim=1, keepdim=True)
        self.assertTrue(
            torch.equal(
                variants["rank1_repeated"].future_inputs,
                rank1_mean.expand(-1, 8, -1),
            )
        )

    def test_transition_rank_repeated_uses_transition_mean_then_fallback(self) -> None:
        future_inputs = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        sample_features = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)
        rank_slots = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        top_indices = torch.tensor([[0, 1, 2, 3], [0, 3, 0, 3]], dtype=torch.long)
        event_classes = (
            "sustain_open::none",
            "transition_close::early",
            "transition_open::late",
            "sustain_closed::none",
        )

        variant = _transition_rank_repeated_variant(
            future_inputs,
            sample_features,
            rank_slots,
            top_indices,
            event_classes,
            num_samples=8,
        )

        row0_transition_mean = future_inputs[0, 2:6, :].mean(dim=0)
        row1_fallback_mean = future_inputs[1].mean(dim=0)
        self.assertTrue(torch.equal(variant.future_inputs[0, 0], row0_transition_mean))
        self.assertTrue(torch.equal(variant.future_inputs[0, -1], row0_transition_mean))
        self.assertTrue(torch.equal(variant.future_inputs[1, 0], row1_fallback_mean))
        self.assertTrue(torch.equal(variant.future_inputs[1, -1], row1_fallback_mean))
        assert variant.sample_features is not None
        self.assertTrue(
            torch.equal(variant.sample_features[0, 0], sample_features[0, 2:6, :].mean(dim=0))
        )
        self.assertTrue(
            torch.equal(variant.sample_features[1, 0], sample_features[1].mean(dim=0))
        )

    def test_tertile_labels_are_stable_for_small_ordered_inputs(self) -> None:
        self.assertEqual(
            _tertile_labels([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ["low", "low", "low", "mid", "mid", "high"],
        )
        self.assertEqual(_tertile_labels([]), [])

    def test_event_entropy_handles_zero_probability(self) -> None:
        entropy = _event_entropy(torch.tensor([[0.5, 0.5], [1.0, 0.0]]))

        self.assertTrue(torch.isfinite(entropy).all())
        self.assertAlmostEqual(float(entropy[0]), math.log(2.0), places=6)
        self.assertAlmostEqual(float(entropy[1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
