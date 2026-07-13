from __future__ import annotations

import unittest

import torch

from scripts.audit_predicted_event_mixture_action_head_usage import FutureInputBundle
from scripts.train_on_generator_geomoco_critic import (
    OnGeneratorCandidateCritic,
    _apply_candidate_control,
    _row_ranking_metrics,
)


class OnGeneratorGeoMoCoCriticTests(unittest.TestCase):
    def test_critic_scores_each_candidate(self) -> None:
        critic = OnGeneratorCandidateCritic(
            context_dim=4,
            motion_dim=5,
            conditioning_dim=2,
            sample_feature_dim=3,
            hidden_dims=(8,),
            dropout=0.0,
        )
        context = torch.zeros((2, 4))
        future_inputs = torch.zeros((2, 6, 5))
        conditioning = torch.zeros((2, 2))
        sample_features = torch.zeros((2, 6, 3))

        scores = critic(context, future_inputs, conditioning, sample_features)

        self.assertEqual(tuple(scores.shape), (2, 6))

    def test_candidate_controls_modify_expected_fields(self) -> None:
        future_inputs = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        sample_features = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
        bundle = FutureInputBundle(
            future_inputs=future_inputs,
            sample_features=sample_features,
            rank_slots=torch.tensor([0, 0, 1, 1]),
            top_indices=torch.zeros((2, 2), dtype=torch.long),
            top_probs=torch.full((2, 2), 0.5),
        )

        mean_repeated = _apply_candidate_control(
            bundle,
            "mean_repeated",
            sample_feature_dim=5,
            event_class_count=3,
        )
        expected_mean = future_inputs.mean(dim=1, keepdim=True).expand_as(future_inputs)
        self.assertTrue(torch.equal(mean_repeated.future_inputs, expected_mean))
        self.assertTrue(torch.equal(mean_repeated.sample_features, sample_features))

        batch_mismatch = _apply_candidate_control(
            bundle,
            "batch_mismatch",
            sample_feature_dim=5,
            event_class_count=3,
        )
        self.assertTrue(torch.equal(batch_mismatch.future_inputs[0], future_inputs[1]))
        assert batch_mismatch.sample_features is not None
        self.assertTrue(torch.equal(batch_mismatch.sample_features[1], sample_features[0]))

        rank_prob_only = _apply_candidate_control(
            bundle,
            "rank_prob_only",
            sample_feature_dim=5,
            event_class_count=3,
        )
        assert rank_prob_only.sample_features is not None
        self.assertTrue(torch.equal(rank_prob_only.sample_features[..., :3], torch.zeros((2, 4, 3))))
        self.assertTrue(torch.equal(rank_prob_only.sample_features[..., 3:], sample_features[..., 3:]))

        shuffled_event = _apply_candidate_control(
            bundle,
            "shuffled_event_identity",
            sample_feature_dim=5,
            event_class_count=3,
        )
        assert shuffled_event.sample_features is not None
        self.assertTrue(
            torch.equal(shuffled_event.sample_features[0, :, :3], sample_features[1, :, :3])
        )
        self.assertTrue(
            torch.equal(shuffled_event.sample_features[..., 3:], sample_features[..., 3:])
        )

        zero_features = _apply_candidate_control(
            bundle,
            "zero_sample_features",
            sample_feature_dim=5,
            event_class_count=3,
        )
        assert zero_features.sample_features is not None
        self.assertTrue(torch.equal(zero_features.sample_features, torch.zeros_like(sample_features)))

    def test_row_ranking_metrics_report_selection_and_oracle_gaps(self) -> None:
        scores = torch.tensor([[0.0, 5.0, 1.0], [4.0, 1.0, 0.0]])
        regrets = torch.tensor([[0.3, 0.1, 0.2], [0.2, 0.5, 0.1]])
        set_mse = torch.tensor([0.15, 0.3])

        metrics = _row_ranking_metrics(scores, regrets, set_mse)

        self.assertTrue(torch.allclose(metrics["critic_selected_mse"], torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.allclose(metrics["candidate_oracle_mse"], torch.tensor([0.1, 0.1])))
        self.assertTrue(torch.allclose(metrics["critic_top1_accuracy"], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(metrics["critic_gap_to_oracle"], torch.tensor([0.0, 0.1])))
        self.assertTrue(torch.allclose(metrics["critic_selected_gain_vs_set"], torch.tensor([0.05, 0.1])))


if __name__ == "__main__":
    unittest.main()
