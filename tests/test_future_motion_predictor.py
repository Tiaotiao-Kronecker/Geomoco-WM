from __future__ import annotations

import unittest

import torch

from scripts.train_future_motion_predictor import _combined_training_loss, _prediction_metrics
from scripts.train_visual_cvae_sample_scorer import (
    CandidateBatch,
    EventTargetData,
    _event_alignment_errors,
    _event_hard_negative_ranking_loss,
    _event_training_metrics,
    _hard_negative_ranking_loss,
    _listwise_ranking_loss,
    _rank_selected_scores,
    _structured_action_errors,
    _structured_oracle_scores,
    _target_scores,
)
from geomoco_wm.data.event_labels import GripperEventConfig, GripperEventLabel
from geomoco_wm.metrics.motion_metrics import future_motion_metrics
from geomoco_wm.models.future_motion_predictor import (
    FutureMotionPredictor,
    StepwiseVisualCrossAttentionFutureMotionPredictor,
    VisualCrossAttentionFutureMotionPredictor,
)
from geomoco_wm.models.geomoco_cvae import (
    VisualConditionedGeoMoCoCVAE,
    gaussian_kl_divergence,
)
from geomoco_wm.models.sample_readout import SampleScoreNet, TemporalSampleScoreNet


class FutureMotionPredictorTests(unittest.TestCase):
    def test_predictor_shape(self) -> None:
        model = FutureMotionPredictor(context_dim=15, motion_dim=48, hidden_dims=(16,))
        context = torch.zeros((3, 15))

        pred = model(context)

        self.assertEqual(tuple(pred.shape), (3, 48))

    def test_conditioned_predictor_shape(self) -> None:
        model = FutureMotionPredictor(
            context_dim=15,
            motion_dim=48,
            hidden_dims=(16,),
            conditioning_dim=4,
        )
        context = torch.zeros((3, 15))
        conditioning = torch.zeros((3, 4))

        pred = model(context, conditioning)

        self.assertEqual(tuple(pred.shape), (3, 48))

    def test_conditioned_predictor_requires_conditioning(self) -> None:
        model = FutureMotionPredictor(
            context_dim=15,
            motion_dim=48,
            hidden_dims=(16,),
            conditioning_dim=4,
        )
        context = torch.zeros((3, 15))

        with self.assertRaisesRegex(ValueError, "conditioning is required"):
            model(context)

    def test_visual_cross_attention_predictor_shape(self) -> None:
        model = VisualCrossAttentionFutureMotionPredictor(
            context_dim=15,
            motion_dim=48,
            visual_token_dim=8,
            visual_token_count=4,
            hidden_dims=(16,),
            conditioning_dim=3,
            query_dim=8,
            num_heads=2,
        )
        context = torch.zeros((3, 15))
        visual = torch.zeros((3, 32))
        conditioning = torch.zeros((3, 3))

        pred = model(context, visual, conditioning)

        self.assertEqual(tuple(pred.shape), (3, 48))

    def test_stepwise_visual_cross_attention_predictor_shape(self) -> None:
        model = StepwiseVisualCrossAttentionFutureMotionPredictor(
            context_dim=15,
            motion_dim=48,
            visual_token_dim=8,
            visual_token_count=4,
            hidden_dims=(16,),
            conditioning_dim=3,
            query_dim=8,
            num_heads=2,
            future_step_dim=6,
        )
        context = torch.zeros((3, 15))
        visual = torch.zeros((3, 32))
        conditioning = torch.zeros((3, 3))

        pred = model(context, visual, conditioning)

        self.assertEqual(tuple(pred.shape), (3, 48))

    def test_stepwise_visual_cross_attention_requires_even_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by future_step_dim"):
            StepwiseVisualCrossAttentionFutureMotionPredictor(
                context_dim=15,
                motion_dim=47,
                visual_token_dim=8,
                visual_token_count=4,
                hidden_dims=(16,),
                conditioning_dim=3,
                query_dim=8,
                num_heads=2,
                future_step_dim=6,
            )

    def test_visual_conditioned_cvae_shape(self) -> None:
        model = VisualConditionedGeoMoCoCVAE(
            context_dim=15,
            motion_dim=48,
            visual_token_dim=8,
            visual_token_count=4,
            conditioning_dim=3,
            latent_dim=5,
            hidden_dims=(16,),
            query_dim=8,
            num_heads=2,
        )
        context = torch.zeros((3, 15))
        visual = torch.zeros((3, 32))
        conditioning = torch.zeros((3, 3))
        motion = torch.zeros((3, 48))

        output = model(context, visual, motion, conditioning)

        self.assertEqual(tuple(output.posterior_reconstruction.shape), (3, 48))
        self.assertEqual(tuple(output.prior_mean_reconstruction.shape), (3, 48))
        self.assertEqual(tuple(output.posterior_mean.shape), (3, 5))
        self.assertEqual(tuple(output.prior_mean.shape), (3, 5))

    def test_sample_score_net_shape(self) -> None:
        model = SampleScoreNet(
            condition_dim=16,
            motion_dim=12,
            action_dim=7,
            horizon=2,
            hidden_dims=(8,),
            dropout=0.0,
        )
        condition = torch.zeros((5, 16))
        motion = torch.zeros((5, 12))
        action = torch.zeros((5, 2, 7))

        scores = model(condition, motion, action)

        self.assertEqual(tuple(scores.shape), (5,))

    def test_sample_score_net_validates_action_shape(self) -> None:
        model = SampleScoreNet(
            condition_dim=16,
            motion_dim=12,
            action_dim=7,
            horizon=2,
            hidden_dims=(8,),
            dropout=0.0,
        )
        condition = torch.zeros((5, 16))
        motion = torch.zeros((5, 12))
        action = torch.zeros((5, 3, 7))

        with self.assertRaisesRegex(ValueError, "action_chunk shape"):
            model(condition, motion, action)

    def test_temporal_sample_score_net_shape(self) -> None:
        model = TemporalSampleScoreNet(
            condition_dim=16,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(8,),
            temporal_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        condition = torch.zeros((5, 16))
        motion = torch.zeros((5, 14))
        action = torch.zeros((5, 2, 7))

        scores = model(condition, motion, action)

        self.assertEqual(tuple(scores.shape), (5,))

    def test_temporal_sample_score_net_requires_motion_dim_divisible_by_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion_dim must be divisible by horizon"):
            TemporalSampleScoreNet(
                condition_dim=16,
                motion_dim=13,
                action_dim=7,
                horizon=2,
                hidden_dims=(8,),
                temporal_dim=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )

    def test_listwise_ranking_loss_prefers_aligned_logits(self) -> None:
        target_scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        aligned_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        reversed_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])

        aligned_loss = _listwise_ranking_loss(aligned_logits, target_scores, 1.0)
        reversed_loss = _listwise_ranking_loss(reversed_logits, target_scores, 1.0)

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_structured_action_errors_zero_for_identical_actions(self) -> None:
        target_actions = torch.zeros((2, 3, 7))
        sample_actions = target_actions.unsqueeze(0).expand(4, -1, -1, -1).clone()

        errors = _structured_action_errors(sample_actions, target_actions)

        self.assertTrue(torch.allclose(errors.translation_m_l2, torch.zeros((4, 2))))
        self.assertTrue(torch.allclose(errors.rotation_geodesic_rad, torch.zeros((4, 2))))
        self.assertTrue(torch.allclose(errors.gripper_mse, torch.zeros((4, 2))))

    def test_structured_action_errors_use_translation_meter_scale(self) -> None:
        target_actions = torch.zeros((1, 2, 7))
        sample_actions = torch.zeros((2, 1, 2, 7))
        sample_actions[1, :, :, 0] = 1.0

        errors = _structured_action_errors(sample_actions, target_actions)

        self.assertAlmostEqual(float(errors.translation_m_l2[0, 0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(errors.translation_m_l2[1, 0]), 0.05, delta=1e-6)

    def test_se3_gripper_target_prefers_lower_gripper_error_when_se3_matches(self) -> None:
        action_errors = torch.zeros((2, 1))
        motion_errors = torch.zeros((2, 1))
        translation_errors = torch.zeros((2, 1))
        rotation_errors = torch.zeros((2, 1))
        gripper_errors = torch.tensor([[0.0], [1.0]])

        target_scores = _target_scores(
            action_errors,
            motion_errors,
            "se3_gripper",
            action_weight=1.0,
            motion_weight=1.0,
            translation_weight=1.0,
            rotation_weight=1.0,
            gripper_weight=1.0,
            translation_m_l2_errors=translation_errors,
            rotation_geodesic_errors=rotation_errors,
            gripper_errors=gripper_errors,
        )

        self.assertGreater(float(target_scores[0, 0]), float(target_scores[1, 0]))

    def test_structured_oracle_scores_and_rank_prefer_lower_geodesic_error(self) -> None:
        candidates = CandidateBatch(
            condition=torch.zeros((1, 4)),
            prior_mean_motion=torch.zeros((1, 6)),
            prior_mean_actions=torch.zeros((1, 2, 7)),
            samples=torch.zeros((2, 1, 6)),
            sample_actions=torch.zeros((2, 1, 2, 7)),
            action_errors=torch.tensor([[0.5], [0.1]]),
            motion_errors=torch.zeros((2, 1)),
            translation_m_l2_errors=torch.tensor([[0.0], [0.0]]),
            rotation_geodesic_errors=torch.tensor([[0.0], [1.0]]),
            gripper_errors=torch.tensor([[1.0], [0.0]]),
        )

        se3_scores = _structured_oracle_scores(candidates, "se3")
        rank = _rank_selected_scores(
            se3_scores,
            selected_indices=torch.tensor([1]),
            batch_indices=torch.tensor([0]),
        )

        self.assertGreater(float(se3_scores[0, 0]), float(se3_scores[1, 0]))
        self.assertEqual(int(rank[0]), 2)

    def test_hard_negative_ranking_loss_prefers_action_oracle_over_structured_negative(
        self,
    ) -> None:
        candidates = CandidateBatch(
            condition=torch.zeros((1, 4)),
            prior_mean_motion=torch.zeros((1, 6)),
            prior_mean_actions=torch.zeros((1, 2, 7)),
            samples=torch.zeros((2, 1, 6)),
            sample_actions=torch.zeros((2, 1, 2, 7)),
            action_errors=torch.tensor([[0.0], [1.0]]),
            motion_errors=torch.zeros((2, 1)),
            translation_m_l2_errors=torch.tensor([[1.0], [0.0]]),
            rotation_geodesic_errors=torch.tensor([[1.0], [0.0]]),
            gripper_errors=torch.zeros((2, 1)),
        )
        aligned_logits = torch.tensor([[2.0], [0.0]])
        reversed_logits = torch.tensor([[0.0], [2.0]])

        aligned_loss = _hard_negative_ranking_loss(
            aligned_logits,
            candidates,
            "se3",
            margin=0.0,
        )
        reversed_loss = _hard_negative_ranking_loss(
            reversed_logits,
            candidates,
            "se3",
            margin=0.0,
        )

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_event_alignment_errors_prefer_matching_transition(self) -> None:
        target = GripperEventLabel(
            event_type="close_transition",
            has_close=True,
            has_open=False,
            close_step=0,
            open_step=None,
            event_step=0,
            event_strength=1.0,
            close_fraction=1.0,
            open_fraction=0.0,
        )
        event_targets = EventTargetData(
            config=GripperEventConfig(command_threshold=0.5, close_sign=1),
            gt_labels={"w0": target},
            previous_commands={"w0": -1.0},
            horizon=2,
        )
        sample_actions = torch.zeros((2, 1, 2, 7))
        sample_actions[0, 0, :, -1] = 1.0
        sample_actions[1, 0, :, -1] = -1.0

        errors = _event_alignment_errors(
            sample_actions,
            {"window_id": ["w0"]},
            event_targets,
            torch.device("cpu"),
        )

        self.assertLess(float(errors[0, 0]), float(errors[1, 0]))

    def test_event_training_metrics_report_event_oracle_rank(self) -> None:
        logits = torch.tensor([[2.0], [0.0]])
        event_errors = torch.tensor([[0.0], [3.0]])

        metrics = _event_training_metrics(logits, event_errors)

        self.assertAlmostEqual(metrics["event_scorer_oracle_rank"], 1.0, delta=1e-6)
        self.assertAlmostEqual(metrics["event_scorer_oracle_match"], 1.0, delta=1e-6)

    def test_event_hard_negative_loss_prefers_composite_positive_over_event_bad_action_plausible(
        self,
    ) -> None:
        target_scores = torch.tensor([[2.0], [0.0], [-2.0]])
        action_errors = torch.tensor([[0.0], [0.01], [1.0]])
        event_errors = torch.tensor([[0.0], [5.0], [0.0]])
        aligned_logits = torch.tensor([[2.0], [0.0], [-1.0]])
        reversed_logits = torch.tensor([[0.0], [2.0], [-1.0]])

        aligned_loss = _event_hard_negative_ranking_loss(
            aligned_logits,
            target_scores,
            action_errors,
            event_errors,
            margin=0.0,
        )
        reversed_loss = _event_hard_negative_ranking_loss(
            reversed_logits,
            target_scores,
            action_errors,
            event_errors,
            margin=0.0,
        )

        self.assertLess(float(aligned_loss), float(reversed_loss))

    def test_gaussian_kl_divergence_zero_for_matching_unit_gaussians(self) -> None:
        mean = torch.zeros((3, 5))
        logvar = torch.zeros((3, 5))

        kl = gaussian_kl_divergence(mean, logvar, mean, logvar)

        self.assertAlmostEqual(float(kl), 0.0, delta=1e-6)

    def test_gaussian_kl_divergence_respects_free_bits(self) -> None:
        mean = torch.zeros((3, 5))
        logvar = torch.zeros((3, 5))

        kl = gaussian_kl_divergence(mean, logvar, mean, logvar, free_bits=0.02)

        self.assertAlmostEqual(float(kl), 0.1, delta=1e-6)

    def test_future_motion_metrics_split_6d_steps(self) -> None:
        pred = torch.zeros((1, 12))
        target = torch.zeros_like(pred)
        pred[0, 0] = 1.0
        pred[0, 3] = 2.0
        pred[0, 6] = 3.0
        pred[0, 9] = 4.0

        metrics = future_motion_metrics(pred, target, step_dim=6)

        self.assertAlmostEqual(metrics["translation_l2"], 2.0, delta=1e-6)
        self.assertAlmostEqual(metrics["orientation_coord_l2"], 3.0, delta=1e-6)
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)

    def test_prediction_metrics_support_future_gripper_mode(self) -> None:
        pred = torch.tensor([[1.0, -1.0]])
        target = torch.tensor([[1.0, 1.0]])

        metrics = _prediction_metrics(pred, target, "future_gripper")

        self.assertAlmostEqual(metrics["mse"], 2.0, delta=1e-6)
        self.assertAlmostEqual(metrics["gripper_mse"], 2.0, delta=1e-6)
        self.assertIn("gripper_mae", metrics)

    def test_prediction_metrics_support_future_delta_gripper_mode(self) -> None:
        pred = torch.zeros((1, 14))
        target = torch.zeros_like(pred)
        pred[0, 0] = 1.0
        pred[0, 12] = 2.0

        metrics = _prediction_metrics(pred, target, "future_delta_gripper")

        self.assertIn("eef_translation_l2", metrics)
        self.assertIn("gripper_gripper_mse", metrics)
        self.assertAlmostEqual(metrics["gripper_gripper_mse"], 2.0, delta=1e-6)

    def test_combined_training_loss_adds_weighted_action_loss(self) -> None:
        motion_loss = torch.tensor(2.0)
        action_loss = torch.tensor(3.0)

        combined = _combined_training_loss(
            motion_loss,
            action_loss,
            action_aware_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(combined), 3.5, delta=1e-6)

    def test_combined_training_loss_defaults_to_motion_loss(self) -> None:
        motion_loss = torch.tensor(2.0)

        combined = _combined_training_loss(
            motion_loss,
            None,
            action_aware_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(combined), 2.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
