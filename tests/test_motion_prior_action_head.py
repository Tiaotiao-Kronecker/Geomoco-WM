from __future__ import annotations

import unittest

import torch

from geomoco_wm.models.motion_prior_action_head import (
    MotionPriorActionHead,
    PostHocActionResidualAdapter,
)
from scripts.train_predicted_event_mixture_action_head import (
    GRIPPER_BOUNDARY_STEP_CLASSES,
    GRIPPER_ROUTE_FAMILIES,
    GRIPPER_STEP_CLASSES,
    _apply_future_input_control,
    _add_gripper_boundary_index_losses,
    _add_aux_gripper_loss,
    _add_gripper_residual_losses,
    _add_gripper_step_residual_losses,
    _add_gripper_trajectory_residual_losses,
    _add_event_time_losses,
    _add_flow_action_losses,
    _add_sample_score_losses,
    _add_temporal_action_losses,
    _boundary_index_predicted_actions,
    _gripper_boundary_index_targets,
    _gripper_boundary_step_targets,
    _gripper_route_target_for_label,
    _gripper_step_targets,
    _oracle_boundary_step_routed_actions,
    _predicted_boundary_step_routed_actions,
    _rank_sample_feature,
    _replace_action_gripper,
    _sample_feature_dim,
    _sample_score_regrets,
    _train_sampling_summary,
    _transition_balanced_sample_weights,
    _weighted_action_loss,
)


class DummyWindow:
    def __init__(self, window_id: str) -> None:
        self.window_id = window_id


class DummyDataset:
    def __init__(self, window_ids: list[str]) -> None:
        self.windows = [DummyWindow(window_id) for window_id in window_ids]


class MotionPriorActionHeadTests(unittest.TestCase):
    def test_posthoc_residual_adapter_shape_and_zero_init(self) -> None:
        adapter = PostHocActionResidualAdapter(
            feature_dim=24,
            action_dim=7,
            horizon=3,
            hidden_dims=(16,),
            step_dim=4,
            dropout=0.0,
        )
        features = torch.randn((5, 24))
        temporal_actions = torch.randn((5, 3, 7))

        output = adapter(features, temporal_actions)

        self.assertEqual(tuple(output["adapter_actions"].shape), (5, 3, 7))
        self.assertEqual(tuple(output["adapter_residual"].shape), (5, 3, 7))
        self.assertTrue(torch.allclose(output["adapter_actions"], temporal_actions))
        self.assertTrue(torch.allclose(output["adapter_residual"], torch.zeros_like(temporal_actions)))

    def test_context_only_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            conditioning_dim=3,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        conditioning = torch.zeros((4, 3))

        actions = model(context, None, conditioning)

        self.assertEqual(tuple(actions.shape), (4, 2, 7))

    def test_sample_set_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=1,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        actions = model(context, future_motions)

        self.assertEqual(tuple(actions.shape), (4, 2, 7))

    def test_transition_balanced_sample_weights_target_fraction(self) -> None:
        dataset = DummyDataset(["w0", "w1", "w2", "w3"])
        event_labels = {
            "w0": {"event_mode": "transition_close::middle"},
            "w1": {"event_mode": "transition_open::early"},
            "w2": {"event_mode": "sustain_open::none"},
            "w3": {"event_mode": "sustain_close::none"},
        }

        weights = _transition_balanced_sample_weights(
            dataset,  # type: ignore[arg-type]
            [0, 1, 2, 3],
            event_labels,
            transition_sampling_fraction=0.75,
        )
        transition_mass = weights[0] + weights[1]
        total_mass = sum(weights)
        summary = _train_sampling_summary(
            dataset,  # type: ignore[arg-type]
            [0, 1, 2, 3],
            event_labels,
            sampling_mode="transition_balanced",
            transition_sampling_fraction=0.75,
        )

        self.assertAlmostEqual(transition_mass / total_mass, 0.75, delta=1e-9)
        self.assertAlmostEqual(summary["sampled_transition_probability"], 0.75, delta=1e-9)
        self.assertEqual(summary["transition_count"], 2)

    def test_sample_set_with_sample_features_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            sample_feature_dim=5,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))
        sample_features = torch.zeros((4, 5, 5))

        actions = model(context, future_motions, sample_features=sample_features)

        self.assertEqual(tuple(actions.shape), (4, 2, 7))

    def test_forward_with_aux_gripper_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            aux_gripper_head=True,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["aux_gripper"].shape), (4, 2))

    def test_forward_with_aux_without_aux_head(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertIsNone(output["aux_gripper"])

    def test_forward_with_gripper_residual_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_residual_mode="event_family",
            gripper_route_count=3,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["routed_actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["gripper_route_logits"].shape), (4, 3))
        self.assertEqual(tuple(output["gripper_route_probs"].shape), (4, 3))
        self.assertEqual(tuple(output["gripper_residuals"].shape), (4, 2, 3))

    def test_forward_with_gripper_step_residual_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_step_residual_mode="event_step",
            gripper_step_class_count=3,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["step_routed_actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["gripper_step_logits"].shape), (4, 2, 3))
        self.assertEqual(tuple(output["gripper_step_probs"].shape), (4, 2, 3))
        self.assertEqual(tuple(output["gripper_step_residuals"].shape), (4, 2, 3))

    def test_forward_with_gripper_trajectory_residual_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_trajectory_residual_mode="temporal_mlp",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["trajectory_routed_actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["gripper_trajectory_residuals"].shape), (4, 2))

    def test_forward_with_temporal_action_decoder_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            temporal_action_decoder_mode="sequence_mlp",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 7))

    def test_forward_with_soft_event_time_conditioning_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            temporal_action_decoder_mode="sequence_mlp",
            event_time_conditioning_mode="soft_boundary",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["event_time_logits"].shape), (4, 2, 3))
        self.assertEqual(tuple(output["event_time_probs"].shape), (4, 2, 3))
        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 7))

    def test_forward_with_temporal_transformer_action_decoder_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=1,
            temporal_action_decoder_mode="temporal_transformer",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 7))

    def test_temporal_transformer_accepts_soft_event_time_conditioning(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=1,
            temporal_action_decoder_mode="temporal_transformer",
            event_time_conditioning_mode="soft_boundary",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["event_time_logits"].shape), (4, 2, 3))
        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 7))

    def test_soft_event_time_conditioning_requires_temporal_decoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires temporal_action_decoder_mode"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=14,
                action_dim=7,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
                temporal_layers=0,
                event_time_conditioning_mode="soft_boundary",
            )

    def test_forward_with_flow_action_decoder_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            temporal_action_decoder_mode="sequence_mlp",
            flow_action_decoder_mode="rectified_mlp",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["flow_actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["flow_action_velocity"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["flow_action_residual"].shape), (4, 2, 7))

    def test_flow_action_decoder_requires_temporal_action_decoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires temporal_action_decoder_mode"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=14,
                action_dim=7,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
                temporal_layers=0,
                flow_action_decoder_mode="rectified_mlp",
                dropout=0.0,
            )

    def test_forward_with_sample_score_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            sample_score_mode="action_regret",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 2, 7))
        self.assertEqual(tuple(output["sample_scores"].shape), (4, 5))
        self.assertEqual(tuple(output["sample_score_probs"].shape), (4, 5))
        self.assertTrue(torch.allclose(output["sample_score_probs"].sum(dim=-1), torch.ones(4)))

    def test_temporal_action_decoder_allows_non_gripper_actions(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=12,
            action_dim=6,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            temporal_action_decoder_mode="sequence_mlp",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 12))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["temporal_actions"].shape), (4, 2, 6))

    def test_future_input_control_mean_repeated(self) -> None:
        future_inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        sample_features = torch.ones((2, 3, 2))

        controlled_inputs, controlled_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            "mean_repeated",
        )

        expected = future_inputs.mean(dim=1, keepdim=True).expand_as(future_inputs)
        self.assertTrue(torch.equal(controlled_inputs, expected))
        self.assertIs(controlled_features, sample_features)

    def test_future_input_control_context_only_drops_prior_inputs(self) -> None:
        future_inputs = torch.zeros((2, 3, 4))
        sample_features = torch.ones((2, 3, 2))

        controlled_inputs, controlled_features = _apply_future_input_control(
            future_inputs,
            sample_features,
            "context_only",
        )

        self.assertIsNone(controlled_inputs)
        self.assertIsNone(controlled_features)

    def test_forward_with_boundary_index_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=21,
            action_dim=7,
            horizon=3,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_step_residual_mode="event_step",
            gripper_step_class_count=3,
            gripper_boundary_index_mode="boundary_index",
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 21))

        output = model.forward_with_aux(context, future_motions)

        self.assertEqual(tuple(output["actions"].shape), (4, 3, 7))
        self.assertEqual(tuple(output["gripper_boundary_index_logits"].shape), (4, 2, 4))
        self.assertEqual(tuple(output["gripper_boundary_index_probs"].shape), (4, 2, 4))

    def test_positive_only_step_residual_ignores_no_boundary_class(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_step_residual_mode="event_step",
            gripper_step_class_count=3,
            gripper_step_residual_blend="positive_only",
            dropout=0.0,
        )
        context = torch.zeros((1, 15))
        future_motions = torch.zeros((1, 5, 14))
        output = model.forward_with_aux(context, future_motions)
        base = output["actions"].clone()
        output["gripper_step_probs"][..., :] = 0.0
        output["gripper_step_probs"][..., 0] = 1.0
        output["gripper_step_residuals"][..., :] = 0.0
        output["gripper_step_residuals"][..., 0] = 100.0

        rerouted = model._step_routed_gripper_actions_from_tensors(
            base,
            output["gripper_step_logits"],
            output["gripper_step_probs"],
            output["gripper_step_residuals"],
        )

        self.assertTrue(torch.allclose(rerouted["step_routed_actions"], base))

    def test_oracle_step_routed_gripper_uses_only_positive_target_residuals(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=21,
            action_dim=7,
            horizon=3,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_step_residual_mode="event_step",
            gripper_step_class_count=3,
            dropout=0.0,
        )
        actions = torch.zeros((1, 3, 7))
        step_targets = torch.tensor([[0, 1, 2]])
        step_residuals = torch.tensor([[[99.0, 1.0, 2.0], [99.0, 3.0, 4.0], [99.0, 5.0, 6.0]]])

        routed = model.oracle_step_routed_gripper_actions_from_targets(
            actions,
            step_targets,
            step_residuals,
        )

        self.assertTrue(torch.equal(routed[..., -1], torch.tensor([[0.0, 3.0, 6.0]])))

    def test_script_oracle_boundary_step_routing_matches_model_helper(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=21,
            action_dim=7,
            horizon=3,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            gripper_step_residual_mode="event_step",
            gripper_step_class_count=3,
            dropout=0.0,
        )
        actions = torch.zeros((2, 3, 7))
        step_targets = torch.tensor([[0, 1, 0], [2, 0, 1]])
        step_residuals = torch.randn((2, 3, 3))

        expected = model.oracle_step_routed_gripper_actions_from_targets(
            actions,
            step_targets,
            step_residuals,
        )
        actual = _oracle_boundary_step_routed_actions(actions, step_targets, step_residuals)

        self.assertTrue(torch.equal(actual, expected))

    def test_predicted_boundary_step_routing_supports_argmax_and_threshold(self) -> None:
        actions = torch.zeros((1, 3, 7))
        logits = torch.tensor(
            [
                [
                    [4.0, 1.0, 0.0],
                    [0.0, 3.0, 1.0],
                    [1.0, 0.0, 3.0],
                ]
            ]
        )
        residuals = torch.tensor(
            [
                [
                    [100.0, 1.0, 2.0],
                    [100.0, 3.0, 4.0],
                    [100.0, 5.0, 6.0],
                ]
            ]
        )

        argmax = _predicted_boundary_step_routed_actions(
            actions,
            logits,
            residuals,
            threshold=None,
        )
        threshold = _predicted_boundary_step_routed_actions(
            actions,
            logits,
            residuals,
            threshold=0.5,
        )

        self.assertTrue(torch.equal(argmax[..., -1], torch.tensor([[0.0, 3.0, 6.0]])))
        self.assertTrue(torch.equal(threshold[..., -1], torch.tensor([[0.0, 3.0, 6.0]])))

    def test_gripper_residual_requires_gripper_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a gripper action channel"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=12,
                action_dim=6,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
                temporal_layers=0,
                gripper_residual_mode="event_family",
                dropout=0.0,
            )

    def test_gripper_trajectory_residual_requires_gripper_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a gripper action channel"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=12,
                action_dim=6,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
                temporal_layers=0,
                gripper_trajectory_residual_mode="temporal_mlp",
                dropout=0.0,
            )

    def test_set_aggregator_variants_shape(self) -> None:
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))
        for aggregator in ("mean_pool", "context_attention", "multi_query_attention"):
            with self.subTest(aggregator=aggregator):
                model = MotionPriorActionHead(
                    context_dim=15,
                    motion_dim=14,
                    action_dim=7,
                    horizon=2,
                    hidden_dims=(16,),
                    token_dim=8,
                    num_heads=2,
                    temporal_layers=0,
                    set_aggregator=aggregator,
                    set_query_count=3,
                    dropout=0.0,
                )

                actions = model(context, future_motions)

                self.assertEqual(tuple(actions.shape), (4, 2, 7))

    def test_single_future_shape(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motion = torch.zeros((4, 14))

        actions = model(context, future_motion)

        self.assertEqual(tuple(actions.shape), (4, 2, 7))

    def test_requires_motion_dim_divisible_by_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion_dim must be divisible"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=15,
                action_dim=7,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
            )

    def test_validates_future_motion_dim(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 13))

        with self.assertRaisesRegex(ValueError, "future_motions dim"):
            model(context, future_motions)

    def test_validates_sample_feature_dim(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=14,
            action_dim=7,
            horizon=2,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            sample_feature_dim=5,
            dropout=0.0,
        )
        context = torch.zeros((4, 15))
        future_motions = torch.zeros((4, 5, 14))
        sample_features = torch.zeros((4, 5, 4))

        with self.assertRaisesRegex(ValueError, "sample_features dim"):
            model(context, future_motions, sample_features=sample_features)

    def test_rejects_unknown_sample_score_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_score_mode"):
            MotionPriorActionHead(
                context_dim=15,
                motion_dim=14,
                action_dim=7,
                horizon=2,
                hidden_dims=(16,),
                token_dim=8,
                num_heads=2,
                sample_score_mode="mystery",
            )

    def test_sample_feature_modes_have_expected_dims(self) -> None:
        event_classes = ("a::none", "b::early", "c::late")

        self.assertEqual(_sample_feature_dim("none", event_classes), 0)
        self.assertEqual(_sample_feature_dim("event_only", event_classes), 3)
        self.assertEqual(_sample_feature_dim("rank_prob_only", event_classes), 2)
        self.assertEqual(_sample_feature_dim("event_rank_prob", event_classes), 5)
        self.assertEqual(_sample_feature_dim("shuffled_event_rank_prob", event_classes), 5)

    def test_shuffled_sample_features_keep_rank_prob_but_roll_events(self) -> None:
        event_one_hot = torch.eye(3)
        probabilities = torch.tensor([0.8, 0.6, 0.4])

        aligned = _rank_sample_feature(
            event_one_hot,
            probabilities,
            rank=1,
            event_top_m=3,
            sample_feature_mode="event_rank_prob",
        )
        shuffled = _rank_sample_feature(
            event_one_hot,
            probabilities,
            rank=1,
            event_top_m=3,
            sample_feature_mode="shuffled_event_rank_prob",
        )

        self.assertTrue(torch.equal(shuffled[:, :3], torch.roll(aligned[:, :3], shifts=1, dims=0)))
        self.assertTrue(torch.equal(shuffled[:, 3:], aligned[:, 3:]))

    def test_transition_weighted_action_loss(self) -> None:
        pred = torch.tensor([[[1.0]], [[2.0]]])
        target = torch.zeros((2, 1, 1))
        batch = {"window_id": ["sustain_window", "transition_window"]}
        event_labels = {
            "sustain_window": "sustain_open::none",
            "transition_window": "transition_open::early",
        }

        loss, metrics, counts = _weighted_action_loss(
            pred,
            target,
            batch,
            event_labels,
            loss_weight_mode="transition",
            transition_loss_weight=3.0,
        )

        self.assertAlmostEqual(float(loss), (1.0 + 3.0 * 4.0) / 4.0)
        self.assertAlmostEqual(metrics["transition_mse"], 4.0)
        self.assertAlmostEqual(metrics["sustain_mse"], 1.0)
        self.assertEqual(counts["transition_mse"], 1)
        self.assertEqual(counts["sustain_mse"], 1)

    def test_replace_action_gripper_only_changes_last_dim(self) -> None:
        pred = torch.zeros((2, 3, 7))
        pred[..., :6] = 5.0
        aux_gripper = torch.ones((2, 3))

        replaced = _replace_action_gripper(pred, aux_gripper)

        self.assertTrue(torch.equal(replaced[..., :6], pred[..., :6]))
        self.assertTrue(torch.equal(replaced[..., -1], aux_gripper))

    def test_aux_gripper_loss_reports_replaced_metrics(self) -> None:
        pred = torch.zeros((1, 2, 7))
        actions = torch.zeros((1, 2, 7))
        actions[..., -1] = 1.0
        aux_gripper = torch.ones((1, 2))

        loss, metrics, counts = _add_aux_gripper_loss(
            torch.tensor(2.0),
            aux_gripper,
            pred,
            actions,
            aux_gripper_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(loss.detach()), 2.0)
        self.assertAlmostEqual(metrics["aux_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["aux_replaced_gripper_mse"], 0.0)
        self.assertEqual(counts["aux_replaced_mse"], 1)

    def test_gripper_route_target_for_label(self) -> None:
        self.assertEqual(
            _gripper_route_target_for_label("sustain_open::none"),
            GRIPPER_ROUTE_FAMILIES.index("sustain"),
        )
        self.assertEqual(
            _gripper_route_target_for_label("transition_close::middle"),
            GRIPPER_ROUTE_FAMILIES.index("transition_close"),
        )
        self.assertEqual(
            _gripper_route_target_for_label("transition_open::late"),
            GRIPPER_ROUTE_FAMILIES.index("transition_open"),
        )
        self.assertEqual(_gripper_route_target_for_label("mixed_transition::early"), -1)

    def test_gripper_step_targets_from_action_commands(self) -> None:
        actions = torch.zeros((2, 3, 7))
        actions[..., -1] = torch.tensor(
            [
                [0.0, 0.75, -0.75],
                [0.49, 0.5, -0.5],
            ]
        )

        targets = _gripper_step_targets(actions)

        expected = torch.tensor(
            [
                [
                    GRIPPER_STEP_CLASSES.index("sustain"),
                    GRIPPER_STEP_CLASSES.index("close"),
                    GRIPPER_STEP_CLASSES.index("open"),
                ],
                [
                    GRIPPER_STEP_CLASSES.index("sustain"),
                    GRIPPER_STEP_CLASSES.index("close"),
                    GRIPPER_STEP_CLASSES.index("open"),
                ],
            ]
        )
        self.assertTrue(torch.equal(targets, expected))

    def test_gripper_boundary_step_targets_from_event_audit_records(self) -> None:
        batch = {"window_id": ["sustain", "close", "open", "mixed"]}
        event_labels = {
            "sustain": {
                "event_mode": "sustain_open::none",
                "close_step": None,
                "open_step": None,
            },
            "close": {
                "event_mode": "transition_close::middle",
                "close_step": 2,
                "open_step": None,
            },
            "open": {
                "event_mode": "transition_open::early",
                "close_step": None,
                "open_step": 1,
            },
            "mixed": {
                "event_mode": "mixed_transition::early",
                "close_step": 0,
                "open_step": 3,
            },
        }

        targets = _gripper_boundary_step_targets(
            batch,
            event_labels,
            horizon=4,
            device=torch.device("cpu"),
        )

        no_boundary = GRIPPER_BOUNDARY_STEP_CLASSES.index("no_boundary")
        close_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")
        open_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")
        expected = torch.tensor(
            [
                [no_boundary, no_boundary, no_boundary, no_boundary],
                [no_boundary, no_boundary, close_start, no_boundary],
                [no_boundary, open_start, no_boundary, no_boundary],
                [close_start, no_boundary, no_boundary, open_start],
            ]
        )
        self.assertTrue(torch.equal(targets, expected))

    def test_gripper_boundary_index_targets_from_event_audit_records(self) -> None:
        batch = {"window_id": ["sustain", "close", "open", "mixed"]}
        event_labels = {
            "sustain": {
                "event_mode": "sustain_open::none",
                "close_step": None,
                "open_step": None,
            },
            "close": {
                "event_mode": "transition_close::middle",
                "close_step": 2,
                "open_step": None,
            },
            "open": {
                "event_mode": "transition_open::early",
                "close_step": None,
                "open_step": 1,
            },
            "mixed": {
                "event_mode": "mixed_transition::early",
                "close_step": 0,
                "open_step": 3,
            },
        }

        targets = _gripper_boundary_index_targets(
            batch,
            event_labels,
            horizon=4,
            device=torch.device("cpu"),
        )

        no_event = 4
        expected = torch.tensor(
            [
                [no_event, no_event],
                [2, no_event],
                [no_event, 1],
                [0, 3],
            ]
        )
        self.assertTrue(torch.equal(targets, expected))

    def test_add_event_time_losses_reports_soft_boundary_metrics(self) -> None:
        actions = torch.zeros((4, 4, 7))
        batch = {"window_id": ["sustain", "close", "open", "mixed"]}
        event_labels = {
            "sustain": {
                "event_mode": "sustain_open::none",
                "close_step": None,
                "open_step": None,
            },
            "close": {
                "event_mode": "transition_close::middle",
                "close_step": 2,
                "open_step": None,
            },
            "open": {
                "event_mode": "transition_open::early",
                "close_step": None,
                "open_step": 1,
            },
            "mixed": {
                "event_mode": "mixed_transition::early",
                "close_step": 0,
                "open_step": 3,
            },
        }
        logits = torch.zeros((4, 2, 5))
        targets = torch.tensor([[4, 4], [2, 4], [4, 1], [0, 3]])
        for row in range(4):
            logits[row, 0, targets[row, 0]] = 6.0
            logits[row, 1, targets[row, 1]] = 6.0
        output = {
            "event_time_logits": logits,
            "event_time_probs": torch.softmax(logits, dim=-1),
        }

        loss, metrics, counts = _add_event_time_losses(
            torch.tensor(1.0),
            output,
            actions,
            batch,
            event_labels,
            event_time_conditioning_mode="soft_boundary",
            event_time_loss_weight=0.1,
        )

        self.assertIsNotNone(loss)
        self.assertLess(metrics["event_time_ce"], 0.02)
        self.assertEqual(metrics["event_time_accuracy"], 1.0)
        self.assertEqual(metrics["event_time_close_within1"], 1.0)
        self.assertEqual(metrics["event_time_open_within1"], 1.0)
        self.assertEqual(counts["event_time_ce"], 4)

    def test_boundary_index_predicted_actions_apply_close_and_open_residuals(self) -> None:
        actions = torch.zeros((1, 4, 7))
        boundary_logits = torch.zeros((1, 2, 5))
        boundary_logits[0, 0, 1] = 5.0
        boundary_logits[0, 1, 3] = 5.0
        residuals = torch.zeros((1, 4, 3))
        residuals[0, :, 0] = 100.0
        residuals[0, 1, 1] = 2.0
        residuals[0, 3, 2] = -3.0

        routed = _boundary_index_predicted_actions(actions, boundary_logits, residuals)

        self.assertTrue(torch.equal(routed[..., -1], torch.tensor([[0.0, 2.0, 0.0, -3.0]])))

    def test_boundary_index_predicted_actions_ignore_no_event(self) -> None:
        actions = torch.zeros((1, 2, 7))
        boundary_logits = torch.zeros((1, 2, 3))
        boundary_logits[..., 2] = 5.0
        residuals = torch.full((1, 2, 3), 100.0)

        routed = _boundary_index_predicted_actions(actions, boundary_logits, residuals)

        self.assertTrue(torch.equal(routed, actions))

    def test_add_gripper_residual_losses_reports_routed_metrics(self) -> None:
        actions = torch.zeros((3, 2, 7))
        actions[..., -1] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        routed_actions = actions.clone()
        base_actions = torch.zeros_like(actions)
        output = {
            "actions": base_actions,
            "routed_actions": routed_actions,
            "gripper_route_logits": torch.tensor(
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
            ),
            "gripper_route_probs": None,
            "gripper_residuals": None,
        }
        batch = {"window_id": ["hold", "close", "open"]}
        event_labels = {
            "hold": "sustain_open::none",
            "close": "transition_close::early",
            "open": "transition_open::late",
        }

        loss, metrics, counts = _add_gripper_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_residual_mode="event_family",
            gripper_residual_loss_weight=0.5,
            gripper_route_loss_weight=0.25,
        )

        self.assertLess(float(loss), 2.1)
        self.assertAlmostEqual(metrics["routed_mse"], 0.0)
        self.assertAlmostEqual(metrics["routed_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["gripper_route_accuracy"], 1.0)
        self.assertEqual(counts["routed_mse"], 3)
        self.assertEqual(counts["gripper_route_accuracy"], 3)

    def test_add_gripper_trajectory_residual_losses_reports_metrics(self) -> None:
        actions = torch.zeros((2, 3, 7))
        actions[..., -1] = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, -2.0]])
        base_actions = torch.zeros_like(actions)
        output = {
            "actions": base_actions,
            "trajectory_routed_actions": actions.clone(),
            "gripper_trajectory_residuals": actions[..., -1].clone(),
        }
        batch = {"window_id": ["sustain", "transition"]}
        event_labels = {
            "sustain": "sustain_open::none",
            "transition": "transition_close::early",
        }

        loss, metrics, counts = _add_gripper_trajectory_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_trajectory_residual_mode="temporal_mlp",
            gripper_trajectory_residual_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(loss.detach()), 2.0)
        self.assertAlmostEqual(metrics["trajectory_routed_mse"], 0.0)
        self.assertAlmostEqual(metrics["trajectory_routed_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["trajectory_routed_transition_mse"], 0.0)
        self.assertAlmostEqual(metrics["gripper_trajectory_residual_mse"], 0.0)
        self.assertEqual(counts["trajectory_routed_mse"], 2)
        self.assertEqual(counts["trajectory_routed_transition_mse"], 1)

    def test_add_temporal_action_losses_reports_metrics(self) -> None:
        actions = torch.zeros((2, 3, 7))
        actions[0, :, 0] = 1.0
        actions[1, :, -1] = -1.0
        output = {"temporal_actions": actions.clone()}
        batch = {"window_id": ["sustain", "transition"]}
        event_labels = {
            "sustain": "sustain_open::none",
            "transition": "transition_open::late",
        }

        loss, metrics, counts = _add_temporal_action_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            temporal_action_decoder_mode="sequence_mlp",
            temporal_action_loss_weight=0.5,
        )

        self.assertAlmostEqual(float(loss.detach()), 2.0)
        self.assertAlmostEqual(metrics["temporal_action_mse"], 0.0)
        self.assertAlmostEqual(metrics["temporal_action_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["temporal_action_transition_mse"], 0.0)
        self.assertEqual(counts["temporal_action_mse"], 2)
        self.assertEqual(counts["temporal_action_transition_mse"], 1)

    def test_add_flow_action_losses_reports_metrics(self) -> None:
        model = MotionPriorActionHead(
            context_dim=15,
            motion_dim=21,
            action_dim=7,
            horizon=3,
            hidden_dims=(16,),
            token_dim=8,
            num_heads=2,
            temporal_layers=0,
            temporal_action_decoder_mode="sequence_mlp",
            flow_action_decoder_mode="rectified_mlp",
            dropout=0.0,
        )
        context = torch.zeros((2, 15))
        future_motions = torch.zeros((2, 5, 21))
        output = model.forward_with_aux(context, future_motions)
        actions = output["flow_actions"].detach().clone()
        batch = {"window_id": ["sustain", "transition"]}
        event_labels = {
            "sustain": "sustain_open::none",
            "transition": "transition_open::late",
        }

        loss, metrics, counts = _add_flow_action_losses(
            torch.tensor(2.0),
            model,
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            flow_action_decoder_mode="rectified_mlp",
            flow_action_loss_weight=0.5,
            flow_matching_loss_weight=0.0,
        )

        self.assertAlmostEqual(loss.detach().item(), 2.0)
        self.assertAlmostEqual(metrics["flow_action_mse"], 0.0)
        self.assertAlmostEqual(metrics["flow_action_transition_mse"], 0.0)
        self.assertIn("flow_action_residual_mse", metrics)
        self.assertEqual(counts["flow_action_mse"], 2)
        self.assertEqual(counts["flow_action_transition_mse"], 1)

    def test_sample_score_motion_regret_targets_oracle_motion(self) -> None:
        model = MotionPriorActionHead(
            context_dim=3,
            motion_dim=4,
            action_dim=2,
            horizon=2,
            hidden_dims=(8,),
            token_dim=4,
            num_heads=2,
            temporal_layers=0,
            sample_score_mode="action_regret",
            dropout=0.0,
        )
        context = torch.zeros((2, 3))
        future_inputs = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            ]
        )
        actions = torch.zeros((2, 2, 2))
        batch = {
            "motion": torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
        }

        regrets = _sample_score_regrets(
            model,
            context,
            None,
            future_inputs,
            None,
            actions,
            batch,
            sample_score_target="motion_regret",
            temporal_action_decoder_mode="none",
        )

        self.assertTrue(torch.equal(regrets.argmin(dim=-1), torch.tensor([0, 1])))
        self.assertTrue(torch.equal(regrets[:, 0], torch.tensor([0.0, 1.0])))

    def test_sample_score_loss_reports_candidate_metrics(self) -> None:
        scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
        output = {
            "sample_scores": scores,
            "sample_score_probs": torch.softmax(scores, dim=-1),
        }
        model = MotionPriorActionHead(
            context_dim=3,
            motion_dim=4,
            action_dim=2,
            horizon=2,
            hidden_dims=(8,),
            token_dim=4,
            num_heads=2,
            temporal_layers=0,
            sample_score_mode="action_regret",
            dropout=0.0,
        )
        context = torch.zeros((2, 3))
        future_inputs = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            ]
        )
        actions = torch.zeros((2, 2, 2))
        batch = {
            "motion": torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
        }

        loss, metrics, counts = _add_sample_score_losses(
            torch.tensor(1.0),
            output,
            model,
            context,
            None,
            future_inputs,
            None,
            actions,
            batch,
            sample_score_mode="action_regret",
            sample_score_loss_weight=0.5,
            sample_score_target="motion_regret",
            sample_score_loss_type="hard_ce",
            sample_score_temperature=0.1,
            temporal_action_decoder_mode="none",
        )

        self.assertGreater(float(loss.detach()), 1.0)
        self.assertAlmostEqual(metrics["sample_score_top1_accuracy"], 1.0)
        self.assertEqual(counts["sample_score_loss"], 2)

    def test_add_gripper_step_residual_losses_reports_metrics(self) -> None:
        actions = torch.zeros((2, 3, 7))
        actions[..., -1] = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]])
        step_routed_actions = actions.clone()
        logits = torch.tensor(
            [
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        output = {
            "step_routed_actions": step_routed_actions,
            "gripper_step_logits": logits,
        }
        batch = {"window_id": ["a", "b"]}
        event_labels = {"a": "transition_open::early", "b": "transition_close::middle"}

        loss, metrics, counts = _add_gripper_step_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_step_residual_mode="event_step",
            gripper_step_residual_loss_weight=0.5,
            gripper_step_loss_weight=0.25,
        )

        self.assertLess(float(loss), 2.1)
        self.assertAlmostEqual(metrics["step_routed_mse"], 0.0)
        self.assertAlmostEqual(metrics["step_routed_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["gripper_step_accuracy"], 1.0)
        self.assertEqual(counts["step_routed_mse"], 2)
        self.assertEqual(counts["gripper_step_accuracy"], 2)

    def test_add_gripper_step_residual_losses_supports_boundary_targets(self) -> None:
        actions = torch.zeros((2, 3, 7))
        step_routed_actions = actions.clone()
        no_boundary = GRIPPER_BOUNDARY_STEP_CLASSES.index("no_boundary")
        close_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")
        open_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")
        logits = torch.zeros((2, 3, len(GRIPPER_BOUNDARY_STEP_CLASSES)))
        logits[0, :, no_boundary] = 5.0
        logits[0, 1, :] = 0.0
        logits[0, 1, close_start] = 5.0
        logits[1, :, no_boundary] = 5.0
        logits[1, 2, :] = 0.0
        logits[1, 2, open_start] = 5.0
        output = {
            "step_routed_actions": step_routed_actions,
            "gripper_step_logits": logits,
        }
        batch = {"window_id": ["close", "open"]}
        event_labels = {
            "close": {"event_mode": "transition_close::early", "close_step": 1},
            "open": {"event_mode": "transition_open::late", "open_step": 2},
        }

        loss, metrics, counts = _add_gripper_step_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_step_residual_mode="event_step",
            gripper_step_residual_loss_weight=0.5,
            gripper_step_loss_weight=0.25,
            gripper_step_target_mode="boundary_start",
        )

        self.assertLess(float(loss), 2.1)
        self.assertAlmostEqual(metrics["step_routed_mse"], 0.0)
        self.assertAlmostEqual(metrics["gripper_step_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["gripper_step_boundary_fraction"], 2 / 6)
        self.assertEqual(counts["gripper_step_accuracy"], 2)

    def test_add_gripper_step_residual_losses_can_optimize_oracle_boundary_readout(self) -> None:
        actions = torch.zeros((1, 3, 7))
        actions[..., -1] = torch.tensor([[0.0, 3.0, -4.0]])
        base_actions = torch.zeros_like(actions)
        no_boundary = GRIPPER_BOUNDARY_STEP_CLASSES.index("no_boundary")
        close_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")
        open_start = GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")
        logits = torch.zeros((1, 3, len(GRIPPER_BOUNDARY_STEP_CLASSES)))
        logits[..., no_boundary] = 5.0
        logits[0, 1, :] = 0.0
        logits[0, 1, close_start] = 5.0
        logits[0, 2, :] = 0.0
        logits[0, 2, open_start] = 5.0
        residuals = torch.zeros_like(logits)
        residuals[0, :, no_boundary] = 100.0
        residuals[0, 1, close_start] = 3.0
        residuals[0, 2, open_start] = -4.0
        output = {
            "actions": base_actions,
            "step_routed_actions": base_actions,
            "gripper_step_logits": logits,
            "gripper_step_residuals": residuals,
        }
        batch = {"window_id": ["mixed"]}
        event_labels = {
            "mixed": {"event_mode": "mixed_transition::early", "close_step": 1, "open_step": 2},
        }

        loss, metrics, counts = _add_gripper_step_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_step_residual_mode="event_step",
            gripper_step_residual_loss_weight=0.0,
            gripper_step_loss_weight=0.0,
            gripper_step_target_mode="boundary_start",
            gripper_step_oracle_boundary_residual_loss_weight=1.0,
        )

        self.assertAlmostEqual(metrics["oracle_step_routed_mse"], 0.0)
        self.assertLess(float(loss), 2.1)
        self.assertEqual(counts["oracle_step_routed_mse"], 1)

    def test_add_gripper_step_residual_losses_upweights_positive_boundary_ce(self) -> None:
        actions = torch.zeros((1, 2, 7))
        step_routed_actions = actions.clone()
        no_boundary = GRIPPER_BOUNDARY_STEP_CLASSES.index("no_boundary")
        logits = torch.zeros((1, 2, len(GRIPPER_BOUNDARY_STEP_CLASSES)))
        logits[..., no_boundary] = 4.0
        output = {
            "step_routed_actions": step_routed_actions,
            "gripper_step_logits": logits,
        }
        batch = {"window_id": ["close"]}
        event_labels = {
            "close": {"event_mode": "transition_close::early", "close_step": 1},
        }

        _, low_metrics, _ = _add_gripper_step_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_step_residual_mode="event_step",
            gripper_step_residual_loss_weight=0.0,
            gripper_step_loss_weight=1.0,
            gripper_step_target_mode="boundary_start",
            gripper_step_positive_loss_weight=1.0,
        )
        _, high_metrics, _ = _add_gripper_step_residual_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_step_residual_mode="event_step",
            gripper_step_residual_loss_weight=0.0,
            gripper_step_loss_weight=1.0,
            gripper_step_target_mode="boundary_start",
            gripper_step_positive_loss_weight=10.0,
        )

        self.assertGreater(high_metrics["gripper_step_ce"], low_metrics["gripper_step_ce"])

    def test_add_gripper_boundary_index_losses_reports_predicted_metrics(self) -> None:
        actions = torch.zeros((1, 4, 7))
        actions[..., -1] = torch.tensor([[0.0, 2.0, 0.0, -3.0]])
        base_actions = torch.zeros_like(actions)
        boundary_logits = torch.zeros((1, 2, 5))
        boundary_logits[0, 0, 1] = 5.0
        boundary_logits[0, 1, 3] = 5.0
        residuals = torch.zeros((1, 4, 3))
        residuals[0, 1, GRIPPER_BOUNDARY_STEP_CLASSES.index("close_start")] = 2.0
        residuals[0, 3, GRIPPER_BOUNDARY_STEP_CLASSES.index("open_start")] = -3.0
        output = {
            "actions": base_actions,
            "gripper_boundary_index_logits": boundary_logits,
            "gripper_step_residuals": residuals,
        }
        batch = {"window_id": ["mixed"]}
        event_labels = {
            "mixed": {
                "event_mode": "mixed_transition::early",
                "close_step": 1,
                "open_step": 3,
            },
        }

        loss, metrics, counts = _add_gripper_boundary_index_losses(
            torch.tensor(2.0),
            output,
            actions,
            batch,
            event_labels,
            loss_weight_mode="none",
            transition_loss_weight=2.0,
            gripper_boundary_index_mode="boundary_index",
            gripper_boundary_index_loss_weight=0.25,
        )

        self.assertLess(float(loss), 2.1)
        self.assertAlmostEqual(metrics["boundary_index_pred_mse"], 0.0)
        self.assertAlmostEqual(metrics["boundary_index_pred_gripper_mse"], 0.0)
        self.assertAlmostEqual(metrics["gripper_boundary_index_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["gripper_boundary_index_close_within1"], 1.0)
        self.assertEqual(counts["boundary_index_pred_mse"], 1)


if __name__ == "__main__":
    unittest.main()
