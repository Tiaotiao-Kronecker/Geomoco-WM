# Experiment Records

This directory is the canonical experiment log for `Geomoco-WM`.

Use it for reproducible run records, metric tables, artifact paths, and
experiment-level interpretations. Discussion notes can stay in
`docs/agent_qa/`; this directory should contain the cleaned experiment ledger.

## Directory Layout

```text
docs/experiments/
  README.md
  plans/
    planned experiment slices and pass/stop criteria
  runs/
    one file per concrete run or run bundle
  comparisons/
    cross-run tables and ablation summaries
  templates/
    reusable record templates
```

## Record Naming

Use:

```text
YYYY-MM-DD_gate-name_short-description.md
```

Example:

```text
2026-06-06_gate1_5_four_suite_oracle_action_decoder.md
```

## Required Fields

Each run record should include:

```text
purpose
dataset slice
exact command(s)
model / training config
split policy
metrics
artifact paths
interpretation
limits
next decision
```

For model-training runs, also record:

```text
seed(s)
device
environment notes
output metrics path
output checkpoint path
```

## Current Runs

| date | gate | record | summary |
| --- | --- | --- | --- |
| 2026-06-06 | Measurement | [action semantics audit and geodesic metrics](runs/2026-06-06_action_semantics_audit_geodesic_metrics.md) | Confirmed four-suite LIBERO `OSC_POSE` action semantics and upgraded action metrics to physical translation plus SO(3) geodesic rotation. |
| 2026-06-06 | Gate 1.5 | [four-suite oracle action decoder](runs/2026-06-06_gate1_5_four_suite_oracle_action_decoder.md) | GT future EEF motion strongly improves action decoding over direct context on a four-suite small formal LIBERO slice. |
| 2026-06-06 | Gate 1.6 | [two-file oracle action with SE(3) metrics](runs/2026-06-06_gate1_6_two_file_oracle_action_se3_metrics.md) | Oracle future EEF motion remains positive at 2 files per suite and shows strongest gains on translation/rotation action metrics. |
| 2026-06-06 | Gate 1.6 | [two-file oracle action geodesic replacement](runs/2026-06-06_gate1_6_two_file_oracle_action_geodesic_replacement.md) | Clean replacement table: oracle future motion reduces mean flat MSE by 52.32%, translation L2 by 60.76%, and SO(3) geodesic rotation by 53.08%. |
| 2026-06-06 | Gate 2 | [deterministic future-motion prior](runs/2026-06-06_gate2_deterministic_future_motion_prior.md) | Context-only future-motion prediction beats zero-motion, but downstream action metrics do not beat the direct-context lower bound. |
| 2026-06-07 | Gate 2.1 | [suite/task-conditioned future-motion prior](runs/2026-06-07_gate2_1_suite_task_conditioned_future_motion_prior.md) | Suite/task conditioning improves motion and action metrics over Gate 2, but still does not beat direct context. |
| 2026-06-07 | Gate 2.2a | [DINOv2 global visual future-motion prior](runs/2026-06-07_gate2_2a_dinov2_global_visual_prior.md) | DINOv2 global visual features push the learned future-motion prior past direct context for the first time. |
| 2026-06-07 | Gate 2.2b | [patch-pooled DINOv2 cross-attention visual prior](runs/2026-06-07_gate2_2b_patchpool_cross_attention_visual_prior.md) | Patch-pooled DINOv2 cross-attention improves over global visual features and closes 47.67% of the direct-to-oracle gap. |
| 2026-06-08 | Gate 2.2c | [visual controls](runs/2026-06-08_gate2_2c_visual_controls.md) | Shuffled DINO features fail while both real single-camera branches beat direct context; two-camera patch grounding remains the best visual route. |
| 2026-06-08 | Gate 2.3a | [action-aware visual future-motion prior](runs/2026-06-08_gate2_3a_action_aware_visual_prior.md) | Frozen action-decoder auxiliary loss improves action MSE to 0.043174 and raises direct-to-oracle gap closure to 66.12%. |
| 2026-06-08 | Gate 2.3b | [action-aware lambda sweep](runs/2026-06-08_gate2_3b_action_aware_lambda_sweep.md) | Sweeping lambda 0.003/0.010/0.030 selects lambda 0.030 as the action-value default, reaching action MSE 0.042090 and 69.26% gap closure. |
| 2026-06-08 | Gate 2.4a | [stepwise multi-query visual predictor](runs/2026-06-08_gate2_4a_stepwise_multi_query_visual_predictor.md) | Stepwise visual queries improve future translation geometry but do not beat the single-query lambda 0.030 action-value default. |
| 2026-06-08 | Gate 2.4b | [visual-conditioned cVAE future-motion prior](runs/2026-06-08_gate2_4b_visual_cvae_future_motion_prior.md) | First visual cVAE prior mean slightly improves action MSE to 0.041579, but KL is near zero so multimodal behavior is not yet proven. |
| 2026-06-08 | Gate 2.4c | [cVAE stochasticity calibration](runs/2026-06-08_gate2_4c_cvae_stochasticity_calibration.md) | Free-bits raises raw KL to 0.442097 and improves best-of-K coverage, but sample readout is still not deployable. |
| 2026-06-08 | Gate 2.4d | [lightweight cVAE sample scorer](runs/2026-06-08_gate2_4d_lightweight_cvae_sample_scorer.md) | First deployable sample scorer beats prior mean and random samples, but closes only 18.09% of the oracle readout gap. |
| 2026-06-08 | Gate 2.4e | [SE(3)+gripper-aware sample scorer](runs/2026-06-08_gate2_4e_se3_gripper_aware_sample_scorer.md) | Structured SE(3)/gripper targets improve over prior mean but regress from the Gate 2.4d flat action-MSE scorer. |
| 2026-06-08 | Gate 2.4f | [structured oracle readout evaluation](runs/2026-06-08_gate2_4f_structured_oracle_readout_eval.md) | SE(3) and gripper-aware oracle ranks confirm structured targets are diagnostic only; flat action-MSE scorer remains the deployable baseline. |
| 2026-06-08 | Gate 2.4g | [hard-negative sample readout](runs/2026-06-08_gate2_4g_hard_negative_readout.md) | Naive SE(3)+gripper hard negatives regress from the flat ScoreNet baseline, so the next readout work needs explicit event/executability proxies. |
| 2026-06-09 | Gate 2.4h-a | [gripper/event label audit](runs/2026-06-09_gate2_4h_a_gripper_event_audit.md) | Transition labels, not command-state labels, provide the usable GeoMoCo phase-boundary signal for close/open timing probes. |
| 2026-06-09 | Gate 2.4h-b | [visual phase/event probe](runs/2026-06-09_gate2_4h_b_visual_phase_event_probe.md) | Real DINO visual grounding strongly predicts transition labels and shuffled visual collapses, validating visual phase/composition signal. |
| 2026-06-09 | Gate 2.4h-c | [cVAE sample event alignment](runs/2026-06-09_gate2_4h_c_cvae_event_alignment.md) | cVAE samples contain limited event-aligned candidates, but transition timing coverage is weak and flat ScoreNet does not explicitly select event-aligned samples. |
| 2026-06-09 | Gate 2.4h-d | [minimal event-aware readout](runs/2026-06-09_gate2_4h_d_event_aware_readout.md) | A weak event-ranking auxiliary does not improve action MSE or event metrics; the next bottleneck is upstream event fidelity. |
| 2026-06-09 | Gate 2.4i | [event-fidelity interface audit](runs/2026-06-09_gate2_4i_event_fidelity_interface_audit.md) | Adding oracle future gripper to future EEF cuts action MSE by 86.65% and gripper MSE by 99.87%, proving the EEF-only interface is missing the event channel. |
| 2026-06-09 | Gate 2.5a | [visual future-gripper/event predictor](runs/2026-06-09_gate2_5a_visual_future_gripper_predictor.md) | Real DINO visual grounding predicts future gripper/event timing and improves the GT-EEF action bridge over the EEF-only oracle interface. |
| 2026-06-09 | Gate 2.5b | [predicted EEF plus predicted gripper bridge](runs/2026-06-09_gate2_5b_predicted_joint_bridge.md) | Modular predicted EEF+gripper preserves visual attribution but does not beat the previous EEF-only learned prior, so the next step is joint training. |
| 2026-06-09 | Gate 2.5b-joint | [joint future-delta-gripper predictor](runs/2026-06-09_gate2_5b_joint_future_delta_gripper_predictor.md) | Joint visual EEF+gripper prediction with stronger action-aware loss beats the modular bridge, shuffled controls, and the previous EEF-only learned prior. |
| 2026-06-09 | Gate 2.5c | [joint cVAE future-delta-gripper](runs/2026-06-09_gate2_5c_joint_cvae_future_delta_gripper.md) | Joint cVAE sample space is valuable and visually grounded, but prior mean does not yet beat the deterministic joint baseline. |
| 2026-06-10 | Gate 2.5d | [joint sample readout](runs/2026-06-10_gate2_5d_joint_sample_readout.md) | Flat action-MSE ScoreNet is weak-positive but does not beat the deterministic joint baseline. |
| 2026-06-10 | Gate 2.5e | [event-aware joint readout pilot](runs/2026-06-10_gate2_5e_event_aware_joint_readout_pilot.md) | Event-aware and event-hard-negative readouts improve transition metrics but regress action MSE, revealing a target conflict. |
| 2026-06-10 | Gate 2.6a | [temporal action-regret readout](runs/2026-06-10_gate2_6a_temporal_action_regret_readout.md) | Temporal ScoreNet v1 improves selected rank and transition step@1 but regresses mean action MSE from the flat readout. |
| 2026-06-10 | Gate 3.0a | [motion-prior-conditioned action head](runs/2026-06-10_gate3_0a_motion_prior_conditioned_action_head.md) | A downstream action head can use real visual cVAE sample sets better than context-only, prior mean, and shuffled sample controls. |
| 2026-06-11 | Gate 3.0b | [K sweep and set aggregator ablation](runs/2026-06-11_gate3_0b_k_sweep_set_aggregator.md) | K=32 slightly improves over K=16, but the gain is tiny; mean/context/multi-query aggregators are close, so readout architecture is not the main bottleneck. |
| 2026-06-11 | Gate 3.0c | [sample-set usage audit](runs/2026-06-11_gate3_0c_sample_set_usage_audit.md) | Original real sample sets beat mean replacement and K=4 subsets, proving the action head uses aligned sample diversity rather than only a mean future. |
| 2026-06-11 | Gate 3.1a | [event mode target audit](runs/2026-06-11_gate3_1a_event_mode_target_audit.md) | Close/open transition timing modes are measurable and balanced enough for the next event-mode probe; mixed-transition modes remain rare diagnostics. |
| 2026-06-11 | Gate 3.1b | [event mode probe](runs/2026-06-11_gate3_1b_event_mode_probe.md) | Real visual/proprio predicts event modes much better than task/proprio and shuffled visual controls, justifying event-conditioned cVAE training. |
| 2026-06-11 | Gate 3.1c | [oracle-event conditioned cVAE](runs/2026-06-11_gate3_1c_oracle_event_conditioned_cvae.md) | Oracle event conditioning strongly improves joint cVAE prior and best-of-K action metrics, while shuffled-event control stays near the unconditional baseline. |
| 2026-06-11 | Gate 3.1d | [predicted event mixture](runs/2026-06-11_gate3_1d_predicted_event_mixture.md) | Predicted top-4 event mixtures nearly recover oracle-event best-of-K coverage, but the wide sample set still needs a downstream action head/planner readout. |
| 2026-06-11 | Gate 3.1e | [predicted event mixture action head](runs/2026-06-11_gate3_1e_predicted_event_mixture_action_head.md) | Predicted event-mixture action heads beat shuffled controls but do not beat the simpler unconditional real sample-set baseline. |
| 2026-06-12 | Gate 3.1f | [event-aware sample consumption](runs/2026-06-12_gate3_1f_event_aware_sample_consumption.md) | Adding event mode/rank/probability metadata to each sample lets the action head beat the Gate 3.0 real sample-set baseline. |
| 2026-06-12 | Gate 3.1g | [event metadata ablation](runs/2026-06-12_gate3_1g_event_metadata_ablation.md) | Full event identity plus rank/probability metadata is the strongest sample-consumption interface; either channel alone is weaker. |
| 2026-06-12 | Gate 3.2a | [group stress audit](runs/2026-06-12_gate3_2a_group_stress_audit.md) | Event-aware top-4 reproduces its global result, but transition/open-close windows dominate the remaining error. |
| 2026-06-12 | Gate 3.2b | [transition-weighted action head](runs/2026-06-12_gate3_2b_transition_weighted_action_head.md) | Transition weighting improves transition MSE but worsens sustain and overall MSE, so it is diagnostic rather than deployable. |
| 2026-06-12 | Gate 3.2c | [auxiliary gripper action head](runs/2026-06-12_gate3_2c_aux_gripper_action_head.md) | A parallel gripper regression head gives only tiny transition gains and worsens overall action MSE. |
| 2026-06-15 | Gate 3.4 | [controlled joint temporal action-sequence decoder](runs/2026-06-15_gate3_4_controlled_joint_temporal_action_sequence_decoder.md) | A small joint temporal decoder is mechanism-positive and beats controls, but mean replacement remains close so sample-diversity attribution is thin. |
| 2026-06-16 | Gate 3.4b | [sample-diversity usage audit](runs/2026-06-16_gate3_4b_sample_diversity_usage_audit.md) | Same-checkpoint eval-time mean collapse hurts full-aligned Gate 3.4, showing runtime K-sample usage while preserving the trained mean-only caveat. |
| 2026-06-16 | Gate 3.4c | [sample-score temporal regret supervision](runs/2026-06-16_gate3_4c_sample_score_temporal_regret.md) | Explicit motion-regret candidate scoring is negative/neutral under short budget and does not beat Gate 3.4, so do not expand the full control matrix. |
| 2026-06-16 | Gate 3.5a | [small residual flow decoder](runs/2026-06-16_gate3_5a_small_residual_flow_decoder.md) | A jointly trained one-step residual flow adapter is negative under short budget; flow readout is worse than temporal_actions and below Gate 3.4. |
| 2026-06-16 | Gate 3.5b | [frozen post-hoc residual adapter](runs/2026-06-16_gate3_5b_frozen_posthoc_residual_adapter.md) | Freezing Gate 3.4 and training only a residual adapter improves overall MSE, but attribution is thin and transition MSE regresses from Gate 3.4. |
| 2026-06-16 | Gate 3.5c | [transition-constrained post-hoc residual](runs/2026-06-16_gate3_5c_transition_constrained_posthoc_residual.md) | Predicted transition gating preserves a small overall adapter gain, but still fails the transition promotion check, so full controls were not expanded. |
| 2026-06-17 | Gate 3.6a | [transition-reserve candidate quality](runs/2026-06-17_gate3_6a_transition_reserve_candidate_quality.md) | Transition MSE improves but overall/sustain regress; later 3.6b/3.6c diagnostics correct the attribution from candidate reserve to transition-MSE checkpoint selection. |
| 2026-06-17 | Gate 3.6b | [Pareto transition candidate allocation sweep](runs/2026-06-17_gate3_6b_pareto_transition_candidate_allocation.md) | Conservative transition-reserve thresholds exactly match Gate 3.4 because the reserve rule never triggers at top-4 on the current split. |
| 2026-06-17 | Gate 3.6c | [top-k transition-selection attribution check](runs/2026-06-17_gate3_6c_topk_transition_selection_attribution.md) | Top-k candidates with transition-MSE checkpoint selection exactly reproduce Gate 3.6a, showing the gain came from selection/early stopping rather than candidate replacement. |

## Current Comparisons

| date | comparison | summary |
| --- | --- | --- |
| 2026-06-06 | [oracle action gate scale-up](comparisons/2026-06-06_oracle_action_gate_scaleup.md) | The direct-context vs oracle-future gap remains stable from 1 file per suite to 2 files per suite. |
| 2026-06-06 | [Gate 1.6 geodesic replacement summary](comparisons/2026-06-06_gate1_6_geodesic_replacement_summary.md) | The corrected measurement contract confirms the two-file oracle-future gap in meters and SO(3) geodesic degrees. |
| 2026-06-06 | [Gate 2 learned prior vs bounds](comparisons/2026-06-06_gate2_learned_prior_vs_bounds.md) | The learned prior improves over zero future motion but fails to move downstream action metrics between direct context and oracle future motion. |
| 2026-06-07 | [Gate 2.1 conditioned prior vs bounds](comparisons/2026-06-07_gate2_1_conditioned_prior_vs_bounds.md) | Task/suite metadata helps, but the learned prior remains worse than direct context as a policy interface. |
| 2026-06-07 | [Gate 2.2a visual prior vs bounds](comparisons/2026-06-07_gate2_2a_visual_prior_vs_bounds.md) | Global DINOv2 visual grounding reduces action MSE below direct context and closes 35.85% of the direct-to-oracle gap. |
| 2026-06-07 | [Gate 2.2 visual grounding summary](comparisons/2026-06-07_gate2_2_visual_grounding_summary.md) | Patch cross-attention is the strongest learned prior so far, but cVAE should wait for shuffled-vision and camera controls. |
| 2026-06-08 | [Gate 2.2 visual controls summary](comparisons/2026-06-08_gate2_2_visual_controls_summary.md) | Visual controls pass: correct visual alignment matters, eye-in-hand is slightly stronger than agentview, and two-camera patch grounding remains the default branch. |
| 2026-06-08 | [Gate 2.3a action-aware vs visual prior](comparisons/2026-06-08_gate2_3a_action_aware_vs_visual_prior.md) | Action-aware training improves over Gate 2.2b without collapsing future-motion MSE; it becomes the new deterministic learned-prior baseline. |
| 2026-06-08 | [Gate 2.3 action-aware lambda selection](comparisons/2026-06-08_gate2_3_action_aware_lambda_selection.md) | Lambda 0.030 is selected for action-value-prioritized runs, while lambda 0.010 remains the balanced geometry reference. |
| 2026-06-08 | [Gate 2.4a stepwise vs single-query](comparisons/2026-06-08_gate2_4a_stepwise_vs_single_query.md) | Stepwise querying is not promoted as the default; the next mainline should move to stochastic or multimodal future-motion priors. |
| 2026-06-08 | [Gate 2.4b visual cVAE vs deterministic](comparisons/2026-06-08_gate2_4b_visual_cvae_vs_deterministic.md) | The first visual cVAE is a weak positive on prior-mean action MSE but needs KL/sample calibration before stochastic claims. |
| 2026-06-08 | [Gate 2.4c cVAE sampling and free-bits](comparisons/2026-06-08_gate2_4c_cvae_sampling_and_freebits.md) | Free-bits validates non-trivial sample coverage, while also showing that a deployable sample scorer/readout is the next bottleneck. |
| 2026-06-08 | [Gate 2.4d sample readout vs oracle](comparisons/2026-06-08_gate2_4d_sample_readout_vs_oracle.md) | Lightweight ScoreNet turns part of best-of-K coverage into deployable readout, but gripper/contact-aware scoring remains the next bottleneck. |
| 2026-06-08 | [Gate 2.4e structured readout vs flat](comparisons/2026-06-08_gate2_4e_structured_readout_vs_flat.md) | Naive SE(3)+gripper target replacement does not beat flat action-MSE ranking; next readout work should add richer hard negatives or executability proxies. |
| 2026-06-08 | [Gate 2.4f structured oracle ranks](comparisons/2026-06-08_gate2_4f_structured_oracle_ranks.md) | Structured scorers improve their matching oracle rank but still lose on deployable action MSE, so Gate 2.4g should add hard-negative / executability-aware supervision. |
| 2026-06-08 | [Gate 2.4g hard negative vs flat](comparisons/2026-06-08_gate2_4g_hard_negative_vs_flat.md) | Minimal SE(3)+gripper hard negatives are not reliable enough; move next to explicit gripper-transition/contact/executability proxy construction. |
| 2026-06-09 | [Gate 2.4h-a command vs transition events](comparisons/2026-06-09_gate2_4h_a_command_vs_transition_events.md) | Command-state labels are shortcut-prone; transition labels isolate actual phase-boundary events and become the event contract for later probes. |
| 2026-06-09 | [Gate 2.4h-b visual phase/event probe summary](comparisons/2026-06-09_gate2_4h_b_visual_phase_event_probe_summary.md) | Visual grounding provides aligned phase/event information; next test whether cVAE samples contain event-aligned candidates. |
| 2026-06-09 | [Gate 2.4h-c event alignment summary](comparisons/2026-06-09_gate2_4h_c_event_alignment_summary.md) | Event oracle best-of-K is positive but small; transition timing and readout remain the bottleneck before event-aware scorer training. |
| 2026-06-09 | [Gate 2.4h-d event-aware vs flat](comparisons/2026-06-09_gate2_4h_d_event_aware_vs_flat.md) | Minimal event-aware ScoreNet is a negative/neutral diagnostic; improve the candidate/event interface before more readout engineering. |
| 2026-06-09 | [Gate 2.4i EEF-only vs gripper interface](comparisons/2026-06-09_gate2_4i_eef_only_vs_gripper_interface.md) | Future EEF-only motion is insufficient for close/open timing; the next target should include future gripper/event channels. |
| 2026-06-09 | [Gate 2.5a future-gripper visual controls](comparisons/2026-06-09_gate2_5a_future_gripper_visual_controls.md) | Real visual future-gripper prediction beats task/proprio and shuffled visual controls, and partially repairs the EEF-only action-bridge gap. |
| 2026-06-09 | [Gate 2.5b joint bridge diagnostics](comparisons/2026-06-09_gate2_5b_joint_bridge_diagnostics.md) | Separate EEF and gripper predictors compound errors inside the joint action decoder; train a joint `future_delta_gripper` predictor next. |
| 2026-06-09 | [Gate 2.5b joint predictor controls](comparisons/2026-06-09_gate2_5b_joint_predictor_controls.md) | Real visual joint EEF+gripper prediction passes controls and becomes the deterministic baseline for joint cVAE. |
| 2026-06-09 | [Gate 2.5c joint cVAE vs deterministic](comparisons/2026-06-09_gate2_5c_joint_cvae_vs_deterministic.md) | cVAE best-of-K is strong, but raw prior mean needs readout/tuning before it can replace deterministic joint. |
| 2026-06-10 | [Gate 2.5d joint readout vs controls](comparisons/2026-06-10_gate2_5d_joint_sample_readout_vs_controls.md) | Joint cVAE sample readout has real but shallow visual signal and remains below deterministic joint performance. |
| 2026-06-10 | [Gate 2.5e event readout vs flat](comparisons/2026-06-10_gate2_5e_event_readout_vs_flat.md) | Event-aware targets improve event alignment while worsening action MSE, so larger event-weight sweeps are not promoted. |
| 2026-06-10 | [Gate 2.6a temporal vs flat readout](comparisons/2026-06-10_gate2_6a_temporal_vs_flat_readout.md) | Temporal scoring alone is insufficient; the next readout should predict calibrated action regret or compare candidates set-wise. |
| 2026-06-10 | [Gate 3.0a action-head prior ablation](comparisons/2026-06-10_gate3_0a_action_head_prior_ablation.md) | Real cVAE sample sets beat context-only and shuffled sample controls under a shared downstream action-head protocol. |
| 2026-06-11 | [Gate 3.0b K and aggregator summary](comparisons/2026-06-11_gate3_0b_k_and_aggregator_summary.md) | Keep context-attention K=16 as the default; K=32 is optional, and the next step should audit sample diversity and action-head usage. |
| 2026-06-11 | [Gate 3.0c usage audit summary](comparisons/2026-06-11_gate3_0c_usage_audit_summary.md) | The useful signal is aligned sample-set diversity, not generic diversity; next branch should make mode/event structure explicit. |
| 2026-06-11 | [Gate 3.1 event-aware prior decision](comparisons/2026-06-11_gate3_1_event_aware_prior_mainline_decision.md) | Next mainline should expose gripper-transition/event timing as an explicit mode structure and evaluate oracle-event versus predicted-event mixture routes. |
| 2026-06-11 | [Gate 3.1a event mode label readiness](comparisons/2026-06-11_gate3_1a_event_mode_label_readiness.md) | Event-mode labels pass readiness checks for close/open transition timing; proceed to the event-mode probe while treating mixed transitions as rare. |
| 2026-06-11 | [Gate 3.1b event mode probe summary](comparisons/2026-06-11_gate3_1b_event_mode_probe_summary.md) | Aligned visual features carry event timing signal; shuffled visual collapses, so event conditioning is a justified next prior structure. |
| 2026-06-11 | [Gate 3.1c oracle event cVAE summary](comparisons/2026-06-11_gate3_1c_oracle_event_cvae_summary.md) | Correct event-mode conditioning is a strong upper bound; next build the deployable predicted-event/top-M mixture route. |
| 2026-06-11 | [Gate 3.1d predicted event mixture summary](comparisons/2026-06-11_gate3_1d_predicted_event_mixture_summary.md) | Predicted top-4 recovers most oracle-event best-of-K coverage, but selection/readout remains the next bottleneck. |
| 2026-06-11 | [Gate 3.1e action head summary](comparisons/2026-06-11_gate3_1e_action_head_summary.md) | Naive action-head consumption of predicted event mixtures is better than shuffled controls but worse than the unconditional real sample-set baseline. |
| 2026-06-12 | [Gate 3.1f event-aware sample consumption summary](comparisons/2026-06-12_gate3_1f_event_aware_sample_consumption_summary.md) | Event identity/rank/probability at the sample-token level turns predicted event mixtures into the current best deployable action-head interface. |
| 2026-06-12 | [Gate 3.1g event metadata ablation summary](comparisons/2026-06-12_gate3_1g_event_metadata_ablation_summary.md) | Metadata ablations show that both aligned event identity and event rank/probability are needed for the best result. |
| 2026-06-12 | [Gate 3.2a group stress summary](comparisons/2026-06-12_gate3_2a_group_stress_summary.md) | Global performance is stable, but transition windows expose gripper/open-close timing as the next bottleneck. |
| 2026-06-12 | [Gate 3.2b transition-weighted summary](comparisons/2026-06-12_gate3_2b_transition_weighted_summary.md) | Simple transition loss weighting confirms the bottleneck is trainable but creates an overall/sustain trade-off. |
| 2026-06-12 | [Gate 3.2c auxiliary gripper summary](comparisons/2026-06-12_gate3_2c_aux_gripper_summary.md) | Separate gripper regression is not enough; the next branch should use event routing or transition-gated residuals. |
| 2026-06-15 | [Gate 3.4 controlled joint temporal decoder summary](comparisons/2026-06-15_gate3_4_controlled_joint_temporal_decoder_summary.md) | Joint temporal decoding gives a small controlled gain, while mean replacement shows that stronger sample-diversity usage remains the next issue. |
| 2026-06-16 | [Gate 3.4b sample-diversity usage summary](comparisons/2026-06-16_gate3_4b_sample_diversity_usage_summary.md) | The full-aligned checkpoint uses K-sample diversity at runtime, but the next step should make that usage stronger under matched training controls. |
| 2026-06-16 | [Gate 3.4c sample-score temporal regret summary](comparisons/2026-06-16_gate3_4c_sample_score_temporal_regret_summary.md) | Motion-regret sample scoring does not convert runtime diversity use into a better action decoder; next mainline should test a small flow/diffusion residual decoder with the same controls. |
| 2026-06-16 | [Gate 3.5a small residual flow decoder summary](comparisons/2026-06-16_gate3_5a_small_residual_flow_decoder_summary.md) | Minimal joint residual flow training does not beat Gate 3.4; the next clean branch is a post-hoc residual adapter over a frozen Gate 3.4 checkpoint. |
| 2026-06-16 | [Gate 3.5b post-hoc residual adapter summary](comparisons/2026-06-16_gate3_5b_posthoc_residual_adapter_summary.md) | Post-hoc residual adaptation is overall-positive, but context-only nearly catches up and full-aligned transition MSE is worse than Gate 3.4. |
| 2026-06-16 | [Gate 3.5c transition-constrained residual summary](comparisons/2026-06-16_gate3_5c_transition_constrained_posthoc_residual_summary.md) | Deployable transition-probability gating does not fix transition/gripper timing; pivot next to upstream event/transition candidate quality while keeping the same controls. |
| 2026-06-17 | [Gate 3.6a transition-reserve candidate quality summary](comparisons/2026-06-17_gate3_6a_transition_reserve_candidate_quality_summary.md) | Transition gains are real but post-hoc corrected to transition-MSE checkpoint selection rather than reserve-trigger candidate replacement. |
| 2026-06-17 | [Gate 3.6b/3.6c candidate-vs-selection attribution summary](comparisons/2026-06-17_gate3_6b_3_6c_candidate_vs_selection_attribution_summary.md) | Fixed-threshold reserve is inert at top-4, while top-k plus transition-based selection recreates the 3.6a trade-off exactly. |
