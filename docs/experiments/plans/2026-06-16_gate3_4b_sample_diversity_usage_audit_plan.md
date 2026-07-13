# Gate 3.4b Sample-Diversity Usage Audit Plan

## Purpose

Gate 3.4 showed a controlled small positive:

```text
full aligned temporal_action_mse = 0.034262
mean_repeated temporal_action_mse = 0.034414
context_only temporal_action_mse = 0.036642
```

The full aligned branch beats same-capacity no-prior and metadata controls, so
GeoMoCo-WM still contributes. But `mean_repeated` is close, so attribution to
K-sample diversity is thin.

Gate 3.4b is a diagnostic step before training a stronger decoder.

## Main Question

Does the Gate 3.4 temporal action decoder actually use the K-sample future
motion distribution, or mostly the per-window mean plus event metadata?

## Audit Variants

Run the trained Gate 3.4 full aligned checkpoints under eval-time variants:

| variant | purpose |
| --- | --- |
| original | normal full aligned K samples |
| permuted | sanity check set-order invariance |
| mean_repeated | remove motion-sample diversity |
| rank1_only | keep only the top predicted event/rank sample family |
| rank1_repeated | repeat the rank-1 mean/sample to K slots |
| subset_k4 | test whether fewer diverse samples preserve value |
| drop_rank1 | test dependence on the top event/rank |
| transition_rank_repeated | collapse transition-rank samples to their mean and repeat it |
| batch_mismatch | destructive alignment control |

For all variants, report both base actions and `temporal_actions` when present.

## Metrics

Overall action metrics:

```text
mse
gripper_mse
transition_mse
sustain_mse
translation_m_mse
rotation_geodesic_deg
```

Usage metrics:

```text
delta/original_vs_variant/action_l2
delta/original_vs_variant/mse
sample/pair_l2
sample/gripper_pair_l2
sample/to_mean_l2
```

Oracle/value diagnostics:

```text
single_sample/base_best_mse
single_sample/base_mean_mse
single_sample/base_best_vs_mean_gap
single_sample/temporal_best_mse
single_sample/temporal_mean_mse
single_sample/temporal_best_vs_mean_gap
```

Window groups:

```text
transition vs sustain
event_family transition_close/open/sustain
sample_diversity tertiles
best_vs_mean_gap tertiles
event_probability_entropy tertiles
```

## Decision Logic

Evidence that K-sample diversity matters:

```text
original beats mean_repeated especially on transition windows;
original beats mean_repeated in high best-vs-mean-gap windows;
original output changes meaningfully when samples are collapsed to mean;
batch_mismatch destroys performance;
permutation does not change output.
```

Evidence that diversity is not yet used:

```text
original and mean_repeated are nearly identical in high-gap windows;
output delta original-vs-mean is tiny;
rank1-only or mean_repeated matches original;
oracle best-vs-mean gap is large but decoder does not exploit it.
```

## Stop / Next Step

If diversity usage is weak, do not jump straight to a large flow/diffusion
decoder. First try a controlled Gate 3.4c objective:

```text
set-wise temporal regret/rank supervision
or candidate-comparison auxiliary loss
```

If diversity usage is already strong but overall MSE remains bottlenecked, then
move to a small flow/diffusion residual decoder with the same controls.

## Executed Artifacts

```text
scripts/audit_predicted_event_mixture_action_head_usage.py
tests/test_predicted_event_mixture_action_head_usage_audit.py
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/usage_audit.json
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/usage_audit.json
docs/experiments/runs/2026-06-16_gate3_4b_sample_diversity_usage_audit.md
docs/experiments/comparisons/2026-06-16_gate3_4b_sample_diversity_usage_summary.md
```
