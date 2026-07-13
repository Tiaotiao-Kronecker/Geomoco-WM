# Gate 3.6b Pareto Transition Candidate Allocation Sweep

- Date: 2026-06-17
- Status: completed, negative / attribution-correcting
- Gate: 3.6b
- Purpose: test whether a more conservative transition-reserve threshold can
  recover the Gate 3.4 overall/sustain Pareto point while keeping the Gate 3.6a
  transition gain.

## Motivation

Gate 3.6a appeared to improve transition MSE:

```text
Gate 3.4 transition MSE: 0.131311
Gate 3.6a transition MSE: 0.126421
```

But it regressed overall and sustain:

```text
Gate 3.4 temporal MSE: 0.034262
Gate 3.6a temporal MSE: 0.036391
Gate 3.4 sustain MSE: 0.022542
Gate 3.6a sustain MSE: 0.025539
```

Gate 3.6b kept the decoder fixed and swept more conservative reserve
thresholds while selecting checkpoints by overall `temporal_action_mse`.

## Config

```text
script: scripts/train_predicted_event_mixture_action_head.py
eval: scripts/evaluate_predicted_event_mixture_action_head.py
event_candidate_policy: transition_reserve
thresholds: 0.25, 0.35
event_top_m: 4
num_samples: 16
sample_feature_mode: event_rank_prob
temporal_action_decoder_mode: sequence_mlp
temporal_action_loss_weight: 1.0
selection_metric: temporal_action_mse
epochs: 20
batch size: 64
seeds: 7, 17
device: cuda
```

## Results

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 top-k, sel=mse | 0.034262 | 0.131311 | 0.022542 | 0.149383 |
| Gate 3.6a reserve t=0.15, sel=transition | 0.036391 | 0.126421 | 0.025539 | 0.159475 |
| Gate 3.6b reserve t=0.25, sel=mse | 0.034262 | 0.131311 | 0.022542 | 0.149383 |
| Gate 3.6b reserve t=0.35, sel=mse | 0.034262 | 0.131311 | 0.022542 | 0.149383 |

The 0.25 and 0.35 threshold branches exactly match the Gate 3.4 top-k
reference. They pass the overall/sustain guard, but fail the transition guard:

```text
required transition MSE < 0.131311
observed transition MSE = 0.131311
```

Therefore no matched controls were expanded.

## Trigger Diagnostic

Post-hoc diagnostics showed why the threshold sweep was inert. The current
`transition_reserve` implementation only replaces a top-M slot when top-M has
no transition candidate. For the Gate 3.1b predicted event probe, top-4 already
contains at least one transition candidate for every train and validation
window.

| seed | split | total | true transition windows | top-4 without transition | reserve triggers at 0.05/0.10/0.15/0.25/0.35 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | train | 12974 | 1296 | 0 | 0 |
| 7 | val | 3544 | 403 | 0 | 0 |
| 17 | train | 13305 | 1374 | 0 | 0 |
| 17 | val | 3213 | 325 | 0 | 0 |

This means Gate 3.6b did not alter candidate sets relative to Gate 3.4, and
the identical metrics are expected.

## Interpretation

Gate 3.6b falsifies the original threshold-sweep hypothesis. The issue is not
that threshold 0.15 was too permissive and 0.25/0.35 were more Pareto-aware.
The reserve rule simply did not trigger at top-4 on this split.

The likely cause of Gate 3.6a's transition improvement is not candidate
replacement. The remaining changed variable is checkpoint selection:

```text
Gate 3.4 / 3.6b: selection_metric = temporal_action_mse
Gate 3.6a:        selection_metric = temporal_action_transition_mse
```

Gate 3.6c tests this directly with top-k candidates and transition-MSE
checkpoint selection.

## Artifacts

```text
outputs/motion_prior_action_head/gate3_6b_transition_reserve_t025_selmse_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_6b_transition_reserve_t025_selmse_top4_k16_seed17/
outputs/motion_prior_action_head/gate3_6b_transition_reserve_t035_selmse_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_6b_transition_reserve_t035_selmse_top4_k16_seed17/
```

## Decision

Do not continue fixed-threshold transition-reserve tuning at `top_m=4`.
Candidate-allocation work should only resume if the policy actually changes the
candidate set, for example by requiring multiple transition slots, comparing
transition timing diversity, or using candidate action regret. The immediate
cleaner finding is Gate 3.6c: transition-based checkpoint selection recreates
the 3.6a trade-off without changing candidates.
