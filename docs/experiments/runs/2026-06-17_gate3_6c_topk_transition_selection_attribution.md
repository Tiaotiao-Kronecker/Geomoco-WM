# Gate 3.6c Top-k Transition-Selection Attribution Check

- Date: 2026-06-17
- Status: completed, attribution-correcting
- Gate: 3.6c
- Purpose: isolate whether Gate 3.6a's transition gain came from
  `transition_reserve` candidate allocation or from selecting checkpoints by
  transition MSE.

## Question

Gate 3.6b showed that `transition_reserve` triggers zero replacements at
`top_m=4`, because top-4 predicted event candidates already contain a
transition class for every train and validation window. Gate 3.6c therefore
keeps the original top-k candidate policy and changes only the selection metric:

```text
event_candidate_policy: topk
selection_metric: temporal_action_transition_mse
```

If Gate 3.6c reproduces Gate 3.6a, the Gate 3.6a transition result should be
attributed to transition-based checkpoint selection, not candidate reserve.

## Config

```text
script: scripts/train_predicted_event_mixture_action_head.py
eval: scripts/evaluate_predicted_event_mixture_action_head.py
event_candidate_policy: topk
event_top_m: 4
num_samples: 16
sample_feature_mode: event_rank_prob
temporal_action_decoder_mode: sequence_mlp
temporal_action_loss_weight: 1.0
selection_metric: temporal_action_transition_mse
epochs: 20
batch size: 64
seeds: 7, 17
device: cuda
```

Best epochs:

| branch | seed 7 | seed 17 |
| --- | ---: | ---: |
| Gate 3.4 top-k, sel=mse | 17 | 20 |
| Gate 3.6a reserve, sel=transition | 16 | 14 |
| Gate 3.6c top-k, sel=transition | 16 | 14 |

## Results

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 top-k, sel=mse | 0.034262 | 0.131311 | 0.022542 | 0.149383 |
| Gate 3.6a reserve t=0.15, sel=transition | 0.036391 | 0.126421 | 0.025539 | 0.159475 |
| Gate 3.6c top-k, sel=transition | 0.036391 | 0.126421 | 0.025539 | 0.159475 |

Base action output shows the same pattern:

| branch | MSE | transition MSE | sustain MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.4 top-k, sel=mse | 0.034303 | 0.131697 | 0.022542 | 0.149826 |
| Gate 3.6a reserve t=0.15, sel=transition | 0.036272 | 0.126023 | 0.025451 | 0.157966 |
| Gate 3.6c top-k, sel=transition | 0.036272 | 0.126023 | 0.025451 | 0.157966 |

## Interpretation

Gate 3.6c exactly reproduces Gate 3.6a while using the original top-k
candidate policy. Combined with the zero-trigger diagnostic from Gate 3.6b,
this corrects the Gate 3.6a attribution:

```text
Observed transition gain: real.
Cause: transition-MSE checkpoint selection / early stopping.
Not supported: transition_reserve candidate replacement at top_m=4.
```

The trade-off is also clear. Selecting by transition MSE improves transition
windows by about `+0.004890` versus Gate 3.4, but worsens:

```text
overall temporal MSE: 0.034262 -> 0.036391
sustain MSE:          0.022542 -> 0.025539
gripper MSE:          0.149383 -> 0.159475
```

So this is a useful diagnostic lever, not a deployable default.

## Artifacts

```text
outputs/motion_prior_action_head/gate3_6c_topk_seltransition_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_6c_topk_seltransition_top4_k16_seed17/
```

## Decision

Do not promote Gate 3.6a/3.6c. The next upstream branch should not spend more
budget on the current reserve rule. A cleaner next step is one of:

```text
1. keep top-k but add explicit Pareto selection/reporting for transition vs
   sustain, treating it as checkpoint-selection analysis rather than a new
   candidate policy;
2. design a candidate policy that actually changes top-4 composition, such as
   transition-timing diversity or multiple transition slots;
3. score candidates by downstream action/transition regret before forcing them
   into the sample set.
```
