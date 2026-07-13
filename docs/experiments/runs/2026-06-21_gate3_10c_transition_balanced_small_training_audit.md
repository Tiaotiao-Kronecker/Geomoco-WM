# Gate 3.10c Transition-Balanced Small Training Audit

## Purpose

Test whether the unreliable transition/open-close timing gains from Gate 3.10b
can become reliable under a matched transition-balanced training protocol on
the existing 2-files slice.

This deliberately does not use full LIBERO training yet. The question is:

```text
Can we make transition-sliced CI reliably positive before spending full-data
DINO/cache/model budget?
```

## Code Change

Added a training-only sampler to:

```text
scripts/train_predicted_event_mixture_action_head.py
```

New arguments:

```text
--train-sampling-mode natural|transition_balanced
--transition-sampling-fraction 0.5
```

The sampler uses `WeightedRandomSampler` on the train split only. Validation
stays the natural episode-level split.

For seed 7 on the existing 2-files slice:

```text
train windows:                 13,086
natural train transition frac:  0.100260
target sampled transition frac: 0.500000
```

Test coverage added in:

```text
tests/test_motion_prior_action_head.py
```

## Matched Branches

All branches use the same transition-balanced training protocol:

```text
temporal_action_decoder_mode = sequence_mlp
temporal_action_loss_weight  = 1.0
selection_metric             = temporal_action_transition_mse
train_sampling_mode          = transition_balanced
transition_sampling_fraction = 0.5
seed                         = 7
```

Branches:

```text
full aligned
context-only / no-prior
mean_repeated
shuffled_event
rankprob_only
```

Artifacts:

```text
outputs/motion_prior_action_head/gate3_10c_transbal_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_10c_transbal_context_only_seed7/
outputs/motion_prior_action_head/gate3_10c_transbal_mean_repeated_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_10c_transbal_shuffled_event_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_10c_transbal_rankprob_top4_k16_seed7/
outputs/bootstrap_ci/gate3_10c_seed7_*.json
outputs/bootstrap_ci/gate3_10c_seed7_*_per_window.jsonl
```

## Aggregate Result

Validation summary:

| branch | temporal MSE | transition MSE | sustain MSE | gripper MSE | SE3 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.4 full reference | 0.036410 | 0.136756 | 0.023536 | 0.163820 | 0.015176 |
| 3.10c full aligned | 0.050925 | 0.122302 | 0.041767 | 0.203485 | 0.025498 |
| 3.10c context-only | 0.051633 | 0.109508 | 0.044208 | 0.221497 | 0.023323 |
| 3.10c mean_repeated | 0.049785 | 0.123127 | 0.040375 | 0.197326 | 0.025195 |
| 3.10c shuffled_event | 0.051097 | 0.117764 | 0.042543 | 0.207548 | 0.025021 |
| 3.10c rankprob_only | 0.074366 | 0.121751 | 0.068287 | 0.203892 | 0.052779 |

Transition-balanced training does reduce full-aligned transition MSE compared
with the Gate 3.4 full reference:

```text
0.136756 -> 0.122302
```

But the matched context-only branch is better:

```text
context-only transition MSE = 0.109508
```

So the transition improvement is not attributable to GeoMoCo prior samples.

## Episode-Bootstrap CI

Loss-style gain definition:

```text
gain = baseline - candidate
```

Positive means full aligned is better than the baseline.

### Prior Gain: Context-Only -> Full

Overall:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | +0.000709 | [-0.001716, +0.003043] | no |
| temporal_action_gripper_mse | +0.018012 | [+0.004050, +0.032110] | yes |
| temporal_action_se3_mse | -0.002175 | [-0.003345, -0.001046] | reliable negative |

Transition close:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.014570 | [-0.027292, -0.003039] | reliable negative |
| temporal_action_gripper_mse | -0.104137 | [-0.190460, -0.025717] | reliable negative |
| temporal_action_se3_mse | +0.000358 | [-0.001071, +0.001811] | no |

Transition open:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.010912 | [-0.024213, +0.001485] | no |
| temporal_action_gripper_mse | -0.055532 | [-0.135337, +0.021448] | no |
| temporal_action_se3_mse | -0.003476 | [-0.007147, -0.000108] | reliable negative |

### Diversity Gain: Mean-Repeated -> Full

Overall:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.001140 | [-0.002032, -0.000270] | reliable negative |
| temporal_action_gripper_mse | -0.006159 | [-0.010952, -0.001567] | reliable negative |
| temporal_action_se3_mse | -0.000303 | [-0.000770, +0.000159] | no |

Transition close:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.002618 | [-0.006915, +0.001878] | no |
| temporal_action_gripper_mse | -0.020965 | [-0.049843, +0.007372] | no |

Transition open:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | +0.004589 | [+0.000737, +0.008404] | yes |
| temporal_action_gripper_mse | +0.023692 | [+0.002225, +0.044527] | yes |

The diversity signal is asymmetric: positive on open transitions, not close,
and negative overall.

### Metadata Gain: Shuffled-Event -> Full

Transition close:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.003300 | [-0.005909, -0.000860] | reliable negative |
| temporal_action_gripper_mse | -0.023715 | [-0.041400, -0.007203] | reliable negative |

Transition open:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| temporal_action_mse | -0.005660 | [-0.010728, -0.001081] | reliable negative |
| temporal_action_gripper_mse | -0.033765 | [-0.062422, -0.007108] | reliable negative |

Under transition-balanced training, aligned event metadata does not produce a
transition timing gain over shuffled metadata. It is reliably worse on the
transition action/gripper metrics in this seed-7 audit.

## Interpretation

Gate 3.10c answers the small-training question negatively.

Transition-balanced training can force the model toward lower transition MSE,
but matched controls show that this is not a clean GeoMoCo-WM attribution:

```text
context-only beats full aligned on transition MSE;
mean_repeated beats full aligned overall;
shuffled_event beats full aligned on close/open transition action/gripper MSE.
```

So the improvement is mainly a training distribution / checkpoint-selection
effect, not evidence that GeoMoCo motion-prior samples or aligned event
metadata solve open-close timing.

This also explains why blind full-data training is risky: more transition data
may improve the absolute transition number, but the current controlled result
says the benefit may still be eaten by a strong context/action decoder rather
than attributable to GeoMoCo-WM.

## Decision

Do not spend full-data DINO/cache/model retraining budget yet.

The next mainline should pivot from "more data for the same decoder" to a more
diagnostic question:

```text
What signal is missing for deployable open-close timing?
```

Recommended next small step:

```text
Gate 3.10d contact/command-transition label audit:
  separate close/open command-transition windows by object/contact/task phase,
  inspect whether current gripper-command labels are noisy or misaligned with
  actual manipulation success/contact,
  and only then decide whether full-data training or a new event/contact
  supervision source is worth it.
```

Alternative if the goal is purely engineering performance:

```text
train a transition-focused context-only/action policy as a strong baseline,
then treat GeoMoCo-WM as a geometry/event auxiliary rather than the main
transition decoder.
```

## Verification

Checks run:

```text
.venv/bin/python -m unittest tests.test_motion_prior_action_head
.venv/bin/ruff check scripts/train_predicted_event_mixture_action_head.py tests/test_motion_prior_action_head.py
```

GPU training completed for five matched Gate 3.10c branches. CPU per-window
eval and episode-bootstrap CI completed for all five branches.
