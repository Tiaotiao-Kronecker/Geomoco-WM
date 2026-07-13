# Gate 3.10b Per-Window Bootstrap CI Audit

## Purpose

Make GeoMoCo-WM attribution claims testable by episode-bootstrap confidence
intervals instead of only reporting aggregate validation means.

This implements the next reliability step after Gate 3.10a:

```text
1. add per-window metric export to Gate 3.4 / Gate 3.9 evaluators;
2. compare key controls with episode-bootstrap CI;
3. decide whether to spend full-data DINO/cache/training budget.
```

## Code Changes

Added shared per-window action metric helpers:

```text
src/geomoco_wm/metrics/window_metrics.py
tests/test_window_metrics.py
```

Extended Gate 3.4 action-head repeated eval:

```text
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/train_predicted_event_mixture_action_head.py
```

New optional argument:

```text
--per-window-output-jsonl path/to/per_window.jsonl
```

Extended Gate 3.9 flow eval-only path:

```text
scripts/train_flow_matching_action_policy.py
```

New optional argument:

```text
--per-window-output-jsonl path/to/per_window.jsonl
```

Per-window export currently requires:

```text
--num-eval-passes 1
```

This avoids duplicate stochastic rows for the same window in bootstrap inputs.

## Metric Contract

Each per-window row includes:

```text
window_id
episode_id
task_id
suite_name
event_type / event_mode / timing_bin / event_step
mse / se3_mse / gripper_mse
temporal_action_* metrics when the action-head checkpoint has temporal_actions
flow_action_* metrics when available
```

The bootstrap CI script uses:

```text
gain = baseline - candidate
```

for loss/MSE metrics, so positive gain means the candidate is better.

## Gate 3.4 CI: Existing 2-Files Slice

Artifacts are under:

```text
outputs/bootstrap_ci/
```

Per-window rows:

```text
seed 7:  3,544 validation windows per branch
seed 17: 3,613 validation windows per branch
```

Branches exported:

```text
full aligned
context-only / no-prior
mean_repeated
shuffled_event
rankprob_only
```

### Seed 7 Overall Attribution

| comparison | metric | observed gain | 95% CI | reliable? |
| --- | --- | ---: | ---: | --- |
| context-only -> full | temporal_action_mse | +0.002250 | [-0.000344, +0.004951] | no |
| context-only -> full | temporal_action_se3_mse | +0.001390 | [+0.000366, +0.002540] | yes |
| mean_repeated -> full | temporal_action_mse | +0.000115 | [-0.000774, +0.000982] | no |
| shuffled_event -> full | temporal_action_mse | +0.001069 | [+0.000182, +0.001946] | yes |
| rankprob_only -> full | temporal_action_mse | +0.001460 | [+0.000081, +0.002739] | yes |

### Seed 17 Overall Attribution

| comparison | metric | observed gain | 95% CI | reliable? |
| --- | --- | ---: | ---: | --- |
| context-only -> full | temporal_action_mse | +0.002595 | [-0.000227, +0.005547] | no |
| context-only -> full | temporal_action_se3_mse | +0.001863 | [+0.000535, +0.003445] | yes |
| mean_repeated -> full | temporal_action_mse | +0.000162 | [-0.000525, +0.000834] | no |
| shuffled_event -> full | temporal_action_mse | +0.001401 | [+0.000474, +0.002387] | yes |
| rankprob_only -> full | temporal_action_mse | +0.001771 | [+0.000843, +0.002706] | yes |

## Gate 3.4 Transition-Sliced CI

Prior gain on close/open transition slices is not stable in either seed.

| seed | slice | comparison | temporal_action_mse gain | 95% CI | reliable? |
| ---: | --- | --- | ---: | ---: | --- |
| 7 | transition_close | context-only -> full | +0.002740 | [-0.007939, +0.012959] | no |
| 7 | transition_open | context-only -> full | +0.006971 | [-0.010942, +0.025381] | no |
| 17 | transition_close | context-only -> full | +0.007305 | [-0.003883, +0.018476] | no |
| 17 | transition_open | context-only -> full | +0.005573 | [-0.010164, +0.021598] | no |

Metadata gain is stronger, especially in seed 17.

| seed | slice | comparison | temporal_action_mse gain | 95% CI | reliable? |
| ---: | --- | --- | ---: | ---: | --- |
| 7 | transition_close | shuffled_event -> full | +0.002049 | [-0.002359, +0.006648] | no |
| 7 | transition_open | shuffled_event -> full | -0.000446 | [-0.005059, +0.004334] | no |
| 17 | transition_close | shuffled_event -> full | +0.005233 | [+0.000446, +0.010437] | yes |
| 17 | transition_open | shuffled_event -> full | +0.009773 | [+0.001810, +0.018635] | yes |

Interpretation:

```text
The event metadata branch has the clearest attribution signal.
The motion-prior/context-only gain is positive in mean but not reliable
under episode-bootstrap CI on the current 2-files validation episodes.
The diversity gain from full samples over mean_repeated is tiny and unreliable.
```

## Gate 3.9 Flow CI: Existing Seed 7 Pair

Compared:

```text
direct_visual_flow -> geomoco_flow
```

Per-window rows:

```text
3,432 validation windows per branch
```

Overall:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| mse | -0.000821 | [-0.004448, +0.002554] | no |
| gripper_mse | -0.014001 | [-0.037225, +0.007423] | no |
| se3_mse | +0.001376 | [+0.000219, +0.002652] | yes |

Transition close:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| mse | -0.021522 | [-0.041921, -0.002088] | reliable negative |
| gripper_mse | -0.159771 | [-0.290523, -0.028596] | reliable negative |
| se3_mse | +0.001520 | [-0.002322, +0.005803] | no |

Transition open:

| metric | observed gain | 95% CI | reliable? |
| --- | ---: | ---: | --- |
| mse | +0.018977 | [-0.001920, +0.038687] | no |
| gripper_mse | +0.068078 | [-0.065406, +0.204411] | no |
| se3_mse | +0.010793 | [+0.003520, +0.018941] | yes |

Interpretation:

```text
GeoMoCo-conditioned flow has a reliable SE(3) gain over direct visual flow,
but it does not have a reliable overall action-MSE gain. On transition_close,
it is reliably worse for action MSE and gripper MSE.
```

This is not a promotion result for flow replacement. It is evidence that the
current flow objective/conditioning can use GeoMoCo geometry, but does not yet
convert it into robust open-close action timing.

## Decision

Do not start full-data DINO/cache/model retraining yet as a blind scale-up.

The current reliability audit says:

```text
stable:
  event metadata attribution
  SE(3)/geometry improvements

not stable:
  prior gain over context-only on action MSE
  sample-diversity gain over mean_repeated
  transition/open-close timing gain

negative:
  Gate 3.9 GeoMoCo-flow close-transition gripper/action MSE
```

The next clean step before full-data training is:

```text
Gate 3.10c: make per-window export the default reliability path for key evals,
then run a transition-balanced or transition-stratified small training audit on
the existing slice. Only promote full-data DINO/cache retraining if the model
shows a reliable transition-sliced gain under CI, or if the explicit goal is
to test whether more transition data alone fixes the instability.
```

## Verification

Checks run:

```text
.venv/bin/python -m unittest tests.test_window_metrics tests.test_bootstrap_episode_ci
.venv/bin/ruff check src/geomoco_wm/metrics/window_metrics.py scripts/evaluate_predicted_event_mixture_action_head.py scripts/train_predicted_event_mixture_action_head.py scripts/train_flow_matching_action_policy.py tests/test_window_metrics.py tests/test_bootstrap_episode_ci.py
```

CPU eval/export runs completed for:

```text
Gate 3.4 seed 7 and seed 17:
  full aligned, context-only, mean_repeated, shuffled_event, rankprob_only

Gate 3.9 seed 7:
  direct_visual_flow, geomoco_flow
```
