# Gate 3.2a Group Stress Audit

## Purpose

Gate 3.1g promoted the predicted event-mixture action-head interface:

```text
context
+ predicted top-4 event-conditioned cVAE samples
+ per-sample event identity/rank/probability
-> action chunk
```

Gate 3.2a asks whether this interface is robust across semantically meaningful
subgroups, rather than only looking good in the global validation mean.

First-principles questions:

```text
1. Does the promoted interface reproduce the Gate 3.1f/3.1g overall result?
2. Is the error distributed evenly across suites/tasks/event modes?
3. If a group is weak, is the weakness geometric SE(3) motion or gripper/event
   timing?
```

## Method

This is an evaluation-only stress audit. It does not retrain the cVAE, event
probe, or action head.

For each validation batch:

```text
visual/proprio context
-> event probe predicts top-4 event modes
-> event-conditioned cVAE samples K=16 future_delta_gripper proposals
-> action head consumes samples plus event/rank/prob metadata
-> predicted action chunk
```

The audit then reports action metrics by:

```text
all windows
suite
task
true event mode
event family
transition vs sustain group
```

## Code

```text
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_predicted_event_mixture_action_head_group_audit.py
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_1f_eventaware_top4_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2a_group_stress_eventaware_top4_seed7/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

Seed 17:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_1f_eventaware_top4_k16_seed17/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2a_group_stress_eventaware_top4_seed17/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Overall Result

Mean over seeds 7 and 17, each with 3 stochastic evaluation passes:

| metric | value |
| --- | ---: |
| action MSE | 0.034773 |
| action MAE | 0.089707 |
| translation m MSE | 0.00007225 |
| rotation geodesic deg | 1.967991 |
| gripper MSE | 0.150159 |

The overall result reproduces Gate 3.1f/3.1g:

```text
Gate 3.1f full event/rank/prob top-4 action MSE: 0.034767
Gate 3.2a group audit overall action MSE:        0.034773
```

So the new audit does not change the promoted interface result; it reveals
where that result is fragile.

## Group Results

Mean over seeds:

| group | count | action MSE | gripper MSE | translation m MSE | rotation geodesic deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 3378.5 | 0.034773 | 0.150159 | 0.00007225 | 1.967991 |
| `transition_group/sustain` | 3014.5 | 0.022793 | 0.068512 | 0.00007037 | 1.978624 |
| `transition_group/transition` | 364.0 | 0.134087 | 0.827336 | 0.00008759 | 1.878719 |
| `event_family/transition_open` | 173.5 | 0.150220 | 0.898143 | 0.00012078 | 2.251408 |
| `event_family/transition_close` | 186.5 | 0.118580 | 0.758840 | 0.00005604 | 1.520447 |
| `event_mode/transition_open::early` | 68.0 | 0.166592 | 0.998885 | 0.00013298 | 2.175561 |
| `event_mode/transition_open::middle` | 66.5 | 0.149313 | 0.905292 | 0.00010855 | 2.372039 |
| `event_mode/transition_open::late` | 39.0 | 0.123820 | 0.712172 | 0.00012223 | 2.189856 |
| `event_mode/transition_close::early` | 72.5 | 0.126030 | 0.818247 | 0.00005026 | 1.481649 |
| `event_mode/transition_close::middle` | 74.0 | 0.122907 | 0.791232 | 0.00005409 | 1.549063 |
| `event_mode/transition_close::late` | 40.0 | 0.096943 | 0.588338 | 0.00007181 | 1.542949 |
| `suite/libero_10` | 1299.5 | 0.031574 | 0.149995 | 0.00005332 | 2.078580 |
| `suite/libero_goal` | 779.0 | 0.029183 | 0.073263 | 0.00010169 | 2.334626 |
| `suite/libero_object` | 851.0 | 0.041072 | 0.203036 | 0.00006706 | 1.526487 |
| `suite/libero_spatial` | 449.0 | 0.042584 | 0.186362 | 0.00008850 | 1.827929 |

## Interpretation

The promoted interface is globally stable but transition-fragile.

The largest split is not between suites; it is between sustain and transition
windows:

```text
sustain action MSE:    0.022793
transition action MSE: 0.134087
```

The transition penalty is primarily gripper/event timing:

```text
sustain gripper MSE:    0.068512
transition gripper MSE: 0.827336
```

The SE(3) geometry does not show an equally large collapse:

```text
sustain translation m MSE:    0.00007037
transition translation m MSE: 0.00008759

sustain rotation geodesic deg:    1.978624
transition rotation geodesic deg: 1.878719
```

This means the current bottleneck is not just geometric future motion quality.
The action head can consume event-structured proposals in the aggregate, but it
still struggles when the correct action depends on precise open/close timing.

Open transitions are the hardest subgroup:

```text
transition_open action MSE: 0.150220
transition_close action MSE: 0.118580
```

## Decision

Gate 3.2a passes the global reproducibility check and fails the subgroup
stress test in an informative way.

Promote the following diagnosis:

```text
event-aware top-4 is the best current deployable interface,
but the next mainline bottleneck is transition/open-close timing,
not generic set aggregation or SE(3) geometry.
```

Next mainline:

```text
Gate 3.2b: transition-focused stress branch.
```

Possible Gate 3.2b routes:

```text
1. transition-balanced action-head training or loss weighting;
2. explicit transition-timing auxiliary target at the action-head output;
3. compare against an oracle-transition metadata upper bound;
4. keep flow/diffusion action heads postponed until transition timing is
   isolated under the deterministic action-head protocol.
```
