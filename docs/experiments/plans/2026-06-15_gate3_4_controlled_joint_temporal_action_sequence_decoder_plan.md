# Gate 3.4 Controlled Joint Temporal Action-Sequence Decoder Plan

## Purpose

Gate 3.3 showed that a shallow additive gripper-only temporal residual is not
enough. It made the decoder slightly stronger in plumbing, but not in metrics:
trajectory-routed output worsened the same checkpoint's base output.

Gate 3.4 tests a qualitatively different but still controlled decoder:

```text
Decode the whole action sequence temporally and jointly,
while keeping the Gate 3.1f/g GeoMoCo-WM sample interface fixed.
```

This is deliberately not a flow/diffusion decoder yet. The first step is a
deterministic temporal sequence readout whose attribution is easier to audit.

## Hypothesis

Close/open transition failures may require joint temporal consistency between
EEF motion and gripper commands. A gripper-only additive residual cannot change
the relation between translation/rotation trajectory and gripper trajectory.

Gate 3.4 asks whether a small joint sequence decoder can use the same aligned
event-mixture motion-prior samples better than the flat MLP head.

## Minimal Model Change

Keep the existing base action head:

```text
features -> base actions [B,H,A]
```

Add one optional temporal action-sequence branch:

```text
features -> learned horizon query tokens -> small Transformer/MLP temporal block
        -> temporal_actions [B,H,A]
```

First implementation:

```text
mode: temporal_action_decoder_mode=sequence_mlp
loss: action loss on temporal_actions
selection metric: temporal_action_mse or temporal_action_transition_mse
```

The branch is joint over all action dimensions. It is not allowed to remove or
overwrite the base output; evaluation must report both base and
`temporal_action_*` metrics.

## Attribution Controls

Only run the full matrix if the aligned branch beats its own base output and
the Gate 3.1f/g reference.

Required controls for any positive result:

| control | purpose |
| --- | --- |
| full event/rank/prob samples | main aligned GeoMoCo-WM interface |
| shuffled event metadata | tests aligned event identity |
| rank/prob-only metadata | separates event confidence/order from event identity |
| mean replacement | tests sample diversity usage |
| context-only/no-prior decoder | tests whether the decoder alone explains the gain |
| same decoder capacity without motion-prior samples | parameter-count control |

Promotion requires:

```text
full aligned GeoMoCo-WM samples > base output
full aligned GeoMoCo-WM samples > Gate 3.1f/g reference
full aligned GeoMoCo-WM samples > shuffled metadata
full aligned GeoMoCo-WM samples > context/no-prior
```

## First Executable Slice

Start with plumbing smoke:

```text
seed 7
max_windows=128
epochs=1
CPU
selection_metric=temporal_action_mse
```

Then short-budget main branch:

```text
seeds 7 and 17
top-4 predicted event mixture
K=16
sample_feature_mode=event_rank_prob
epochs=20
GPU
```

Run repeated eval and group audit before deciding whether controls are worth
expanding.

## Metrics

Primary:

```text
temporal_action_mse
temporal_action_gripper_mse
temporal_action_transition_mse
temporal_action_sustain_mse
```

Secondary:

```text
base mse/gripper/transition/sustain
physical translation/rotation metrics
group audit by transition/sustain and event family
```

## Stop Conditions

Stop before controls if:

```text
temporal_action_mse is worse than same checkpoint base MSE on both seeds;
temporal_action_transition_mse is worse than same checkpoint base transition MSE;
the branch improves only geometry while worsening gripper/transition;
training becomes unstable or produces non-finite metrics.
```

## Expected Interpretation

Positive:

```text
The GeoMoCo-WM event-aware sample interface is useful, but consuming it needs
a joint temporal action-sequence decoder rather than scalar gripper residuals.
```

Negative:

```text
The bottleneck is not solved by deterministic temporal decoder capacity alone;
move to stricter contact/event supervision or a controlled flow/diffusion
action residual with the same attribution matrix.
```
