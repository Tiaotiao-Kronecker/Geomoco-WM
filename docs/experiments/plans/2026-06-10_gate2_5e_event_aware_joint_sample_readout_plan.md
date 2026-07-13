# Gate 2.5e Event-Aware Joint Sample Readout Plan

- Date: 2026-06-10
- Gate: Gate 2.5e-a
- Status: pilot completed

## Objective

Improve the deployable readout over joint GeoMoCo-cVAE samples by training the
ScoreNet target to prefer samples that are both action-useful and aligned with
future gripper transition timing.

## Main Question

Can explicit transition-event ranking close more of the gap between:

```text
prior mean action MSE: 0.043816
flat ScoreNet action MSE: 0.043414
oracle best-of-K action MSE: 0.022192
deterministic joint baseline: 0.040688
```

without losing the real-vs-shuffled visual attribution established in Gate 2.5d?

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| motion mode | `future_delta_gripper` |
| split policy | episode |
| seeds | 7, 17 |
| sample count | 16 |

## Scorer Target

The current Gate 2.5d scorer uses:

```text
target_score_k = standardized negative flat action MSE(a_k, a_gt)
```

Gate 2.5e-a keeps that action target and adds transition-event alignment:

```text
target_score_k =
  standardized negative flat action MSE(a_k, a_gt)
  + event_target_weight
    * standardized negative event_alignment_error(event_k, event_gt)
```

The event label is the transition label from Gate 2.4h-a:

```text
close_transition / open_transition / mixed_transition
sustain_close / sustain_open / hold
```

The alignment error penalizes wrong event type first, then wrong transition
step. This is still a readout target, not a new action head.

## Sweep

Run real visual and shuffled visual controls with:

```text
event_target_weight in {0.05, 0.10, 0.30}
seeds in {7, 17}
```

Start from the existing Gate 2.5c joint cVAE checkpoints and the same frozen
`future_delta_gripper` action decoder.

## Promotion Criteria

Promote only if all hold:

- real visual scorer beats Gate 2.5d flat scorer action MSE `0.043414`;
- ideally beats deterministic joint baseline `0.040688`;
- real visual remains better than shuffled visual on action MSE;
- event metrics improve without being the only gain;
- selected rank closes a meaningful part of the oracle best-of-K gap.

## Stop Criteria

If event-aware readout only improves event metrics while action MSE regresses,
do not promote it. Move to Gate 2.5e-b hard-negative ranking or Gate 2.5e-c
executability/contact proxy scoring.

## Pilot Outcome

The seed-7 pilot hit the stop criterion.

Event-aware and event-hard-negative variants improved event metrics but
regressed action MSE relative to the Gate 2.5d flat scorer. The full result is
archived in:

```text
docs/experiments/runs/2026-06-10_gate2_5e_event_aware_joint_readout_pilot.md
docs/experiments/comparisons/2026-06-10_gate2_5e_event_readout_vs_flat.md
```

