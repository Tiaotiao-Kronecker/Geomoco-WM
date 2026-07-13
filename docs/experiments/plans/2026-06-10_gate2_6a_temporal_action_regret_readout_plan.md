# Gate 2.6a Temporal Action-Regret Readout Plan

- Date: 2026-06-10
- Gate: Gate 2.6a
- Status: executed

## Objective

Test whether the Gate 2.5d readout bottleneck is caused by the shallow
flattened MLP ScoreNet. Keep the joint GeoMoCo-cVAE and frozen action decoder
unchanged, and replace only the readout architecture with a temporal scorer.

## Main Question

Can a temporal scorer over candidate future-motion and decoded-action chunks
select better cVAE samples than the Gate 2.5d flat ScoreNet?

## Architecture

```text
future_delta_gripper sample m_k: [H, 7]
decoded action chunk a_k:        [H, 7]
condition c:                     visual/proprio/task condition embedding

[m_k step, a_k step] over H
  -> step embedding
  -> TransformerEncoder
  -> mean/last temporal pooling
  -> concat condition embedding + summary features
  -> MLP score head
  -> scalar score_k
```

The module is still a readout/scorer. It does not train the cVAE and does not
output actions directly.

## Dataset And Controls

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| windows | 16,518 |
| horizon | 8 |
| motion mode | `future_delta_gripper` |
| sample count | 16 |
| seeds | 7, 17 |

## Promotion Criteria

- beat Gate 2.5d flat ScoreNet mean action MSE `0.043414`;
- preferably beat deterministic joint baseline `0.040688`;
- improve selected oracle rank without event collapse;
- if positive, run shuffled-visual controls.

## Outcome

The temporal scorer did not pass promotion. It slightly improved selected rank
and transition step-within-1, but mean action MSE regressed from `0.043414` to
`0.043636`.

