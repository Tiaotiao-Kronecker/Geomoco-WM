# Gate 3.0c Sample-Set Usage Audit

## Purpose

Audit whether the Gate 3.0 action head actually uses multimodal future-motion
sample sets, or mostly collapses them to a mean statistic.

## Code

- Audit script: `scripts/audit_motion_prior_action_head_usage.py`
- Tests: `tests/test_motion_prior_action_head_audit.py`

## Checkpoints

Real visual sample-set action heads:

```text
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/model.pt
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed17/model.pt
```

Shuffled visual sample-set action heads:

```text
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed7/model.pt
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed17/model.pt
```

## Command

Example:

```bash
.venv/bin/python scripts/audit_motion_prior_action_head_usage.py \
  --checkpoint outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_0c_usage_audit_real_k16_seed7.json \
  --num-eval-passes 3 \
  --device cuda \
  --seed 3007
```

## Artifacts

```text
outputs/motion_prior_action_head/gate3_0c_usage_audit_real_k16_seed7.json
outputs/motion_prior_action_head/gate3_0c_usage_audit_real_k16_seed17.json
outputs/motion_prior_action_head/gate3_0c_usage_audit_shuffled_k16_seed7.json
outputs/motion_prior_action_head/gate3_0c_usage_audit_shuffled_k16_seed17.json
```

## Variant Metrics

Mean across seeds:

| source | original | mean repeated | subset K=4 | batch mismatch | damage mean-original | damage subset-original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real visual cVAE | 0.037061 | 0.041893 | 0.040585 | 0.317347 | 0.004832 | 0.003524 |
| shuffled visual cVAE | 0.042665 | 0.053407 | 0.048454 | 0.221706 | 0.010742 | 0.005789 |

Permutation invariance passed:

```text
delta/original_vs_permuted/action_l2 ~= 2.4e-7
```

This means the set head is order-invariant as intended.

## Diversity Metrics

Mean across seeds:

| source | sample pair L2 | gripper pair L2 | single-sample action-to-mean L2 | best single-sample action MSE |
| --- | ---: | ---: | ---: | ---: |
| real visual cVAE | 0.611503 | 0.599218 | 0.404625 | 0.019049 |
| shuffled visual cVAE | 1.385104 | 1.369719 | 0.729488 | 0.023408 |

Shuffled samples are much more diverse, but this diversity is not aligned with
the task/context and leads to worse action prediction. Real samples are less
spread out but more usable.

## Event Sensitivity

Transition windows are much harder than no-transition windows:

| source | transition original | no-transition original | transition mean | no-transition mean |
| --- | ---: | ---: | ---: | ---: |
| real visual cVAE | 0.139793 | 0.026555 | 0.167666 | 0.028998 |
| shuffled visual cVAE | 0.147233 | 0.032024 | 0.175142 | 0.041012 |

Replacing the sample set by its mean hurts transition windows more than
no-transition windows. This supports the idea that sample-set information is
especially useful around gripper/event timing.

## Interpretation

Gate 3.0c answers the main concern from Gate 3.0b:

```text
The action head is not merely using a mean feature.
```

Evidence:

- `original` beats `mean_repeated`;
- `original` beats a K=4 subset;
- `batch_mismatch` is much worse;
- permutation changes almost nothing;
- real samples beat shuffled samples even though shuffled samples have larger
  raw diversity.

The more precise conclusion is:

```text
Aligned, moderate sample diversity is useful; unaligned large diversity is harmful.
```

This strengthens the GeoMoCo-WM story. The value is not simply "more futures";
it is visually grounded, context-aligned future-motion hypotheses that the
downstream action head can use.

## Next Decision

The next mainline should not be another set-readout tweak. A better next step is
to make the multimodal structure more explicit:

```text
Gate 3.1: mode-structured / event-aware future-motion prior
```

Possible directions:

- expose event/transition mode labels as diagnostics or weak supervision;
- separate gripper-event timing modes from continuous EEF motion modes;
- evaluate mode coverage on LIBERO-10 and transition-heavy windows;
- keep real-vs-shuffled controls and action-head usage audits as standard
  checks.
