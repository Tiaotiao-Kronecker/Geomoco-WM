# Gate 3.4b Sample-Diversity Usage Audit

## Purpose

Gate 3.4 showed that the small joint temporal action decoder is a controlled
positive, but the separately trained `mean_repeated` control stayed close to
full aligned. Gate 3.4b asks a narrower diagnostic question:

```text
Given the trained full-aligned Gate 3.4 checkpoint, does the decoder depend on
the K-sample future-motion distribution at evaluation time?
```

This is different from retraining a mean-only control. It probes runtime sample
usage inside the same full-aligned checkpoint.

## Code

- Audit script: `scripts/audit_predicted_event_mixture_action_head_usage.py`
- Tests: `tests/test_predicted_event_mixture_action_head_usage_audit.py`

## Inputs

Full-aligned Gate 3.4 checkpoints:

```text
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/model.pt
```

Interface:

```text
predicted top-4 event/rank/prob conditioning
K=16 future-motion samples
sample_feature_mode=event_rank_prob
temporal_action_decoder_mode=sequence_mlp
```

## Commands

Smoke:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_usage.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/usage_audit_smoke.json \
  --num-eval-passes 1 \
  --max-batches 1 \
  --batch-size 16 \
  --device cpu
```

Full audit:

```bash
for seed in 7 17; do
  .venv/bin/python scripts/audit_predicted_event_mixture_action_head_usage.py \
    --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/model.pt \
    --output-json outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed${seed}/usage_audit.json \
    --num-eval-passes 3 \
    --device cuda
done
```

## Artifacts

```text
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/usage_audit_smoke.json
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/usage_audit.json
outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed17/usage_audit.json
```

## Overall Results

Mean over seeds 7 and 17, 3-pass eval, using `temporal_actions`:

| eval-time variant | temporal MSE | delta vs original |
| --- | ---: | ---: |
| original | 0.034218 | 0.000000 |
| mean_repeated | 0.042594 | +0.008376 |
| rank1_only | 0.047154 | +0.012936 |
| rank1_repeated | 0.047719 | +0.013501 |
| subset_k4 | 0.071022 | +0.036804 |
| drop_rank1 | 0.157422 | +0.123204 |
| transition_rank_repeated | 0.175443 | +0.141225 |
| batch_mismatch | 0.314274 | +0.280056 |

Sanity and usage metrics:

| metric | value |
| --- | ---: |
| `delta/original_vs_permuted/temporal/action_l2` | 0.000000263 |
| `delta/original_vs_mean_repeated/temporal/action_l2` | 0.475171 |
| `delta/original_vs_subset_k4/temporal/action_l2` | 0.808006 |
| `sample/pair_l2` | 2.693237 |
| `sample/gripper_pair_l2` | 2.687866 |
| `single_sample/temporal_best_vs_mean_gap` | 0.124484 |

Permutation is effectively invariant, so the set model still behaves as
order-insensitive. Mean collapse, K=4 subsampling, top-rank removal, and batch
mismatch all change the output substantially.

## Group Results

Mean over seeds 7 and 17:

| group | original | mean_repeated | eval-time diversity gain | output delta L2 |
| --- | ---: | ---: | ---: | ---: |
| transition windows | 0.131027 | 0.141792 | +0.010765 | 0.656585 |
| sustain windows | 0.022528 | 0.030694 | +0.008166 | 0.453970 |
| high sample diversity | 0.050112 | 0.062638 | +0.012527 | 0.560969 |
| low sample diversity | 0.012129 | 0.018115 | +0.005987 | 0.377713 |
| high best-vs-mean gap | 0.046607 | 0.061768 | +0.015162 | 0.577012 |
| low best-vs-mean gap | 0.021016 | 0.025209 | +0.004194 | 0.385090 |

The strongest evidence for sample-distribution use appears exactly where it
should: transition windows, high-diversity windows, and high single-sample
best-vs-mean-gap windows.

## Interpretation

Gate 3.4b strengthens the runtime-usage claim:

```text
The full-aligned Gate 3.4 checkpoint is not simply ignoring K-sample diversity.
When its sample distribution is collapsed at eval time, performance degrades
and the predicted action sequence moves substantially.
```

But this does not erase the earlier Gate 3.4 caveat. The separately trained
`mean_repeated` control stayed close to full aligned:

```text
full aligned trained/evaluated normally: 0.034262
separately trained mean_repeated control: 0.034414
same-checkpoint eval-time mean collapse: 0.042594
```

So the clean interpretation is:

```text
The trained full checkpoint uses sample distribution at runtime, but current
training does not yet force an irreducible advantage over a model adapted to
mean-only inputs.
```

## Decision

Do not jump directly to a larger black-box decoder. The next cleaner branch is
Gate 3.4c:

```text
add set-wise temporal/regret/rank supervision or candidate-comparison auxiliary
loss, while keeping the same predicted top-4 event/rank/prob interface and the
same attribution controls.
```

Move to a small flow/diffusion residual decoder only after this controlled
sample-diversity objective is tested or clearly insufficient.

## Verification

```text
.venv/bin/python -m compileall scripts/audit_predicted_event_mixture_action_head_usage.py
.venv/bin/ruff check scripts/audit_predicted_event_mixture_action_head_usage.py tests/test_predicted_event_mixture_action_head_usage_audit.py
.venv/bin/python -m pytest tests/test_predicted_event_mixture_action_head_usage_audit.py tests/test_predicted_event_mixture_action_head_group_audit.py
```
