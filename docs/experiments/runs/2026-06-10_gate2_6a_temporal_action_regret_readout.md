# Gate 2.6a Temporal Action-Regret Readout

- Date: 2026-06-10
- Status: completed
- Gate: Gate 2.6a
- Purpose: replace the Gate 2.5d flat MLP ScoreNet with a temporal scorer while
  keeping the joint cVAE and action decoder frozen.

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
| windows | 16,518 |
| horizon | 8 |
| motion mode | `future_delta_gripper` |
| cVAE samples | 16 |
| split policy | episode |
| seeds | 7, 17 |
| device | cuda |

## Code

```text
src/geomoco_wm/models/sample_readout.py
scripts/train_visual_cvae_sample_scorer.py
scripts/evaluate_visual_cvae_sample_scorer.py
scripts/evaluate_cvae_event_alignment.py
tests/test_future_motion_predictor.py
```

New architecture:

```text
TemporalSampleScoreNet
```

It keeps the same call interface as the original ScoreNet:

```text
score = scorer(condition, motion_candidate, decoded_action_chunk)
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_6a_temporal_action_rank_k16_seed7 \
  --scorer-arch temporal \
  --temporal-dim 128 \
  --temporal-layers 2 \
  --temporal-heads 4 \
  --target-kind action \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Seed 17 uses the matching seed-17 cVAE checkpoint and `--seed 17`.

Event alignment:

```bash
.venv/bin/python scripts/evaluate_cvae_event_alignment.py \
  --checkpoint <matching-cvae-checkpoint> \
  --scorer-checkpoint <temporal-scorer-model.pt> \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/event_alignment/gate2_6a_temporal_action_rank_seed<seed>_k16.json \
  --num-samples 16 \
  --batch-size 128 \
  --device cuda \
  --seed <seed> \
  --quiet
```

## Artifacts

```text
outputs/visual_cvae_sample_scorer/gate2_6a_temporal_action_rank_k16_seed7/
outputs/visual_cvae_sample_scorer/gate2_6a_temporal_action_rank_k16_seed17/
outputs/event_alignment/gate2_6a_temporal_action_rank_seed7_k16.json
outputs/event_alignment/gate2_6a_temporal_action_rank_seed17_k16.json
```

## Results

### Action Readout

| method | prior action MSE | scorer action MSE | oracle action MSE | top-1 match | selected rank | regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.5d flat mean | 0.043816 | 0.043414 | 0.022192 | 0.255727 | 5.637193 | 0.021222 |
| Gate 2.6a temporal mean | 0.043816 | 0.043636 | 0.022158 | 0.252489 | 5.551251 | 0.021478 |

Seed-level:

| run | scorer action MSE | selected rank | best epoch |
| --- | ---: | ---: | ---: |
| flat seed 7 | 0.045230 | 5.981643 | 19 |
| flat seed 17 | 0.041597 | 5.292743 | 13 |
| temporal seed 7 | 0.044917 | 6.031469 | 7 |
| temporal seed 17 | 0.042355 | 5.071033 | 15 |

### Event Diagnostics

| method | event acc | macro-F1 | transition acc | step@1 |
| --- | ---: | ---: | ---: | ---: |
| Gate 2.5d flat mean | 0.900561 | 0.639790 | 0.647706 | 0.264299 |
| Gate 2.6a temporal mean | 0.899412 | 0.635961 | 0.636113 | 0.289056 |

## Interpretation

Gate 2.6a is a weak negative.

The temporal scorer does improve selected oracle rank slightly and improves
transition step-within-1, which suggests it is using some temporal structure.
However, it does not improve the primary downstream action metric. Mean action
MSE regresses from the Gate 2.5d flat ScoreNet `0.043414` to `0.043636`, and it
still does not beat the deterministic joint baseline `0.040688`.

This means the readout bottleneck is not solved by simply replacing the flat MLP
with a small temporal encoder. The next mainline should either:

- add explicit action-regret regression / calibration rather than only listwise
  ranking; or
- revisit cVAE sample training so the action-useful samples are easier to
  identify from deployable candidate features.

## Decision

Do not promote `TemporalSampleScoreNet` v1.

Do not run shuffled visual controls for this exact v1 because it failed the
real-visual promotion criterion. Keep the implementation for the next readout
iteration, likely with explicit action-regret prediction or set-wise
candidate-comparison attention.

