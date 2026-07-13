# Gate 2.4e SE(3)+Gripper-Aware cVAE Sample Scorer

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4e
- Purpose: test whether replacing the Gate 2.4d flat action-MSE ranking target
  with structured SE(3) and SE(3)+gripper ranking targets improves deployable
  cVAE sample readout.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

Summary:

| field | value |
| --- | ---: |
| suites | 4 |
| HDF5 task files | 8 |
| demos | 400 |
| windows | 16,518 |
| context length | 2 |
| horizon | 8 |
| split policy | episode |

## Method

Frozen modules:

```text
Gate 2.4c visual cVAE
Gate 1.6 geodesic ActionDecoder
```

For each sampled future-motion candidate:

```text
future_motion_k = cVAE.sample(c_t, z_k)
action_k = frozen_action_decoder(context, future_motion_k)
score_k = SampleScoreNet(c_t, future_motion_k, action_k)
```

New structured candidate errors:

```text
translation_m_l2_k
rotation_geodesic_rad_k
gripper_mse_k
```

Ranking targets:

```text
se3:
  score target = zscore(-translation_m_l2) + zscore(-rotation_geodesic_rad)

se3_gripper:
  score target = zscore(-translation_m_l2)
               + zscore(-rotation_geodesic_rad)
               + zscore(-gripper_mse)
```

The main evaluation still reports the same deployable readouts as Gate 2.4d:

- prior mean;
- random sample mean;
- scorer argmax;
- scorer soft motion;
- oracle best-of-K action as a non-deployable upper-bound diagnostic.

## Code Changes

- `scripts/train_visual_cvae_sample_scorer.py`
  - added `--target-kind se3` and `--target-kind se3_gripper`;
  - added `--translation-target-weight`, `--rotation-target-weight`, and
    `--gripper-target-weight`;
  - computes structured candidate errors using meter-scaled translation,
    SO(3) geodesic rotation, and gripper MSE.
- `tests/test_future_motion_predictor.py`
  - added structured target tests for zero error, translation scaling, and
    gripper-sensitive ranking.

## Commands

Smoke:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4e_smoke_se3_gripper_seed7 \
  --target-kind se3_gripper \
  --num-samples 4 \
  --epochs 1 \
  --batch-size 8 \
  --max-windows 64 \
  --seed 7 \
  --device cpu \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Formal runs:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed7 \
  --target-kind se3 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt

.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed17 \
  --target-kind se3 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt

.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed7 \
  --target-kind se3_gripper \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt

.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed17 \
  --target-kind se3_gripper \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Artifacts

| target | seed | metrics | checkpoint |
| --- | ---: | --- | --- |
| `se3` | 7 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed7/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed7/model.pt` |
| `se3` | 17 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed17/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_k16_seed17/model.pt` |
| `se3_gripper` | 7 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed7/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed7/model.pt` |
| `se3_gripper` | 17 | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed17/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4e_se3_gripper_k16_seed17/model.pt` |

## Results

| target | seed | best epoch | prior action MSE | scorer argmax action MSE | oracle best action MSE | trans L2 m | rot geod deg | gripper MSE | top1 oracle | oracle rank | regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `se3` | 7 | 20 | 0.042666 | 0.042350 | 0.038578 | 0.014417 | 2.009419 | 0.178766 | 0.163 | 7.657 | 0.003772 |
| `se3` | 17 | 20 | 0.039196 | 0.038815 | 0.035213 | 0.014195 | 2.070731 | 0.155743 | 0.195 | 7.392 | 0.003602 |
| `se3` mean | - | - | 0.040931 | 0.040582 | 0.036895 | 0.014306 | 2.040075 | 0.167254 | 0.179 | 7.525 | 0.003687 |
| `se3_gripper` | 7 | 19 | 0.042666 | 0.042301 | 0.038578 | 0.014455 | 2.022427 | 0.177994 | 0.154 | 7.575 | 0.003723 |
| `se3_gripper` | 17 | 20 | 0.039196 | 0.038547 | 0.035213 | 0.014168 | 2.063609 | 0.154282 | 0.217 | 6.903 | 0.003334 |
| `se3_gripper` mean | - | - | 0.040931 | 0.040424 | 0.036895 | 0.014311 | 2.043018 | 0.166138 | 0.186 | 7.239 | 0.003529 |

Reference mean from Gate 2.4d flat action-MSE target:

| target | action MSE | trans L2 m | rot geod deg | gripper MSE | top1 oracle | oracle rank | regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `action` flat MSE | 0.040201 | 0.014234 | 2.091603 | 0.165270 | 0.238 | 6.614 | 0.003306 |

## Interpretation

The structured targets do not beat Gate 2.4d flat action-MSE ranking.

Mean action MSE:

```text
Gate 2.4d flat target:       0.040201
Gate 2.4e SE(3) target:      0.040582
Gate 2.4e SE(3)+gripper:     0.040424
Prior mean baseline:         0.040931
Oracle best-of-K diagnostic: 0.036895
```

Gap closure from prior mean to oracle best-of-K:

| target | gap closed |
| --- | ---: |
| flat action-MSE | 18.09% |
| SE(3) | 8.64% |
| SE(3)+gripper | 12.57% |

The gripper-aware target improves over pure SE(3), especially in seed 17, but
does not improve over the flat action-MSE scorer. It also does not improve the
mean gripper metric relative to the flat target.

## Decision

Do not promote structured target replacement as the default scorer.

Keep Gate 2.4d flat action-MSE ScoreNet as the current deployed readout
baseline, and treat Gate 2.4e as a negative but informative ablation:

- the readout bottleneck is not solved by simply changing scalar ranking target
  units;
- gripper/contact/executability likely needs explicit labels, diagnostics, or
  hard-negative construction rather than a naive gripper-MSE term;
- the next readout branch should focus on hard-negative ranking or real
  executability/contact proxies before moving to multimodal action heads.

## Verification

```text
.venv/bin/python -m unittest tests.test_future_motion_predictor
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall src scripts tests
.venv/bin/ruff check src scripts tests
```

All checks passed before the formal GPU runs.
