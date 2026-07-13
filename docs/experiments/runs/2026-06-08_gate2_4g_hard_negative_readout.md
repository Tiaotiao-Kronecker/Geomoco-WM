# Gate 2.4g Hard-Negative cVAE Sample Readout

- Date: 2026-06-08
- Status: completed
- Device: NVIDIA GeForce RTX 5090
- Purpose: test a minimal hard-negative auxiliary loss over calibrated visual
  cVAE samples.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| split policy | episode |
| K | 16 prior samples |

## Code Changes

- `scripts/train_visual_cvae_sample_scorer.py`
  - added `--hard-negative-target-kind`;
  - added `--hard-negative-weight`;
  - added `--hard-negative-margin`;
  - added pairwise hard-negative loss;
  - added `--quiet` for long GPU runs.
- `scripts/evaluate_visual_cvae_sample_scorer.py`
  - preserves hard-negative scorer config when re-evaluating checkpoints.
- `tests/test_future_motion_predictor.py`
  - added hard-negative loss direction test.

## Commands

Smoke:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_smoke_seed7 \
  --target-kind action \
  --hard-negative-target-kind se3_gripper \
  --hard-negative-weight 0.5 \
  --num-samples 4 \
  --epochs 1 \
  --batch-size 8 \
  --max-windows 64 \
  --seed 7 \
  --device cpu \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

Formal branch template:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint <gate2_4c-cvae-checkpoint> \
  --output-dir <gate2_4g-output-dir> \
  --target-kind action \
  --hard-negative-target-kind se3_gripper \
  --hard-negative-weight <0.1-or-0.5> \
  --hard-negative-margin 0.0 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --seed <7-or-17> \
  --device cuda \
  --action-decoder-checkpoint <gate1_6-action-decoder-checkpoint> \
  --quiet
```

## Artifacts

| branch | seed | metrics | checkpoint |
| --- | ---: | --- | --- |
| w=0.1 | 7 | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed7/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed7/model.pt` |
| w=0.1 | 17 | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed17/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w01_k16_seed17/model.pt` |
| w=0.5 | 7 | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed7/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed7/model.pt` |
| w=0.5 | 17 | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed17/metrics.json` | `outputs/visual_cvae_sample_scorer/gate2_4g_hardneg_se3gripper_w05_k16_seed17/model.pt` |

## Results

Mean over seed 7 and seed 17:

| branch | action MSE | trans L2 m | rot geod deg | gripper MSE | flat rank | flat top1 | SE(3) rank | SE(3)+gripper rank | gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.4d flat ScoreNet | 0.040190 | 0.014229 | 2.093158 | 0.165335 | 6.624765 | 0.235384 | 7.768547 | 7.451381 | 18.36% |
| Gate 2.4g hardneg w=0.1 | 0.040344 | 0.014264 | 2.101104 | 0.165828 | 6.943479 | 0.226036 | 7.974585 | 7.717540 | 14.55% |
| Gate 2.4g hardneg w=0.5 | 0.040479 | 0.014301 | 2.117862 | 0.166149 | 7.263220 | 0.210748 | 8.297672 | 8.190786 | 11.20% |

## Interpretation

The naive hard-negative construction is a negative ablation.

Lowering the auxiliary weight from `0.5` to `0.1` reduces the damage, but still
does not beat the Gate 2.4d flat ScoreNet baseline. The auxiliary loss likely
pushes against the already useful flat-action ranking target because the
structured negative is not a reliable proxy for real execution failure.

The lesson is precise: not all "geometry-plausible but action-worse" negatives
are useful. A better next branch should construct negatives around actual
execution-sensitive events, such as gripper transitions, contact timing, or
phase/order violations.

## Decision

Do not promote Gate 2.4g hard-negative auxiliary loss.

Keep Gate 2.4d flat ScoreNet as the current deployable readout baseline.

Next readout direction:

1. build explicit gripper-transition / contact-proxy labels from existing
   action and proprio streams;
2. score candidates on event timing and action regret, not only structured
   `SE(3)` proximity;
3. keep the Gate 2.4f structured oracle ranks as diagnostics.

## Verification

```text
.venv/bin/python -m unittest tests.test_future_motion_predictor
.venv/bin/python -m compileall src scripts tests
.venv/bin/ruff check src scripts tests
```

These checks passed before the formal GPU runs. Full final checks are tracked
in the session close-out.
