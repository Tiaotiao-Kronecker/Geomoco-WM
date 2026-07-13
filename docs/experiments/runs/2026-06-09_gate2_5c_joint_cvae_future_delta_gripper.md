# Gate 2.5c Joint GeoMoCo-cVAE Future-Delta-Gripper

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.5c
- Purpose: train and evaluate visual-conditioned cVAE priors over joint
  `future_delta_ee + future_gripper/event` futures.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8_shuffled_seed7.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| motion dim | 56 |
| split policy | episode |
| seeds | 7, 17 |
| training device | cuda |
| eval device | cpu |

## Code

```text
scripts/train_visual_cvae_future_motion.py
scripts/evaluate_visual_cvae_samples.py
scripts/evaluate_visual_cvae_gripper_events.py
```

Implementation changes:

- cVAE training now supports `--motion-mode`, including
  `future_delta_gripper`;
- cVAE sample evaluation now uses motion-mode-aware split metrics;
- cVAE event evaluation extracts the gripper channel from joint predictions.

## Commands

Real visual, `prior_recon_weight=0.5`, seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7 \
  --motion-mode future_delta_gripper \
  --condition-on suite_task \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --action-aware-loss-weight 0.3 \
  --beta-kl 0.001 \
  --beta-kl-start 0.0 \
  --beta-kl-warmup-epochs 5 \
  --free-bits 0.02 \
  --prior-recon-weight 0.5 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Sample evaluation:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt \
  --output-json outputs/visual_cvae_samples/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7_k16.json \
  --num-samples 16 \
  --batch-size 64 \
  --device cpu \
  --seed 7
```

Event evaluation:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_gripper_events.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-json outputs/event_alignment/gate2_5c_joint_cvae_prw05_seed7_prior_mean.json \
  --device cpu \
  --seed 7 \
  --quiet
```

## Artifacts

Real visual:

```text
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_lam03_seed7/
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_lam03_seed17/
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed17/
```

Shuffled visual control:

```text
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_shuffled_freebits002_warmup5_prw05_lam03_seed7/
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_shuffled_freebits002_warmup5_prw05_lam03_seed17/
```

Sample evaluations:

```text
outputs/visual_cvae_samples/gate2_5c_joint_cvae_*_k16.json
```

Event evaluations:

```text
outputs/event_alignment/gate2_5c_joint_cvae_*_prior_mean.json
```

## Prior-Recon Sweep

Mean over seeds 7 and 17:

| branch | prior action MSE | action MAE | SE(3) MSE | gripper MSE | prior recon loss | raw KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cVAE prw=1.0 | 0.050771 | 0.111992 | 0.025748 | 0.200906 | 0.029325 | 1.924330 |
| cVAE prw=0.5 | 0.043816 | 0.100310 | 0.020473 | 0.183879 | 0.026787 | 1.672331 |
| deterministic joint | 0.040688 | 0.101703 | 0.020486 | 0.161903 | 0.023847 | 0.000000 |

Lowering `prior_recon_weight` from `1.0` to `0.5` improves prior-mean action
MSE by `13.70%`, but the cVAE prior mean still does not beat the deterministic
joint baseline.

## Real vs Shuffled Cvae

Mean over seeds 7 and 17, `prior_recon_weight=0.5`:

| branch | prior action MSE | prior SE(3) MSE | prior gripper MSE | sample mean action MSE | best-of-K action MSE | best-of-K SE(3) MSE | best-of-K gripper MSE | sample variance | pair L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real visual | 0.043816 | 0.020473 | 0.183879 | 0.061517 | 0.022139 | 0.017020 | 0.052857 | 0.014827 | 0.612986 |
| shuffled visual | 0.068816 | 0.035716 | 0.267414 | 0.132051 | 0.030576 | 0.028068 | 0.045624 | 0.039918 | 1.380726 |

Real visual improves over shuffled visual:

| metric | improvement |
| --- | ---: |
| prior action MSE | 36.33% |
| best-of-K action MSE | 27.59% |
| prior gripper MSE | 31.24% |
| transition accuracy | +0.347720 |

## Event Fidelity

Mean prior-mean event metrics:

| branch | event acc | macro-F1 | transition acc | step within 1 |
| --- | ---: | ---: | ---: | ---: |
| real visual | 0.889838 | 0.618679 | 0.571094 | 0.233047 |
| shuffled visual | 0.786677 | 0.456603 | 0.223375 | 0.070012 |
| deterministic joint reference | 0.878580 | 0.612640 | 0.560270 | 0.234025 |

The cVAE prior mean slightly improves transition accuracy over the deterministic
joint reference, while staying similar on step-within-1.

## Main Interpretation

Gate 2.5c is mixed-positive:

- The joint cVAE prior mean does not yet beat the deterministic joint baseline
  on deployable action MSE.
- Real visual cVAE clearly beats shuffled visual cVAE, so the cVAE branch still
  depends on aligned visual grounding.
- Best-of-K action MSE is very strong (`0.022139`), showing that the stochastic
  sample set contains high-value futures.
- Random sample mean is worse than prior mean, so naive sampling is not
  deployable.
- Event fidelity is positive: prior-mean transition accuracy is slightly above
  deterministic joint reference and much better than shuffled visual.

## Decision

Do not promote raw cVAE prior mean as the final deployable branch yet.

Promote the joint cVAE sample space and move next to:

```text
Gate 2.5d: joint cVAE sample readout/scorer
```

The target is to convert the large best-of-K gap into a deployable readout,
using the deterministic joint baseline `0.040688` and cVAE prior mean
`0.043816` as references.

## Verification

```text
.venv/bin/python -m compileall scripts/train_visual_cvae_future_motion.py scripts/evaluate_visual_cvae_samples.py scripts/evaluate_visual_cvae_gripper_events.py
.venv/bin/ruff check scripts/train_visual_cvae_future_motion.py scripts/evaluate_visual_cvae_samples.py scripts/evaluate_visual_cvae_gripper_events.py tests/test_future_motion_predictor.py
.venv/bin/python -m unittest tests.test_future_motion_predictor tests.test_cvae_event_alignment
```

All passed before the formal runs.

