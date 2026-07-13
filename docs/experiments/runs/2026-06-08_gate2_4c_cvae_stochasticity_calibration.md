# Gate 2.4c cVAE Stochasticity Calibration

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4c
- Purpose: test whether the Gate 2.4b visual-conditioned cVAE has meaningful
  prior-sample coverage, and whether KL free-bits plus beta warmup can increase
  latent usage without collapsing deployable prior quality.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Visual cache:

```text
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
| future horizon | 8 |
| stride | 4 |
| split policy | episode |

This is the two-files-per-suite formal slice, not the full LIBERO dataset.

## Method

Gate 2.4c keeps the Gate 2.4b visual cVAE architecture:

```text
base = [proprio, suite_task one-hot]
base -> query
query attends DINO patch tokens -> g_t
condition c_t = [base, g_t]
posterior q(z | c_t, future_motion)
prior     p(z | c_t)
decoder   Dec(c_t, z) -> future_motion
```

Two evaluations were added:

1. Evaluate prior samples from the existing Gate 2.4b checkpoints with
   `K=16`.
2. Retrain a calibrated branch with free-bits and beta warmup, then evaluate
   the same `K=16` sample metrics.

Best-of-K is a diagnostic coverage metric:

```text
best-of-K motion: choose the sample closest to GT future motion
best-of-K action: choose the sample whose decoded action is closest to GT action
```

Both use ground truth for selection. They are not deployable policy metrics.

## Code Changes

- `scripts/evaluate_visual_cvae_samples.py`
  - evaluates prior mean, sample mean, best-of-K motion, best-of-K action, and
    sample diversity.
- `scripts/train_visual_cvae_future_motion.py`
  - added `--beta-kl-start`, `--beta-kl-warmup-epochs`, and `--free-bits`.
  - logs both `kl_loss` after free-bits and `raw_kl_loss` before free-bits.
- `src/geomoco_wm/models/geomoco_cvae.py`
  - `gaussian_kl_divergence(..., free_bits=...)` clamps per-latent-dimension
    KL when free-bits is enabled.
- `tests/test_future_motion_predictor.py`
  - added coverage for free-bits KL behavior.

## Training Config

Baseline sample evaluation:

```text
checkpoints:
  outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/model.pt
  outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/model.pt
num_samples: 16
device: cuda
```

Free-bits calibration:

```text
model: VisualConditionedGeoMoCoCVAE
visual grounding: single-query DINO patch cross-attention
visual tokens: 64 x 384D
latent dim: 32
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
beta_kl_start: 0.0
beta_kl: 0.001
beta_kl_warmup_epochs: 5
free_bits: 0.02
prior_recon_weight: 1.0
action-aware loss weight: 0.030
seeds: 7, 17
device: cuda
downstream decoder: frozen Gate 1.6 geodesic ActionDecoder
```

## Commands

Existing Gate 2.4b sample evaluation:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/model.pt \
  --output-json outputs/visual_cvae_future_motion/gate2_4c_sample_eval_gate2_4b_seed7_k16.json \
  --num-samples 16 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/model.pt \
  --output-json outputs/visual_cvae_future_motion/gate2_4c_sample_eval_gate2_4b_seed17_k16.json \
  --num-samples 16 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

Free-bits retraining:

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --latent-dim 32 \
  --split-by episode \
  --condition-on suite_task \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt \
  --action-aware-loss-weight 0.03 \
  --beta-kl 0.001 \
  --beta-kl-start 0.0 \
  --beta-kl-warmup-epochs 5 \
  --free-bits 0.02 \
  --prior-recon-weight 1.0
```

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --latent-dim 32 \
  --split-by episode \
  --condition-on suite_task \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt \
  --action-aware-loss-weight 0.03 \
  --beta-kl 0.001 \
  --beta-kl-start 0.0 \
  --beta-kl-warmup-epochs 5 \
  --free-bits 0.02 \
  --prior-recon-weight 1.0
```

Free-bits sample evaluation:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-json outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed7_k16.json \
  --num-samples 16 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed7/model.pt
```

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt \
  --output-json outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed17_k16.json \
  --num-samples 16 \
  --batch-size 64 \
  --seed 17 \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed17/model.pt
```

## Artifacts

| branch | seed | metrics / sample eval | checkpoint |
| --- | ---: | --- | --- |
| Gate 2.4b sample eval | 7 | `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_gate2_4b_seed7_k16.json` | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/model.pt` |
| Gate 2.4b sample eval | 17 | `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_gate2_4b_seed17_k16.json` | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/model.pt` |
| Gate 2.4c free-bits | 7 | `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/metrics.json`; `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed7_k16.json` | `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt` |
| Gate 2.4c free-bits | 17 | `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/metrics.json`; `outputs/visual_cvae_future_motion/gate2_4c_sample_eval_freebits002_seed17_k16.json` | `outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed17/model.pt` |

## Per-Seed Results

| branch | seed | raw KL | logged KL | prior motion MSE | sample motion MSE | best-of-K motion MSE | sample var | sample pair L2 | prior action MSE | sample action MSE | best-of-K action MSE | prior grip MSE | best-of-K grip MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.4b baseline | 7 | 0.000743 | 0.000743 | 0.000824 | 0.000826 | 0.000763 | 0.00000182 | 0.012731 | 0.044342 | 0.044390 | 0.042106 | 0.180472 | 0.172250 |
| 2.4b baseline | 17 | 0.000736 | 0.000736 | 0.000779 | 0.000780 | 0.000724 | 0.00000151 | 0.011518 | 0.038815 | 0.038855 | 0.036945 | 0.150758 | 0.144454 |
| 2.4c free-bits | 7 | 0.438041 | 0.647082 | 0.000839 | 0.000885 | 0.000579 | 0.00004631 | 0.056396 | 0.042666 | 0.042965 | 0.038587 | 0.179010 | 0.162951 |
| 2.4c free-bits | 17 | 0.446153 | 0.649124 | 0.000764 | 0.000808 | 0.000524 | 0.00004331 | 0.054779 | 0.039196 | 0.039434 | 0.035200 | 0.156329 | 0.141463 |

## Mean Results

| branch | raw KL | logged KL | prior motion MSE | sample motion MSE | best-of-K motion MSE | sample var | sample pair L2 | prior action MSE | sample action MSE | best-of-K action MSE | prior grip MSE | best-of-K grip MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.4b baseline | 0.000740 | 0.000740 | 0.000802 | 0.000803 | 0.000744 | 0.00000167 | 0.012125 | 0.041579 | 0.041622 | 0.039526 | 0.165615 | 0.158352 |
| 2.4c free-bits | 0.442097 | 0.648103 | 0.000801 | 0.000847 | 0.000552 | 0.00004481 | 0.055588 | 0.040931 | 0.041199 | 0.036894 | 0.167670 | 0.152207 |

## Interpretation

Free-bits does what Gate 2.4c asked for on latent usage:

```text
raw KL: 0.000740 -> 0.442097
sample variance: 0.00000167 -> 0.00004481
sample pair L2: 0.012125 -> 0.055588
```

It also improves coverage:

```text
2.4b best-of-K motion MSE: 0.000744
2.4c best-of-K motion MSE: 0.000552

2.4b best-of-K action MSE: 0.039526
2.4c best-of-K action MSE: 0.036894
```

The best-of-K action gain means the prior distribution now contains samples
that are much closer to action-useful futures. But this is still an oracle
selection result, because GT action is used to choose the best sample.

The deployable prior mean also improves slightly:

```text
2.4b prior-mean action MSE: 0.041579
2.4c prior-mean action MSE: 0.040931
```

However, random sample mean remains worse than prior mean:

```text
2.4c prior-mean action MSE: 0.040931
2.4c sample-mean action MSE: 0.041199
```

So the model has gained coverage, but not yet a deployable sample-selection
mechanism.

## Decision

Gate 2.4c passes as a stochasticity calibration gate:

- latent usage is no longer collapsed;
- sample diversity is non-trivial;
- best-of-K coverage improves both motion and downstream action;
- prior mean does not collapse and slightly improves action MSE.

It does not yet justify claiming a deployable stochastic policy module:

- best-of-K uses ground-truth selection;
- random prior samples are not better than the prior mean;
- gripper prior-mean MSE does not improve relative to Gate 2.4b.

Promote the free-bits cVAE as the calibrated stochastic branch, but keep the
deterministic single-query `lambda_action=0.030` and Gate 2.4b prior mean as
comparison baselines.

## Next Decision

The next mainline should test a deployable readout from the stochastic prior:

1. score or rank sampled future motions without GT action labels at test time;
2. add action-aware or risk-sensitive sample aggregation instead of plain
   random sample mean;
3. add gripper/contact diagnostics, because the current stochastic gains are
   stronger in SE(3) coverage than in gripper prediction;
4. only after that connect the calibrated branch into a GeoMoCo-cVAE policy
   validation path.
