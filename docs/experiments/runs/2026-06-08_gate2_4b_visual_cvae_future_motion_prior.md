# Gate 2.4b Visual-Conditioned cVAE Future-Motion Prior

- Date: 2026-06-08
- Status: completed
- Gate: Gate 2.4b
- Purpose: test the first stochastic / cVAE future-motion prior on the
  validated visual-action route.

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
| windows | 16,518 |
| context dim | 15 |
| future-motion dim | 48 |
| future horizon | 8 |
| action dim | 7 |
| split policy | episode |

## Method

Condition builder:

```text
base = [proprio, suite_task one-hot]
base -> query
query attends DINO patch tokens -> g_t
condition c_t = [base, g_t]
```

cVAE:

```text
posterior q(z | c_t, future_motion)
prior     p(z | c_t)
decoder   future_motion_hat = Dec(c_t, z)
```

Training objective:

```text
posterior_recon_loss = MSE(Dec(c_t, z_q), gt_future_motion)
prior_mean_motion = Dec(c_t, prior_mean)
prior_recon_loss = MSE(prior_mean_motion, gt_future_motion)
prior_action_loss = MSE(
  frozen_action_decoder(context, prior_mean_motion),
  gt_action_chunk
)
loss = posterior_recon_loss
     + 1.0 * prior_recon_loss
     + 0.001 * KL(q || p)
     + 0.030 * prior_action_loss
```

The main deployable metric is the prior mean:

```text
prior_mean_motion -> frozen action decoder -> action metrics
```

Posterior reconstruction is recorded as a diagnostic, not as the policy-ready
result.

## Code Changes

- `src/geomoco_wm/models/geomoco_cvae.py`
  - added `VisualConditionedGeoMoCoCVAE`;
  - added `gaussian_kl_divergence`.
- `scripts/train_visual_cvae_future_motion.py`
  - new training script for visual-conditioned cVAE future-motion priors.
- `tests/test_future_motion_predictor.py`
  - added visual-conditioned cVAE shape and KL tests.

## Model And Training Config

```text
script: scripts/train_visual_cvae_future_motion.py
model: VisualConditionedGeoMoCoCVAE
visual grounding: single-query DINO patch cross-attention
visual tokens: 64 x 384D
latent dim: 32
hidden dims: 256,256
epochs: 20
batch size: 64
lr: 1e-3
weight decay: 0
beta_kl: 0.001
prior_recon_weight: 1.0
action-aware loss weight: 0.030
split policy: episode
seed(s): 7, 17
device: cuda
downstream decoder: frozen Gate 1.6 geodesic oracle future-motion ActionDecoder
```

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7 \
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
  --prior-recon-weight 1.0
```

Seed 17:

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17 \
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
  --prior-recon-weight 1.0
```

## Artifacts

| seed | metrics | checkpoint |
| ---: | --- | --- |
| 7 | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/metrics.json` | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed7/model.pt` |
| 17 | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/metrics.json` | `outputs/visual_cvae_future_motion/gate2_4b_visual_cvae_lam003_beta0001_prior1_seed17/model.pt` |

## Per-Seed Prior-Mean Results

| seed | prior MSE | prior trans L2 | prior orient L2 | KL | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.000824 | 0.018401 | 0.051518 | 0.000743 | 0.044342 | 0.116193 | 0.015381 | 2.019127 | 0.180472 |
| 17 | 0.000779 | 0.016440 | 0.050053 | 0.000736 | 0.038815 | 0.107142 | 0.014491 | 2.101555 | 0.150758 |

## Mean Results

| branch | prior/future MSE | trans L2 | orient L2 | posterior MSE | KL | action MSE | action MAE | action trans L2 (m) | rot geo (deg) | gripper MSE | gap closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.3b deterministic single-query lambda 0.030 | 0.000782 | 0.018767 | 0.050640 | n/a | n/a | 0.042090 | 0.110949 | 0.014598 | 2.016930 | 0.174519 | 69.26% |
| Gate 2.4a deterministic stepwise lambda 0.030 | 0.000776 | 0.017155 | 0.051082 | n/a | n/a | 0.042687 | 0.112300 | 0.014750 | 2.042381 | 0.177896 | 67.53% |
| Gate 2.4b visual cVAE prior mean | 0.000802 | 0.017420 | 0.050785 | 0.000802 | 0.000740 | 0.041579 | 0.111667 | 0.014936 | 2.060341 | 0.165615 | 70.74% |

## Interpretation

Gate 2.4b is mildly positive on the main action-value metric:

```text
deterministic single-query lambda 0.030 action MSE: 0.042090
visual cVAE prior-mean action MSE: 0.041579
```

It also improves gripper MSE:

```text
deterministic single-query lambda 0.030 gripper MSE: 0.174519
visual cVAE prior-mean gripper MSE: 0.165615
```

But the improvement is not yet strong or uniformly stable:

```text
seed 7 cVAE action MSE: 0.044342
seed 17 cVAE action MSE: 0.038815
```

The KL is also very small:

```text
mean KL: 0.000740
```

Posterior and prior metrics are almost identical, which means this first cVAE
run behaves more like a deterministic prior regularized through a VAE-shaped
objective than a clearly multimodal future-motion model.

## Decision

Do not yet claim multimodal GeoMoCo-WM behavior.

Promote Gate 2.4b only as a promising cVAE entry point:

- it slightly improves mean action MSE over the deterministic default;
- it improves gripper MSE;
- it does not yet show meaningful stochastic latent usage.

## Next Decision

Run a cVAE calibration gate before policy claims:

1. add sample-based evaluation, including prior samples and best-of-K
   future-motion coverage diagnostics;
2. test KL/free-bits or beta schedules to avoid posterior-prior collapse;
3. add gripper/contact auxiliary targets or diagnostics;
4. keep deterministic `cross_attention + lambda_action=0.030` as the main
   baseline until stochastic coverage is demonstrated.
