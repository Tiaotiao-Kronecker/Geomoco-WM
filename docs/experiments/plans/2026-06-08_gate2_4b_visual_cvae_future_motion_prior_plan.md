# Gate 2.4b Visual-Conditioned cVAE Future-Motion Prior Plan

- Date: 2026-06-08
- Status: planned
- Gate: Gate 2.4b
- Purpose: test the first stochastic / multimodal GeoMoCo-WM future-motion
  prior after deterministic visual/action-aware gates.

## Why This Gate

Gate 2.4a showed that deterministic query structure alone is not the main
remaining bottleneck. Stepwise multi-query attention improved future-motion
translation geometry but did not beat the single-query action-value branch.

The next hypothesis is that the remaining oracle gap is partly multimodal:
from the same visual/proprio/task context, multiple plausible future EEF
motions can exist, especially around grasp timing, contact phase, and
long-horizon composition.

## Model

Use the validated single-query visual grounding path as the condition builder:

```text
base = [proprio, suite_task one-hot]
base -> query
query attends DINO patch tokens -> g_t
condition c_t = [base, g_t]
```

Then train a conditional VAE:

```text
posterior q(z | c_t, future_motion)
prior     p(z | c_t)
decoder   future_motion_hat = Dec(c_t, z)
```

## Training Objective

Primary training objective:

```text
posterior_recon_loss = MSE(Dec(c_t, z_q), gt_future_motion)
kl_loss = KL(q(z | c_t, gt_future_motion) || p(z | c_t))
```

Because our main downstream metric is action value, add a deployable-prior
auxiliary branch using the prior mean:

```text
prior_mean_motion = Dec(c_t, prior_mean)
prior_action_loss = MSE(
  frozen_action_decoder(context, prior_mean_motion),
  gt_action_chunk
)
```

Total:

```text
loss = posterior_recon_loss
     + beta_kl * kl_loss
     + prior_recon_weight * MSE(prior_mean_motion, gt_future_motion)
     + lambda_action * prior_action_loss
```

Initial settings:

```text
beta_kl = 0.001
prior_recon_weight = 1.0
lambda_action = 0.030
latent_dim = 32
```

The prior-mean branch makes the cVAE comparable to deterministic predictors.

## Dataset And References

Dataset:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Visual cache:

```text
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

References:

| branch | action MSE | future trans L2 | gap closure |
| --- | ---: | ---: | ---: |
| Gate 2.3b deterministic single-query lambda 0.030 | 0.042090 | 0.018767 | 69.26% |
| Gate 2.4a deterministic stepwise lambda 0.030 | 0.042687 | 0.017155 | 67.53% |

## Metrics

Main deployable metric:

```text
prior_mean_motion -> frozen action decoder -> action metrics
```

Also report:

- posterior reconstruction future-motion metrics;
- prior-mean future-motion metrics;
- KL;
- optional best-of-K sample future-motion metrics as coverage diagnostics.

Posterior and best-of-K are not policy-ready metrics because they depend on GT
future motion or GT-based sample selection.

## Pass / Stop Criteria

Pass if the prior mean improves action MSE over the deterministic single-query
`lambda_action=0.030` baseline without severe future-motion metric regression.

Stop or redesign if:

- prior mean action MSE is worse than deterministic baselines;
- posterior reconstruction is good but prior mean is poor, indicating KL/prior
  mismatch;
- KL collapses to near zero and samples do not cover meaningful variation.

## Next If Positive

Promote visual-conditioned cVAE as the GeoMoCo-WM stochastic future-motion
prior and then add:

1. best-of-K / risk-sensitive action readout;
2. gripper/contact auxiliary branch;
3. closed-loop policy integration.
