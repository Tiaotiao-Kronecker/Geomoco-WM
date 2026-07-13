# Gate 3.0a Motion-Prior-Conditioned Action Head Plan

## Purpose

Gate 2.5c showed that the joint GeoMoCo-cVAE sample set contains strong future
motions (`best-of-K` action MSE around `0.022`), while Gate 2.5d/2.6a showed
that selecting one sample with a lightweight ScoreNet does not reliably close
the readout gap. Gate 3.0a changes the interface: instead of forcing GeoMoCo-WM
to choose one future, train a downstream action head that consumes the full set
of sampled `future_delta_gripper` hypotheses.

This tests the current mainline positioning:

```text
GeoMoCo-WM is a visually grounded multimodal future-motion prior.
The action head / planner may aggregate or choose among its rollouts.
```

## Dataset

- Slice: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Suites: four standard LIBERO suites present in the two-file slice.
- Motion mode: `future_delta_gripper`.
- Split: episode-level train/validation split.
- Seeds: start with seed 7 pilot, expand to seed 17 if attribution is positive.

## Frozen Priors

Real visual cVAE:

```text
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed17/model.pt
```

Shuffled visual cVAE control:

```text
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_shuffled_freebits002_warmup5_prw05_lam03_seed7/model.pt
outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_shuffled_freebits002_warmup5_prw05_lam03_seed17/model.pt
```

## Model

New model: `MotionPriorActionHead`.

Inputs:

```text
context: [B, C]
conditioning: optional suite/task one-hot
future_motions: None, [B, M], or [B, K, M]
```

Future encoding:

```text
future [B,K,M] -> per-sample temporal tokens over horizon
context/task -> query token
query attends over K sample tokens
[context token, attended token, mean sample token, summary] -> action chunk [B,H,A]
```

The first Gate 3.0a version intentionally does not feed raw DINO features
directly into the action head. Visual contribution is tested by comparing real
cVAE samples against shuffled-visual cVAE samples.

## Ablations

Use the same action-head architecture and training protocol, changing only the
future input:

| mode | future input | role |
| --- | --- | --- |
| `context_only` | no future set | direct action-head lower bound |
| `prior_mean` | frozen cVAE prior mean as `K=1` | deterministic cVAE interface |
| `sample_set` real | `K=16` real visual cVAE samples | main test |
| `sample_set` shuffled | `K=16` shuffled visual cVAE samples | visual attribution control |
| `gt_future` | ground-truth future `future_delta_gripper` as `K=1` | privileged upper bound |

## Metrics

Primary:

```text
validation action MSE
```

Also report the existing action metric contract:

```text
MAE
translation_m_l2
rotation_geodesic_deg
gripper_mse / gripper_mae
```

## Promotion Criteria

Promote the branch only if:

```text
real sample_set action head beats context_only
real sample_set action head beats shuffled sample_set
real sample_set is competitive with or better than prior_mean
```

If real samples do not beat context-only or shuffled controls, record the result
as a negative and do not claim that the current motion prior helps downstream
action prediction.

## Initial Commands

Dry-run:

```bash
.venv/bin/python scripts/train_motion_prior_action_head.py \
  --input-mode sample_set \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_0a_dryrun \
  --num-samples 16 \
  --dry-run
```

Pilot seed 7:

```bash
.venv/bin/python scripts/train_motion_prior_action_head.py \
  --input-mode sample_set \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed7 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7
```

## Expected Interpretation

This gate decides whether the multimodal motion prior can be useful without
solving sample selection inside GeoMoCo-WM itself. A positive result supports
the world-model/motion-prior positioning. A negative result means the sample set
has oracle coverage but is not yet presented in a form a simple downstream
action head can exploit.
