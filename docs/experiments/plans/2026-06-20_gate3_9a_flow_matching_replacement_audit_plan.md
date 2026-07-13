# Gate 3.9a Flow-Matching Replacement Audit Plan

## Purpose

Directly test whether a stronger action-chunk policy can replace the
GeoMoCo-WM motion-prior interface.

This gate is not intended to maximize final policy performance. It is an
attribution audit:

```text
Does GeoMoCo-conditioned flow matching beat a direct visual flow policy under
matched capacity and data?
```

## Motivation

Gate 3.7a and Gate 3.8a both failed as small decoder-side repairs:

```text
Gate 3.7a soft event-time latent:
  temporal_action_mse = 0.037941
  transition_mse      = 0.147882

Gate 3.8a tiny temporal transformer:
  temporal_action_mse = 0.038275
  transition_mse      = 0.162345
```

The current question is therefore no longer just how to repair a weak decoder.
It is whether the GeoMoCo-WM prior remains useful when the action policy itself
is stronger.

## Minimal Model

Train a conditional rectified-flow / flow-matching policy over action chunks:

```text
x1 = ground-truth action chunk [B,H,A]
x0 = Gaussian noise
t  ~ Uniform(0, 1)
xt = (1 - t) x0 + t x1
target velocity = x1 - x0
loss = MSE(v_theta(xt, t, condition), target velocity)
```

Inference:

```text
x = Gaussian noise
for t in Euler grid 0 -> 1:
    x = x + dt * v_theta(x, t, condition)
```

Evaluate the sampled action chunk with the same action metrics used by Gate 3.

## Condition Modes

First implementation supports two seed-7 short-budget branches:

```text
direct_visual
  context/proprio + suite/task conditioning + DINO visual feature

geomoco
  direct_visual condition
  + predicted top-4 event-mixture GeoMoCo samples
  + per-sample event/rank/prob metadata
```

The direct branch must see DINO visual features. Otherwise the replacement
audit would be unfair.

## Short-Budget Stop Rule

Run seed 7 only:

```text
direct_visual_flow seed7
geomoco_conditioned_flow seed7
```

Expand only if:

```text
geomoco transition MSE < direct_visual transition MSE
geomoco gripper MSE <= direct_visual gripper MSE
geomoco overall MSE is not materially worse than direct_visual
```

Optional positive reference:

```text
Gate 3.4 seed7 temporal_action_mse            = 0.036502
Gate 3.4 seed7 temporal_action_transition_mse = 0.137827
Gate 3.4 seed7 temporal_action_gripper_mse    = 0.164389
```

The flow policy does not need to beat Gate 3.4 immediately to justify the
replacement audit, but GeoMoCo-conditioned flow must beat direct visual flow to
justify further controls.

## Controls If Positive

If seed 7 passes, run seed 17. If the two-seed signal remains positive, expand:

```text
full geomoco
direct_visual
shuffled event metadata
rank/prob-only
mean_repeated samples
no-prior/context-only strong flow
```

Attribution ledger:

```text
replacement gap = direct_visual_flow - full_geomoco_flow
prior gain      = no-prior/context-only - full_geomoco_flow
metadata gain   = shuffled/rank-prob-only - full_geomoco_flow
diversity gain  = mean_repeated - full_geomoco_flow
```

Use positive MSE reductions as gains.

## Negative Interpretation

If direct visual flow matches or beats GeoMoCo-conditioned flow:

```text
In this setting, the in-distribution action-MSE benefit of GeoMoCo-WM is
replaceable by a stronger direct policy.
```

That does not kill GeoMoCo-WM as a research direction. It changes the value
claim toward:

```text
sample efficiency
OOD/task generalization
interpretable future-motion candidates
failure diagnosis
planning/reranking interfaces
contact/transition candidate quality
```

## Initial Commands

Direct visual:

```bash
.venv/bin/python scripts/train_flow_matching_action_policy.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --output-dir outputs/flow_matching_action_policy/gate3_9a_direct_visual_seed7 \
  --condition-mode direct_visual \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --quiet
```

GeoMoCo-conditioned:

```bash
.venv/bin/python scripts/train_flow_matching_action_policy.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --output-dir outputs/flow_matching_action_policy/gate3_9a_geomoco_top4_k16_seed7 \
  --condition-mode geomoco \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --epochs 20 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --quiet
```

Use `--num-eval-passes 5` for repeated validation.
