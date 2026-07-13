# Gate 3.9a Flow-Matching Replacement Audit Initial Runs

## Purpose

Gate 3.9a starts the strong-policy replacement audit:

```text
Does GeoMoCo-WM still help when the action policy is a stronger conditional
flow-matching action-chunk model?
```

This initial slice only runs the seed-7 pair:

```text
direct_visual_flow
geomoco_conditioned_flow
```

No seed17 or attribution controls are run unless the pair gives a clean positive
signal.

## Implementation

Added:

```text
scripts/train_flow_matching_action_policy.py
```

The model trains a rectified-flow objective over normalized action chunks:

```text
x1 = ground-truth action chunk [B,H,A]
x0 = Gaussian noise
t  ~ Uniform(0, 1)
xt = (1 - t) x0 + t x1
target velocity = x1 - x0
loss = MSE(v_theta(xt, t, condition), target velocity)
```

Inference uses Euler integration from noise to action chunk. Initial runs use
8 Euler steps.

Condition modes:

```text
direct_visual:
  context/proprio + suite/task + DINO visual feature

geomoco:
  direct_visual condition
  + predicted top-4 event-mixture samples
  + event/rank/prob sample metadata
```

## Commands

Direct visual:

```bash
.venv/bin/python scripts/train_flow_matching_action_policy.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt \
  --output-dir outputs/flow_matching_action_policy/gate3_9a_direct_visual_seed7 \
  --condition-mode direct_visual \
  --epochs 20 \
  --batch-size 64 \
  --eval-steps 8 \
  --num-eval-passes 5 \
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
  --eval-steps 8 \
  --num-eval-passes 5 \
  --seed 7 \
  --device cuda \
  --quiet
```

## Results

Seed 7, 5-pass repeated validation:

| branch | overall MSE | transition MSE | sustain MSE | gripper MSE | flow MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_visual_flow | 0.059032 | 0.227686 | 0.037597 | 0.266500 | 0.120325 |
| geomoco_flow | 0.058218 | 0.216756 | 0.038069 | 0.267212 | 0.125919 |

GeoMoCo minus direct:

```text
overall MSE:    -0.000814
transition MSE: -0.010929
sustain MSE:    +0.000472
gripper MSE:    +0.000713
flow MSE:       +0.005594
```

Reference Gate 3.4 seed 7:

```text
temporal_action_mse            = 0.036502
temporal_action_transition_mse = 0.137827
temporal_action_gripper_mse    = 0.164389
```

## Interpretation

This is a weak mixed signal, not a promotion result.

Positive part:

```text
GeoMoCo-conditioned flow improves overall MSE and transition MSE relative to
the direct visual flow baseline.
```

Negative part:

```text
GeoMoCo-conditioned flow slightly worsens gripper MSE.
Both flow branches are much weaker than Gate 3.4, so this is not yet a strong
replacement-policy baseline.
```

The current flow policy therefore does not answer the strong-policy replacement
question cleanly. It is too weak to say that a strong policy replaces
GeoMoCo-WM, and the GeoMoCo gain is not clean enough to justify full
attribution controls yet.

## Decision

Do not expand immediately to seed17/full controls.

The next useful step is to calibrate the flow policy itself before using it as
the strong-policy replacement opponent. Minimal calibration options:

```text
1. evaluate/train with more Euler steps, such as 16 or 32;
2. add a deterministic action-readout auxiliary head for action-MSE
   calibration while keeping the flow objective;
3. reduce stochastic-eval variance with an eval mean over multiple initial
   noises per pass;
4. only after direct_visual_flow is a credible strong baseline, repeat the
   GeoMoCo vs direct replacement audit.
```

## Verification

Checks run:

```text
.venv/bin/python -m compileall -q scripts/train_flow_matching_action_policy.py
.venv/bin/ruff check scripts/train_flow_matching_action_policy.py
direct_visual dry-run
geomoco dry-run
direct_visual CPU smoke
geomoco CPU smoke
```

Artifacts:

```text
outputs/flow_matching_action_policy/gate3_9a_direct_visual_seed7/
outputs/flow_matching_action_policy/gate3_9a_geomoco_top4_k16_seed7/
outputs/flow_matching_action_policy/gate3_9a_direct_visual_smoke_seed7/
outputs/flow_matching_action_policy/gate3_9a_geomoco_smoke_seed7/
```
