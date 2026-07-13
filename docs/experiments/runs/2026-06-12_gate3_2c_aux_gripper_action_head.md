# Gate 3.2c Auxiliary Gripper Action Head

## Purpose

Gate 3.2b showed that transition timing is trainable but scalar transition
loss weighting is too blunt: it improves transition windows while damaging
sustain and overall action MSE.

Gate 3.2c tests a more structured but still minimal idea:

```text
keep the main action head and flat action MSE
+ add an auxiliary future-gripper readout
+ optionally replace the action gripper dimension with the auxiliary readout
```

First-principles motivation:

```text
SE(3) action dimensions are continuous geometric controls.
The gripper dimension is close to a discrete open/close timing variable.
A single flat MSE head may under-model this timing boundary.
```

## Model Change

`MotionPriorActionHead` now has an optional auxiliary gripper head.

Default behavior is unchanged:

```text
model(context, samples, conditioning, sample_features) -> actions [B,H,7]
```

Auxiliary behavior is opt-in:

```text
model.forward_with_aux(...) -> {
  "actions": [B,H,7],
  "aux_gripper": [B,H] or None
}
```

The auxiliary head is enabled only when:

```text
--aux-gripper-loss-weight > 0
```

## Training Objective

Main loss stays flat action MSE:

```text
L_action = mean((a_pred - a_gt)^2)
```

Auxiliary loss:

```text
L_aux = mean((g_aux - g_gt)^2)
```

Total loss:

```text
L = L_action + lambda_aux * L_aux
```

Swept:

```text
lambda_aux = 0.3
lambda_aux = 1.0
```

Both branches use:

```text
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
selection_metric = aux_replaced_mse
seeds = 7, 17
```

## Code

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
scripts/audit_predicted_event_mixture_action_head_groups.py
tests/test_motion_prior_action_head.py
```

## Commands

Training, for each `weight in {0.3,1.0}` and `seed in {7,17}`:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_2c_auxgripper_w${tag}_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --aux-gripper-loss-weight ${weight} \
  --selection-metric aux_replaced_mse \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-dims 512,512 \
  --token-dim 256 \
  --num-heads 4 \
  --temporal-layers 1 \
  --set-aggregator context_attention \
  --dropout 0.1 \
  --seed ${seed} \
  --device cuda \
  --quiet
```

Repeated eval:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2c_auxgripper_w${tag}_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2c_auxgripper_w${tag}_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

Group audit:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2c_auxgripper_w${tag}_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2c_auxgripper_w${tag}_top4_k16_seed${seed}/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Results

Mean over seeds 7 and 17:

| branch | readout | overall MSE | overall gripper MSE | sustain MSE | transition MSE | transition gripper MSE | transition-open MSE | transition-close MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | main | 0.034767 | 0.150052 | 0.022793 | 0.134087 | 0.827336 | 0.150220 | 0.118580 |
| Gate 3.2c aux w0.3 | main | 0.036377 | 0.153010 | 0.024518 | 0.134748 | 0.824031 | 0.152771 | 0.117548 |
| Gate 3.2c aux w0.3 | aux | 0.036388 | 0.153091 | 0.024624 | 0.133936 | 0.818348 | 0.151008 | 0.117525 |
| Gate 3.2c aux w1.0 | main | 0.036503 | 0.154335 | 0.024248 | 0.137979 | 0.850726 | 0.153282 | 0.123531 |
| Gate 3.2c aux w1.0 | aux | 0.036599 | 0.155007 | 0.024403 | 0.137685 | 0.848667 | 0.152008 | 0.124143 |

## Interpretation

The auxiliary gripper readout is a negative ablation.

It produces a tiny transition improvement in one branch:

```text
aux w0.3 transition MSE:
main readout 0.134748
aux readout  0.133936
```

But it does not beat the Gate 3.1f baseline:

```text
baseline transition MSE: 0.134087
aux w0.3 transition MSE: 0.133936
```

The difference is too small to matter, and the overall result gets worse:

```text
baseline overall MSE: 0.034767
aux w0.3 aux-readout MSE: 0.036388
aux w1.0 aux-readout MSE: 0.036599
```

So simply adding another gripper regression head does not solve the timing
problem. It still treats open/close timing as regression rather than as an
event-conditional decision.

## Decision

Do not promote the auxiliary gripper head.

Keep Gate 3.1f full event/rank/prob top-4 as the deployable default.

Promote this diagnosis:

```text
The transition bottleneck is not solved by scalar loss weighting or by a
parallel gripper regression head. The next branch should explicitly route or
condition behavior on transition state, rather than merely adding another
regression target.
```

Next mainline:

```text
Gate 3.2d: transition-gated residual action head or event-routed action head.
```

Candidate routes:

```text
1. predict a transition gate and use it to blend sustain-action and
   transition-action residuals;
2. condition the action head on predicted event family at the output layer,
   not only at the sample-token level;
3. train separate gripper residuals for sustain/open-transition/close-transition
   families while keeping shared SE(3) action regression.
```
