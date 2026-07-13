# Gate 3.2b Transition-Weighted Action Head

## Purpose

Gate 3.2a showed that the promoted event-aware top-4 action-head interface is
globally stable, but its remaining error is concentrated in gripper transition
windows:

```text
sustain action MSE:    0.022793
transition action MSE: 0.134087
```

Gate 3.2b asks whether this is simply a training-distribution imbalance. From
first principles, if most windows are sustain windows, flat action MSE can learn
a good average policy while under-serving rare open/close boundary windows.

So this gate changes only the training loss:

```text
flat action MSE
-> transition-weighted action MSE
```

It keeps the same cVAE, event probe, predicted event-mixture samples, per-sample
event/rank/prob metadata, action-head architecture, and evaluation protocol.

## Method

Training loss:

```text
per_item_mse_i = mean((a_pred_i - a_gt_i)^2)

weight_i = transition_loss_weight if true_event_i is transition
           1 otherwise

L = sum_i weight_i * per_item_mse_i / sum_i weight_i
```

Swept weights:

```text
transition_loss_weight = 2
transition_loss_weight = 4
```

Both branches use:

```text
event_top_m = 4
num_samples = 16
sample_feature_mode = event_rank_prob
selection_metric = weighted_loss
seeds = 7, 17
```

## Code

```text
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
tests/test_motion_prior_action_head.py
```

New training options:

```text
--loss-weight-mode transition
--transition-loss-weight <2|4>
--selection-metric weighted_loss
--event-mode-audit-json <optional; defaults to cVAE checkpoint audit JSON>
```

## Commands

Training, for each `weight in {2,4}` and `seed in {7,17}`:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed${seed}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed${seed}/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_2b_transition_weight${weight}_top4_k16_seed${seed} \
  --event-top-m 4 \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --loss-weight-mode transition \
  --transition-loss-weight ${weight}.0 \
  --selection-metric weighted_loss \
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
  --checkpoint outputs/motion_prior_action_head/gate3_2b_transition_weight${weight}_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2b_transition_weight${weight}_top4_k16_seed${seed}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

Group stress audit:

```bash
.venv/bin/python scripts/audit_predicted_event_mixture_action_head_groups.py \
  --checkpoint outputs/motion_prior_action_head/gate3_2b_transition_weight${weight}_top4_k16_seed${seed}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_2b_transition_weight${weight}_top4_k16_seed${seed}/group_stress_3pass.json \
  --num-eval-passes 3 \
  --device cuda
```

## Repeated Eval Results

Mean over seeds 7 and 17:

| branch | action MSE | gripper MSE | transition MSE | sustain MSE | weighted loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | 0.034767 | 0.150052 | - | - | - |
| Gate 3.2b weight=2 | 0.035974 | 0.157348 | 0.124905 | 0.025232 | 0.044410 |
| Gate 3.2b weight=4 | 0.037963 | 0.160325 | 0.121979 | 0.027817 | 0.057744 |

## Group Stress Results

Mean over seeds:

| branch | overall MSE | sustain MSE | transition MSE | transition open MSE | transition close MSE | transition gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 3.1f baseline | 0.034773 | 0.022793 | 0.134087 | 0.150220 | 0.118580 | 0.827336 |
| Gate 3.2b weight=2 | 0.035986 | 0.025233 | 0.125002 | 0.132274 | 0.117343 | 0.766794 |
| Gate 3.2b weight=4 | 0.037981 | 0.027829 | 0.122045 | 0.135566 | 0.108494 | 0.742795 |

## Interpretation

Transition weighting is a mechanism-positive but deployable-negative result.

It does improve the targeted subgroup:

```text
transition MSE:
baseline 0.134087
weight=2 0.125002
weight=4 0.122045

transition gripper MSE:
baseline 0.827336
weight=2 0.766794
weight=4 0.742795
```

So the transition bottleneck is not completely untouchable. The action head can
shift capacity toward open/close timing when the loss asks for it.

But the improvement is not free:

```text
overall MSE:
baseline 0.034773
weight=2 0.035986
weight=4 0.037981

sustain MSE:
baseline 0.022793
weight=2 0.025233
weight=4 0.027829
```

This means coarse loss weighting mostly moves error between regimes. It does
not create a strictly better deployable interface.

## Decision

Do not promote transition-weighted action heads as the default.

Promote the diagnosis:

```text
The transition bottleneck is real and trainable, but simple scalar weighting is
too blunt. The next action-head step should separate transition timing from
sustain action regression rather than forcing one flat MSE head to serve both.
```

Next mainline:

```text
Gate 3.2c: transition-aware action head or auxiliary transition/gripper timing
readout.
```

Candidate routes:

```text
1. output an auxiliary future gripper/event-transition head and train it with
   event/timing supervision;
2. use a small transition gate to route between sustain and transition action
   residuals;
3. keep the action MSE head flat, but add a separate transition-timing loss
   instead of upweighting all transition action dimensions;
4. postpone flow/diffusion action heads until this deterministic interface can
   localize and repair the transition trade-off.
```
