# Gate 3.1f Event-Aware Sample Consumption

## Purpose

Gate 3.1e showed that predicted event-mixture samples contain signal but that
anonymous sample-set aggregation is not enough. Gate 3.1f tests whether the
action head can use the predicted event-mixture better when each future sample
carries event metadata.

Each sample token receives:

```text
future_delta_gripper sample
event-mode one-hot
event rank among top-M
event probability within top-M
```

The action head still does not receive oracle event labels. Event metadata comes
from the deployable visual event predictor.

## Code

```text
src/geomoco_wm/models/motion_prior_action_head.py
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
tests/test_motion_prior_action_head.py
```

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_1f_eventaware_top{2,4}_k16_seed{7,17} \
  --event-top-m {2,4} \
  --num-samples 16 \
  --sample-feature-mode event_rank_prob \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-dims 512,512 \
  --token-dim 256 \
  --num-heads 4 \
  --temporal-layers 1 \
  --set-aggregator context_attention \
  --dropout 0.1 \
  --seed {7,17} \
  --device cuda \
  --quiet
```

Repeated evaluation:

```bash
.venv/bin/python scripts/evaluate_predicted_event_mixture_action_head.py \
  --checkpoint outputs/motion_prior_action_head/gate3_1f_eventaware_top{2,4}_k16_seed{7,17}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_1f_eventaware_top{2,4}_k16_seed{7,17}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

## Results

Mean across seeds 7 and 17, 5-pass stochastic evaluation:

| branch | action MSE | action MAE | translation m MSE | rotation geodesic deg | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.0 real sample-set K=16 | 0.036675 | 0.088318 | 0.000072 | 1.970024 | 0.164061 |
| Gate 3.0 shuffled sample-set K=16 | 0.042633 | 0.100501 | 0.000088 | 2.060474 | 0.186272 |
| Gate 3.1e anonymous top-2 | 0.038052 | 0.092523 | 0.000075 | 1.965227 | 0.169784 |
| Gate 3.1e anonymous top-4 | 0.038024 | 0.094724 | 0.000077 | 1.997687 | 0.167432 |
| Gate 3.1f event-aware top-2 | 0.036671 | 0.089319 | 0.000071 | 1.954026 | 0.165363 |
| Gate 3.1f event-aware top-4 | 0.034767 | 0.089716 | 0.000072 | 1.968135 | 0.150052 |

## Artifacts

```text
outputs/motion_prior_action_head/gate3_1f_eventaware_top2_k16_seed7/
outputs/motion_prior_action_head/gate3_1f_eventaware_top2_k16_seed17/
outputs/motion_prior_action_head/gate3_1f_eventaware_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_1f_eventaware_top4_k16_seed17/
```

## Interpretation

Gate 3.1f is a positive result.

Event metadata turns the predicted event-mixture from a weak anonymous sample
set into a useful structured proposal set. Top-4 is best:

```text
anonymous top-4 action MSE: 0.038024
event-aware top-4 action MSE: 0.034767
Gate 3.0 real sample-set action MSE: 0.036675
```

This means the action head can exploit predicted event structure when event
identity/rank/probability are exposed at the sample-token level.

Top-2 roughly ties the Gate 3.0 real sample-set baseline, while top-4 clearly
beats it. This matches Gate 3.1d: top-4 has much stronger event-mode coverage,
and the event-aware consumer can now handle the wider set.

## Decision

Promote Gate 3.1f event-aware top-4 as the current best deployable
action-head interface.

Next mainline should test robustness:

```text
event metadata ablations:
event one-hot only
rank/prob only
probability-weighted sample allocation
shuffled event metadata control
```

Do not jump to flow matching yet; first lock down which event metadata carries
the gain.
