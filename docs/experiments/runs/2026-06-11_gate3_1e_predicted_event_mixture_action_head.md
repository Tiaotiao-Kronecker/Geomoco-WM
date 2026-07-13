# Gate 3.1e Predicted Event Mixture Action Head

## Purpose

Gate 3.1d showed that predicted top-4 event-mixture samples contain good
futures under oracle best-of-K. Gate 3.1e asks whether a deployable action head
can consume those samples without oracle event labels.

The action head receives:

```text
context + suite/task conditioning + K future_delta_gripper samples
```

It does not receive oracle event one-hot labels. Event one-hot vectors are used
only inside the frozen event-conditioned cVAE to generate candidate futures.

## Dataset And Inputs

```text
windows:
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl

visual cache:
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5

cVAE:
outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt

event probe:
outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/model.pt
```

## Code

```text
scripts/train_predicted_event_mixture_action_head.py
scripts/evaluate_predicted_event_mixture_action_head.py
```

The script uses the same `MotionPriorActionHead` architecture as Gate 3.0.

## Commands

Training:

```bash
.venv/bin/python scripts/train_predicted_event_mixture_action_head.py \
  --checkpoint outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17}/model.pt \
  --event-probe-checkpoint outputs/event_mode_probe/gate3_1b_visual_proprio_seed{7,17}/model.pt \
  --output-dir outputs/motion_prior_action_head/gate3_1e_pred_event_top{2,4}_k16_seed{7,17} \
  --event-top-m {2,4} \
  --num-samples 16 \
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
  --checkpoint outputs/motion_prior_action_head/gate3_1e_pred_event_top{2,4}_k16_seed{7,17}/model.pt \
  --output-json outputs/motion_prior_action_head/gate3_1e_pred_event_top{2,4}_k16_seed{7,17}/repeated_eval_5pass.json \
  --num-eval-passes 5 \
  --device cuda
```

## Results

Mean across seeds 7 and 17, using 5-pass repeated stochastic evaluation:

| branch | action MSE | action MAE | translation m MSE | rotation geodesic deg | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate 3.0 real sample-set K=16 | 0.036675 | 0.088318 | 0.000072 | 1.970024 | 0.164061 |
| Gate 3.0 shuffled sample-set K=16 | 0.042633 | 0.100501 | 0.000088 | 2.060474 | 0.186272 |
| Gate 3.1e predicted top-2 | 0.038052 | 0.092523 | 0.000075 | 1.965227 | 0.169784 |
| Gate 3.1e predicted top-4 | 0.038024 | 0.094724 | 0.000077 | 1.997687 | 0.167432 |

## Artifacts

```text
outputs/motion_prior_action_head/gate3_1e_pred_event_top2_k16_seed7/
outputs/motion_prior_action_head/gate3_1e_pred_event_top2_k16_seed17/
outputs/motion_prior_action_head/gate3_1e_pred_event_top4_k16_seed7/
outputs/motion_prior_action_head/gate3_1e_pred_event_top4_k16_seed17/
```

## Interpretation

Gate 3.1e is a weak positive versus shuffled controls, but not a promoted
replacement for the Gate 3.0 unconditional real sample-set baseline.

The predicted event-mixture samples are aligned enough to beat shuffled samples:

```text
predicted top-4 action MSE: 0.038024
shuffled sample-set action MSE: 0.042633
```

But they do not beat the simpler unconditional real sample set:

```text
predicted top-4 action MSE: 0.038024
Gate 3.0 real sample-set action MSE: 0.036675
```

Top-2 and top-4 are nearly tied. This suggests the bottleneck is not only
whether the event proposal is too wide; the current action-head interface does
not yet exploit the structured event mixture well enough.

## Decision

Do not promote predicted event-mixture action-head v1 as the main result.

Keep Gate 3.1d as evidence that predicted event modes can place good futures in
the sample set. Treat Gate 3.1e as a stop signal for naive sample-set
aggregation over a wide event mixture.

Next work should improve the sample-set consumption interface, for example by
passing event-rank/probability tokens, using probability-weighted sample
allocation, or training a set-wise planner/readout that is aware of event mode
identity.
