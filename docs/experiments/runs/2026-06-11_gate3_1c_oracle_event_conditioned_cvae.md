# Gate 3.1c Oracle-Event Conditioned cVAE

## Purpose

Gate 3.1b showed that aligned visual context predicts gripper-event timing.
Gate 3.1c tests whether explicit event-mode conditioning improves the joint
`future_delta_gripper` cVAE sample space.

This run is an upper-bound diagnostic because it uses oracle event modes derived
from future action chunks. It is not deployable yet.

## Inputs

```text
windows:
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl

visual cache:
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5

event labels:
outputs/event_modes/gate3_1a_event_modes_2files.json

action decoders:
outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed7/model.pt
outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed17/model.pt
```

## Model Change

The cVAE architecture itself is unchanged. Gate 3.1c appends event-mode one-hot
conditioning to the existing suite/task conditioning:

```text
condition = [suite_task_one_hot, event_mode_one_hot, visual_grounding]
```

For this first cVAE run, `event_class_set=all_observed` is used so the 14 rare
mixed-transition windows do not require a separate dataset filter.

## Commands

Oracle-event cVAE:

```bash
.venv/bin/python scripts/train_visual_cvae_future_motion.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --output-dir outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed{7,17} \
  --motion-mode future_delta_gripper \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --event-conditioning-mode oracle \
  --event-class-set all_observed \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-dims 256,256 \
  --latent-dim 32 \
  --beta-kl 0.001 \
  --beta-kl-start 0.0 \
  --beta-kl-warmup-epochs 5 \
  --free-bits 0.02 \
  --prior-recon-weight 0.5 \
  --action-aware-loss-weight 0.3 \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed{7,17}/model.pt \
  --seed {7,17} \
  --device cuda \
  --split-by episode \
  --condition-on suite_task \
  --quiet
```

Shuffled-event control uses the same command with:

```text
--event-conditioning-mode shuffled
--event-shuffle-seed 7
```

Evaluation:

```bash
.venv/bin/python scripts/evaluate_visual_cvae_samples.py \
  --checkpoint <checkpoint>/model.pt \
  --output-json outputs/visual_cvae_samples/<tag>_k16.json \
  --event-mode-audit-json outputs/event_modes/gate3_1a_event_modes_2files.json \
  --num-samples 16 \
  --batch-size 64 \
  --device cuda \
  --seed {7,17} \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate2_4i_future_delta_gripper_seed{7,17}/model.pt
```

## Results

Mean across seeds:

| branch | prior action MSE | best-of-K action MSE | prior gripper MSE | best-of-K gripper MSE | prior flat MSE | best-of-K flat MSE | sample pair L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.5c unconditional | 0.043816 | 0.022139 | 0.183879 | 0.052857 | 0.026787 | 0.007716 | 0.612986 |
| Gate 3.1c oracle event | 0.018448 | 0.014656 | 0.030296 | 0.010571 | 0.004970 | 0.001873 | 0.179034 |
| Gate 3.1c shuffled event | 0.042406 | 0.021296 | 0.169936 | 0.049051 | 0.024883 | 0.007204 | 0.574333 |

Per-seed action MSE:

| branch | seed 7 prior | seed 7 best-of-K | seed 17 prior | seed 17 best-of-K |
| --- | ---: | ---: | ---: | ---: |
| Gate 2.5c unconditional | 0.042954 | 0.021053 | 0.044679 | 0.023226 |
| Gate 3.1c oracle event | 0.018684 | 0.014754 | 0.018212 | 0.014558 |
| Gate 3.1c shuffled event | 0.045546 | 0.022360 | 0.039266 | 0.020232 |

## Artifacts

```text
outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/
outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed17/
outputs/visual_cvae_future_motion/gate3_1c_shuffled_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/
outputs/visual_cvae_future_motion/gate3_1c_shuffled_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed17/

outputs/visual_cvae_samples/gate3_1c_oracle_event_joint_cvae_seed7_k16.json
outputs/visual_cvae_samples/gate3_1c_oracle_event_joint_cvae_seed17_k16.json
outputs/visual_cvae_samples/gate3_1c_shuffled_event_joint_cvae_seed7_k16.json
outputs/visual_cvae_samples/gate3_1c_shuffled_event_joint_cvae_seed17_k16.json
```

## Interpretation

Gate 3.1c is a strong positive upper-bound result.

Oracle event conditioning dramatically improves both prior mean and best-of-K
coverage:

```text
prior action MSE: 0.043816 -> 0.018448
best-of-K action MSE: 0.022139 -> 0.014656
prior gripper MSE: 0.183879 -> 0.030296
best-of-K gripper MSE: 0.052857 -> 0.010571
```

The shuffled-event control stays close to the unconditional cVAE:

```text
best-of-K action MSE:
unconditional 0.022139
shuffled event 0.021296
oracle event 0.014656
```

So the gain is not explained by a larger conditioning vector. The useful signal
is aligned event timing.

## Decision

Proceed to Gate 3.1d predicted-event / top-M event mixture.

Keep the claim scoped:

```text
oracle event mode is an upper bound
predicted event-mode mixture is the deployable route
```

Gate 3.1c justifies the event-mode structure. It does not yet solve deployment.
