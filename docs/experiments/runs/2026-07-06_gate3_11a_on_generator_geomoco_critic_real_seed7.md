# Gate 3.11a On-Generator GeoMoCo Critic Real Seed 7

- Date: 2026-07-06
- Status: completed seed-7 real branch; controls paused by stop rule
- Gate: 3.11a
- Purpose: test whether a GraspGen-style on-generator critic can extract
  downstream action value from current GeoMoCo event-mixture candidates.

## Dataset Slice

Source:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Summary:

| field | value |
| --- | ---: |
| windows | 16,518 |
| train windows | 12,974 |
| validation windows | 3,544 |
| split | episode |
| horizon | 8 |
| motion mode | future_delta_gripper |

## Model And Training Config

```text
script: scripts/train_on_generator_geomoco_critic.py
action-head checkpoint:
  outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt
cVAE checkpoint:
  outputs/visual_cvae_future_motion/gate3_1c_oracle_event_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt
event probe:
  outputs/event_mode_probe/gate3_1b_visual_proprio_seed7/model.pt
candidate control: real
event top-M: 4
num samples: 16
sample feature mode: event_rank_prob
critic: MLP, hidden_dims=(512,512), soft-CE ranking target
epochs: 10
batch size: 64
selection metric: critic_selected_mse
seed: 7
checkpoint split seed: 7
device: cuda
```

## Commands

```bash
.venv/bin/python scripts/train_on_generator_geomoco_critic.py \
  --checkpoint outputs/motion_prior_action_head/gate3_4_temporal_action_top4_k16_seed7/model.pt \
  --output-dir outputs/on_generator_geomoco_critic/gate3_11a_real_seed7 \
  --candidate-control real \
  --epochs 10 \
  --batch-size 64 \
  --seed 7 \
  --device cuda \
  --quiet
```

The restricted shell could not see CUDA, so the same command was rerun in the
approved GPU-visible execution mode.

## Results

Best epoch: 5.

| metric | value |
| --- | ---: |
| set_temporal_mse | 0.036371 |
| candidate_mean_mse | 0.166857 |
| candidate_oracle_mse | 0.026081 |
| oracle_gain_vs_set | 0.010291 |
| candidate_best_vs_mean_gap | 0.140776 |
| candidate_oracle_beats_set | 0.319131 |
| critic_selected_mse | 0.050388 |
| critic_selected_gain_vs_set | -0.014017 |
| critic_gain_vs_mean | 0.116468 |
| critic_gap_to_oracle | 0.024308 |
| critic_top1_accuracy | 0.184819 |
| critic_selected_beats_set | 0.216986 |

Group metrics:

| group | count | set_mse | oracle_mse | oracle_gain | critic_mse | critic_gain_vs_set | critic_gap_to_oracle | top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sustain | 3,141 | 0.023453 | 0.020855 | 0.002598 | 0.032656 | -0.009203 | 0.011801 | 0.198663 |
| transition | 403 | 0.137058 | 0.066808 | 0.070250 | 0.188597 | -0.051539 | 0.121789 | 0.076923 |

## Artifacts

```text
metrics:
  outputs/on_generator_geomoco_critic/gate3_11a_real_seed7/metrics.json
checkpoint:
  outputs/on_generator_geomoco_critic/gate3_11a_real_seed7/model.pt
plan:
  docs/experiments/plans/2026-07-06_gate3_11a_on_generator_geomoco_critic_audit_plan.md
```

## Interpretation

The current GeoMoCo generator candidate set has real oracle headroom:
`candidate_oracle_mse=0.026081` beats the Gate 3.4 set readout
`set_temporal_mse=0.036371`. The headroom is especially large on transition
windows, where the oracle candidate improves `0.137058 -> 0.066808`.

The lightweight critic does not extract that value. Its selected candidate
regresses to `critic_selected_mse=0.050388`, worse than the set readout, and
transition selection is particularly bad (`0.188597` vs set `0.137058`).

This is therefore:

```text
candidate-space positive
critic-extraction negative
```

Do not expand the full control matrix yet. The real branch already fails the
predeclared deployable selection check.

## Limits

This run does not prove that on-generator critics are ineffective. It only
tests a first MLP critic with per-candidate temporal-action regret labels. It
does not use transition-specific targets, pairwise/set-wise comparison, contact
or object-state labels, or task-phase proxies.

## Next Decision

Stop the broad Gate 3.11a control expansion. The next variant should redesign
the critic target before controls, with priority on:

```text
transition-specific regret
gripper/contact regret
pairwise or set-wise candidate comparison
contact/object-state/task-phase proxy labels
```
