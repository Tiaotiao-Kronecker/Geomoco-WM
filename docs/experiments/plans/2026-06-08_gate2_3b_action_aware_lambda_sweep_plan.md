# Gate 2.3b Action-Aware Lambda Sweep Plan

- Date: 2026-06-08
- Status: planned, then execute immediately
- Slice: four LIBERO suites, 2 HDF5 task files per suite, all demos, horizon 8
- Windows: `outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl`
- Visual cache: `outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5`
- Split: episode-level train/validation split
- Seeds: 7 and 17

## Purpose

Gate 2.3a showed that adding a small frozen-action-decoder auxiliary loss
improves downstream action value:

```text
motion_loss = MSE(pred_future_ee_delta, gt_future_ee_delta)
action_loss = MSE(frozen_action_decoder(context, pred_future_ee_delta), gt_action_chunk)
total_loss = motion_loss + lambda_action * action_loss
```

Gate 2.3b sweeps `lambda_action` to find whether `0.01` is a stable sweet spot
or whether a weaker/stronger action-aware signal is better.

## Sweep Matrix

| lambda | status | reason |
| ---: | --- | --- |
| 0.003 | run now | Weaker action-aware signal; should preserve motion geometry better. |
| 0.010 | already completed as Gate 2.3a | Current best branch. |
| 0.030 | run now | Stronger action-aware signal; may improve action metrics but risks motion distortion. |

## Reference Results

| branch | future MSE | action MSE | action MAE | trans L2 (m) | rot geo (deg) | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.2b visual MSE-only | 0.000772 | 0.049547 | 0.120370 | 0.014859 | 2.030450 | 0.222467 |
| Gate 2.3a lambda 0.010 | 0.000770 | 0.043174 | 0.113432 | 0.014835 | 2.037468 | 0.177930 |
| Oracle future motion | n/a | 0.031474 | 0.079508 | 0.007466 | 1.048033 | 0.184683 |

## Command Template

```bash
.venv/bin/python scripts/train_future_motion_predictor.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl \
  --visual-feature-cache outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5 \
  --visual-fusion cross_attention \
  --output-dir outputs/future_motion_predictor/<RUN_TAG> \
  --epochs 20 \
  --batch-size 64 \
  --hidden-dims 256,256 \
  --split-by episode \
  --condition-on suite_task \
  --seed <SEED> \
  --device cuda \
  --action-decoder-checkpoint outputs/oracle_action_decoder/gate1_6_geodesic_future_seed<SEED>/model.pt \
  --action-aware-loss-weight <LAMBDA>
```

## Selection Rule

Use the lambda with the best mean action MSE if future-motion metrics do not
collapse. Treat this as a tradeoff, not a one-number leaderboard:

- primary: downstream action MSE / MAE;
- secondary: translation L2 and SO(3) geodesic action metrics;
- guardrail: future-motion MSE and translation/orientation future-motion L2;
- diagnostic: gripper MSE, because Gate 2.3a unexpectedly improved it.

If stronger lambda improves action MSE but sharply worsens future-motion
geometry, keep `lambda=0.01` as the default deterministic baseline and move the
stronger branch to an action-interface diagnostic.
