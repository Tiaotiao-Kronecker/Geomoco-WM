# Gate 2.5d Joint GeoMoCo-cVAE Sample Readout

- Date: 2026-06-10
- Status: completed
- Gate: Gate 2.5d
- Purpose: train a lightweight readout over joint cVAE future-motion samples
  and test whether it can convert best-of-K coverage into deployable action
  value.

## Dataset Slice

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
outputs/event_audits/gate2_4h_gripper_transitions_2files.json
```

| field | value |
| --- | ---: |
| suites | 4 |
| task files | 8 |
| demos | 400 |
| windows | 16,518 |
| horizon | 8 |
| motion dim | 56 |
| motion mode | `future_delta_gripper` |
| split policy | episode |
| seeds | 7, 17 |
| training device | cuda |
| eval device | cuda |

## Code

```text
scripts/train_visual_cvae_sample_scorer.py
scripts/evaluate_visual_cvae_sample_scorer.py
scripts/evaluate_cvae_event_alignment.py
```

Implementation changes:

- scorer training/evaluation now inherit `motion_mode` from the cVAE checkpoint
  and support `future_delta_gripper` end to end;
- readout motion metrics now use motion-mode-aware scoring instead of hard-coded
  EEF-only metrics;
- event-alignment evaluation now uses the dataset horizon explicitly instead of
  inferring it from EEF-only motion dimensionality.

## Commands

Real visual, seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_5d_joint_action_rank_k16_seed7 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --target-kind action \
  --seed 7 \
  --device cuda \
  --quiet
```

Real visual, seed 17:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_5c_joint_cvae_freebits002_warmup5_prw05_lam03_seed17/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_5d_joint_action_rank_k16_seed17 \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --target-kind action \
  --seed 17 \
  --device cuda \
  --quiet
```

Shuffled visual controls used the matching shuffled cVAE checkpoints and the
same scorer recipe.

## Results

### Real visual, mean over seeds 7 and 17

| metric | value |
| --- | ---: |
| prior mean action MSE | 0.043816 |
| scorer argmax action MSE | 0.043414 |
| oracle best action MSE | 0.022192 |
| scorer top-1 oracle match | 0.255727 |
| scorer selected oracle rank | 5.637193 |
| scorer action regret to oracle | 0.021222 |

### Real visual, event readout mean over seeds 7 and 17

| metric | prior mean | scorer argmax |
| --- | ---: | ---: |
| event accuracy | 0.889522 | 0.900561 |
| transition accuracy | 0.568301 | 0.647706 |
| transition step within 1 | 0.239717 | 0.264299 |
| macro-F1 | 0.617667 | 0.639790 |
| balanced accuracy | 0.601051 | 0.633401 |

### Shuffled visual, mean over seeds 7 and 17

| metric | value |
| --- | ---: |
| prior mean action MSE | 0.068816 |
| scorer argmax action MSE | 0.070023 |
| oracle best action MSE | 0.030778 |
| scorer top-1 oracle match | 0.316459 |
| scorer selected oracle rank | 4.693893 |
| scorer action regret to oracle | 0.039245 |

### Shuffled visual, event readout mean over seeds 7 and 17

| metric | prior mean | scorer argmax |
| --- | ---: | ---: |
| event accuracy | 0.787503 | 0.837445 |
| transition accuracy | 0.224003 | 0.491305 |
| transition step within 1 | 0.069768 | 0.192786 |
| macro-F1 | 0.455225 | 0.551454 |
| balanced accuracy | 0.409029 | 0.515370 |

## Seed-Level Notes

Real visual:

- seed 7 scorer action MSE: `0.045230`
- seed 17 scorer action MSE: `0.041597`

Shuffled visual:

- seed 7 scorer action MSE: `0.067943`
- seed 17 scorer action MSE: `0.072103`

## Main Interpretation

Gate 2.5d is a weak-positive / not-promoted result:

- real visual scorer is slightly better than real visual prior mean on action
  MSE, but still worse than the deterministic joint baseline `0.040688`;
- shuffled visual scorer is worse than shuffled prior mean on action MSE;
- real visual does help the readout relative to shuffled, so the scorer is not
  pure noise;
- event fidelity improves modestly on real visual and more strongly on shuffled,
  but this is still not enough to justify promotion;
- oracle best-of-K remains much better than the learned readout, so the main
  bottleneck is readout quality, not candidate existence.

## Decision

Do not promote the current flat action-ranking scorer.

Next step should be one of:

```text
Gate 2.5e: event/contact/executability-aware readout
Gate 2.5e-alt: stronger hard-negative ranking or structure-aware scorer
```

The current sample space is useful, but the deployable readout still has a
large gap to oracle best-of-K.

