# Gate 2.4h-d Minimal Event-Aware Readout

- Date: 2026-06-09
- Status: completed
- Gate: Gate 2.4h-d
- Purpose: test whether a weak transition-event ranking auxiliary can improve
  cVAE sample readout without hurting deployable action MSE.

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
| windows | 16,518 |
| horizon | 8 |
| split policy | episode |
| seeds | 7, 17 |
| cVAE samples | 16 |
| device | cuda |

## Code

```text
scripts/train_visual_cvae_sample_scorer.py
scripts/evaluate_cvae_event_alignment.py
tests/test_future_motion_predictor.py
```

New optional training path:

```text
--event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json
--event-target-weight 0.1
```

The default ScoreNet path remains unchanged when `event_target_weight=0`.

## Commands

Seed 7:

```bash
.venv/bin/python scripts/train_visual_cvae_sample_scorer.py \
  --checkpoint outputs/visual_cvae_future_motion/gate2_4c_visual_cvae_freebits002_warmup5_lam003_seed7/model.pt \
  --output-dir outputs/visual_cvae_sample_scorer/gate2_4h_d_event_w01_k16_seed7 \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --event-target-weight 0.1 \
  --target-kind action \
  --num-samples 16 \
  --epochs 20 \
  --batch-size 64 \
  --device cuda \
  --seed 7 \
  --quiet
```

Seed 17 uses matching `seed17` cVAE and `--seed 17`.

Event-alignment evaluation:

```bash
.venv/bin/python scripts/evaluate_cvae_event_alignment.py \
  --checkpoint <matching-cvae-checkpoint> \
  --scorer-checkpoint <gate2_4h_d_event_w01_scorer> \
  --event-audit-json outputs/event_audits/gate2_4h_gripper_transitions_2files.json \
  --output-json outputs/event_alignment/gate2_4h_d_event_w01_seed<seed>_k16.json \
  --num-samples 16 \
  --batch-size 128 \
  --device cuda \
  --seed <seed> \
  --quiet
```

## Artifacts

Training outputs:

```text
outputs/visual_cvae_sample_scorer/gate2_4h_d_event_w01_k16_seed7/
outputs/visual_cvae_sample_scorer/gate2_4h_d_event_w01_k16_seed17/
```

Event evaluation:

```text
outputs/event_alignment/gate2_4h_d_event_w01_seed7_k16.json
outputs/event_alignment/gate2_4h_d_event_w01_seed17_k16.json
```

## Results

Mean over seeds 7 and 17:

| readout | action MSE | event acc | macro-F1 | transition acc | step within 1 | event rank | event error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate 2.4d flat ScoreNet | 0.040201 | 0.839062 | 0.553349 | 0.408688 | 0.175082 | 1.256558 | 2.401051 |
| Gate 2.4h-d event w=0.1 | 0.040255 | 0.838916 | 0.552820 | 0.407396 | 0.173790 | 1.254738 | 2.401610 |

Seed-level action MSE:

| seed | flat ScoreNet | event w=0.1 |
| ---: | ---: | ---: |
| 7 | 0.042053 | 0.042132 |
| 17 | 0.038350 | 0.038377 |

## Interpretation

Gate 2.4h-d is a negative / neutral diagnostic.

The weak event target slightly improves mean selected event-oracle rank
(`1.256558 -> 1.254738`), but this is too small to matter and does not improve
the actual event metrics. It also slightly worsens deployable action MSE
(`0.040201 -> 0.040255`).

This suggests the bottleneck is not merely that the existing ScoreNet ignores a
clean event-aligned candidate. The upstream candidate set and the
motion-to-action interface still do not encode close/open timing strongly
enough.

## Decision

Do not promote event-aware ScoreNet as a method component yet.

Next mainline should move upstream:

```text
Gate 2.4i: event-fidelity interface audit
```

Candidate directions:

- add explicit future gripper / event channels to the world-motion target;
- train an event-aware action decoder diagnostic;
- test whether GT future motion plus event labels gives a stronger oracle
  action interface than GT EEF-only future motion;
- only return to event-aware readout if the candidate samples actually contain
  stronger transition timing signal.

## Verification

```text
.venv/bin/python -m unittest tests.test_future_motion_predictor tests.test_cvae_event_alignment tests.test_event_labels
.venv/bin/python -m compileall scripts/train_visual_cvae_sample_scorer.py tests/test_future_motion_predictor.py
.venv/bin/ruff check scripts/train_visual_cvae_sample_scorer.py tests/test_future_motion_predictor.py
```

All passed before formal CUDA runs.

