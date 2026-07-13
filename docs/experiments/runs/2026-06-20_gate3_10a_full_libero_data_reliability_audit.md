# Gate 3.10a Full LIBERO Data Reliability Audit

## Purpose

Before continuing decoder, timing, or strong-policy replacement work, verify
whether the local full LIBERO data can support stable claims about the current
main bottleneck:

```text
transition/open-close action timing
```

This follows the Gate 3.9a observation that the 2-files-per-suite slice had
only `1,699` transition windows, with close/open counts below `1,000`.

## Full Window Export

Command:

```bash
.venv/bin/python scripts/export_libero_windows.py \
  --input-root /home/user/dataset/libero_official \
  --all-libero-suites \
  --output-dir outputs/libero_windows/libero_all_suites_full_h8 \
  --context-len 2 \
  --horizon 8 \
  --stride 4
```

Result:

```text
source HDF5 task files: 40
episodes:               2,000
windows:                80,883
frames:                 338,575
warnings:               none
```

Artifacts:

```text
outputs/libero_windows/libero_all_suites_full_h8/windows.jsonl
outputs/libero_windows/libero_all_suites_full_h8/episodes.jsonl
outputs/libero_windows/libero_all_suites_full_h8/summary.json
```

## Event-Mode Audit

Command:

```bash
.venv/bin/python scripts/audit_event_modes.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_full_h8/windows.jsonl \
  --output-json outputs/event_modes/gate3_10a_event_modes_full_h8.json \
  --output-md outputs/event_modes/gate3_10a_event_modes_full_h8.md \
  --train-ratio 0.8 \
  --split-by episode \
  --seed 7
```

Event-mode counts:

| event mode | count |
| --- | ---: |
| sustain_close::none | 34,591 |
| sustain_open::none | 37,144 |
| transition_close::early | 1,792 |
| transition_close::middle | 1,847 |
| transition_close::late | 1,210 |
| transition_open::early | 1,508 |
| transition_open::middle | 1,659 |
| transition_open::late | 1,079 |
| mixed_transition::early | 34 |
| mixed_transition::middle | 18 |
| mixed_transition::late | 1 |

Train/validation split:

```text
train windows: 64,138
val windows:   16,745
split-by:      episode
seed:          7
```

All common sustain/transition close/open modes appear in both train and val.
The missing val mode is only `mixed_transition::late`, which has exactly one
window in the whole dataset and should remain diagnostic only.

## Dataset Sufficiency Audit

Command:

```bash
.venv/bin/python scripts/audit_dataset_sufficiency.py \
  --event-mode-audit-json outputs/event_modes/gate3_10a_event_modes_full_h8.json \
  --output-json outputs/dataset_audits/gate3_10a_full_h8_sufficiency_audit.json \
  --output-md outputs/dataset_audits/gate3_10a_full_h8_sufficiency_audit.md
```

Summary:

```text
windows:             80,883
episodes:            2,000
suites:              4
tasks:               40
transition windows:  9,148 = 11.31%
close transitions:   4,849
open transitions:    4,246
mixed transitions:   53
```

Split coverage:

```text
train transition windows: 7,315
train close/open:         3,873 / 3,401
val transition windows:   1,833
val close/open:           976 / 845
```

The pre-declared desired targets were:

```text
transition windows >= 8,000
close transition >= 4,000
open transition >= 4,000
```

The full dataset passes these targets globally.

## Remaining Data Caveats

Transition windows are still a minority:

```text
transition fraction = 11.31%
```

So training and reporting must keep transition/sustain slices separate.
Otherwise a policy can improve sustain/continuous-motion regions while still
hurting the actual close/open timing problem.

Two tasks have zero transition windows under the current gripper-command event
definition:

```text
open_the_middle_drawer_of_the_cabinet
push_the_plate_to_the_front_of_the_stove
```

These tasks should not be used as evidence that open/close timing improved.
They can remain in full-distribution metrics, but transition claims should
either exclude zero-transition tasks or report them separately.

Mixed transitions remain too rare:

```text
mixed_transition total = 53
mixed_transition::late = 1
```

Keep mixed-transition labels diagnostic or merged; do not train separate
high-stakes branches around them.

## Gate 3.10b Tooling Added

Implemented a reusable episode-bootstrap CI utility:

```text
scripts/bootstrap_episode_ci.py
tests/test_bootstrap_episode_ci.py
```

The tool compares two aligned per-window metric files by resampling validation
episodes with replacement.

Default loss-style gain definition:

```text
gain = baseline metric - candidate metric
```

So positive gain means the candidate is better for MSE/loss metrics.

It reports:

```text
observed gain
bootstrap mean gain
95% CI low/high
whether CI crosses zero
effect-to-CI-half-width
```

This is intentionally generic: it can compare Gate 3.4 temporal heads, Gate
3.9 flow policies, or future full-data reruns once per-window evaluation
artifacts are emitted.

## Interpretation

The user's concern about earlier small-slice conclusions was justified, but
full LIBERO gives enough transition examples for the next reliability pass.

The right conclusion is not that earlier gates were wrong; it is that they
should be treated as mechanism-screening results until key claims are rerun
with:

```text
full or larger data
episode-level splits
per-task/per-suite breakdowns
transition/sustain slices
episode-bootstrap confidence intervals
```

The full data audit also shifts the immediate bottleneck from "do we have
enough transition examples?" to:

```text
1. build full-data DINO feature/cache and model artifacts;
2. make model evaluators emit aligned per-window metrics;
3. rerun only the key claims under bootstrap CI.
```

## Decision

Promote full LIBERO as the next reliability dataset for serious claims.

Do not launch a large model sweep yet. First add per-window metric export to the
key evaluators, then run a small full-data reproduction path:

```text
1. Gate 3.4-style temporal action head:
   full aligned, context-only/no-prior, mean_repeated,
   shuffled/rank-prob controls.
2. Gate 3.9-style strong-policy replacement:
   direct visual flow vs GeoMoCo-conditioned flow.
3. Report decoder/prior/metadata/diversity gains with bootstrap CI.
```

## Verification

Checks run:

```text
.venv/bin/python scripts/export_libero_windows.py ... --all-libero-suites
.venv/bin/python scripts/audit_event_modes.py ... full_h8
.venv/bin/python scripts/audit_dataset_sufficiency.py ... full_h8
.venv/bin/python -m unittest tests.test_bootstrap_episode_ci
.venv/bin/ruff check scripts/bootstrap_episode_ci.py tests/test_bootstrap_episode_ci.py
```
