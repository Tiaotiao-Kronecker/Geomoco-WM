# Gate 3.10 Data Reliability First Plan

## Purpose

Reassess GeoMoCo-WM conclusions under a more reliable data protocol before
continuing decoder, timing, or strong-policy work.

The current 2-files-per-suite slice was useful for fast mechanism tests, but it
has only:

```text
16,518 windows
400 episodes
8 tasks
1,699 transition windows = 10.29%
878 close-transition windows
807 open-transition windows
```

This is too sparse for stable claims about the main bottleneck:

```text
transition/open-close action timing
```

## Main Risk

Many previous positive and negative results may be small-slice observations
rather than stable conclusions. Windows are correlated within episode, and the
effective independent sample count is closer to episodes/tasks than raw window
count.

Therefore future claims should be evaluated by:

```text
episode-level statistics
per-task/per-suite breakdowns
transition-sliced metrics
confidence intervals over episode resampling
```

## Gate 3.10a: Full LIBERO Data Audit

Use the local full LIBERO official dataset:

```text
/home/user/dataset/libero_official
40 HDF5 task files
4 suites
```

First execute only the data layer:

```text
1. export full 40-task h=8 windows
2. materialize/audit event-mode labels
3. run dataset sufficiency audit
4. inspect transition/open-close coverage by task/suite/train/val
```

Do not train DINO/cVAE/action heads until this audit is complete.

Desired sufficiency targets:

```text
transition windows >= 8,000
close transition >= 4,000
open transition >= 4,000
train/val both cover all common event modes
zero-transition tasks are identified and not mixed blindly into transition claims
```

## Gate 3.10b: Episode Bootstrap CI Tooling

Implement a reusable reliability tool that compares two prediction artifacts by
episode bootstrap:

```text
1. sample validation episodes with replacement
2. aggregate their windows
3. compute baseline-new gain
4. repeat many times
5. report mean gain and 95% confidence interval
```

Interpretation:

```text
CI crosses 0      -> not a stable gain
CI fully above 0  -> reliable positive gain
effect < CI width -> weak signal, even if mean is positive
```

Use this first on existing 2-files results to validate the tool, then use it on
larger data.

## Gate 3.10c: Data Scaling Curve

Run the same data audits for:

```text
2 files/suite   already available
4 or 5 files/suite
full 40 task files
```

This separates:

```text
small-slice artifacts
task coverage effects
true model effects
```

## Claims To Re-Run Later

Do not re-run every historical gate. Once data reliability is established,
repeat only the key claims:

```text
Gate 3.4-style temporal action head:
  full aligned
  context-only/no-prior
  mean_repeated
  shuffled/rank-prob controls

Gate 3.9-style flow replacement audit:
  direct visual flow
  GeoMoCo-conditioned flow

transition-balanced sampler variant:
  direct visual
  GeoMoCo-conditioned
```

All future reports should include:

```text
overall MSE
transition MSE
sustain MSE
gripper MSE
per-task metrics
per-suite metrics
episode-bootstrap CI
```

## Immediate Execution

Start with:

```bash
.venv/bin/python scripts/export_libero_windows.py \
  --input-root /home/user/dataset/libero_official \
  --all-libero-suites \
  --output-dir outputs/libero_windows/libero_all_suites_full_h8 \
  --context-len 2 \
  --horizon 8 \
  --stride 4

.venv/bin/python scripts/audit_event_modes.py \
  --windows-jsonl outputs/libero_windows/libero_all_suites_full_h8/windows.jsonl \
  --output-json outputs/event_modes/gate3_10a_event_modes_full_h8.json \
  --output-md outputs/event_modes/gate3_10a_event_modes_full_h8.md \
  --train-ratio 0.8 \
  --split-by episode \
  --seed 7

.venv/bin/python scripts/audit_dataset_sufficiency.py \
  --event-mode-audit-json outputs/event_modes/gate3_10a_event_modes_full_h8.json \
  --output-json outputs/dataset_audits/gate3_10a_full_h8_sufficiency_audit.json \
  --output-md outputs/dataset_audits/gate3_10a_full_h8_sufficiency_audit.md
```

Only after these audits should the project decide whether to build full DINO
features and retrain event probe/cVAE/action policies.
