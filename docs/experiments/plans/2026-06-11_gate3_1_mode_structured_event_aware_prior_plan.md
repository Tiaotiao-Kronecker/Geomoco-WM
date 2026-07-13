# Gate 3.1 Mode-Structured / Event-Aware Future-Motion Prior Plan

## Purpose

Gate 3.0c showed that the downstream action head uses aligned sample-set
diversity, but generic diversity is not enough. Shuffled samples are more
diverse and worse. The next step is therefore to make the useful modes more
explicit, with a focus on gripper/event timing.

Gate 3.1 asks:

```text
Can GeoMoCo-WM generate future_delta_gripper samples whose modes align with
gripper transition/event timing, and do those samples improve downstream action
prediction beyond the current joint cVAE?
```

## First-Principles Motivation

For manipulation, many futures differ not only in EEF geometry but in when the
gripper closes, opens, or sustains. A continuous Gaussian latent can represent
this implicitly, but Gate 3.0c suggests the action head benefits when sample-set
diversity is aligned with context and event timing.

So the next prior should separate:

```text
continuous motion variation: how the EEF moves
discrete/structured event variation: whether/when the gripper transition happens
```

This is still a future-motion prior, not a direct policy. The action head remains
the downstream consumer.

## Proposed Event Mode Labels

Use weak labels derived from exported action chunks, not simulator object state.

Base event class:

```text
transition_close
transition_open
mixed_transition
sustain_close
sustain_open
hold
```

Timing bins:

```text
none
early: step 0-2
middle: step 3-5
late: step 6-7
```

Combined event mode:

```text
event_mode = event_type + "::" + timing_bin
```

Examples:

```text
transition_close::early
transition_open::middle
sustain_close::none
hold::none
```

Use labels first as diagnostics and optional conditioning. Do not overclaim
semantic contact; these are gripper-command transition proxies.

## Gate 3.1a: Event Mode Target Audit

Goal:

```text
Materialize and audit event-mode labels on the current two-file four-suite slice.
```

Outputs:

```text
outputs/event_modes/gate3_1a_event_modes_2files.json
outputs/event_modes/gate3_1a_event_modes_2files.md
```

Checks:

```text
class counts
per-suite counts
per-task counts
transition-step histogram
train/val class balance
rare-class threshold
```

Pass criteria:

```text
transition classes are not too rare to measure
train/val splits both contain key modes
no step-0 shortcut dominates the combined label
```

Stop/adjust:

```text
If classes are too sparse, merge to coarser labels:
transition_any / sustain_close / sustain_open / hold
and timing_bin early_or_middle / late / none.
```

## Gate 3.1b: Event Probe Baseline

Before changing the cVAE, train/evaluate a lightweight event-mode probe from:

```text
context + task
context + task + visual grounded features
context + task + cVAE prior mean
context + task + cVAE samples summary
```

Purpose:

```text
Confirm that event mode is predictable from visual context and/or existing
motion-prior samples.
```

Metrics:

```text
macro-F1
transition-vs-nontransition F1
timing-bin accuracy for transition windows
real-vs-shuffled visual controls
```

Pass criteria:

```text
real visual > shuffled visual
cVAE sample summary carries event signal beyond context-only
```

## Gate 3.1c: Event-Conditioned cVAE

Train a cVAE whose prior can be conditioned on an event mode.

Training-time posterior:

```text
q(z | context, visual, event_mode, future_delta_gripper)
```

Prior:

```text
p(z | context, visual, event_mode)
```

Decoder:

```text
future_delta_gripper = decode(context, visual, event_mode, z)
```

Event mode enters as a small embedding or one-hot appended to the cVAE
condition. This is the simplest interpretable version.

Important: at deployment/eval time, the event mode should not require an oracle
future label. We need two eval paths:

1. Oracle-event upper bound:

```text
Use GT event_mode to sample futures.
```

This measures whether event conditioning can help if the mode is known.

2. Predicted/prior event mixture:

```text
Predict p(event_mode | context, visual), sample or enumerate top modes,
then sample future_delta_gripper from each mode.
```

This is the deployable route.

## Gate 3.1d: Mode Mixture Sampling

Instead of sampling K futures from one unconditional prior, allocate samples by
event mode:

```text
top M event modes from event prior
for each mode, sample K_m futures
combine into K total future hypotheses
```

Default first setting:

```text
M = 3 modes
K = 16 total
allocation: proportional to event prior probability, minimum 2 per selected mode
```

Baselines:

```text
unconditional Gate 2.5c joint cVAE
event-conditioned cVAE with oracle event
event-conditioned cVAE with predicted top-M event modes
event-conditioned cVAE with shuffled event modes
```

## Gate 3.1e: Downstream Action Head Evaluation

Use the same Gate 3 action-head contract:

```text
context + sampled future_delta_gripper set -> action chunk
```

Compare:

| branch | role |
| --- | --- |
| Gate 3.0 default real K=16 | current default |
| Gate 3.0 real K=32 | optional stronger reference |
| event-conditioned oracle-event | upper bound for mode structure |
| event-conditioned predicted-event mixture | deployable main branch |
| shuffled event-mode control | shortcut/control |

Primary metrics:

```text
action MSE
translation_m_l2
rotation_geodesic_deg
gripper_mse
transition-window action MSE
no-transition-window action MSE
```

Usage diagnostics:

```text
mean replacement damage
subset damage
batch mismatch damage
sample pair L2
gripper pair L2
event-mode coverage
```

Pass criteria:

```text
predicted-event mixture beats Gate 3.0 default real K=16
predicted-event mixture beats shuffled event-mode control
oracle-event branch shows meaningful headroom
transition-window MSE improves without hurting no-transition windows badly
usage audit still shows original set > mean replacement
```

## Risks And Controls

Risk: labels are action-derived and may overfit to gripper command shortcuts.

Control:

```text
report transition vs sustain separately
keep timing-bin diagnostics
use shuffled event modes
check no-transition windows do not regress heavily
```

Risk: oracle-event conditioning is not deployable.

Control:

```text
always separate oracle-event upper bound from predicted-event mixture
```

Risk: event conditioning collapses continuous EEF diversity.

Control:

```text
track EEF pair L2 and action-head usage audit
compare best-of-K and original action-head MSE
```

## Immediate Implementation Order

1. Build event-mode target materializer/audit.
2. Run Gate 3.1a on current two-file four-suite slice.
3. If labels are usable, add event-mode probe baseline.
4. Add event-mode conditioning to visual cVAE.
5. Train oracle-event cVAE first as an upper bound.
6. Train event prior / top-M mixture route.
7. Evaluate with Gate 3 action head and Gate 3.0c usage audit.

## Expected Outcome

This gate should clarify whether GeoMoCo-WM's useful multimodality is primarily
about gripper/event timing. If positive, it gives a sharper paper story:

```text
GeoMoCo-WM learns visually grounded future-motion modes over EEF geometry and
gripper-event timing, and downstream action heads can exploit those aligned
modes.
```
