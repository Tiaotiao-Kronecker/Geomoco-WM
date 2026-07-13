# Gate 2.5b Joint Bridge Diagnostics

- Date: 2026-06-09
- Scope: modular predicted EEF plus predicted gripper action bridge.

## Main Table

| branch | action MSE | SE(3) MSE | action gripper MSE | EEF MSE | gripper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| task/proprio | 0.079633 | 0.039495 | 0.320460 | 0.000929 | 0.324415 |
| visual patchpool | 0.050333 | 0.029825 | 0.173377 | 0.000782 | 0.172088 |
| shuffled visual patchpool | 0.065466 | 0.038015 | 0.230170 | 0.000985 | 0.233726 |

## What Passed

Real visual grounding remains necessary and useful:

```text
visual action MSE 0.050333 < shuffled 0.065466 < task/proprio 0.079633
```

The same ordering holds for EEF prediction, gripper prediction, and action
gripper metrics.

## What Did Not Pass

The modular joint bridge does not beat the previous best learned EEF-only
action interface:

```text
Gate 2.3b visual EEF-only learned prior: 0.042090
Gate 2.5b modular predicted EEF+gripper: 0.050333
```

So the current result is not a method promotion. It is a diagnostic that
separate EEF and gripper prediction errors compound inside an oracle joint
action decoder.

## Diagnostic Decomposition

| input | action MSE | SE(3) MSE | gripper MSE |
| --- | ---: | ---: | ---: |
| GT EEF + predicted gripper | 0.028987 | 0.004943 | 0.173254 |
| predicted EEF + GT gripper | 0.025443 | 0.029632 | 0.000312 |
| predicted EEF + predicted gripper | 0.050333 | 0.029825 | 0.173377 |

This says:

- predicted gripper helps when EEF is oracle-quality;
- predicted EEF is not precise enough for the joint oracle decoder;
- predicted EEF and predicted gripper together create a noisy joint
  representation.

## Next Decision

Train a joint deterministic predictor:

```text
context + visual tokens -> future_delta_ee + future_gripper
```

Use `future_delta_gripper` action-aware loss so the predictor learns the joint
interface that the action decoder actually consumes.

