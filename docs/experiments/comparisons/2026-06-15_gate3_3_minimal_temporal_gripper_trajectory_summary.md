# Gate 3.3 Minimal Temporal Gripper Trajectory Summary

## Question

Can a minimal temporal gripper residual sequence decoder repair close/open
transition shape while keeping the Gate 3.1f/g predicted event-mixture sample
interface fixed?

## Result

No.

| readout | overall MSE | gripper MSE | sustain MSE | transition MSE |
| --- | ---: | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.022793 | 0.134087 |
| Gate 3.3 base output | 0.035516 | 0.153991 | 0.023151 | 0.137769 |
| Gate 3.3 trajectory-routed output | 0.035655 | 0.154966 | 0.023274 | 0.138046 |

The trajectory branch worsens the same checkpoint's base output:

```text
overall MSE:     0.035516 -> 0.035655
gripper MSE:     0.153991 -> 0.154966
transition MSE:  0.137769 -> 0.138046
sustain MSE:     0.023151 -> 0.023274
```

## Interpretation

The plumbing is now available for trajectory-routed gripper metrics, but this
minimal additive residual is not enough. It neither beats the Gate 3.1f/g
reference nor improves over its own base action output.

This means the branch is negative before attribution controls are needed. The
main risk discussed in the plan, decoder capacity swallowing the motion-prior
contribution, does not arise because the stronger branch does not help.

## Decision

Do not promote Gate 3.3 `temporal_mlp`.

Keep Gate 3.1f/g as the deployable reference. If continuing the richer decoder
direction, make the next change qualitatively different rather than another
shallow additive gripper residual:

```text
joint temporal action-sequence decoder;
strictly controlled flow/diffusion action residual;
or richer contact/object/event supervision.
```
