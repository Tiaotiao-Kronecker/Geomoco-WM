# Gate 2.5d Joint Sample Readout vs Controls

- Date: 2026-06-10
- Scope: joint cVAE sample readout over `future_delta_gripper`

## Action Readout

| branch | prior mean action MSE | scorer argmax action MSE | oracle best action MSE | top-1 match | selected rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| real visual | 0.043816 | 0.043414 | 0.022192 | 0.255727 | 5.637193 |
| shuffled visual | 0.068816 | 0.070023 | 0.030778 | 0.316459 | 4.693893 |

## Event Readout

| branch | prior event acc | scorer event acc | prior transition acc | scorer transition acc | prior step@1 | scorer step@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real visual | 0.889522 | 0.900561 | 0.568301 | 0.647706 | 0.239717 | 0.264299 |
| shuffled visual | 0.787503 | 0.837445 | 0.224003 | 0.491305 | 0.069768 | 0.192786 |

## Takeaways

1. Real visual readout is a genuine signal: it improves over the real prior mean
   a little, while shuffled degrades on action.
2. The gain is too small to beat the deterministic joint baseline `0.040688`.
3. The oracle gap remains large: learned readout closes only a tiny fraction of
   the `prior mean -> oracle best` gap.
4. Event fidelity improves modestly, but not enough to claim a deployable
   sample selector.

## Mainline Decision

The current flat action ranking scorer should not be promoted.

The next mainline step should target richer readout supervision:

```text
event/contact/executability-aware scorer
or
hard-negative ranking with stronger structure
```

