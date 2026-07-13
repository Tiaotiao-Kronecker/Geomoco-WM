# Gate 3.0a Action Head Prior Ablation

## Question

If GeoMoCo-WM provides a set of visually grounded future-motion hypotheses, can a
simple downstream action head use that set better than direct context or a
shuffled visual prior?

## Shared Protocol

- Same dataset slice: two files per LIBERO suite, all demos.
- Same episode-level split policy.
- Same `MotionPriorActionHead` architecture and action MSE training loss.
- Same action metrics.
- Only the future-prior input changes.
- For stochastic sample-set modes, report 5-pass repeated-eval mean.

## Result

| input | seed 7 MSE | seed 17 MSE | mean MSE | decision |
| --- | ---: | ---: | ---: | --- |
| context only | 0.039253 | 0.035685 | 0.037469 | lower bound |
| real prior mean | 0.038547 | 0.035574 | 0.037061 | weak positive |
| real sample set, K=16 | 0.038277 | 0.035072 | 0.036675 | promote |
| shuffled sample set, K=16 | 0.043066 | 0.042200 | 0.042633 | fails control |
| GT future | 0.004421 | 0.004827 | 0.004624 | privileged upper bound |

Relative to context-only, real sample-set improves mean action MSE by about
`2.12%`. Relative to real prior mean, it improves by about `1.04%`. Relative to
shuffled sample-set, it improves by about `13.98%`.

## Interpretation

The main positive is not the absolute gain over prior mean; that gain is small.
The stronger evidence is the controlled ordering:

```text
GT future << real sample set < real prior mean < context-only << shuffled sample set
```

This says the action head can extract useful signal from aligned real visual
motion-prior hypotheses, while shuffled visual cVAE samples hurt. That supports
the current positioning of GeoMoCo-WM as a visually grounded multimodal
future-motion prior.

## Limits

- The action head is now stronger than the earlier frozen `ActionDecoder`, so
  Gate 3.0a numbers should not be mixed with Gate 2 frozen-decoder numbers.
- The real sample-set gain over real prior mean is modest. This proves the set
  interface is viable, not that the current set aggregator is optimal.
- The GT-future upper bound is still much lower, so there is substantial
  remaining room in future-motion quality, aggregation, and action decoding.
- This is offline action prediction, not closed-loop success rate.

## Decision

Promote Gate 3.0a as the new mainline interface:

```text
frozen visual joint cVAE -> K future_delta_gripper hypotheses -> downstream action head/planner
```

Next work should improve the set/action interface without hiding the motion-prior
contribution. Good next ablations:

```text
K sweep: 4 / 8 / 16 / 32
set aggregator variants: mean-pooling, context-query attention, multi-query attention
action head capacity sweep
event/gripper diagnostic reporting for the action head
```

Do not add raw DINO to the action head in the next immediate step; that would
make it unclear whether the improvement came from GeoMoCo-WM samples or direct
visual imitation.
