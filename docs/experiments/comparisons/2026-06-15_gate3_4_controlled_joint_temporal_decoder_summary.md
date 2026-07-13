# Gate 3.4 Controlled Joint Temporal Decoder Summary

## Question

After Gate 3.3 failed, does replacing the shallow gripper-only additive
residual with a small joint temporal action-sequence decoder improve the
event-aware predicted-event action head without losing attribution?

## Result

Yes, but only as a small mechanism-positive result.

Mean over seeds 7 and 17, 5-pass repeated eval:

| branch | overall MSE | gripper MSE | transition MSE |
| --- | ---: | ---: | ---: |
| Gate 3.1f/g reference | 0.034767 | 0.150052 | 0.134087 |
| Gate 3.4 full aligned base | 0.034303 | 0.149826 | 0.131697 |
| Gate 3.4 full aligned temporal | 0.034262 | 0.149383 | 0.131311 |

The temporal branch improves over the same checkpoint's base output by only
`0.000040` overall MSE, but it is directionally positive and improves
transition MSE by `0.000386`.

## Attribution Controls

Mean over seeds 7 and 17, 5-pass repeated eval, using `temporal_actions`:

| control | overall MSE | transition MSE | interpretation |
| --- | ---: | ---: | --- |
| full event/rank/prob | 0.034262 | 0.131311 | main aligned branch |
| shuffled event/rank/prob | 0.035529 | 0.135399 | aligned event identity still matters |
| rank/prob-only | 0.035875 | 0.135893 | rank/prob without identity is weaker |
| mean repeated | 0.034414 | 0.132199 | close control; sample diversity gain is thin |
| context-only/no-prior | 0.036642 | 0.136922 | decoder capacity alone does not explain gain |

## Interpretation

Gate 3.4 answers the immediate question from Gate 3.3:

```text
The failure was not simply "any richer temporal branch is useless."
Joint temporal action-sequence decoding is better than gripper-only additive residuals.
```

But the result is not strong enough to claim that the current decoder deeply
uses the full multimodal K-sample distribution. The mean-replacement control is
too close to full aligned. The safer claim is:

```text
Aligned event metadata plus motion-prior mean/sample structure remains useful,
and a small joint temporal decoder can extract a little more from it.
```

## Decision

Keep Gate 3.4 as the current best short-budget action-head result, but do not
promote it as a final architecture.

Next work should focus on one of two controlled directions:

```text
1. make sample-diversity usage stronger and measurable;
2. test a small flow/diffusion residual decoder under the same controls.
```

Any future richer decoder still needs the same attribution matrix:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
mean replacement
context-only/no-prior
same-capacity no-prior
```
