# Gate 3.4b Sample-Diversity Usage Summary

## Question

Gate 3.4 had two facts in tension:

```text
full aligned beats context-only, shuffled metadata, and rank/prob-only controls;
but a separately trained mean_repeated control is very close.
```

Gate 3.4b separates two meanings of `mean_repeated`:

```text
trained mean-only control:
  can a same-capacity model adapt if it only ever sees mean-repeated inputs?

eval-time same-checkpoint collapse:
  does the trained full-aligned model depend on the K-sample distribution?
```

## Main Result

Mean over seeds 7 and 17:

| comparison | temporal MSE | interpretation |
| --- | ---: | --- |
| Gate 3.4 full aligned, normal eval | 0.034218 | same full checkpoint under usage audit |
| same checkpoint, mean_repeated eval | 0.042594 | runtime sample distribution matters |
| same checkpoint, rank1_only eval | 0.047154 | top rank alone is insufficient |
| same checkpoint, subset_k4 eval | 0.071022 | the head depends on the K=16 set structure |
| same checkpoint, batch_mismatch eval | 0.314274 | alignment is essential |

Permutation sanity passed:

```text
delta/original_vs_permuted/temporal/action_l2 = 2.63e-7
```

The model is order-invariant but not sample-distribution-invariant.

## Where Diversity Matters

Eval-time diversity gain:

```text
gain = mean_repeated_eval_MSE - original_eval_MSE
```

| group | gain |
| --- | ---: |
| transition windows | +0.010765 |
| sustain windows | +0.008166 |
| high sample diversity | +0.012527 |
| low sample diversity | +0.005987 |
| high best-vs-mean gap | +0.015162 |
| low best-vs-mean gap | +0.004194 |

This is the expected pattern if sample diversity is being used: the largest
damage from collapsing to a mean appears in high-diversity/high-gap windows and
is also visible on transition windows.

## Interpretation

Gate 3.4b changes the nuance, not the whole decision.

Before this audit, the conservative statement was:

```text
sample-diversity attribution is thin because the trained mean_repeated control
is close to full aligned.
```

After this audit, the stronger but still careful statement is:

```text
The full-aligned temporal decoder does use the K-sample distribution at runtime.
However, the training objective still does not make that usage robust enough to
clearly beat a separately trained mean-only control by a large margin.
```

So the next problem is not "the decoder ignores samples"; it is:

```text
make sample-distribution usage more causally necessary and more deployably
valuable under matched training controls.
```

## Mainline Decision

Proceed to Gate 3.4c before a larger flow/diffusion residual decoder:

```text
set-wise temporal regret/rank supervision
or candidate-comparison auxiliary loss
```

Keep the attribution matrix:

```text
full aligned event/rank/prob
shuffled event metadata
rank/prob-only
trained mean_repeated
eval-time mean collapse
context-only/no-prior
batch mismatch and permutation sanity
```

If Gate 3.4c increases the gap between full aligned and trained mean-only while
preserving metadata/prior controls, then sample-diversity attribution becomes
much stronger. If it does not, then a small flow/diffusion residual decoder is
justified as the next controlled test.
