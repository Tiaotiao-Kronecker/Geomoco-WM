# Sample-Set Usage Diagnostics Explained

## Context

After Gate 3.0c, we clarified three diagnostic perturbations used to test
whether the downstream action head truly uses the cVAE sample set:

```text
sample permutation
mean repeated
batch mismatch
```

These are not new training losses. They are evaluation-time sanity checks.

## Sample Permutation

Original input:

```text
[sample_1, sample_2, ..., sample_16]
```

Permutation input:

```text
[sample_7, sample_2, sample_15, ..., sample_1]
```

The content is identical; only order changes.

Purpose:

```text
Check whether the action head treats futures as a set rather than as ordered slots.
```

Result:

```text
delta/original_vs_permuted/action_l2 ~= 2.4e-7
```

Interpretation: the action head is effectively permutation-invariant. It is not
assigning special meaning to "sample 1" or "sample 2".

## Mean Repeated

Original input:

```text
[s1, s2, ..., s16]
```

Mean-repeated input:

```text
m = mean(s1, s2, ..., s16)
[m, m, ..., m]
```

Purpose:

```text
Check whether the model only needs the average future.
```

If `mean_repeated` matched `original`, then the multimodal sample set would not
be doing much beyond providing a mean future. Gate 3.0c showed the opposite:

```text
real original:      0.037061
real mean repeated: 0.041893
```

Interpretation: replacing the sample set by its mean hurts. The spread and
structure of the sample set contain action-useful information.

## Batch Mismatch

Original alignment:

```text
context_A + futures_A -> action_A
context_B + futures_B -> action_B
```

Batch mismatch:

```text
context_A + futures_B -> action_A
context_B + futures_A -> action_B
```

Purpose:

```text
Check whether future samples must be aligned with the current context.
```

Result:

```text
real original:       0.037061
real batch mismatch: 0.317347
```

Interpretation: mismatching futures with the wrong context destroys
performance. The action head depends on context-aligned future hypotheses, not
arbitrary sample diversity.

## Main Takeaway

Gate 3.0c supports this statement:

```text
The action head uses aligned sample-set diversity.
```

It does not support the weaker or wrong statement:

```text
More random diversity is always better.
```

In fact, shuffled samples are more diverse but worse:

```text
real sample pair L2:     0.611503
shuffled sample pair L2: 1.385104

real original MSE:       0.037061
shuffled original MSE:   0.042665
```

So the next research target is not generic diversity. It is structured,
visually grounded, event-aware diversity.

## Implication For Next Mainline

Move to:

```text
Gate 3.1: mode-structured / event-aware future-motion prior
```

The core idea is to make gripper/event timing modes explicit, because Gate 3.0c
showed that sample-set information is especially valuable around transition
windows.
