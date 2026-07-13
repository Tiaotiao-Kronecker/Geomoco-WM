# cVAE KL / Prior / Posterior Explanation And Dataset Slice

- Date: 2026-06-08
- Context: after Gate 2.4b visual-conditioned cVAE future-motion prior.

## Question

What does this conclusion mean?

```text
KL is only 0.000740, and posterior and prior are almost identical. Therefore
we cannot yet claim the model learned a true multimodal future-motion
distribution. More accurately: the cVAE-form prior mean gives a slight
action-value gain, but stochastic / multimodal ability is not proven yet.
```

Also: what dataset did the recent training runs use, and did it cover the four
LIBERO suites?

## Prior

The prior is the distribution the model can use at deployment time:

```text
p(z | c_t)
```

where:

```text
c_t = visual grounding + proprio + task conditioning
```

It only sees current information. It does not see the ground-truth future
motion. In deployment, future motion must come from this branch:

```text
current visual/proprio/task context -> prior -> z -> decoded future motion
```

In Gate 2.4b, the main deployable readout was even stricter:

```text
current visual/proprio/task context -> prior mean -> decoded future motion
```

That is why the `prior_mean_action_metrics` are the main policy-relevant
metrics.

## Posterior

The posterior is the training-time distribution that sees the answer:

```text
q(z | c_t, future_motion_GT)
```

It receives both current context and the ground-truth future motion. Its role is
to encode the observed future trajectory into a latent variable `z`.

The posterior is useful for training and reconstruction diagnostics, but it is
not directly deployable because `future_motion_GT` is not available at test
time.

## KL

KL divergence measures how different the posterior is from the prior:

```text
KL(q(z | c_t, future_motion_GT) || p(z | c_t))
```

Intuition:

```text
large KL: posterior and prior are different
small KL: posterior and prior are similar
near-zero KL: posterior adds almost no extra information beyond the prior
```

In a useful multimodal cVAE, the posterior should use the ground-truth future
motion to encode which future mode happened, and the prior should learn to
predict the distribution of possible modes from current context.

Gate 2.4b had:

```text
mean KL: 0.000740
```

This is very small. Posterior reconstruction and prior-mean reconstruction were
also almost identical. Therefore, the latent variable is likely underused. The
model behaves more like a deterministic visual future-motion prior trained with
a VAE-shaped objective than a clearly stochastic / multimodal future-motion
model.

## Current Interpretation

What Gate 2.4b supports:

```text
cVAE-form prior mean gives a slight action-value improvement.
```

Mean result:

```text
deterministic single-query lambda 0.030 action MSE: 0.042090
visual cVAE prior-mean action MSE: 0.041579
```

What Gate 2.4b does not yet support:

```text
sampling different z values produces meaningful diverse future motions.
```

So the correct claim is a weak positive entry point, not a completed
multimodal world-motion result.

## Dataset Used In Recent Training

The recent gates used the same four-suite small formal slice:

```text
outputs/libero_windows/libero_all_suites_2files_all_demos_h8/windows.jsonl
```

Visual runs additionally used:

```text
outputs/visual_features/gate2_2b_dinov2_vits14_reg_patchpool4_2files_h8.h5
```

This dataset was exported from:

```text
/home/user/dataset/libero_official
```

Export config:

```text
context_len = 2
horizon = 8
stride = 4
num_suites = 4
num_files = 8
num_episodes = 400
num_windows = 16,518
num_frames = 69,073
```

It covers the four standard local suites:

| suite | files | episodes | windows | tasks |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 2 | 100 | 2,546 | 2 |
| `libero_object` | 2 | 100 | 3,602 | 2 |
| `libero_goal` | 2 | 100 | 4,125 | 2 |
| `libero_10` | 2 | 100 | 6,245 | 2 |

So yes, these runs cover all four suites.

Important caveat: this is not the full LIBERO dataset. It is `2 files per
suite`, with all demos inside those selected files. That means it covers four
suites and eight tasks total, but not every task file from every suite.

## Runs Using This Slice

This same slice is used by:

- Gate 2.3b action-aware lambda sweep;
- Gate 2.4a stepwise / multi-query visual predictor;
- Gate 2.4b visual-conditioned cVAE future-motion prior.

The split policy for formal comparisons is episode-level split, so adjacent
windows from the same demonstration are not split across train and validation.
