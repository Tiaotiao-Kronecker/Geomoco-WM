# Gate 3.0c Sample-Set Usage Audit Plan

## Purpose

Gate 3.0b showed that increasing `K` or changing small set aggregators is not a
strong next lever. Gate 3.0c audits whether the downstream action head actually
uses multimodal future-motion samples, or whether it mostly collapses the set to
a coarse mean statistic.

## Method

Use trained Gate 3.0a default checkpoints:

```text
outputs/motion_prior_action_head/gate3_0a_sample_set_real_k16_seed{7,17}/model.pt
outputs/motion_prior_action_head/gate3_0a_sample_set_shuffled_k16_seed{7,17}/model.pt
```

For each validation batch, generate the same `K=16` prior samples and compare:

| variant | meaning |
| --- | --- |
| `original` | normal K-sample set |
| `permuted` | sample order shuffled inside the set |
| `mean_repeated` | replace all K samples by the set mean |
| `mean_single` | feed only the mean future |
| `first_single` | feed only the first sampled future |
| `subset` | feed a random K=4 subset |
| `batch_mismatch` | give each context another batch row's sample set |

Key questions:

```text
Does replacing the set with the mean hurt?
Does reducing K to a subset hurt?
Does permutation leave output unchanged?
Does batch mismatch destroy performance?
Are real samples less noisy/more aligned than shuffled samples?
Are transition windows more sensitive to sample-set usage?
```

## Promotion Criteria

This is an audit, not a new promoted model. Evidence is positive if:

```text
original < mean_repeated
original < subset
batch_mismatch is much worse
permuted is unchanged
real original < shuffled original
```

If these hold, the action head is using aligned sample-set information, not only
a mean feature.
