# Gate 3.0b K And Aggregator Summary

## Question

After Gate 3.0a, should the next improvement come from sampling more futures or
from a stronger set aggregator?

## K Sweep Result

| K | real mean MSE | shuffled mean MSE |
| ---: | ---: | ---: |
| 4 | 0.037166 | 0.038868 |
| 8 | 0.037772 | 0.040161 |
| 16 | 0.036675 | 0.042633 |
| 32 | 0.036565 | 0.041596 |

K=32 is best for real samples, but only improves over K=16 by `0.000110`.
The trend is not monotonic, so this does not justify a simple "larger K solves
the readout" claim.

## Aggregator Result

K fixed to 16.

| aggregator | real mean MSE | shuffled mean MSE |
| --- | ---: | ---: |
| `context_attention` | 0.036675 | 0.042633 |
| `mean_pool` | 0.036691 | 0.042385 |
| `multi_query_attention` | 0.036949 | 0.042333 |

No aggregator clearly beats the Gate 3.0a `context_attention` default. Mean
pooling almost matches attention, which suggests the action head may be using a
coarse set statistic rather than deeply exploiting multimodal structure.

## Decision

Keep `context_attention`, `K=16` as the default for now, with `K=32` as an
optional stronger but slightly more expensive reference. Do not promote
multi-query attention.

The next mainline should not be another small aggregator tweak. Move to an
audit of sample-set diversity and action-head usage:

```text
Gate 3.0c: sample-set mode/diversity and action-head usage audit
```

This should answer whether the current samples actually expose distinct
action-useful futures, and whether the downstream action head is using them or
mostly collapsing them to an average.
