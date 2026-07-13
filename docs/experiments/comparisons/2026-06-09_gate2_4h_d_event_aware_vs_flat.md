# Gate 2.4h-d Event-Aware Readout vs Flat ScoreNet

- Date: 2026-06-09
- Scope: compare the original Gate 2.4d flat action-rank ScoreNet with the
  minimal Gate 2.4h-d event-aware auxiliary readout.

## Mean Results

| method | action MSE | event acc | macro-F1 | transition acc | step within 1 | event rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat ScoreNet | 0.040201 | 0.839062 | 0.553349 | 0.408688 | 0.175082 | 1.256558 |
| event w=0.1 ScoreNet | 0.040255 | 0.838916 | 0.552820 | 0.407396 | 0.173790 | 1.254738 |

## Takeaway

The minimal event-aware readout is not promoted.

It does not improve deployable action MSE, event accuracy, macro-F1, transition
accuracy, or transition timing. The tiny event-rank improvement is not enough
to justify adding it to the method.

## Mainline Update

The next bottleneck is upstream event fidelity:

```text
cVAE sampled future motion
  -> frozen action decoder
  -> decoded gripper transition timing
```

If this interface does not represent close/open events clearly, adding more
readout losses will only reshuffle weak candidates.

