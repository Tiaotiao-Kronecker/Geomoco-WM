# Motion-To-Action And Decoder Plan

- Date: 2026-06-05
- Context: discussion after Gate 1 exporter and first oracle action-decoder
  smoke

## Questions

1. How do ZipMotion / ZipMo and AMPLIFY recover actions from motion?
2. Is the current MLP `ActionDecoder` too weak? Should we use Diffusion Policy,
   flow matching, or MeanFlow immediately?

## Interpretation

ZipMotion / ZipMo and AMPLIFY do not reduce to direct RGB-to-action policies.
They both use motion as an intermediate interface.

AMPLIFY explicitly separates:

```text
forward dynamics:
  observation + task -> latent keypoint motion tokens

inverse dynamics:
  predicted latent motion tokens -> robot actions
```

ZipMo emphasizes learned long-horizon motion embeddings and then attaches
LIBERO policy heads to map the generated/planned motion representation into
actions.

For `Geomoco-WM`, this supports the current route:

```text
future motion interface first
strong action decoder later
```

## Decoder Decision

The current MLP `ActionDecoder` is intentionally simple. It is a diagnostic
tool, not the final action model.

The immediate question is:

```text
Does GT future EEF motion help action prediction over direct context?
```

Therefore the first fair comparison should use the same training script and
same simple decoder:

```text
direct-context baseline:
  context -> action chunk

oracle future-motion baseline:
  context + GT future EEF delta -> action chunk
```

If oracle future motion does not help a simple decoder, do not immediately move
to cVAE training. First check whether the interface or decoder capacity is the
bottleneck.

## Escalation Order

Use this order:

```text
1. MLP diagnostic
2. temporal MLP / TCN / small Transformer
3. Diffusion Policy / flow matching / MeanFlow
```

Do not start with Diffusion or Flow as the main method, because a strong action
decoder can hide whether gains come from GeoMoCo-WM or from the action head.

## Code Decision

`scripts/train_oracle_action_decoder.py` now supports:

```text
--motion-mode future_delta
--motion-mode none
```

This lets the same script run both the oracle future-motion condition and the
direct-context baseline.
