# Changelog

## v0.1.0-alpha.1

First experimental source release of the D1 checkpoint tools.

- `make_manifest.py` pins explicitly selected source metadata and checkpoint hashes.
- `adapter.py project` prepares 266 language tensors for the native LFM2 converter.
- `adapter.py assemble` restores D1 names, replaces 166 language-body groups and
  preserves 541 original BF16 tensors byte-for-byte.
- Strict topology, quantization-layout, digest, free-disk and fresh-output checks;
  per-tensor receipts distinguish structural success from runtime acceptance.
- Eleven dependency-free synthetic tests, including CLI round trips and corrupt
  inputs, plus optional AMD patches checked against pinned public source.

Requires Python 3.11+; tests verified on Python 3.12. No runtime, weights,
quantizer, server or automatic deployment is included. Real checkpoint loading,
typed-output quality, GPU compatibility and capacity are separate validation work.
Remote CI is not configured. Original scripts/tests are MIT; LiquidAI-derived
sources and patches retain the LFM Open License v1.0 in `upstream/LICENSE`.
