# Historical execution scope — not a new portable-runtime qualification

These are aggregate synthetic/capacity observations from a workstation-specific
integration on 2026-10-17. The source-release audit rechecked the relevant source
hashes against that run's recorded source index; it did **not** reload weights,
repeat GPU inference, remeasure speed or independently replay every raw result.
No private photo benchmark or photo-level answer is included.

## Exact identities

- Author model revision: `051bcc464b01b9f92942b364d9586b0ef5912432`.
- Original checkpoint SHA256: `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`.
- Native 8-bit quantizer output SHA256: `783acff756827ef91685bc53ba1582f884bd868bc1ca93cdf41271cd6891bb2a`.
- Reassembled author-format weight SHA256: `27d0f9c2d2c341d755f95a91be42f4cfb06b14aec3564712d0cd8f756dfa43bf`.
- Frozen runtime source-index SHA256: `5c9909b374dc668764666a540fd9a1795a83400772b1049a71333f627232f515`.
- Historical modified `hybrid.py` SHA256: `b06122e855d73c10523ef0f715df27bb07c17733fd9e49cc737602942276ddbb`.
- Historical modified `lfm2_vl.py` SHA256: `00291c8e69df9dd980a492f271f16981056bf198e12f9441c051c1ecc5171daa`.

The patches in this repository add attribution comments, so their output bytes
have new SHA256 values. Their Python ASTs are identical to those historical two
files. Automated tests apply the patches with fuzz=0 to the pinned public author
source, then check the complete AST hash. This is source-logic equivalence, not a
new GPU parity test. The newly extracted `adapter.py`/`make_manifest.py` are not
those historical runtime entry points and inherit no GPU qualification.

## What the old receipt records

Hardware/software: Radeon RX 6800 XT / gfx1030, ExLlamaV3 1.6.2, Torch
2.13.0+rocm10.1.0, HIP 7.16.26385. Mixed precision: 166 EXL3 8-bit language-body
modules; original BF16 vision, embeddings/head and other residual tensors.
Explicit attention used FP32 products; that is not all-BF16 inference.

- Eleven factual typed controls passed in that scoped test.
- Two full targets completed in one model residency: 32,512 trunk tokens,
  65,536 packed branch tokens, 256 branches of 256 tokens, 98,048 packed tokens,
  32,768 tokens per path, one synthetic 1024×1024 RAM image.
- Each output was finite with shape `[256, 128000]`.
- Recorded first/warm target wall times: 266.298 / 258.170 seconds.
- Recorded Torch peaks: allocated 10,367,571,968 bytes; reserved 11,928,600,576.
- Device-wide sampled peak: 14,874,836,992 bytes; samples every 0.25 seconds.
  This includes unrelated device users and cannot bound unsampled peaks.
- The tokenwise MLP chunk check at 8193 tokens recorded max difference 0.0078125,
  mean 1.3733414e-7, against limits 0.02 / 0.002. That is finite-precision agreement,
  not bit identity or a universal accuracy bound.
- The receipt records a successful owned worker exit and no retained owned model.

## What it does not establish

No clean installation, general model accuracy, best throughput, arbitrary long
branches, tensor-batched images, other bit rates, quantized vision/head, TP,
DFlash, Tabby serving or other model support is established here. Historical
individual image requests were serial B1, not true tensor image batching.
The old runtime also contained five ExLlamaV3 Python compatibility patches and
an environment linked to an external Torch installation. This source package
neither silently bundles those workstation patches nor promises an independent
wheel/build environment. A new GPU environment must pass separate installation,
exact loading/head-tie, typed probability, quality and full-capacity gates.
