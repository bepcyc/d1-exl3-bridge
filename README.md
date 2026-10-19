# D1 ↔ ExLlamaV3 bridge (experimental source tools)

Byte-preserving conversion glue for LiquidAI/d1-3B's author-format checkpoint and
ExLlamaV3 1.6.2's native LFM2 quantizer, plus separately scoped AMD compatibility
patches. No weights, photos, datasets, service manager, network server or automatic
GPU launcher are included.

**Status:** the portable CLI is tested with fabricated checkpoints, not qualified
as a clean-machine GPU installation. An earlier workstation-specific integration
executed the same AMD patch logic on a real RX 6800 XT. That historical result does
not qualify this extracted CLI, a new runtime build, every GPU, model accuracy or
all D1 features. See [historical scope](HISTORICAL_RESULTS.md) and
[remaining gates](OSS_AUDIT.md).

## What the bridge preserves

Supported topology: `LiquidAI/d1-3B` revision
`051bcc464b01b9f92942b364d9586b0ef5912432`, original 707 tensors.

1. `project` validates language shapes/dtypes/head tying and streams 266 language
   tensors into the native `model.*` LFM2 namespace. It does not quantize anything.
2. Run the official ExLlamaV3 quantizer separately, after checking your hardware
   and full memory requirements. This repository neither starts nor admits it.
3. `assemble` replaces exactly 166 language-body matrices with native 8-bit
   trellis/mul1 buffers and copies the other 541 original BF16 tensor payloads
   byte-for-byte. Vision/projector, embedding and tied-head semantics are retained.
   Native FP16 copies of embeddings/norms are not substituted.

The supported assembly format is EXL3 1.6.2, `mul1`, automatic output scales and a
16-bit head. Trellis axes, input/output scale vectors and codebook marker bytes are
validated. Other bit rates/versions/codebooks and quantized vision are rejected.
The result is **mixed precision**, not an entirely 8-bit multimodal model. These
are structural/byte checks, not proof of trained numerical quality.

## Requirements and tests

Python 3.11+ and `uv`; the 11 tests were verified locally on Python 3.12.
Remote CI is not configured in this source-only publication scope; no remote CI
result is claimed. The CLI has no third-party Python dependency and imports no
Torch, Transformers or ExLlama runtime. Patch tests also require the standard
`patch` executable. Run from a checkout:

```sh
uv run --no-project --python 3.12 python -B -m unittest discover -s tests -v
```

Tests fabricate tiny shapes with the complete tensor topology, run the actual CLI,
check 707→266/1205 counts and all 541 preserved plus 664 quantized payloads, assert input immutability,
check rejected corruption/metadata/path/codebook/axis/disk/replay cases, and verify
that both AMD patches apply exactly. No downloaded model or GPU is needed.

## Prepare an explicit input manifest

Obtain the author snapshot at the exact revision above yourself. The original
single-file checkpoint SHA256 used by the historical integration is
`50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`.
This tool does not download weights, execute source code or authenticate a model
publisher: verify the revision/digest through your trusted source first.

Pin metadata explicitly, including the license and every author code/tokenizer/
processor file needed by your intended runtime. The source snapshot used here has
these files; review the list rather than adding unrelated files from a model folder:

```sh
uv run --no-project --python 3.12 python make_manifest.py \
  --source /path/to/d1-source --output /path/to/source-manifest.json \
  --model-sha256 50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c \
  --metadata LICENSE --metadata config.json --metadata modeling_d1.py \
  --metadata api.py --metadata runner.py --metadata prompt.py \
  --metadata hybrid.py --metadata lfm2_vl.py --metadata tokenizer.json \
  --metadata tokenizer_config.json --metadata processor_config.json \
  --metadata chat_template.jinja
```

This checks the checkpoint and metadata hashes and writes a fresh JSON manifest.
A manifest records the caller's selected files; it is not a signed vendor
attestation. `project`/`assemble` recheck those inputs. Extra files are not copied.
The supplied license/code remains subject to its own terms.

```sh
uv run --no-project --python 3.12 python adapter.py project \
  --source /path/to/d1-source --source-manifest /path/to/source-manifest.json \
  --output /path/to/fresh-native-projection

# Perform official, separately admitted GPU quantization outside this tool.

uv run --no-project --python 3.12 python adapter.py assemble \
  --source /path/to/d1-source --source-manifest /path/to/source-manifest.json \
  --native /path/to/native-exl3-output --output /path/to/fresh-author-exl3
```

Each output directory must be fresh. Estimated payload/metadata/header space plus
64 MiB free headroom is required; `--headroom-bytes` adjusts the *disk* margin. It
is not a reservation or a GPU-memory guarantee. Source checkpoint/native shards
are fully rehashed after streaming. A partial failure leaves the new directory
for inspection, without a success receipt; inspect it and choose a fresh namespace
rather than overwriting or silently resuming it. Source inputs are never written.

Successful output contains `adapter-receipt.json` with per-tensor source key,
byte count and SHA256, and explicit `neural_execution=false` /
`typed_api_gpu_acceptance=false`. Preserve this receipt, but do not confuse it with
runtime acceptance. Generated config changes are documented in `MODIFICATIONS.md`.
Concurrent unrelated modification of input/output directories is unsupported.

## Runtime integration and AMD patches

This is an **HF-plugin integration**, not upstream-native D1/Tabby chat support.
The original D1 `system_one` / `answer` / packed-tree semantics must be preserved.
In the qualified historical route, explicitly calling
`exllamav3.integration.transformers.register()` was necessary before loading.
Reject missing/unexpected/mismatched weights, require 166 actual `Exl3HfLinear`
body replacements, call the author's head-tying operation after loading, and check
that `lm_head.weight is get_input_embeddings().weight`. These requirements are
instructions for a separately admitted GPU test, not an executable admission here.

`patches/hybrid.py.patch` and `patches/lfm2_vl.py.patch` apply only to the matching
pinned author files (identities in `upstream/manifest.json`). On a disposable copy:

```sh
patch --batch --fuzz=0 -d /path/to/model-code-copy -p1 -i "$PWD/patches/hybrid.py.patch"
patch --batch --fuzz=0 -d /path/to/model-code-copy -p1 -i "$PWD/patches/lfm2_vl.py.patch"
```

They use the author's explicit GPU attention math in query chunks of 256 and
chunk the tokenwise FFN by 8192. CPU learned fallback is rejected. These are
compatibility/memory changes, not a proven fastest kernel. Own-branch attention
still grows quadratically with an individual branch; arbitrary long branches are
not qualified. The causal attention replacement also affects CUDA; **do not apply
these AMD-focused patches to a CUDA deployment assuming unchanged performance**.

No new model should be loaded from this README without separate per-device
weights/vision/head/KV/workspace/runtime/background memory and thermal admission.
The historical environment also had additional ExLlamaV3 compatibility patches
and non-self-contained runtime linking. Those are not bundled as a reproducible
wheel here. Clean GPU installation/parity/capacity remain open release gates.

## Licensing and upstream return

Original bridge scripts and synthetic tests: MIT, see `LICENSE`. LiquidAI-derived
files under `upstream/` and `patches/` remain under the **LFM Open License v1.0**;
see `upstream/LICENSE` and `NOTICE`. In particular it has commercial-use conditions
including a revenue threshold; this repository does not relicense model/code as
unrestricted OSI open source. ExLlamaV3 is a separately obtained dependency with
its own license. No weights or calibration corpus are redistributed here.

No upstream merge, release or production deployment is implied. Keep any reviewed
fork/patch commit pinned until equivalent upstream functionality is released and
passes the same structural, typed numerical and full-capacity checks; then return
to the verified upstream release. Questions or contributions should preserve the
scope distinctions above and include a reproducible public/synthetic example.
