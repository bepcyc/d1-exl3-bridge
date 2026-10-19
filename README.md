# D1 ↔ ExLlamaV3 bridge

Tools for adapting **LiquidAI/d1-3B to ExLlamaV3-quantized language layers while
retaining D1's original multimodal decision model**. The bridge prepares and
reassembles checkpoints; your application continues to use the author's D1
implementation for image inputs, typed questions and decision probabilities.

The Python package and resident JSONL CLI expose D1's original typed decisions
and probabilities, using EXL3 for its language body. Supply your own state,
questions and images; checkpoint path, device and capacity are configurable.

**Validation:** conversion and runtime contracts pass CPU tests. GPU acceptance
of the new portable runtime is pending. Historical GPU results remain separate in
[HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

- [The deployment experience we want](#the-deployment-experience-we-want)
- [Why simply loading D1 is not enough](#why-simply-loading-d1-is-not-enough)
- [What this repository supplies](#what-this-repository-supplies)
- [Checkpoint contract](#what-the-bridge-preserves)
- [Get the code and run the tests](#get-the-code)
- [Conversion workflow](#prepare-an-explicit-input-manifest)
- [Runtime integration](#runtime-integration)
- [Optional AMD compatibility patches](#optional-amd-compatibility-patches)
- [License and upstream return](#licensing-and-upstream-return)

## Run typed decisions

Use Python 3.12+ with the installed GPU build of Torch and ExLlamaV3 1.6.2,
Transformers 5.14.1, Accelerate, Safetensors and Pillow. Install the bridge without
replacing those compiled dependencies:

```sh
uv pip install --python "$D1_PYTHON" --no-deps .
uv run --offline --no-project --python "$D1_PYTHON" python -m d1_exl3 \
  --model "$D1_MODEL" --device cuda:0 \
  --minimum-free-gib 14 --allocator-gib 13 --stats < requests.jsonl
```

D1_MODEL is your assembled q8/mul1 checkpoint directory, including the verified
author code, tokenizer and processor files. Loading is local-only. Set memory
flags from your own per-GPU calculation and measured workload. This example
approximates the historical 16 GiB full-context profile, not a universal fit claim.

One JSON object per line requests one answer; an array requests a native batch.
Output is one JSON object/array per line, in input order:

```json
{"state":{"request":"Refund the duplicate order"},"questions":{"refund":{"type":"noul","instructions":"Is a refund explicitly requested?"}}}
```

For images, add an images list containing path or base64 objects. JSON/text
state, images and named noul/choice/score questions use the author's decision API,
without chat generation or intermediate captions.

The first request loads once; later lines reuse that model. EOF or 300 seconds of
input inactivity closes the runtime and exits. CPU decoding/tokenization is
allowed; learned execution stays on the chosen GPU. Torch uses cuda:N for both
CUDA and ROCm. Independent processes can serve different GPUs without hostname
or GPU-brand assumptions.

```python
from pathlib import Path
from d1_exl3 import D1Runtime, RuntimeConfig

config = RuntimeConfig(
    model=Path("./my-d1-exl3"),
    minimum_free_bytes=15_000_000_000,
    allocator_limit_bytes=14_000_000_000,
    device="cuda:0",
)
with D1Runtime(config) as d1:
    result = d1.system_one(
        {"total":23},
        {"large":{"type":"noul","instructions":"Is total greater than 20?"}},
    )
    batch = d1.system_one_batch([
        ({"total":8},{"large":{"type":"noul","instructions":"Is total greater than 20?"}}),
        ({"total":42},{"large":{"type":"noul","instructions":"Is total greater than 20?"}}),
    ])
    probabilities = d1.probabilities(
        {"total":23},
        [{"type":"noul","instructions":"Is total greater than 20?"}],
    )
```

Native batching is preserved: single-question image-free requests pack together;
multiple questions share their own state; image requests run native serial B1.
This does not claim tensor batching of images. Path, packed-token, branch and
tile limits are checked before learned execution without truncation. Default
capacity: model 32768-token paths,98048 packed tokens,65536 branch budget,
256 branches and11 total vision tiles. Raising limits requires a matching memory
plan and qualification. Models are not moved or offloaded after loading.

The optional runtime extra declares dependencies for new environments. CUDA/ROCm
EXL3 builds and optional AMD patches must match the installed Torch backend.
The 11 conversion tests remain dependency-free; new runtime tests use injected
fakes and original author readout without models or GPU. Tests cover typed
results, ordering, admission, native EXL3-loader contracts and residency.

## The deployment experience we want

Ideally, you would download D1, select ExLlamaV3 as its backend, deploy it, and
immediately use **all of D1's capabilities**: images and text, named typed
questions, answer probabilities, and batches of related questions or states.
Choosing a quantized checkpoint would change its resource requirements without
requiring you to rebuild the model's application API.

That is the problem this project addresses. D1's published checkpoint and custom
execution path do not become a native ExLlamaV3 deployment just because its text
backbone belongs to a supported model family. There are several integration
steps between those two points, and this repository currently supplies the
checkpoint conversion and runtime integration.

If the author's original Transformers implementation already meets your needs,
you can use it directly. This bridge is for developers who specifically want to
use EXL3 quantization for D1's language-body matrices while keeping the rest of
the decision model. It does not replace the decision model with a chat model.

## Why simply loading D1 is not enough

The explanation here is specific to **ExLlamaV3 1.6.2** and the
[supported D1 revision](https://huggingface.co/LiquidAI/d1-3B/tree/051bcc464b01b9f92942b364d9586b0ef5912432).
It is not a claim that future ExLlamaV3 versions cannot support D1 natively.

### 1. Supporting the text backbone does not implement the whole D1 model

ExLlamaV3 1.6.2 has a native text-only LFM2 implementation. D1's checkpoint
describes an LFM2-VL model and maps `AutoModel` to the author's custom `D1Model`.
That implementation retains the vision tower and projector and replaces the
language model with the author's hybrid execution code. The native architecture
registry at this ExLlamaV3 revision does not supply that complete D1/LFM2-VL
implementation. Changing a model name or an `architectures` field cannot add
the missing execution path. Sources: [native architecture registry](https://github.com/turboderp-org/exllamav3/blob/4c84273155359ebb305accc39f74ab683eaaadf5/exllamav3/architecture/architectures.py),
[D1 configuration](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/config.json),
[author model implementation](upstream/lfm2_vl.py).

### 2. D1's decision API is more than text generation

D1 exposes `system_one()` and `system_one_batch()`. They accept named questions
and return typed answers and probabilities. The author's runner shares a common
input prefix and packs question/answer branches together without padding; its
attention and convolution paths respect that tree. A text-generation endpoint
does not automatically reproduce these methods, their outputs or their batching
semantics. Successful loading of some language weights would therefore be
insufficient evidence of a working D1 deployment. Sources:
[D1 model API](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/modeling_d1.py),
[typed answer definitions](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/api.py),
[question/batch runner](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/runner.py),
[packed-tree execution](upstream/hybrid.py).

### 3. The checkpoints have different layouts and responsibilities

D1 stores text weights under `model.language_model.*`, alongside its vision and
other tensors. The native LFM2 converter works with the text model's `model.*`
layout. A complete D1 checkpoint cannot be treated as a text-only checkpoint and
then used as the complete multimodal model again without a reconstruction step.

The intended transformation here quantizes selected language-body matrices,
retains the other trained tensors and restores D1's names. Embedding/head tying
also needs to survive loading: producing text with an accidentally substituted
head would not establish that answers came from the intended trained model. Sources:
[native LFM2 layout](https://github.com/turboderp-org/exllamav3/blob/4c84273155359ebb305accc39f74ab683eaaadf5/exllamav3/architecture/lfm2.py),
[EXL3 conversion configuration](https://github.com/turboderp-org/exllamav3/blob/4c84273155359ebb305accc39f74ab683eaaadf5/exllamav3/conversion/quant_config.py),
[bridge topology checks](contract.py).

These architecture, API and checkpoint issues exist independently of whether
your GPU is NVIDIA or AMD. GPU-specific kernel compatibility is a further,
separate question.

## What this repository supplies

The integration route preserves the author's D1 model and uses ExLlamaV3's
**Transformers integration** to replace supported linear layers with EXL3 layers.
This differs from adding a complete native D1 architecture to ExLlamaV3's model
loader or adding a D1 endpoint to a serving application. The upstream plugin
matches quantized module names to the actual model's linear layers; that is why
checkpoint names and reconstruction matter. See the
[pinned Transformers integration](https://github.com/turboderp-org/exllamav3/blob/4c84273155359ebb305accc39f74ab683eaaadf5/exllamav3/integration/transformers.py).

```text
Author D1 checkpoint
    → project: prepare a text-only LFM2 view for the converter
    → official ExLlamaV3 quantization, performed separately
    → assemble: restore D1's layout and preserved multimodal tensors
    → author D1 implementation + EXL3 Transformers integration
    → your application or serving integration
```

| What you need | What is available here | What remains outside this repository |
|---|---|---|
| Prepare D1 weights for the text-model quantizer | `project`, with topology and hash checks | Obtain and verify the author's checkpoint |
| Quantize the language body | A compatible projected checkpoint and a strict assembly contract | Run the official quantizer with a suitable runtime and memory budget |
| Reconstruct the multimodal checkpoint | `assemble`, preserving non-body payloads and restoring names | Validate loading and trained head tying in the intended runtime |
| Use D1 decisions, probabilities and images | d1_exl3 Python API and resident JSONL CLI | Qualify the chosen GPU/backend and capacity |
| Deploy a complete service | No server or automatic launcher is included | Serving, lifecycle, concurrency and end-to-end feature validation |
| Address particular AMD execution issues | Optional, separately scoped patches | Validate them on the intended GPU/runtime and workload |

Reducing language-body weight storage can help that component's memory usage.
Total inference memory still includes vision, projector, embeddings/head,
activations, branch state and runtime buffers. This repository does not establish
a universal VRAM saving, speedup or supported-GPU matrix.

Quantization can change logits, probabilities and answers. Retaining D1's API,
execution path and trained head connections does not guarantee identical outputs
or probability calibration. Evaluate the quantized model against the original
on representative inputs for your task.

The portable CLI is verified on fabricated checkpoints. Historical evidence for
an earlier integration is recorded separately in [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md);
it does not certify a clean installation of this extracted tool or complete
feature parity. The remaining runtime and quality checks are explicit in
[OSS_AUDIT.md](OSS_AUDIT.md).

## Get the code

The experimental source release is `v0.1.0-alpha.1`. Use its fixed tag to reproduce
the checkpoint-tool tests:

```sh
git clone https://github.com/bepcyc/d1-exl3-bridge.git
cd d1-exl3-bridge
git checkout v0.1.0-alpha.1
```

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

Checkpoint tools require Python3.11+; the optional runtime package requires
Python 3.12+. All 31 CPU tests pass, including the original 11 conversion/patch
tests. Remote CI is not configured. The checkpoint CLI has no third-party dependency and imports no
Torch, Transformers or ExLlama runtime. Patch tests also require the standard
`patch` executable. Run from a checkout:

```sh
uv run --no-project --python 3.12 python -B -m unittest discover -s tests -v
uvx --from ruff==0.17.1 ruff check adapter.py contract.py make_manifest.py tests
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

# Run the official quantizer separately with a supported configuration.

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

## Runtime integration

This is an **HF-plugin integration**, not upstream-native D1/Tabby chat support.
The original D1 `system_one` / `answer` / packed-tree semantics must be preserved.
In the qualified historical route, explicitly calling
`exllamav3.integration.transformers.register()` was necessary before loading.
Reject missing/unexpected/mismatched weights, require 166 actual `Exl3HfLinear`
body replacements, call the author's head-tying operation after loading, and check
that `lm_head.weight is get_input_embeddings().weight`. These are runtime
validation requirements implemented by the runtime loader, including exact
replacement names and quantized-buffer placement. The checkpoint-conversion
CLI remains separate and never loads a model.

Test the features you intend to expose through the resulting service: typed
answers and probability outputs, image questions, multiple questions sharing a
state, and `system_one_batch()`. Quantized-layer counts and a successful forward
pass alone does not establish complete D1 behavior. CPU tests cover original
author readout, arbitrary states/questions, ordering, raw probabilities, native
loader guards and teardown. GPU numerical/parity/capacity checks are the next
acceptance step for each backend profile.

## Optional AMD compatibility patches

This section concerns device/runtime execution after the checkpoint integration
described above. **AMD hardware is not required to use the checkpoint tools.**
If you are investigating a CUDA deployment, do not apply these patches merely
because you are using D1 or EXL3.

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

The same patch logic ran during an earlier integration on an RX 6800 XT. That is
one historical test environment, not this project's target audience or a claim
that the tools require that card. The earlier environment also had additional
ExLlamaV3 compatibility patches and external runtime linking, so this repository
does not reproduce a complete ROCm installation by itself. See the exact scope
in [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

For either GPU vendor, validate the intended runtime and calculate memory for
the full workload before loading: weights, vision, head, cache/branch state,
workspace, runtime and existing allocations. Clean installation, typed-output
parity and full-capacity validation remain open checks for this extracted package.

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
