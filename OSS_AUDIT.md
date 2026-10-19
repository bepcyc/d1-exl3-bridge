# D1 EXL3 source-package audit

## Repository summary

Scope: portable byte-stream projection/assembly glue and separately attributed
AMD source patches; Python standard library, CLI entry points `adapter.py` and
`make_manifest.py`, no binary/package-registry release. No runtime backend, weights,
training corpus or private fixtures are included. Verification is `unittest`
synthetic CLI/format tests plus Ruff. `LICENSE` covers original glue/tests; author
snapshots/patches retain the LFM Open License in `upstream/LICENSE` and `NOTICE`.

The extracted CLI is new and only CPU/synthetic-tested. Historical exact AMD
logic executed on one GPU profile, described narrowly in `HISTORICAL_RESULTS.md`.
This is reviewable experimental source, not a turnkey runtime or model release.

## Correctness

- **P0 for a ready-to-run GPU release:** the new wrapper has no clean GPU install,
  strict real-loader, typed-probability or capacity run. Evidence: tests import
  only stdlib and fabricate weight payloads; historical entry points differ.
  Impact: a structural success could be confused with model correctness. Smallest
  fix: preserve experimental boundary now; later qualify an independently built
  runtime using exact model/software hashes, real 166-module checks and head tie.
- **P1 addressed:** the old machine adapter lacked exact trellis-axis/marker-value
  checks and automatically selected metadata by suffix. `adapter.py` now validates
  exact eight-bit axes, four buffers including mul1 marker, explicit metadata
  digests, topology/head tie, original/native post-stream hashes and fresh output.

## Maintainability

- **P1 addressed:** original coordinator used fixed filesystem devices, paths,
  receipt hashes and service lifecycle. The public tools accept explicit source,
  output, native directory, manifest and free-disk headroom, without coordinating
  services, CPU/GPU policies, driver state or authentication.
- **P2:** only the documented D1 topology/EXL3 1.6.2 format is supported. Evidence:
  exact 707/266/166/541 counts and quant config checks. Smallest future change:
  add a separately tested format/profile rather than silently weakening checks.

## Testability

- **P1 addressed:** synthetic end-to-end 707→266/1205 payload tests, all 541
  preserved bytes, real CLI calls, corrupt headers, duplicate shards, wrong hashes,
  path names, disk refusal, partial failure/replay, tied-head/config and quant
  layout/marker refusals. Patch tests use exact public source and complete ASTs.
- **P1 remaining:** all 11 tests were verified locally on Python 3.12 only.
  Remote CI is not configured in this source-only publication scope; executable
  workflows are excluded, and no remote CI result is claimed. Impact: portability
  outside the tested interpreter is not yet proven. Smallest follow-up: expand
  actual interpreter coverage and configure CI in a separately scoped change.

## Security

- **P1 addressed:** no Torch/custom model-code execution or weights download;
  explicit metadata allowlist/digests, simple basenames, no generated indices or
  weights masquerading as metadata, non-overwriting fresh outputs and no success
  receipt on write failure. Files originate only from caller-selected trusted
  model folders; concurrent mutation is unsupported and post-copy drift refuses.
- **P1 publication gate:** inspect complete staged tree/history and final public
  files for personal paths, credentials, images or private receipts. Only public
  source snapshots, original glue, synthetic tests and sanitized aggregate history
  may be staged. No PhotoLab worktree history may be imported.

## Performance

- **P1 addressed:** streaming tensor payloads in at most 1 MiB blocks; bounded
  header size 16 MiB; no model construction or per-tensor Torch allocations. The
  full tiny synthetic suite used maxRSS 27,980 KiB on this host. This is a test
  process observation, not a memory bound for every real checkpoint/disk setup.
- **P1 remaining:** no portable full-checkpoint I/O throughput or new GPU speed
  benchmark exists. Historical AMD math route is compatibility-focused and may
  be slow, especially if applied to CUDA. Smallest fix: measure comparable public
  data after the independent runtime gate; do not call this an optimum.

## Observability

- **P1 addressed:** actionable nonzero CLI errors, success only after final input
  rechecks, per-tensor output receipt and explicit false NN/GPU acceptance fields.
  Partial output directories are retained for inspection without a success receipt.
- **P2:** no automatic resume/cleanup daemon exists. Evidence: fresh namespace is
  mandatory. This is deliberate bounded CLI behavior, not unattended production
  recovery. Future resume requires a reviewed journal, not overwrite/retry logic.

## Documentation

- **P1 addressed:** README names exact revision/format, two-stage use, metadata
  manifest creation, separate GPU quantization/loading gates, patch effects and
  author-license commercial-use conditions. Historical results separate modified
  wrapper from historical runtime code. No standalone installation claim.
- **P0 for a weight redistribution release:** no derivative model/license/model
  card/public synthetic quality run/clean reproducible runtime package is prepared.
  Impact: a weight release would overstate its acceptance and permissions. Smallest
  fix: do not publish weights as part of this source release; prepare separately.

## File-level changes

| Priority | Path | Action | Reason |
|---|---|---|---|
| P1 | adapter.py | add | Portable strict streaming projection/assembly |
| P1 | contract.py | add | Exact author language topology/head validation |
| P1 | make_manifest.py | add | Explicit source digest and metadata selection |
| P1 | tests/test_adapter.py | add | Synthetic end-to-end/refusal/failure regression proof |
| P1 | tests/test_patches.py | add | Exact public input and historical AST identity |
| P1 | patches/*.patch | add | Isolated AMD modifications with author notices |
| P1 | upstream/* | add | Pinned public-only sources/config/license/provenance |
| P1 | LICENSE, NOTICE | add | Separate glue and author-derived terms/attribution |
| P1 | README.md, HISTORICAL_RESULTS.md | add | Reproduction, scope and explicit runtime gaps |
| P2 | .gitignore | add | Exclude weights, cache, virtualenv and accidental secrets |

## Do first / do later

Source-release checks: independent source/privacy/license review and all 11 CPU
tests passed before publication. Release contents contain only the isolated source
tree; no weights or private inputs are included. The release is experimental, and
remote CI is not configured. See `CHANGELOG.md` for its supported scope.

Do later: independent GPU runtime/build recipe, typed quality/parity/capacity,
performance benchmarking, supported-device matrix and any separately licensed
weight release. Do not weaken these gates or substitute mock checks for them.
