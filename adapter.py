"""Byte-preserving native LFM2 projection and author D1 EXL3 reassembly.

No Torch import or neural execution. Checkpoints remain immutable inputs.
"""

import argparse
from dataclasses import dataclass
import hashlib
from contract import check
import json
import math
import re
import os
from pathlib import Path
import struct
import shutil

PREFIX = "model.language_model."
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
QUANT_SUFFIXES = {"trellis", "suh", "svh", "su", "sv", "mcg", "mul1"}


@dataclass(frozen=True)
class TensorReference:
    path: Path
    key: str
    metadata: dict
    payload_start: int


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_tensors(path: Path) -> dict[str, TensorReference]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header length")
        size = struct.unpack("<Q", prefix)[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Unbounded checkpoint header")
        header = json.loads(stream.read(size), object_pairs_hook=unique_keys)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    payload_size = path.stat().st_size - 8 - size
    tensors = {}
    intervals = []
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        if (
            not isinstance(metadata, dict)
            or not {"dtype", "shape", "data_offsets"} <= metadata.keys()
        ):
            raise ValueError("Missing tensor metadata: " + key)
        if (
            not isinstance(metadata["dtype"], str)
            or not isinstance(metadata["shape"], list)
            or not isinstance(metadata["data_offsets"], list)
            or len(metadata["data_offsets"]) != 2
        ):
            raise ValueError("Invalid tensor metadata types: " + key)
        dtype = metadata["dtype"]
        shape = metadata["shape"]
        start, end = metadata["data_offsets"]
        if dtype not in DTYPE_BYTES or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("Invalid tensor dtype/shape: " + key)
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= payload_size
        ):
            raise ValueError("Tensor outside payload: " + key)
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError("Tensor byte count disagrees with shape: " + key)
        tensors[key] = TensorReference(path, key, metadata, 8 + size)
        if start != end:
            intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Overlapping/gapped checkpoint payload")
        cursor = end
    if cursor != payload_size:
        raise ValueError("Unaccounted checkpoint payload")
    return tensors


def write_tensors(path: Path, tensors: dict[str, TensorReference]) -> list[dict]:
    header = {}
    offset = 0
    for key, reference in sorted(tensors.items()):
        metadata = reference.metadata
        start, end = metadata["data_offsets"]
        header[key] = {**metadata, "data_offsets": [offset, offset + end - start]}
        offset += end - start
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    ledger = []
    with path.open("xb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for key, reference in sorted(tensors.items()):
            start, end = reference.metadata["data_offsets"]
            remaining = end - start
            checksum = hashlib.sha256()
            with reference.path.open("rb") as source:
                source.seek(reference.payload_start + start)
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Source truncated during streaming copy")
                    output.write(block)
                    checksum.update(block)
                    remaining -= len(block)
            ledger.append(
                {
                    "key": key,
                    "source_key": reference.key,
                    "bytes": end - start,
                    "sha256": checksum.hexdigest(),
                }
            )
        output.flush()
        os.fsync(output.fileno())
    return ledger


def author_contract(source: Path, expected_sha: str):
    checkpoint = source / "model.safetensors"
    if sha(checkpoint) != expected_sha:
        raise ValueError("Author full checkpoint SHA mismatch")
    tensors = read_tensors(checkpoint)
    config = json.loads(
        (source / "config.json").read_text(), object_pairs_hook=unique_keys
    )
    check(config, {key: value.metadata for key, value in tensors.items()})
    if not config.get("tie_word_embeddings") or config.get("quantization_config"):
        raise ValueError("Unquantized author config with tied trained head required")
    linears = {
        key[:-7]
        for key in tensors
        if key.startswith(PREFIX)
        and key.endswith(".weight")
        and len(tensors[key].metadata["shape"]) == 2
        and key != PREFIX + "embed_tokens.weight"
    }
    if any(ref.metadata["dtype"] != "BF16" for ref in tensors.values()):
        raise ValueError("All original author tensors must retain BF16")
    if len(linears) != 166 or len(tensors) != 707:
        raise ValueError("Qualified author topology changed")
    return config, tensors, linears


def copy_metadata(
    source: Path, destination: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Copy only explicitly pinned metadata; LICENSE must always be included."""
    for name, expected in pins.items():
        target = destination / name
        payload = (source / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Source metadata changed: " + name)
        with target.open("xb") as stream:
            stream.write(payload)
    return dict(pins)


def assemble_tensors(
    author: dict[str, TensorReference],
    native: dict[str, TensorReference],
    linears: set[str],
) -> dict[str, TensorReference]:
    output = {
        key: reference
        for key, reference in author.items()
        if key not in {name + ".weight" for name in linears}
    }
    groups = {}
    for key, reference in native.items():
        if not key.startswith("model."):
            if key.rsplit(".", 1)[-1] in QUANT_SUFFIXES:
                raise ValueError("Unreviewed non-body quantized group: " + key)
            continue
        renamed = PREFIX + key[len("model.") :]
        module, suffix = renamed.rsplit(".", 1)
        if module not in linears:
            if suffix in QUANT_SUFFIXES:
                raise ValueError("Unreviewed model quantized group: " + key)
            continue
        if suffix not in QUANT_SUFFIXES:
            raise ValueError(
                "Converted target has unquantized/unexpected payload: " + key
            )
        groups.setdefault(module, set()).add(suffix)
        if renamed in output:
            raise ValueError("Quantized tensor collides with preserved author tensor")
        output[renamed] = reference
    if set(groups) != linears or any(
        "trellis" not in suffixes for suffixes in groups.values()
    ):
        raise ValueError("Missing native quantized projection group")
    for module, suffixes in groups.items():
        if suffixes != {"trellis", "suh", "svh", "mul1"}:
            raise ValueError(
                "Current native converter FP16 sign/scales required: " + module
            )
        out_features, in_features = author[module + ".weight"].metadata["shape"]
        trellis = output[module + ".trellis"].metadata
        expected_shape = [in_features // 16, out_features // 16, 128]
        if (
            in_features % 16
            or out_features % 16
            or trellis["dtype"] != "I16"
            or trellis["shape"] != expected_shape
        ):
            raise ValueError(
                "Packed trellis axis order/shape is not the reviewed eight-bit layout: "
                + module
            )
        marker = output[module + ".mul1"]
        if marker.metadata["dtype"] != "I32" or marker.metadata["shape"] != []:
            raise ValueError("Invalid mul1 codebook marker shape/dtype")
        with marker.path.open("rb") as stream:
            stream.seek(marker.payload_start + marker.metadata["data_offsets"][0])
            if stream.read(4) != struct.pack("<I", 0x83DCD12D):
                raise ValueError("Unexpected mul1 codebook marker value")
        for suffix, width in [("suh", in_features), ("svh", out_features)]:
            metadata = output[module + "." + suffix].metadata
            if metadata["dtype"] != "F16" or metadata["shape"] != [width]:
                raise ValueError(
                    "Projection sign/scale shape or dtype mismatch: " + module
                )

    return output


def native_tensors(directory: Path):
    paths = sorted(directory.glob("*.safetensors"))
    tensors = {}
    if not paths:
        raise ValueError("No compiled native checkpoint shards")
    pins = {path.name: sha(path) for path in paths}
    for path in paths:
        shard = read_tensors(path)
        if set(tensors) & set(shard):
            raise ValueError("Duplicate tensor between native shards")
        tensors.update(shard)
    return tensors, pins


def execute(
    action: str,
    source: Path,
    output: Path,
    expected_sha: str,
    native: Path | None,
    metadata: dict[str, str],
    headroom: int,
):
    config, author, linears = author_contract(source, expected_sha)
    if output.exists():
        raise ValueError("Fresh output namespace required")
    if action == "project":
        tensors = {
            "model." + key[len(PREFIX) :]: ref
            for key, ref in author.items()
            if key.startswith(PREFIX)
        }
        output_config = dict(config["text_config"])
        output_config["architectures"] = ["Lfm2ForCausalLM"]
        native_pins = {}
    else:
        if native is None:
            raise ValueError("Compiled native input required")
        converted, native_pins = native_tensors(native)
        native_config_sha = sha(native / "config.json")
        quant = json.loads((native / "config.json").read_text()).get(
            "quantization_config", {}
        )
        required_quant = {
            "quant_method": "exl3",
            "version": "1.6.2",
            "head_bits": 16,
            "out_scales": "auto",
            "codebook": "mul1",
        }
        if any(quant.get(key) != value for key, value in required_quant.items()):
            raise ValueError("Expected reviewed EXL3 1.6.2 mul1/head16 configuration")
        tensors = assemble_tensors(author, converted, linears)
        output_config = {**config, "quantization_config": quant}
    required = sum(
        ref.metadata["data_offsets"][1] - ref.metadata["data_offsets"][0]
        for ref in tensors.values()
    )
    required += sum((source / name).stat().st_size for name in metadata)
    # Header, receipt and user-selected free-space headroom are separate from tensor payloads.
    required += 16 * 1024 * 1024 + headroom
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError("Insufficient output space including metadata/header/headroom")
    output.mkdir(mode=0o700)
    metadata_pins = copy_metadata(source, output, metadata)
    (output / "MODIFICATIONS.md").write_text(
        "Modified by d1-exl3-bridge: "
        + action
        + " for EXL3 1.6.2; source revision "
        + REVISION
        + ".\n"
        "Language-body quantization only; original non-body tensors/head embedding preserved on assembly.\n"
        "This byte-conversion receipt is not a GPU inference or quality qualification.\n"
    )
    ledger = write_tensors(output / "model.safetensors", tensors)
    if sha(source / "model.safetensors") != expected_sha:
        raise ValueError("Author changed during copy; output NOT accepted")
    if native is not None:
        if sha(native / "config.json") != native_config_sha or any(
            sha(native / name) != pin for name, pin in native_pins.items()
        ):
            raise ValueError("Native output changed during assembly; NOT accepted")
    (output / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    receipt = {
        "status": action.upper() + "_BYTES_VERIFIED",
        "source_sha256": expected_sha,
        "native_shards": native_pins,
        "original_metadata_sha256": metadata_pins,
        "tensor_ledger": ledger,
        "author_tensor_count": len(author),
        "replaced_author_weights": len(linears) if action == "assemble" else 0,
        "preserved_original_tensors": len(author) - len(linears)
        if action == "assemble"
        else None,
        "output_sha256": sha(output / "model.safetensors"),
        "typed_api_gpu_acceptance": False,
        "neural_execution": False,
    }
    (output / "adapter-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"


def validate_metadata_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or name
        in {".", "..", "model.safetensors", "adapter-receipt.json", "MODIFICATIONS.md"}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
    ):
        raise ValueError("Metadata must be an allowed simple basename")
    if name not in {"LICENSE", "NOTICE"} and Path(name).suffix not in {
        ".py",
        ".json",
        ".jinja",
        ".txt",
        ".md",
    }:
        raise ValueError("Unsupported metadata name: " + name)
    if name.endswith(".safetensors.index.json"):
        raise ValueError(
            "Source weight indices cannot be copied to a single-file output"
        )


def admit_source(source: Path, manifest: Path) -> tuple[str, dict[str, str]]:
    """Check an explicit source manifest before output creation or copying code."""
    receipt = json.loads(manifest.read_text(), object_pairs_hook=unique_keys)
    if not isinstance(receipt, dict):
        raise ValueError("Source manifest must be a JSON object")
    if receipt.get("model_revision") != REVISION:
        raise ValueError("Only the documented D1 revision is supported")
    expected = receipt["model_sha256"]
    pins = receipt["metadata_sha256"]
    if not isinstance(pins, dict) or not {"LICENSE", "config.json"} <= pins.keys():
        raise ValueError("Pinned config and LICENSE are required")
    for name in pins:
        validate_metadata_name(name)
    for name, digest in {"model.safetensors": expected, **pins}.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Expected a complete lowercase SHA256 digest")
        if name != "model.safetensors" and sha(source / name) != digest:
            raise ValueError("Metadata SHA mismatch: " + name)
    return expected, pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["project", "assemble"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument(
        "--headroom-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Additional free disk space above payload/header estimate (default: 64 MiB)",
    )
    args = parser.parse_args()
    try:
        if args.headroom_bytes < 0:
            raise ValueError("Headroom cannot be negative")
        if args.action == "project" and args.native is not None:
            raise ValueError("--native applies only to assembly")
        source_sha, metadata = admit_source(args.source, args.source_manifest)
        receipt = execute(
            args.action,
            args.source,
            args.output,
            source_sha,
            args.native,
            metadata,
            args.headroom_bytes,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "tensor_ledger"}
        )
    )


if __name__ == "__main__":
    main()
