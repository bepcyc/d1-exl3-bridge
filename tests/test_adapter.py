"""All checkpoint payloads here are fabricated bytes, never trained weights."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import adapter

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path, entries):
    header, payload = {}, bytearray()
    for key, (dtype, shape, value) in entries.items():
        start = len(payload)
        payload.extend(value)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return adapter.read_tensors(path)


def payload(reference):
    begin, end = reference.metadata["data_offsets"]
    with reference.path.open("rb") as stream:
        stream.seek(reference.payload_start + begin)
        return stream.read(end - begin)


def distinct_bytes(key, length):
    seed = hashlib.sha256(key.encode()).digest()
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def fixture(source, native):
    source.mkdir()
    native.mkdir()
    config = json.loads((ROOT / "upstream/config.json").read_text())
    c = config["text_config"]
    c.update(
        hidden_size=32,
        intermediate_size=32,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    dim, inter, head, kv = 32, 32, 16, 16
    shapes = {"embed_tokens.weight": [32, dim], "embedding_norm.weight": [dim]}
    linears = {}
    for index, kind in enumerate(c["layer_types"]):
        prefix = f"layers.{index}."
        shapes[prefix + "operator_norm.weight"] = [dim]
        shapes[prefix + "ffn_norm.weight"] = [dim]
        projections = {
            "feed_forward.w1": [inter, dim],
            "feed_forward.w3": [inter, dim],
            "feed_forward.w2": [dim, inter],
        }
        if kind == "conv":
            projections.update(
                {"conv.in_proj": [3 * dim, dim], "conv.out_proj": [dim, dim]}
            )
            shapes[prefix + "conv.conv.weight"] = [dim, 1, c["conv_L_cache"]]
        else:
            projections.update(
                {
                    "self_attn.q_proj": [dim, dim],
                    "self_attn.k_proj": [kv, dim],
                    "self_attn.v_proj": [kv, dim],
                    "self_attn.out_proj": [dim, dim],
                }
            )
            shapes[prefix + "self_attn.q_layernorm.weight"] = [head]
            shapes[prefix + "self_attn.k_layernorm.weight"] = [head]
        for key, shape in projections.items():
            shapes[prefix + key + ".weight"] = shape
            linears[adapter.PREFIX + prefix + key] = shape
    import math

    author = {
        adapter.PREFIX + key: ("BF16", shape, distinct_bytes(key, 2 * math.prod(shape)))
        for key, shape in shapes.items()
    }
    for index in range(707 - len(author)):
        author[f"model.vision_tower.synthetic_{index}"] = (
            "BF16",
            [1],
            distinct_bytes(str(index), 2),
        )
    checkpoint(source / "model.safetensors", author)
    (source / "config.json").write_text(json.dumps(config))
    (source / "LICENSE").write_text("Synthetic fixture only; no actual weights.")
    quantized = {}
    for module, (out_dim, in_dim) in linears.items():
        name = "model." + module[len(adapter.PREFIX) :]
        quantized[name + ".trellis"] = (
            "I16",
            [in_dim // 16, out_dim // 16, 128],
            distinct_bytes(name + ".trellis", in_dim * out_dim),
        )
        quantized[name + ".suh"] = (
            "F16",
            [in_dim],
            distinct_bytes(name + ".suh", 2 * in_dim),
        )
        quantized[name + ".svh"] = (
            "F16",
            [out_dim],
            distinct_bytes(name + ".svh", 2 * out_dim),
        )
        quantized[name + ".mul1"] = ("I32", [], struct.pack("<I", 0x83DCD12D))
    quantized["model.embed_tokens.weight"] = ("F16", [1], b"XX")
    quantized["lm_head.weight"] = ("F16", [1], b"HH")
    checkpoint(native / "model.safetensors", quantized)
    (native / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "version": "1.6.2",
                    "head_bits": 16,
                    "out_scales": "auto",
                    "codebook": "mul1",
                    "bits": 8.0,
                }
            }
        )
    )
    pins = {name: adapter.sha(source / name) for name in ["config.json", "LICENSE"]}
    manifest = source.parent / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_revision": adapter.REVISION,
                "model_sha256": adapter.sha(source / "model.safetensors"),
                "metadata_sha256": pins,
            }
        )
    )
    return manifest, pins


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.native = self.root / "source", self.root / "native"
        self.manifest, self.pins = fixture(self.source, self.native)
        self.source_sha = adapter.sha(self.source / "model.safetensors")

    def tearDown(self):
        self.temp.cleanup()

    def command(self, action, output, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "adapter.py"),
                action,
                "--source",
                str(self.source),
                "--output",
                str(output),
                "--source-manifest",
                str(self.manifest),
                *extra,
            ],
            text=True,
            capture_output=True,
        )

    def test_cli_project_assemble_and_all_preserved_payloads(self):
        before = {
            p: adapter.sha(p)
            for directory in [self.source, self.native]
            for p in directory.iterdir()
        }
        projected = self.root / "projection"
        result = self.command("project", projected)
        self.assertEqual(result.returncode, 0, result.stderr)
        projection = adapter.read_tensors(projected / "model.safetensors")
        self.assertEqual(len(projection), 266)
        self.assertEqual(
            json.loads((projected / "config.json").read_text())["architectures"],
            ["Lfm2ForCausalLM"],
        )
        assembled = self.root / "assembled"
        result = self.command("assemble", assembled, "--native", str(self.native))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = adapter.read_tensors(assembled / "model.safetensors")
        original = adapter.read_tensors(self.source / "model.safetensors")
        self.assertEqual(len(output), 1205)
        preserved = [key for key in original if key in output]
        self.assertEqual(len(preserved), 541)
        for key in preserved:
            self.assertEqual(payload(output[key]), payload(original[key]), key)
        self.assertNotIn("lm_head.weight", output)
        native = adapter.read_tensors(self.native / "model.safetensors")
        copied_quantized = 0
        for key in output:
            if key not in original:
                source_key = "model." + key[len(adapter.PREFIX) :]
                self.assertEqual(payload(output[key]), payload(native[source_key]), key)
                copied_quantized += 1
        self.assertEqual(copied_quantized, 664)
        self.assertEqual(
            payload(output[adapter.PREFIX + "embed_tokens.weight"]),
            payload(original[adapter.PREFIX + "embed_tokens.weight"]),
        )
        receipt = json.loads((assembled / "adapter-receipt.json").read_text())
        self.assertFalse(receipt["neural_execution"])
        self.assertFalse(receipt["typed_api_gpu_acceptance"])
        self.assertEqual(receipt["preserved_original_tensors"], 541)
        self.assertEqual(
            (assembled / "LICENSE").read_bytes(), (self.source / "LICENSE").read_bytes()
        )
        self.assertTrue((assembled / "MODIFICATIONS.md").is_file())
        self.assertEqual(before, {p: adapter.sha(p) for p in before})
        self.assertNotEqual(
            self.command(
                "assemble", assembled, "--native", str(self.native)
            ).returncode,
            0,
        )

    def test_bad_model_and_metadata_hashes_refuse_before_output(self):
        original = json.loads(self.manifest.read_text())
        for field in ["model_sha256", "metadata_sha256"]:
            manifest = copy.deepcopy(original)
            if field == "model_sha256":
                manifest[field] = "0" * 64
            else:
                manifest[field]["config.json"] = "0" * 64
            self.manifest.write_text(json.dumps(manifest))
            output = self.root / field
            self.assertEqual(self.command("project", output).returncode, 2)
            self.assertFalse(output.exists())

    def test_manifest_requires_license_safe_names_and_supported_revision(self):
        original = json.loads(self.manifest.read_text())
        cases = []
        bad = copy.deepcopy(original)
        del bad["metadata_sha256"]["LICENSE"]
        cases.append(bad)
        bad = copy.deepcopy(original)
        bad["metadata_sha256"]["../secret.json"] = "0" * 64
        cases.append(bad)
        bad = copy.deepcopy(original)
        bad["metadata_sha256"]["model.safetensors"] = "0" * 64
        cases.append(bad)
        bad = copy.deepcopy(original)
        bad["model_revision"] = "other"
        cases.append(bad)
        for case in cases:
            self.manifest.write_text(json.dumps(case))
            with self.subTest(case=case), self.assertRaises(ValueError):
                adapter.admit_source(self.source, self.manifest)

    def test_source_topology_and_tied_head_refuse_changes(self):
        config = json.loads((self.source / "config.json").read_text())
        for change in ["tie_word_embeddings", "quantization_config"]:
            altered = copy.deepcopy(config)
            altered[change] = (
                False if change == "tie_word_embeddings" else {"quant_method": "other"}
            )
            (self.source / "config.json").write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                adapter.author_contract(self.source, self.source_sha)

    def test_disk_admission_precedes_output_creation(self):
        with patch.object(
            adapter.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()
        ):
            with self.assertRaisesRegex(ValueError, "Insufficient output space"):
                adapter.execute(
                    "project",
                    self.source,
                    self.root / "full",
                    self.source_sha,
                    None,
                    self.pins,
                    0,
                )
        self.assertFalse((self.root / "full").exists())

    def test_missing_wrong_axes_marker_or_unknown_quantized_group_refuse(self):
        _, author, linears = adapter.author_contract(self.source, self.source_sha)
        native, _ = adapter.native_tensors(self.native)
        for edit in ["missing", "axis", "marker", "extra_head"]:
            modified = dict(native)
            module = next(iter(sorted(linears)))
            key = "model." + module[len(adapter.PREFIX) :]
            if edit == "missing":
                del modified[key + ".mul1"]
            elif edit == "axis":
                ref = modified[key + ".trellis"]
                metadata = copy.deepcopy(ref.metadata)
                metadata["shape"][:2] = reversed(metadata["shape"][:2])
                if metadata["shape"] == ref.metadata["shape"]:
                    metadata["shape"][0] += 1
                modified[key + ".trellis"] = adapter.TensorReference(
                    ref.path, ref.key, metadata, ref.payload_start
                )
            elif edit == "marker":
                ref = modified[key + ".mul1"]
                marker = self.root / "bad-marker"
                marker.write_bytes(b"\0" * 4)
                modified[key + ".mul1"] = adapter.TensorReference(
                    marker,
                    ref.key,
                    {"dtype": "I32", "shape": [], "data_offsets": [0, 4]},
                    0,
                )
            else:
                modified["lm_head.trellis"] = next(iter(native.values()))
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                adapter.assemble_tensors(author, modified, linears)

    def test_invalid_headers_offsets_duplicates_and_payload_refuse(self):
        path = self.root / "bad"
        headers = [
            ("[]", b""),
            ('{"x":null}', b""),
            ('{"x":{"dtype":[],"shape":[],"data_offsets":[]}}', b""),
            ('{"x":{"dtype":"BF16","shape":[1],"data_offsets":[-1,1]}}', b"XX"),
            ('{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}', b"XX"),
            ('{"x":{"dtype":"BAD","shape":[1],"data_offsets":[0,2]}}', b"XX"),
            ('{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{}}', b"XX"),
            ("{}", b"unaccounted"),
        ]
        for header, data in headers:
            encoded = header.encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
            with self.subTest(header=header), self.assertRaises(ValueError):
                adapter.read_tensors(path)
        path.write_bytes(struct.pack("<Q", 17 * 1024 * 1024))
        with self.assertRaises(ValueError):
            adapter.read_tensors(path)
        path.write_bytes(b"123")
        with self.assertRaises(ValueError):
            adapter.read_tensors(path)

    def test_manifest_builder_cli_and_reserved_metadata_refusal(self):
        output = self.root / "generated.json"
        command = [
            sys.executable,
            "-B",
            str(ROOT / "make_manifest.py"),
            "--source",
            str(self.source),
            "--output",
            str(output),
            "--model-sha256",
            self.source_sha,
            "--metadata",
            "LICENSE",
            "--metadata",
            "config.json",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(output.read_text()), json.loads(self.manifest.read_text())
        )
        result = subprocess.run(
            command + ["--metadata", "../private.json"], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Metadata must be", result.stderr)

    def test_interrupted_output_has_no_success_receipt_and_refuses_replay(self):
        output = self.root / "partial"
        with patch.object(
            adapter, "write_tensors", side_effect=OSError("injected write failure")
        ):
            with self.assertRaises(OSError):
                adapter.execute(
                    "project", self.source, output, self.source_sha, None, self.pins, 0
                )
        self.assertTrue(output.exists())
        self.assertFalse((output / "adapter-receipt.json").exists())
        self.assertEqual(self.command("project", output).returncode, 2)
        self.assertEqual(
            adapter.sha(self.source / "model.safetensors"), self.source_sha
        )

    def test_duplicate_native_shards_refuse(self):
        (self.native / "duplicate.safetensors").write_bytes(
            (self.native / "model.safetensors").read_bytes()
        )
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            adapter.native_tensors(self.native)


if __name__ == "__main__":
    unittest.main()
