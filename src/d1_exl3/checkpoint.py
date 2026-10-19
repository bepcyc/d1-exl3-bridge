"""Bounded metadata inspection before importing an accelerator runtime."""

import hashlib
import json
import struct


def expected_linears(config):
    prefix = "model.language_model.layers."
    result = set()
    for index, kind in enumerate(config["text_config"]["layer_types"]):
        names = ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
        if kind == "conv":
            names += ["conv.in_proj", "conv.out_proj"]
        elif kind == "full_attention":
            names += [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
            ]
        else:
            raise ValueError("Unknown D1 layer topology")
        result.update(prefix + str(index) + "." + name for name in names)
    if len(result) != 166:
        raise ValueError("Unsupported D1 language-body topology")
    return frozenset(result)


def inspect_checkpoint(config):
    metadata = json.loads((config.model / "config.json").read_text())
    quant = metadata.get("quantization_config", {})
    if (
        metadata.get("auto_map", {}).get("AutoModel") != "modeling_d1.D1Model"
        or quant.get("quant_method") != "exl3"
        or quant.get("bits") != 8
        or quant.get("codebook") != "mul1"
        or quant.get("version") != "1.6.2"
        or not metadata.get("tie_word_embeddings")
        or not metadata["text_config"].get("tie_word_embeddings")
    ):
        raise ValueError("Expected the supported tied-head D1 q8/mul1 EXL3 checkpoint")
    path = config.model / "model.safetensors"
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated checkpoint header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= 2 * 1024**2:
            raise ValueError("Checkpoint header exceeds metadata bound")
        header = json.loads(stream.read(length))
    expected = metadata["text_config"]
    if (
        config.maximum_path_tokens is not None
        and config.maximum_path_tokens > expected["max_position_embeddings"]
    ):
        raise ValueError("Requested path capacity exceeds the author model context")
    embedding = header.get("model.language_model.embed_tokens.weight", {})
    if (
        embedding.get("dtype") != "BF16"
        or embedding.get("shape") != [expected["vocab_size"], expected["hidden_size"]]
        or "lm_head.weight" in header
    ):
        raise ValueError("Trained embedding/tied-head checkpoint provenance differs")
    if config.expected_model_sha256:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != config.expected_model_sha256:
            raise ValueError("Checkpoint SHA256 differs from the pinned model")
    return metadata, expected_linears(metadata)
