"""Metadata-only topology contract for the supported D1 revision."""


def check(config, header):
    c = config["text_config"]
    dim = c["hidden_size"]
    inter = c["intermediate_size"]
    vocab = c["vocab_size"]
    head = dim // c["num_attention_heads"]
    kv = c["num_key_value_heads"] * head
    if c["block_auto_adjust_ff_dim"]:
        raise ValueError("Unqualified native FFN adjustment")
    if c["rope_parameters"] != {"rope_theta": 1000000.0, "rope_type": "default"}:
        raise ValueError("Unqualified RoPE projection")
    expected = {"embed_tokens.weight": [vocab, dim], "embedding_norm.weight": [dim]}
    linears = []
    for i, kind in enumerate(c["layer_types"]):
        p = f"layers.{i}."
        expected.update(
            {p + "operator_norm.weight": [dim], p + "ffn_norm.weight": [dim]}
        )
        projections = {
            "feed_forward.w1": [inter, dim],
            "feed_forward.w3": [inter, dim],
            "feed_forward.w2": [dim, inter],
        }
        if kind == "conv":
            projections.update(
                {"conv.in_proj": [3 * dim, dim], "conv.out_proj": [dim, dim]}
            )
            expected[p + "conv.conv.weight"] = [dim, 1, c["conv_L_cache"]]
        elif kind == "full_attention":
            projections.update(
                {
                    "self_attn.q_proj": [dim, dim],
                    "self_attn.k_proj": [kv, dim],
                    "self_attn.v_proj": [kv, dim],
                    "self_attn.out_proj": [dim, dim],
                }
            )
            expected[p + "self_attn.q_layernorm.weight"] = [head]
            expected[p + "self_attn.k_layernorm.weight"] = [head]
        else:
            raise ValueError("Unknown layer type")
        for key, shape in projections.items():
            expected[p + key + ".weight"] = shape
            linears.append("model.language_model." + p + key)
    prefix = "model.language_model."
    actual = {k[len(prefix) :]: v for k, v in header.items() if k.startswith(prefix)}
    if set(actual) != set(expected):
        raise ValueError("Missing/extra language tensor keys")
    for key, shape in expected.items():
        if actual[key]["shape"] != shape or actual[key]["dtype"] != "BF16":
            raise ValueError("Shape/dtype mismatch: " + key)
    if not c["tie_word_embeddings"] or "lm_head.weight" in header:
        raise ValueError("Unexpected head/tie contract")
    # Native converter tied head is emitted under lm_head; HF plugin removes tied-key bookkeeping for quantized head.
    return {
        "text_tensors": len(expected),
        "quantizable_author_linears": len(linears),
        "project_text_prefix": {"from": "model.language_model.", "to": "model."},
        "restore_quant_prefix": {"from": "model.", "to": "model.language_model."},
        "head_source": "model.language_model.embed_tokens.weight",
        "head_export": "lm_head",
        "auxiliary_tensors_preserve_bf16": len(header)
        - len(actual)
        - ("__metadata__" in header),
        "shape_contract": "PASS",
        "runtime_api_equivalence": "NOT_RUN",
        "full_context_fit": "NOT_ADMITTED",
    }
