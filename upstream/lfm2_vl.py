"""LFM2-VL and LFM2 on the hybrid stack.

HF's classes stay the frame (the SigLIP2 tower, the projector, the image
features, loading and saving); only the language model is replaced, with HF's
module names, so checkpoints load and save unchanged. A state with any number
of questions runs as one `hybrid.Tree`, with the same outputs as HF's own
implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Lfm2Config, Lfm2VlConfig
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM as HfLfm2
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration as HfLfm2Vl

from .hybrid import LanguageModel, RMSNorm, Tree, attention, causal_conv, rope


class Attention(nn.Module):
    """GQA with per-head q/k norms and RoPE over the whole head."""

    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.theta = cfg.rope_parameters["rope_theta"]
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, pos: torch.Tensor, tree: Tree | None) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).view(batch, length, self.heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim))
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim)
        q, k = rope(q, pos, self.theta, self.head_dim // 2), rope(k, pos, self.theta, self.head_dim // 2)
        y = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), tree, None, self.scale)
        return self.out_proj(y.transpose(1, 2).flatten(2))


class ShortConv(nn.Module):
    """`out(C * conv(B * x))`; HF's `nn.Conv1d` holds the taps, `causal_conv` runs them."""

    def __init__(self, cfg):
        super().__init__()
        dim, taps = cfg.hidden_size, cfg.conv_L_cache
        self.conv = nn.Conv1d(dim, dim, taps, groups=dim, bias=cfg.conv_bias, padding=taps - 1)
        self.in_proj = nn.Linear(dim, 3 * dim, bias=cfg.conv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=cfg.conv_bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, tree: Tree | None) -> torch.Tensor:
        b, c, u = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
        z = causal_conv(b * u, self.conv.weight, tree)
        if self.conv.bias is not None:
            z = z + self.conv.bias[:, None]
        return self.out_proj((c * z).transpose(1, 2))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.intermediate_size
        if cfg.block_auto_adjust_ff_dim:  # HF's sizing rule, kept exactly
            hidden = int(2 * hidden / 3)
            if cfg.block_ffn_dim_multiplier is not None:
                hidden = int(cfg.block_ffn_dim_multiplier * hidden)
                hidden = cfg.block_multiple_of * ((hidden + cfg.block_multiple_of - 1) // cfg.block_multiple_of)
        self.w1 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def language_model(cfg) -> LanguageModel:
    return LanguageModel(cfg, lambda kind: ShortConv(cfg) if kind == "conv" else Attention(cfg),
                         lambda: MLP(cfg), cfg.norm_eps)


class Lfm2VlForConditionalGeneration(HfLfm2Vl):
    """HF's LFM2-VL with the hybrid language model, under HF's name so a saved
    checkpoint reads as HF's."""

    def __init__(self, config: Lfm2VlConfig):
        super().__init__(config)
        self.model.language_model = language_model(config.text_config)

    def answer(self, trunk: torch.Tensor, questions: torch.Tensor, lengths: torch.Tensor, **vision) -> torch.Tensor:
        """Logits at the last token of every question, the state (and its image)
        read once: `trunk` `(1, P)` ids, `questions` their ids back to back."""
        embeds = self.get_input_embeddings()(trunk)
        if vision:
            features = torch.cat(self.model.get_image_features(**vision).pooler_output).to(embeds.dtype)
            slots = self.model.get_placeholder_mask(trunk, inputs_embeds=embeds, image_features=features)
            embeds = embeds.masked_scatter(slots, features)
        return self.lm_head(self.model.language_model.answer(embeds, questions, lengths))


class Lfm2ForCausalLM(HfLfm2):
    """HF's LFM2 with the hybrid language model, under HF's name."""

    def __init__(self, config: Lfm2Config):
        super().__init__(config)
        self.model = language_model(config)

    def answer(self, trunk: torch.Tensor, questions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model.answer(self.model.embed_tokens(trunk), questions, lengths))

