"""The pieces every LFM hybrid is made of, and the one shape that makes many
questions cost about one forward.

A model runs a batch of right-padded chains `(B, T)`, or a tree: a trunk of P
tokens (the state) and N branches (the questions) that each continue it, packed
back to back on one axis with no padding, `[trunk | branch 0 | branch 1 | ...]`.
Everything that works token by token (norms, projections, the MLP) runs once
over the packed axis. A convolution continues each branch from the trunk's last
inputs. Attention runs by parts: the trunk causally, every branch token over the
trunk in one flash call that reads the trunk's keys once for all questions
(under sliding attention, one fp32 product over the trunk's last window-1 keys),
each branch over its own tokens in one varlen flash call, the two merged by
their log-sum-exp. Nothing is cached, copied or padded.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

Tensor = torch.Tensor


class Tree:
    """A trunk of `p` tokens and branches of `lengths` tokens, packed on one axis."""

    def __init__(self, p: int, lengths: Tensor):
        self.p, self.n = p, len(lengths)
        self.cu = F.pad(lengths.cumsum(0), (1, 0)).int()  # where each branch starts, past the trunk
        self.longest = int(lengths.max())
        self.branch = torch.repeat_interleave(torch.arange(self.n, device=lengths.device), lengths)
        self.offset = torch.arange(len(self.branch), device=lengths.device) - self.cu[self.branch]
        # a branch starts where the trunk ends
        self.positions = torch.cat((torch.arange(p, device=lengths.device), p + self.offset))
        self.leaves = p + self.cu[1:].long() - 1  # each branch's last token
        self._conv = {}

    def conv_order(self, keep: int) -> tuple[Tensor, Tensor]:
        """Gather and keep indices that run one causal conv over all branches,
        each led by the trunk's last `keep` inputs. Indices are into the whole
        input left-padded by `keep` zeros."""
        if keep not in self._conv:
            p, total, dev = self.p, len(self.branch), self.branch.device
            slot = torch.arange(total, device=dev) + keep * (self.branch + 1)
            order = torch.empty(total + self.n * keep, dtype=torch.long, device=dev)
            order[slot] = keep + p + torch.arange(total, device=dev)
            lead = (self.cu[:-1] + keep * torch.arange(self.n, device=dev))[:, None]
            order[lead + torch.arange(keep, device=dev)] = p + torch.arange(keep, device=dev)
            self._conv[keep] = order, slot - keep
        return self._conv[keep]


# --------------------------------------------------------------------------- #
# norms and rotary
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def cos_sin(angles: Tensor) -> tuple[Tensor, Tensor]:
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-split RoPE in fp32; `cos`/`sin` broadcast over the head axis."""
    x32 = x.float()
    a, b = x32.chunk(2, dim=-1)
    return (x32 * cos + torch.cat((-b, a), dim=-1) * sin).type_as(x)


def rope(x: Tensor, pos: Tensor, theta: float, rotated: int) -> Tensor:
    """RoPE over `(B, T, H, D)` at positions `pos` `(T,)`, pairs past `rotated` left as they are."""
    dim = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    inv[rotated:] = 0.0
    cos, sin = cos_sin(pos.float()[:, None] * inv)
    return rotate(x, cos[:, None], sin[:, None])


# --------------------------------------------------------------------------- #
# causal convolution
# --------------------------------------------------------------------------- #


def causal_conv(x: Tensor, weight: Tensor, tree: Tree | None) -> Tensor:
    """Depthwise causal conv over `(B, C, T)`; a tree's branches continue its trunk."""
    keep = weight.shape[-1] - 1
    padded = F.pad(x, (keep, 0))
    if tree is None:
        return F.conv1d(padded, weight, groups=x.shape[1])
    order, kept = tree.conv_order(keep)
    branches = F.conv1d(padded[..., order], weight, groups=x.shape[1])[..., kept]
    if tree.p == 0:
        return branches
    return torch.cat((F.conv1d(padded[..., : keep + tree.p], weight, groups=x.shape[1]), branches), dim=-1)


# --------------------------------------------------------------------------- #
# attention, `(B, H, T, D)`
# --------------------------------------------------------------------------- #


# queries per explicit product off the flash kernels, which bounds its fp32 scores
CHUNK = 1024


def _fused(q: Tensor) -> bool:
    """Whether the flash kernels take these queries: a GPU in half precision, ROCm or CUDA from Ampere on."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return bool(torch.version.hip) or torch.cuda.get_device_capability(q.device)[0] >= 8


def _explicit(q: Tensor, k: Tensor, v: Tensor, allowed: Tensor | None, scale: float) -> tuple[Tensor, Tensor]:
    """Attention with its log-sum-exp, in fp32, for small key sets. GQA by broadcast."""
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1).float(), v.repeat_interleave(groups, dim=1).float()
    scores = (q.float() @ k.transpose(-1, -2)) * scale
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1).nan_to_num() @ v, torch.logsumexp(scores, dim=-1)


def _cat(parts: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    return torch.cat([o for o, _ in parts], dim=2), torch.cat([lse for _, lse in parts], dim=2)


def _over_trunk(q: Tensor, k: Tensor, v: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    """Every branch token over all of the trunk, with its log-sum-exp. Each KV
    head's query group is folded into the query axis, so the trunk's keys are
    read once and never repeated."""
    batch, heads, length, dim = q.shape
    kv = k.shape[1]
    if not _fused(q):
        return _cat([_explicit(part, k, v, None, scale) for part in q.split(CHUNK, dim=2)])
    folded = q.reshape(batch, kv, heads // kv * length, dim)
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(folded, k, v, 0.0, False, False, scale=scale)[:2]
    return out.reshape(batch, heads, length, dim).float(), lse.reshape(batch, heads, length)


def _own(q: Tensor, k: Tensor, v: Tensor, tree: Tree, window: int | None, scale: float) -> tuple[Tensor, Tensor]:
    """Each branch over its own tokens, causally (within `window`), with its log-sum-exp."""
    if not _fused(q) and len(tree.branch) <= CHUNK:  # small enough for one masked product
        same, off = tree.branch[:, None] == tree.branch[None], tree.offset
        allowed = same & (off[None] <= off[:, None])
        if window is not None:
            allowed &= off[None] > off[:, None] - window
        return _explicit(q, k, v, allowed, scale)
    if not _fused(q):  # branch by branch: memory grows with the longest question, not with the batch
        parts = []
        for a, b in zip(tree.cu[:-1].tolist(), tree.cu[1:].tolist()):
            off = torch.arange(b - a, device=q.device)
            allowed = off[None] <= off[:, None]
            if window is not None:
                allowed &= off[None] > off[:, None] - window
            parts.append(_explicit(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], allowed, scale))
        return _cat(parts)
    local = {} if window is None else {"window_size_left": window - 1, "window_size_right": 0}
    out, lse = torch.ops.aten._flash_attention_forward(
        *(t[0].transpose(0, 1) for t in (q, k, v)), tree.cu, tree.cu, tree.longest, tree.longest,
        0.0, True, False, scale=scale, **local)[:2]
    return out.transpose(0, 1)[None].float(), lse[None]


def _merge(a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor]) -> Tensor:
    (oa, la), (ob, lb) = a, b
    m = torch.maximum(la, lb)
    wa, wb = (la - m).exp(), (lb - m).exp()
    return (oa * wa[..., None] + ob * wb[..., None]) / (wa + wb)[..., None]


def _windowed(q: Tensor, k: Tensor, v: Tensor, window: int, scale: float) -> Tensor:
    """Causal attention within `window`, as local blocks: each block of `window`
    queries sees its own block and the one before it."""
    batch, heads, length, dim = q.shape
    blocks = -(-length // window)

    def split(t: Tensor) -> Tensor:  # (B, H, T, D) -> (B, blocks, H, window, D)
        t = F.pad(t, (0, 0, 0, blocks * window - length))
        return t.reshape(batch, t.shape[1], blocks, window, dim).transpose(1, 2)

    def with_previous(t: Tensor) -> Tensor:
        return torch.cat((F.pad(t, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], t), dim=3)

    qb, kb, vb = split(q), with_previous(split(k)), with_previous(split(v))
    qpos = torch.arange(window, device=q.device)[:, None] + window
    kpos = torch.arange(2 * window, device=q.device)[None, :]
    allowed = (kpos <= qpos) & (kpos > qpos - window)
    first = allowed & (kpos >= window)  # block 0 has no block before it
    mask = torch.stack([first] + [allowed] * (blocks - 1))[:, None].repeat(batch, 1, 1, 1)
    out = F.scaled_dot_product_attention(qb.flatten(0, 1), kb.flatten(0, 1), vb.flatten(0, 1),
                                         attn_mask=mask, scale=scale, enable_gqa=True)
    out = out.reshape(batch, blocks, heads, window, dim).transpose(1, 2)
    return out.reshape(batch, heads, blocks * window, dim)[:, :, :length]


def causal_attention(q: Tensor, k: Tensor, v: Tensor, window: int | None, scale: float) -> Tensor:
    """Right-padded chains: causal, within `window` if set."""
    if window is None or q.shape[2] <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    return _windowed(q, k, v, window, scale)


def attention(q: Tensor, k: Tensor, v: Tensor, tree: Tree | None, window: int | None, scale: float) -> Tensor:
    """Chains, or a tree: the trunk causally, each branch over the trunk and over
    its own tokens."""
    if tree is None:
        return causal_attention(q, k, v, window, scale)
    p = tree.p
    qb = q[:, :, p:]
    mine = _own(qb, k[:, :, p:], v[:, :, p:], tree, window, scale)
    if p == 0:
        return mine[0].to(q.dtype)
    trunk = causal_attention(q[:, :, :p], k[:, :, :p], v[:, :, :p], window, scale)
    if window is None:
        seen = _over_trunk(qb, k[:, :, :p], v[:, :, :p], scale)
    else:  # only the trunk's last window-1 keys are in reach of a branch
        tail = min(p, window - 1)
        allowed = torch.arange(p - tail, p, device=q.device)[None] > tree.positions[p:, None] - window
        seen = _explicit(qb, k[:, :, p - tail:p], v[:, :, p - tail:p], allowed, scale)
    return torch.cat((trunk, _merge(seen, mine).to(q.dtype)), dim=2)


# --------------------------------------------------------------------------- #
# the stack
# --------------------------------------------------------------------------- #


class DecoderLayer(GradientCheckpointingLayer):
    """Norm, operator (`conv` or `self_attn`), residual; norm, MLP, residual.
    `sandwich` adds a norm on each branch's output."""

    def __init__(self, kind: str, operator: nn.Module, mlp: nn.Module, dim: int, eps: float, sandwich: bool):
        super().__init__()
        self.operator_name = "conv" if kind == "conv" else "self_attn"
        setattr(self, self.operator_name, operator)
        self.feed_forward = mlp
        self.operator_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.post_operator_norm = RMSNorm(dim, eps) if sandwich else nn.Identity()
        self.post_ffn_norm = RMSNorm(dim, eps) if sandwich else nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, tree: Tree | None = None) -> Tensor:
        h = x + self.post_operator_norm(getattr(self, self.operator_name)(self.operator_norm(x), pos, tree))
        return h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))


class LanguageModel(nn.Module):
    """Embedding, the hybrid stack and the final norm, called like HF's text models.

    Rows are right-padded, so the causal operators never let a pad reach a real
    token and there is no attention mask to apply.
    """

    def __init__(self, cfg, operator: Callable[[str], nn.Module], mlp: Callable[[], nn.Module],
                 eps: float, sandwich: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(kind, operator(kind), mlp(), cfg.hidden_size, eps, sandwich) for kind in cfg.layer_types)
        self.embedding_norm = RMSNorm(cfg.hidden_size, eps)  # the final norm, despite its name

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def run(self, h: Tensor, tree: Tree | None = None) -> Tensor:
        pos = tree.positions if tree is not None else torch.arange(h.shape[1], device=h.device)
        for layer in self.layers:
            h = layer(h, pos, tree)
        return self.embedding_norm(h)

    def forward(self, input_ids=None, inputs_embeds=None, **_) -> BaseModelOutputWithPast:
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return BaseModelOutputWithPast(last_hidden_state=self.run(h))

    def answer(self, trunk: Tensor, questions: Tensor, lengths: Tensor) -> Tensor:
        """Final hidden states at the last token of every question, in one pass.
        `trunk` `(1, P, d)` embeddings; `questions` the ids of all questions back
        to back, `(sum(lengths),)`."""
        tree = Tree(trunk.shape[1], lengths)
        h = self.run(torch.cat((trunk, self.embed_tokens(questions)[None]), dim=1), tree)
        return h[0, tree.leaves]
