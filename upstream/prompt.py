"""The prompt and the readout. A System One decision is a single forward pass that stops at the answer
slot: everything the model sees is built here, and the answer is a softmax over its options' tokens.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

# The system turn: none by default.
SYSTEMS: dict[str, str | None] = {
    "none": None,
    "isolated": (
        "You are a System One decision model. Answer only the current isolated "
        "question from the shared state. Other questions do not exist."
    ),
}
DEFAULT_SYSTEM = "none"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# --------------------------------------------------------------------------- #
# question types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str | None]
    type: str = "choice"


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, Any] | None = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: Sequence[str]
    type: str = "score"


Question = Choice | Noul | Score


# --------------------------------------------------------------------------- #
# verbalizer
# --------------------------------------------------------------------------- #


def option_codes(labels: Sequence[str]) -> list[str]:
    """Native letters when the labels already are letters, else A..Z, else 00..; one rule for every
    cardinality."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    # Where a tokenizer splits digits, "00".."99" and "#i" are two tokens, and two capital letters are
    # often one. Last in the pool, so a tokenizer with digit pairs never reaches it.
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)


@lru_cache(maxsize=4096)
def _aliases_cached(tokenizer_key: int, codes: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    tokenizer = _TOKENIZERS[tokenizer_key]
    used: set[int] = set()
    out: list[tuple[str, int]] = []

    def take(raw: str) -> bool:
        enc = tokenizer.encode(raw, add_special_tokens=False)
        if len(enc) != 1 or enc[0] in used:
            return False
        out.append((raw, enc[0]))
        used.add(enc[0])
        return True

    for code in codes:
        if take(code):
            continue
        if not any(take(raw) for raw in _FALLBACK_POOL):
            raise RuntimeError(f"no single-token alias left for {len(codes)} options")
    return tuple(out)


_TOKENIZERS: dict[int, object] = {}


def aliases(tokenizer, labels: Sequence[str]) -> list[tuple[str, int]]:
    """Assign every label a distinct single-token code: [(code, token_id)].

    Memoised on the codes rather than on the labels: the codes are positional
    unless the labels are already letters, so every option list of one length
    shares an entry.
    """
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    codes = tuple(option_codes(labels))
    return list(_aliases_cached(id(tokenizer), codes))


@lru_cache(maxsize=8192)
def _ids_cached(tokenizer_key: int, texts: tuple[str, ...]) -> tuple[int, ...]:
    tokenizer = _TOKENIZERS[tokenizer_key]
    out, seen = [], set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in seen:
            out.append(enc[0])
            seen.add(enc[0])
    return tuple(out)


def _ids(tokenizer, texts: Sequence[str]) -> list[int]:
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    return list(_ids_cached(id(tokenizer), tuple(texts)))


def as_question(q: Mapping | Question) -> Question:
    """A question in the Decision Index's JSON, `{"type": "noul" | "choice" | "score", "instructions",
    "criteria"}`, as one of the classes above; a class passes through."""
    if not isinstance(q, Mapping):
        return q
    kind = q.get("type", "choice")
    if kind == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    if kind == "score":
        return Score(q["instructions"], list(q["criteria"]))
    return Choice(q["instructions"], q["criteria"])


YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")


def readout_ids(tokenizer, q: Question) -> list[list[int]]:
    """Token ids to score, one group per option, max-pooled: the answer is a softmax over these and
    nothing else."""
    if isinstance(q, Noul):
        yes, no = _ids(tokenizer, YES_FORMS), _ids(tokenizer, NO_FORMS)
        if not yes or not no:
            raise RuntimeError("tokenizer has no single-token yes/no")
        return [yes, no]
    if isinstance(q, Score):
        groups = [_ids(tokenizer, [str(i)]) for i in range(len(q.criteria))]
        if any(not g for g in groups):
            raise RuntimeError(
                f"score with {len(q.criteria)} levels needs single-token digits; "
                "the primitive is defined for 2 to 10"
            )
        return groups
    groups = []
    for code, tid in aliases(tokenizer, list(q.criteria.keys())):
        extra = _ids(tokenizer, [f" {code}"])
        groups.append([tid] + [i for i in extra if i != tid])
    if not groups:
        raise RuntimeError("choice with no options")
    return groups


def readout(tokenizer, q: Question, logz, calibration=None) -> list[float]:
    """Option probabilities from the log-probabilities at the answer slot.

    `logz` is indexed by token id: a vocabulary tensor, or a dict holding at
    least the question's option tokens. Each option scores its best form.
    """
    scores = [max(float(logz[i]) for i in g) for g in readout_ids(tokenizer, q)]
    if calibration is not None:
        scores = calibration.apply(q, scores)
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return [e / sum(exps) for e in exps]


# --------------------------------------------------------------------------- #
# state and question rendering
# --------------------------------------------------------------------------- #


DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-3B"

# How a state is rendered: `json_only`, the default, writes every state as the object it is; `json` keeps
# three shortcuts (`Message:`, `Passage:` / `Asked:`, a lone question's text); `sections` writes nested
# states as labelled blocks.
DEFAULT_STATE_STYLE = "json_only"


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool)) and "\n" not in str(v)


def _sections(obj: Any, path: str, out: list[str]) -> None:  # noqa: C901
    """Flatten a nested state into labelled blocks, keeping real newlines.

    `json.dumps` escapes every newline inside a log line or a record, so a
    multi-line record would arrive as one string of `\n`; this keeps it readable.
    """
    head = f"[{path}]\n" if path else ""
    if isinstance(obj, dict):
        scalars = [(k, v) for k, v in obj.items() if _is_scalar(v)]
        rest = [(k, v) for k, v in obj.items() if not _is_scalar(v)]
        if scalars:
            body = "\n".join(f"{k}: {'' if v is None else v}" for k, v in scalars)
            out.append(f"{head}{body}")
        for k, v in rest:
            _sections(v, f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(_is_scalar(v) for v in obj):
            body = "\n".join(f"- {'' if v is None else v}" for v in obj)
            out.append(f"{head}{body}")
            return
        for i, v in enumerate(obj, start=1):
            _sections(v, f"{path} {i}/{len(obj)}" if path else f"{i}/{len(obj)}", out)
        return
    out.append(f"{head}{'' if obj is None else obj}")


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    out: list[str] = []
    _sections(state, "", out)
    return "\n\n".join(out)


def state_block(state: Any, style: str = DEFAULT_STATE_STYLE) -> str:
    """Flatten a state into the block that precedes QUESTION:.

    Without `_only`, three shapes get a shortcut: a bare utterance becomes
    `Message:`, a passage and a question `Passage:` / `Asked:`, a lone question
    its text. The `_only` styles (the default) render every state as the object
    it is.
    """
    if isinstance(state, dict) and not style.endswith("_only"):
        keys = set(state.keys())
        if keys == {"text"}:
            return f"Message: {state['text']}\n\n"
        if {"passage", "question"} <= keys and len(keys) == 2:
            return f"Passage: {state['passage']}\n\nAsked: {state['question']}\n\n"
        if keys == {"question"}:
            return f"{state['question']}\n\n"
    if style.startswith("json"):
        if isinstance(state, str):
            return f"{state}\n\n"
        return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"
    return f"{render_state(state)}\n\n"


_PLACEHOLDER = re.compile(r"^opt\d+$")


def _option_line(code: str, label: str, desc: str | None, style: str) -> str:
    text = desc or label.replace("_", " ")
    if style == "name_desc" and not _PLACEHOLDER.match(label) and label != text:
        return f"{code} {label}: {text}"
    return f"{code} {text}"


def question_block(tokenizer, q: Question, option_style: str = "desc") -> str:
    if isinstance(q, Choice):
        labels = list(q.criteria.keys())
        codes = aliases(tokenizer, labels)
        lines = "\n".join(
            _option_line(codes[i][0], lab, q.criteria[lab], option_style)
            for i, lab in enumerate(labels)
        )
        return (
            f"{q.instructions}\n\nOptions:\n{lines}\n\n"
            "Reply with the option code only."
        )
    if isinstance(q, Noul):
        extra = ""
        if q.criteria:
            extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}"
        return f"{q.instructions}{extra}\n\nReply with yes or no only."
    if isinstance(q, Score):
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
        return (
            f"{q.instructions}\n\n{legend}\n\n"
            f"Reply with a single digit 0-{len(q.criteria) - 1} only."
        )
    raise TypeError(f"unknown question type {type(q)}")


def prefix_text(
    tokenizer,
    state: Any,
    bos: str = "",
    style: str = DEFAULT_STATE_STYLE,
    system: str = DEFAULT_SYSTEM,
    images: str = "",
) -> str:
    """Everything before the question, shared by all questions on one state: the pictures' markup
    (`images`, as the chat template writes them) at the head of the user turn, then the state. With no
    state (`None`) the question follows the pictures directly."""
    text = SYSTEMS[system]
    turn = "" if text is None else f"{IM_START}system\n{text}{IM_END}\n"
    body = "" if state is None else f"{state_block(state, style)}\nQUESTION:\n"
    return f"{bos}{turn}{IM_START}user\n{images}{body}"


# What sits between the assistant header and the answer slot, per model type: nothing on LFM2-VL, whose
# template opens no reasoning block.
DEFAULT_LEAD = ""
LEADS: dict[str, str] = {}


def default_lead(model_type: str | None) -> str:
    """What a checkpoint's own template writes before a non-thinking answer."""
    return LEADS.get(model_type, DEFAULT_LEAD)


def suffix_text(
    tokenizer, q: Question, lead: str = DEFAULT_LEAD, option_style: str = "desc"
) -> str:
    """The question and the assistant header, up to the answer slot."""
    body = question_block(tokenizer, q, option_style)
    return f"{body}{IM_END}\n{IM_START}assistant\n{lead}"


def render(
    tokenizer,
    state: Any,
    q: Question,
    bos: str = "",
    lead: str = DEFAULT_LEAD,
    style: str = DEFAULT_STATE_STYLE,
    system: str = DEFAULT_SYSTEM,
    option_style: str = "desc",
    images: str = "",
) -> str:
    return prefix_text(tokenizer, state, bos, style, system, images) + suffix_text(
        tokenizer, q, lead, option_style
    )
