"""The System One API: named questions over a state in, typed answers out.

    model.system_one(state, {"refund": {"type": "noul", "instructions": "Is this a refund request?"}})
    # {"answers": {"refund": {"type": "noul", "noul": p}}, "usage": {"input_tokens": n, "output_tokens": 0}}

A state is a string, any JSON value, or None with pictures alone. A question is a dict in the Decision
Index's schema (`type`, `instructions`, `criteria`) or one of `prompt`'s classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .prompt import Choice, Noul, Question, as_question


def answer(q: Question, probs: Sequence[float]) -> dict:
    """A noul's P(yes); a choice's pick and its probabilities; a score's expected level."""
    if isinstance(q, Noul):
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if isinstance(q, Choice):
        names = list(q.criteria)
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): text for i, text in enumerate(q.criteria)}}


class SystemOneApi:
    """`system_one` and `system_one_batch`, and the plain probabilities under them, over a model's
    `run(requests)`: each `(state, questions, images)` request's probabilities (in its questions' order)
    and the tokens it read."""

    def run(self, requests: Sequence[tuple[Any, list[Question], Sequence]]) -> list[tuple[list[list[float]], int]]:
        raise NotImplementedError

    def probabilities(self, state: Any, questions: Sequence, images: Sequence | None = None) -> list[list[float]]:
        """Each question's distribution over its options (`yes`, `no` for a noul), in one pass."""
        return self.run([(state, [as_question(q) for q in questions], images or ())])[0][0]

    def probabilities_batch(self, requests: Sequence[tuple]) -> list[list[list[float]]]:
        """`probabilities` for many `(state, questions[, images])` requests, packed as `system_one_batch`."""
        reqs = [(r[0], [as_question(q) for q in r[1]], (r[2] if len(r) > 2 else None) or ()) for r in requests]
        return [probs for probs, _ in self.run(reqs)]

    def system_one(self, state: Any, questions: Mapping[str, Any], images: Sequence | None = None) -> dict:
        """Named questions over one state, and its pictures if any, in one forward pass."""
        return self.system_one_batch([(state, questions, images)])[0]

    def system_one_batch(self, requests: Sequence[tuple]) -> list[dict]:
        """Many `(state, questions)` or `(state, questions, images)` requests. Single-question requests
        are packed together with no padding; a request with several questions reads its state once."""
        named = [(r[0], {n: as_question(q) for n, q in r[1].items()}, (r[2] if len(r) > 2 else None) or ())
                 for r in requests]
        done = self.run([(state, list(qs.values()), images) for state, qs, images in named])
        return [{"answers": {n: answer(q, p) for (n, q), p in zip(qs.items(), probs)},
                 "usage": {"input_tokens": read, "output_tokens": 0}}
                for (_, qs, _), (probs, read) in zip(named, done)]
