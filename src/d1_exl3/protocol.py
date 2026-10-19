"""Validate native typed answer contracts without importing model dependencies."""

import math
from collections.abc import Mapping


def validate_questions(questions):
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Named questions must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in questions):
        raise ValueError("Every question must have a nonempty string name")


def validate_response(response, questions):
    validate_questions(questions)
    if not isinstance(response, Mapping) or set(response.get("answers", {})) != set(
        questions
    ):
        raise RuntimeError("D1 answer names differ from requested questions")
    usage = response.get("usage", {})
    if (
        usage.get("output_tokens") != 0
        or not isinstance(usage.get("input_tokens"), int)
        or usage["input_tokens"] <= 0
    ):
        raise RuntimeError(
            "D1 decision usage must report positive input and zero generated tokens"
        )
    for name, question in questions.items():
        answer = response["answers"][name]
        kind = question.get("type") if isinstance(question, Mapping) else question.type
        if answer.get("type") != kind:
            raise RuntimeError("D1 answer type differs from requested type")
        if kind == "noul":
            values = [answer["noul"], 1 - answer["noul"]]
        else:
            criteria = (
                question.get("criteria")
                if isinstance(question, Mapping)
                else question.criteria
            )
            expected = (
                set(criteria)
                if kind == "choice"
                else {str(i) for i in range(len(criteria))}
            )
            probabilities = answer["probabilities"]
            if set(probabilities) != expected:
                raise RuntimeError(
                    "D1 probability labels differ from requested criteria"
                )
            values = list(probabilities.values())
            if (
                not math.isfinite(answer["confidence"])
                or not 0 <= answer["confidence"] <= 1
            ):
                raise RuntimeError("Invalid typed confidence")
            if (
                abs(sum(values) - 1) > 1e-5
                or abs(answer["confidence"] - max(values)) > 1e-6
            ):
                raise RuntimeError("D1 probabilities/confidence are inconsistent")
            if kind == "choice" and answer["choice"] not in expected:
                raise RuntimeError("D1 returned an unknown choice")
            if kind == "choice" and probabilities[answer["choice"]] != max(values):
                raise RuntimeError("D1 choice differs from its probability argmax")
            if kind == "score" and not 0 <= answer["score"] <= len(criteria) - 1:
                raise RuntimeError("D1 score is outside its requested ordinal range")
            if kind == "score" and (
                abs(answer["score"] - sum(int(i) * p for i, p in probabilities.items()))
                > 1e-5
                or answer.get("legend") != {str(i): v for i, v in enumerate(criteria)}
            ):
                raise RuntimeError(
                    "D1 score expectation/legend differs from native semantics"
                )
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        ):
            raise RuntimeError("D1 returned invalid probabilities")
    return response


def validate_probability_rows(rows, questions):
    if len(rows) != len(questions):
        raise RuntimeError("Probability question cardinality changed")
    for row, question in zip(rows, questions, strict=True):
        kind = question.get("type") if isinstance(question, Mapping) else question.type
        criteria = (
            question.get("criteria")
            if isinstance(question, Mapping)
            else question.criteria
        )
        expected = 2 if kind == "noul" else len(criteria)
        if (
            len(row) != expected
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row)
            or abs(sum(row) - 1) > 1e-5
        ):
            raise RuntimeError("Native probability distribution changed")
