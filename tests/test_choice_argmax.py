"""Native choice validation, independent of model and GPU dependencies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from d1_exl3.protocol import validate_response


QUESTION = {"decision": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}}


def response(choice, yes, no):
    return {
        "usage": {"input_tokens": 7, "output_tokens": 0},
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "confidence": max(yes, no),
                "probabilities": {"yes": yes, "no": no},
            }
        },
    }


@pytest.mark.parametrize("choice,yes,no", [("no", 0.8, 0.2), ("yes", 0.2, 0.8)])
def test_choice_cannot_disagree_with_probability_argmax(choice, yes, no):
    with pytest.raises(RuntimeError, match="argmax"):
        validate_response(response(choice, yes, no), QUESTION)


@pytest.mark.parametrize(
    "choice,yes,no",
    [
        ("yes", 0.8, 0.2),
        ("no", 0.2, 0.8),
        ("yes", 0.5, 0.5),
        ("no", 0.5, 0.5),
    ],
)
def test_valid_choices_and_exact_ties_keep_native_response(choice, yes, no):
    value = response(choice, yes, no)
    assert validate_response(value, QUESTION) is value
