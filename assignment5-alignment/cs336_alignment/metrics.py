import regex as re

from typing import Any


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.
    """
    match = re.search(r"The correct answer is\s*([A-D])", model_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if numbers:
        return numbers[-1]
    return None
