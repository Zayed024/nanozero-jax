"""Countdown task: reward + data pipeline.

The scoring functions (`extract_solution`, `validate_equation`, `evaluate_equation`,
`compute_score`) are vendored from rLLM's `rllm/rewards/countdown_reward.py`
(Apache-2.0) so the RL signal is byte-faithful to the recipe. Only the debug
logging / random sampling was removed to make the reward pure and deterministic.

This is the environment for NanoZero: given a target and a set of numbers, the
model must emit an arithmetic equation in <answer>...</answer> that hits the target
using each number exactly once. reward = 1.0 (correct) / 0.1 (valid format, wrong)
/ 0.0 (no answer).
"""

from __future__ import annotations

import re

# --- vendored from rllm/rewards/countdown_reward.py (Apache-2.0) -----------------------


def extract_solution(solution_str: str) -> str | None:
    """Extract the equation from inside the last <answer>...</answer>."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Each available number must be used exactly once (multiset match)."""
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r"\d+", equation_str))
        return numbers_in_eq == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str: str):
    """Safely eval an arithmetic-only equation; None on any error."""
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})  # noqa: S307 - guarded above
    except Exception:
        return None


def compute_score(solution_str: str, ground_truth: dict, *, format_score: float = 0.1, score: float = 1.0) -> float:
    """0.0 no answer | format_score valid-format-but-wrong | score correct."""
    target, numbers = ground_truth["target"], ground_truth["numbers"]
    equation = extract_solution(solution_str)
    if equation is None:
        return 0.0
    if not validate_equation(equation, numbers):
        return format_score
    result = evaluate_equation(equation)
    if result is None:
        return format_score
    return score if abs(result - target) < 1e-5 else format_score


# --- data pipeline ---------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number exactly once. "
    "Show your reasoning in <think> </think> tags, and return the final equation in "
    "<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def build_user_prompt(numbers: list[int], target: int) -> str:
    """The user-turn text. The trainer wraps this with the model's chat template."""
    return _PROMPT_TEMPLATE.format(numbers=list(numbers), target=target)


def reward(completion_str: str, numbers: list[int], target: int) -> float:
    """Convenience wrapper: score a raw completion against a countdown instance."""
    return compute_score(completion_str, {"target": target, "numbers": list(numbers)})


def load_countdown(n: int, *, split: str = "train", repo: str = "Jiayi-Pan/Countdown-Tasks-3to4"):
    """Load `n` countdown instances -> list of {numbers, target, prompt}.

    The HF dataset (used by TinyZero) has columns `nums` (list[int]) and `target` (int).
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        numbers = row.get("nums") or row.get("numbers")
        target = row["target"]
        out.append({"numbers": list(numbers), "target": int(target), "prompt": build_user_prompt(numbers, target)})
    return out
