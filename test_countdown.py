"""Unit tests for the vendored Countdown reward + prompt builder (pure, no network)."""

from countdown import build_user_prompt, compute_score, extract_solution, reward, validate_equation


def _gt(target, numbers):
    return {"target": target, "numbers": numbers}


class TestExtractSolution:
    def test_last_answer_tag(self):
        assert extract_solution("<answer>1+1</answer> junk <answer>4 * 6</answer>") == "4 * 6"

    def test_strips_assistant_prefix(self):
        assert extract_solution("User: hi\nAssistant: <answer>2+2</answer>") == "2+2"

    def test_strips_qwen_marker(self):
        assert extract_solution("...<|im_start|>assistant\n<answer>3+3</answer>") == "3+3"

    def test_no_answer(self):
        assert extract_solution("just some reasoning, no tags") is None


class TestValidateEquation:
    def test_exact_multiset(self):
        assert validate_equation("4 * 6", [6, 4]) is True

    def test_reused_or_extra_number(self):
        assert validate_equation("4 * 6 * 2", [4, 6]) is False

    def test_missing_number(self):
        assert validate_equation("4 + 6", [4, 6, 2]) is False


class TestComputeScore:
    def test_correct(self):
        assert compute_score("<answer>4 * 6</answer>", _gt(24, [4, 6])) == 1.0

    def test_correct_with_parens(self):
        assert compute_score("<answer>(1 + 2 + 3) * 4</answer>", _gt(24, [1, 2, 3, 4])) == 1.0

    def test_valid_format_wrong_answer(self):
        assert compute_score("<answer>4 + 6</answer>", _gt(24, [4, 6])) == 0.1

    def test_reused_number_gets_format_score(self):
        assert compute_score("<answer>4 * 6 * 2</answer>", _gt(24, [4, 6])) == 0.1

    def test_no_answer_zero(self):
        assert compute_score("I think it's 24 but forgot the tags", _gt(24, [4, 6])) == 0.0

    def test_division(self):
        assert compute_score("<answer>12 / 2</answer>", _gt(6, [12, 2])) == 1.0

    def test_reward_wrapper(self):
        assert reward("<answer>4 * 6</answer>", [4, 6], 24) == 1.0
        assert reward("<answer>4 + 6</answer>", [4, 6], 24) == 0.1


class TestPrompt:
    def test_contains_numbers_target_and_format(self):
        p = build_user_prompt([3, 7, 11], 28)
        assert "[3, 7, 11]" in p
        assert "28" in p
        assert "<answer>" in p
