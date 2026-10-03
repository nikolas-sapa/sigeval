"""Offline example grading and reporter session isolation."""

from examples.test_llm_eval_example import complete
from sigeval import make_judge, run_case
from sigeval import pytest_plugin


def test_example_judge_grades_output_instead_of_criterion():
    judge = make_judge(complete, criterion="answer is about the refund")
    assert judge("Your refund will arrive tomorrow.") is True
    assert judge("I like turtles.") is False


def test_reporter_starts_each_session_empty():
    for _ in range(2):
        pytest_plugin.record(run_case("old", lambda _: True, None, 30, 0.8))
        pytest_plugin.pytest_sessionstart(None)
        assert pytest_plugin._COLLECTED == []
