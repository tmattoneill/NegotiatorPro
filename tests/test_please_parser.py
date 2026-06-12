"""Unit tests for the PLEASE self-assessment parser."""
import pytest
pytestmark = pytest.mark.unit

from backend.please_parser import parse_and_strip

_BLOCK = (
    "```please\n"
    "polite=4 logical=5 empathetic=3 assertive=2 strategic=4 engaging=3\n"
    "```"
)


# ------------------------------------------------------------------
# Valid blocks
# ------------------------------------------------------------------

def test_parses_six_scores_and_total():
    _clean, score = parse_and_strip(f"Some advice.\n\n{_BLOCK}")
    assert score is not None
    assert score["polite"] == 4
    assert score["logical"] == 5
    assert score["empathetic"] == 3
    assert score["assertive"] == 2
    assert score["strategic"] == 4
    assert score["engaging"] == 3
    assert score["total"] == 21


def test_strips_block_from_answer():
    clean, _score = parse_and_strip(f"Some advice.\n\n{_BLOCK}")
    assert "please" not in clean.lower()
    assert clean == "Some advice."


def test_weakest_lists_soft_spots_lowest_first():
    _clean, score = parse_and_strip(f"x\n{_BLOCK}")
    # Assertive=2 is lowest, then the 3s (Empathetic, Engaging) → letters dedup.
    assert score["weakest"] == ["A", "E"]


def test_removes_dangling_heading_above_block():
    answer = f"Advice body.\n\n### PLEASE Self-Assessment\n\n{_BLOCK}"
    clean, score = parse_and_strip(answer)
    assert clean == "Advice body."
    assert score["total"] == 21


def test_equal_scores_have_no_weakest():
    block = (
        "```please\n"
        "polite=5 logical=5 empathetic=5 assertive=5 strategic=5 engaging=5\n"
        "```"
    )
    _clean, score = parse_and_strip(f"x\n{block}")
    assert score["total"] == 30
    assert score["weakest"] == []


def test_bare_line_without_fence_still_parses():
    answer = "Advice.\npolite=3 logical=3 empathetic=3 assertive=4 strategic=2 engaging=3"
    clean, score = parse_and_strip(answer)
    assert score is not None
    assert score["total"] == 18
    assert "polite=" not in clean


# ------------------------------------------------------------------
# No block / malformed → unchanged answer, None
# ------------------------------------------------------------------

def test_no_block_returns_none_and_unchanged():
    answer = "Just the tactical move and the words to use."
    clean, score = parse_and_strip(answer)
    assert score is None
    assert clean == answer


def test_partial_block_is_ignored():
    answer = "Advice.\n```please\npolite=4 logical=5\n```"
    clean, score = parse_and_strip(answer)
    assert score is None
    assert clean == answer


def test_out_of_range_score_is_ignored():
    answer = (
        "Advice.\n```please\n"
        "polite=9 logical=5 empathetic=3 assertive=2 strategic=4 engaging=3\n```"
    )
    _clean, score = parse_and_strip(answer)
    assert score is None


def test_empty_answer():
    clean, score = parse_and_strip("")
    assert clean == ""
    assert score is None
