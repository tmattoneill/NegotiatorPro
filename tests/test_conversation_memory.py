"""Unit tests for the per-negotiation conversation memory helpers."""
import pytest
pytestmark = pytest.mark.unit

from backend.conversation_memory import (
    build_memory_block,
    estimate_tokens,
    format_dropped_turns,
    plan_memory_fold,
    refresh_memory,
    select_history_window,
    window_turn_count,
)


class _Resp:
    def __init__(self, content):
        self.content = content


class _LLM:
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._error:
            raise self._error
        return _Resp(self._content)


def _rows(n_turns: int, chars_per_message: int = 40) -> list[dict]:
    """n_turns of (user, assistant) rows, chronological, with ids."""
    rows = []
    for turn in range(n_turns):
        rows.append({"id": 2 * turn + 1, "role": "user", "content": f"u{turn} " + "x" * chars_per_message})
        rows.append({"id": 2 * turn + 2, "role": "assistant", "content": f"a{turn} " + "y" * chars_per_message})
    return rows


# ------------------------------------------------------------------
# select_history_window
# ------------------------------------------------------------------

def test_window_empty_input():
    assert select_history_window([], 1000, 2, 10) == []


def test_window_keeps_whole_turns_oldest_first():
    window = select_history_window(_rows(3), 100_000, 4, 12)
    assert [m["role"] for m in window] == ["user", "assistant"] * 3
    assert window[0]["content"].startswith("u0")
    assert window[-1]["content"].startswith("a2")


def test_window_steps_rather_than_slides():
    # min 2 / max 10 -> step 8. 12 turns: keep 2 + (10 % 8) = 4 newest.
    window = select_history_window(_rows(12), 100_000, 2, 10)
    assert len(window) == 8
    assert window[0]["content"].startswith("u8")


def test_window_turn_count_is_stable_between_steps():
    # min 4 / max 12 -> step 8. The window start (total - keep) only moves
    # every 8 turns, so the cached prefix survives a run of turns.
    starts = {n: n - window_turn_count(n, 4, 12) for n in range(1, 30)}
    assert [window_turn_count(n, 4, 12) for n in range(1, 13)] == list(range(1, 12)) + [4]
    assert all(starts[n] == 0 for n in range(1, 12))
    assert all(starts[n] == 8 for n in range(12, 20))
    assert all(starts[n] == 16 for n in range(20, 28))
    assert max(window_turn_count(n, 4, 12) for n in range(1, 100)) == 11


def test_window_uses_total_turns_for_the_grid():
    # Only the newest 10 turns are fetched, but the conversation has 20.
    rows = _rows(10)
    window = select_history_window(rows, 100_000, 4, 12, total_turns=20)
    assert len(window) == 2 * window_turn_count(20, 4, 12) == 8


def test_window_budget_trims_oldest_but_keeps_min_turns():
    # Each turn costs ~ (44+44)//4 = 22 tokens. Budget of 50 fits two turns.
    window = select_history_window(_rows(6), 50, 2, 12)
    assert len(window) == 4
    assert window[0]["content"].startswith("u4")

    # Even a tiny budget keeps min_turns.
    window = select_history_window(_rows(6), 1, 2, 12)
    assert len(window) == 4


def test_window_drops_trailing_orphan_user_row():
    rows = _rows(2) + [{"id": 99, "role": "user", "content": "unanswered"}]
    window = select_history_window(rows, 100_000, 4, 12)
    assert len(window) == 4
    assert all(m["content"] != "unanswered" for m in window)


def test_window_skips_system_rows():
    rows = _rows(1) + [{"id": 50, "role": "system", "content": "note"}] + _rows(1)
    window = select_history_window(rows, 100_000, 4, 12)
    assert [m["role"] for m in window] == ["user", "assistant", "user", "assistant"]


# ------------------------------------------------------------------
# plan_memory_fold
# ------------------------------------------------------------------

def test_fold_first_turn_is_none():
    # 2 messages total, both still in the window.
    assert plan_memory_fold(total=2, covered=0, window_rows=2) is None


def test_fold_nothing_new_since_last_refresh():
    assert plan_memory_fold(total=30, covered=10, window_rows=20) is None


def test_fold_returns_uncovered_range():
    # 26 messages, window holds 20, memory covers 4 -> fold rows 4..6
    assert plan_memory_fold(total=26, covered=4, window_rows=20) == (4, 2)


def test_fold_from_scratch():
    assert plan_memory_fold(total=24, covered=0, window_rows=20) == (0, 4)


# ------------------------------------------------------------------
# refresh_memory
# ------------------------------------------------------------------

def test_refresh_returns_summary_and_sends_context():
    llm = _LLM(content="## State of play\n- early days")
    out = refresh_memory("old memory", "User: hi\n\nAdvisor: hello", "## Briefing\nTitle: Deal", llm)
    assert out == "## State of play\n- early days"
    system, user = llm.calls[0]
    assert system["role"] == "system"
    assert "old memory" in user["content"]
    assert "Advisor: hello" in user["content"]
    assert "Title: Deal" in user["content"]


def test_refresh_strips_code_fence():
    llm = _LLM(content="```markdown\n## State of play\n- x\n```")
    assert refresh_memory("", "t", "", llm) == "## State of play\n- x"


def test_refresh_returns_none_when_llm_raises():
    assert refresh_memory("", "t", "", _LLM(error=RuntimeError("boom"))) is None


def test_refresh_returns_none_on_empty_reply():
    assert refresh_memory("", "t", "", _LLM(content="   ")) is None


def test_refresh_truncates_to_max_chars():
    out = refresh_memory("", "t", "", _LLM(content="z" * 500), max_chars=100)
    assert out is not None and len(out) <= 100


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------

def test_build_memory_block():
    assert build_memory_block(None) == ""
    assert build_memory_block({}) == ""
    assert build_memory_block({"summary_md": "  "}) == ""
    assert build_memory_block({"summary_md": "- x"}) == "## Negotiation Memory\n\n- x"


def test_format_dropped_turns():
    text = format_dropped_turns([
        {"role": "user", "content": " hi "},
        {"role": "assistant", "content": "hello"},
    ])
    assert text == "User: hi\n\nAdvisor: hello"


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 400) == 100
