"""Unit tests for the cache-friendly message layout built by get_advice.

The layout matters more than any single string: the cacheable prefix
(system + history) must only ever grow, and every per-turn part must sit in
the final user message.
"""
import pytest
pytestmark = pytest.mark.unit

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.prompt_manager import PromptParts
from backend.rag_engine import EnhancedNegotiationRAG


PARTS = PromptParts(
    static_system="META + PERSONA\n",
    negotiation_block="## Negotiation Briefing\n\nTitle: Deal\n\n## Negotiation Memory\n\n- open: price",
    context_block="## Reference Material from Knowledge Base\n\nSource 1: chunk\n\n## Response Format\n\nAnswer directly.",
    user_prompt="What about the equity point?",
)

HISTORY = [
    {"role": "user", "content": "u0"},
    {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
]


def _cache_markers(messages: list[BaseMessage]) -> int:
    count = 0
    for message in messages:
        if isinstance(message.content, list):
            count += sum(1 for block in message.content if isinstance(block, dict) and "cache_control" in block)
    return count


def test_anthropic_layout_and_breakpoints():
    messages = EnhancedNegotiationRAG._build_messages("anthropic", PARTS, HISTORY, "")

    assert isinstance(messages[0], SystemMessage)
    blocks = messages[0].content
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["text"] == PARTS.static_system and "cache_control" in blocks[0]
    assert blocks[1]["text"] == PARTS.negotiation_block and "cache_control" in blocks[1]

    assert [type(m) for m in messages[1:5]] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    # Only the last history message carries a breakpoint.
    assert isinstance(messages[4].content, list) and "cache_control" in messages[4].content[0]
    assert all(isinstance(m.content, str) for m in messages[1:4])

    final = messages[-1]
    assert isinstance(final, HumanMessage) and isinstance(final.content, str)
    assert "Reference Material" in final.content
    assert final.content.endswith(PARTS.user_prompt)
    assert _cache_markers(messages) == 3


def test_anthropic_first_turn_has_no_history_breakpoint():
    messages = EnhancedNegotiationRAG._build_messages("anthropic", PARTS, [], "")
    assert len(messages) == 2
    assert _cache_markers(messages) == 2


def test_anthropic_omits_empty_negotiation_block():
    parts = PARTS._replace(negotiation_block="")
    messages = EnhancedNegotiationRAG._build_messages("anthropic", parts, [], "")
    assert len(messages[0].content) == 1
    assert _cache_markers(messages) == 1


def test_openai_layout_is_plain_strings():
    messages = EnhancedNegotiationRAG._build_messages("openai", PARTS, HISTORY, "")
    assert isinstance(messages[0], SystemMessage) and isinstance(messages[0].content, str)
    assert PARTS.static_system.rstrip() in messages[0].content
    assert "## Negotiation Briefing" in messages[0].content
    assert "Reference Material" not in messages[0].content
    assert all(isinstance(m.content, str) for m in messages)
    assert _cache_markers(messages) == 0
    assert messages[-1].content.endswith(PARTS.user_prompt)


def test_runpod_gets_base_messages_with_string_content():
    messages = EnhancedNegotiationRAG._build_messages("runpod", PARTS, HISTORY, "")
    assert all(isinstance(m, BaseMessage) for m in messages)
    assert all(isinstance(m.content, str) for m in messages)


def test_extra_context_leads_the_final_message():
    messages = EnhancedNegotiationRAG._build_messages("openai", PARTS, [], "[File: brief.pdf]\ncontents")
    final = messages[-1].content
    assert final.startswith("[File: brief.pdf]")
    assert final.index("[File: brief.pdf]") < final.index("Reference Material") < final.index(PARTS.user_prompt)


def test_prefix_grows_by_appending_only():
    """Turn N+1's prefix must equal turn N's prefix plus turn N's exchange."""
    turn_n = EnhancedNegotiationRAG._build_messages("openai", PARTS, HISTORY[:2], "")
    turn_n1 = EnhancedNegotiationRAG._build_messages("openai", PARTS, HISTORY, "")
    prefix_n = [m.content for m in turn_n[:-1]]
    prefix_n1 = [m.content for m in turn_n1[:-1]]
    assert prefix_n1[: len(prefix_n)] == prefix_n


def test_retrieval_query_uses_last_user_turn():
    assert EnhancedNegotiationRAG._build_retrieval_query("q", []) == "q"
    query = EnhancedNegotiationRAG._build_retrieval_query("q", HISTORY)
    assert query == "u1\nq"
    long_history = [{"role": "user", "content": "x" * 5000}, {"role": "assistant", "content": "a"}]
    assert len(EnhancedNegotiationRAG._build_retrieval_query("q", long_history)) <= 1500 + 2


class _Resp:
    def __init__(self, content):
        self.content = content


def test_response_text_joins_text_blocks_and_skips_thinking():
    blocks = [
        {"type": "thinking", "thinking": "private"},
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
    ]
    assert EnhancedNegotiationRAG._response_text(_Resp(blocks)) == "Hello world"
    assert EnhancedNegotiationRAG._response_text(_Resp("plain")) == "plain"
    assert EnhancedNegotiationRAG._response_text("bare") == "bare"
