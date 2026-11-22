"""Tests for the three-layer mode-aware PromptManager."""

from pathlib import Path

import pytest
pytestmark = pytest.mark.unit

from backend.prompt_manager import PromptManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """
    Build a PromptManager rooted in tmp_path, but with the persona YAMLs and
    meta.md symlinked from the real project so we test against canonical content.
    """
    project_prompts = Path(__file__).parent.parent / "prompts"
    if not (project_prompts / "amfonica-meta.md").exists():
        pytest.skip("Project prompts/ not populated")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for name in ("amfonica-meta.md", "sales-mentor.yaml", "negotiation-strategist.yaml"):
        (prompts_dir / name).write_text((project_prompts / name).read_text())

    config_dir = tmp_path / ".config"
    config_dir.mkdir()

    return PromptManager(prompts_dir=str(prompts_dir), config_dir=str(config_dir))


def test_auto_mode_contains_meta_but_no_persona(manager):
    system, _ = manager.get_prompts_for_chat(
        question="anything", context="some context", mode="auto"
    )
    assert "Amfonica" in system, "meta must be present"
    assert "Sales Mentor" not in system, "sales persona must NOT appear in auto"
    assert "Negotiation Strategist" not in system, "negotiation persona must NOT appear in auto"


def test_sales_mode_includes_meta_plus_sales_persona(manager):
    system, _ = manager.get_prompts_for_chat(
        question="anything", context="", mode="sales"
    )
    assert "Amfonica" in system
    assert "Sales Mentor" in system
    assert "Negotiation Strategist" not in system


def test_negotiation_mode_includes_meta_plus_negotiation_persona(manager):
    system, _ = manager.get_prompts_for_chat(
        question="anything", context="", mode="negotiation"
    )
    assert "Amfonica" in system
    assert "Negotiation Strategist" in system
    assert "Sales Mentor" not in system


def test_each_mode_produces_distinct_system_prompts(manager):
    sales, _ = manager.get_prompts_for_chat("q", "", mode="sales")
    nego, _ = manager.get_prompts_for_chat("q", "", mode="negotiation")
    auto, _ = manager.get_prompts_for_chat("q", "", mode="auto")
    assert sales != nego
    assert sales != auto
    assert nego != auto


def test_context_is_injected_into_system_prompt(manager):
    ctx = "SENTINEL_CONTEXT_PHRASE_123"
    system, _ = manager.get_prompts_for_chat(
        question="q", context=ctx, mode="sales"
    )
    assert ctx in system
    assert "Reference Material from Knowledge Base" in system


def test_empty_context_does_not_emit_reference_header(manager):
    system, _ = manager.get_prompts_for_chat(
        question="q", context="", mode="sales"
    )
    assert "Reference Material from Knowledge Base" not in system


def test_unknown_mode_falls_back_to_auto(manager, caplog):
    system_unknown, _ = manager.get_prompts_for_chat(
        question="q", context="", mode="bogus-mode"
    )
    system_auto, _ = manager.get_prompts_for_chat(
        question="q", context="", mode="auto"
    )
    assert system_unknown == system_auto


def test_user_prompt_substitutes_question(manager):
    _, user = manager.get_prompts_for_chat(
        question="What is the BATNA here?", context="", mode="auto"
    )
    assert "What is the BATNA here?" in user


def test_test_prompt_detection_unchanged(manager):
    assert manager.is_test_prompt("hello")
    assert manager.is_test_prompt("ping")
    assert manager.is_test_prompt("test test test")
    assert not manager.is_test_prompt(
        "I have a real negotiation question with substantial content that wouldn't be a test"
    )


def test_test_system_prompt_does_not_load_personae(manager):
    """The test path should bypass mode-aware logic."""
    test_sys = manager.get_test_system_prompt()
    assert "Amfonica" in test_sys
    assert "Sales Mentor" not in test_sys
    assert "Negotiation Strategist" not in test_sys


def test_update_system_prompt_writes_meta_file(manager, tmp_path):
    new_meta = "# Overridden Meta\nYou are different now.\n"
    manager.update_system_prompt(new_meta)
    system, _ = manager.get_prompts_for_chat(question="q", context="", mode="auto")
    assert "Overridden Meta" in system
    # And the file on disk has it
    assert "Overridden Meta" in manager.meta_file.read_text()


def test_get_raw_prompts_returns_meta_under_system_key(manager):
    raw = manager.get_raw_prompts()
    assert "system" in raw
    assert "user" in raw
    assert "Amfonica" in raw["system"]


def test_validate_prompts_reports_loaded_state(manager):
    v = manager.validate_prompts()
    assert v["meta_present"]
    assert v["sales_persona_loaded"]
    assert v["negotiation_persona_loaded"]
    assert v["user_template_has_question"]


# ------------------------------------------------------------------
# get_prompt_parts: negotiation block, follow-up handling
# ------------------------------------------------------------------

from backend.intent_classifier import FOLLOWUP_NOTE, USER_TEMPLATES
from backend.prompt_manager import NEGOTIATION_BLOCK_MAX_CHARS


def test_prompt_parts_negotiation_block_holds_briefing_and_memory_only(manager):
    parts = manager.get_prompt_parts(
        "q", "REF CHUNK", mode="negotiation", intent="GENERAL",
        briefing="## Negotiation Briefing\n\nTitle: Deal", memory="## Negotiation Memory\n\n- open: price",
    )
    assert parts.negotiation_block.startswith("## Negotiation Briefing")
    assert "## Negotiation Memory" in parts.negotiation_block
    assert "REF CHUNK" not in parts.negotiation_block
    assert "Deal" not in parts.static_system
    assert "REF CHUNK" in parts.context_block


def test_prompt_parts_empty_briefing_and_memory_gives_empty_block(manager):
    parts = manager.get_prompt_parts("q", "", mode="negotiation")
    assert parts.negotiation_block == ""


def test_prompt_parts_followup_note_only_with_history(manager):
    cold = manager.get_prompt_parts("q", "ctx", mode="negotiation", intent="GENERAL", has_history=False)
    warm = manager.get_prompt_parts("q", "ctx", mode="negotiation", intent="GENERAL", has_history=True)
    assert FOLLOWUP_NOTE not in cold.context_block
    assert FOLLOWUP_NOTE in warm.context_block
    # Format instructions still follow the note so the PLEASE trailer survives.
    assert "```please" in warm.context_block


def test_prompt_parts_general_followup_sends_bare_question(manager):
    question = "What about the equity point?"
    cold = manager.get_prompt_parts(question, "", mode="negotiation", intent="GENERAL", has_history=False)
    warm = manager.get_prompt_parts(question, "", mode="negotiation", intent="GENERAL", has_history=True)
    assert cold.user_prompt == USER_TEMPLATES["GENERAL"].format(question=question)
    assert warm.user_prompt == question


def test_prompt_parts_non_general_followup_keeps_template(manager):
    question = "They countered at 80k, what do I say?"
    warm = manager.get_prompt_parts(question, "", mode="negotiation", intent="TACTICAL", has_history=True)
    assert warm.user_prompt == USER_TEMPLATES["TACTICAL"].format(question=question)


def test_prompt_parts_static_system_is_history_independent(manager):
    cold = manager.get_prompt_parts("q", "ctx", mode="negotiation", has_history=False, briefing="b")
    warm = manager.get_prompt_parts("q", "ctx", mode="negotiation", has_history=True, briefing="b", memory="m")
    assert cold.static_system == warm.static_system


def test_prompt_parts_oversized_memory_is_trimmed(manager, caplog):
    briefing = "B" * 1000
    memory = "M" * (NEGOTIATION_BLOCK_MAX_CHARS + 500)
    parts = manager.get_prompt_parts("q", "", mode="negotiation", briefing=briefing, memory=memory)
    assert len(parts.negotiation_block) <= NEGOTIATION_BLOCK_MAX_CHARS + 2
    assert parts.negotiation_block.startswith(briefing)
    assert "oversized" in caplog.text
