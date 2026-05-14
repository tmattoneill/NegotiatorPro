"""Tests for the YAML → Markdown renderer used to build persona system prompts."""
from pathlib import Path

import pytest
import yaml

from backend.prompt_renderer import render_yaml_to_markdown


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def test_empty_dict_returns_empty_string():
    assert render_yaml_to_markdown({}) == ""


def test_none_input_returns_empty_string():
    assert render_yaml_to_markdown(None) == ""


def test_flat_dict_renders_each_key_as_header():
    data = {"identity": "You are a helper.", "style": "Be concise."}
    out = render_yaml_to_markdown(data)
    assert "# Identity" in out
    assert "# Style" in out
    assert "You are a helper." in out
    assert "Be concise." in out


def test_single_top_level_key_becomes_document_title():
    data = {"sales_mentor": {"role": "Mentor", "style": "Direct"}}
    out = render_yaml_to_markdown(data)
    # Title from the wrapper
    assert "# Sales Mentor" in out
    # Children rendered one heading level in
    assert "## Role" in out
    assert "## Style" in out


def test_list_of_strings_becomes_bullets():
    data = {"principles": ["Be honest", "Be clear", "Be useful"]}
    out = render_yaml_to_markdown(data)
    assert "- Be honest" in out
    assert "- Be clear" in out
    assert "- Be useful" in out


def test_list_of_dicts_uses_title_key_when_present():
    data = {
        "objections": [
            {"objection": "Too expensive", "response": "Reframe value"},
            {"objection": "Need to think", "response": "Surface concerns"},
        ]
    }
    out = render_yaml_to_markdown(data)
    assert "- **Too expensive**" in out
    assert "- **Need to think**" in out
    assert "*Response*: Reframe value" in out


def test_nested_dicts_increase_header_depth():
    data = {
        "section_a": {
            "subsection_one": {"detail": "value"},
        }
    }
    out = render_yaml_to_markdown(data)
    assert "# Section A" in out
    assert "## Subsection One" in out
    assert "### Detail" in out


def test_humanises_snake_case_keys_to_title_case():
    data = {"tactical_empathy_patterns": "go"}
    out = render_yaml_to_markdown(data)
    assert "Tactical Empathy Patterns" in out


def test_collapses_multiple_blank_lines():
    data = {"a": "one", "b": "two"}
    out = render_yaml_to_markdown(data)
    assert "\n\n\n" not in out


@pytest.fixture(scope="module")
def sales_yaml():
    p = PROMPTS_DIR / "sales-mentor.yaml"
    if not p.exists():
        pytest.skip(f"Persona YAML missing: {p}")
    return yaml.safe_load(p.read_text())


@pytest.fixture(scope="module")
def negotiation_yaml():
    p = PROMPTS_DIR / "negotiation-strategist.yaml"
    if not p.exists():
        pytest.skip(f"Persona YAML missing: {p}")
    return yaml.safe_load(p.read_text())


def test_sales_yaml_renders_to_non_trivial_markdown(sales_yaml):
    out = render_yaml_to_markdown(sales_yaml)
    assert len(out) > 2000, "rendered sales prompt should be substantial"
    # Spot-check distinctive sales content
    assert "Sales Mentor" in out
    assert "Buyer Psychology" in out or "buyer_psychology" not in out  # may have been humanised
    # Some bullet content should appear
    assert "- " in out


def test_negotiation_yaml_renders_to_non_trivial_markdown(negotiation_yaml):
    out = render_yaml_to_markdown(negotiation_yaml)
    assert len(out) > 2000, "rendered negotiation prompt should be substantial"
    assert "Negotiation Strategist" in out
    # Voss tactics should show up somewhere in the negotiation YAML
    assert "Calibrated Questions" in out or "Tactical Empathy" in out


def test_renders_are_distinct_per_persona(sales_yaml, negotiation_yaml):
    sales_out = render_yaml_to_markdown(sales_yaml)
    neg_out = render_yaml_to_markdown(negotiation_yaml)
    assert sales_out != neg_out
    # Persona headers should be in only their own rendering
    assert "Sales Mentor" in sales_out and "Sales Mentor" not in neg_out
    assert "Negotiation Strategist" in neg_out and "Negotiation Strategist" not in sales_out
