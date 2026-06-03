"""Shared helpers for measuring how much real context a persona carries.

Two callers rely on one definition of "what counts as content":
  - the chat briefing builder (chat.py), which must not inject placeholder junk
    like "Role: N/A" into the prompt; and
  - persona create/update validation (models/personas.py, routes/personas.py),
    which requires a minimum amount of descriptive context per persona.
"""
from typing import Optional

# Placeholder strings users type when they have nothing to say. Treated as empty
# everywhere: never injected into the prompt, never counted toward the minimum.
PLACEHOLDER_VALUES = {"n/a", "na", "n.a.", "-", "—", "none", "tbd", "nil", "?"}

# Minimum combined length of a persona's descriptive free-text fields. Forces
# users to give the model enough to personalise advice. The persona name is an
# identifier and is deliberately not counted.
MIN_PERSONA_CHARS = 240


def is_placeholder(value: Optional[str]) -> bool:
    """True if the value is empty or a known placeholder (so it carries no info)."""
    if value is None:
        return True
    v = value.strip()
    return not v or v.lower() in PLACEHOLDER_VALUES


def meaningful_value(value: Optional[str]) -> str:
    """Return the trimmed value, or '' if it is empty or a placeholder."""
    return "" if is_placeholder(value) else value.strip()


def meaningful_len(*values: Optional[str]) -> int:
    """Total character length of the meaningful (non-placeholder) values."""
    return sum(len(meaningful_value(v)) for v in values)
