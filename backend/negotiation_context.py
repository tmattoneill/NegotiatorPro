"""
Infer the negotiation-level context for the stats gutter: leverage, parties,
and vitals.

Parties come straight from the negotiation's personas — no model needed.
Leverage and vitals need judgement, so they come from a single structured LLM
call over the conversation transcript. The result is cached on the negotiation
(see the context endpoint) and only recomputed when new messages land.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .persona_text import is_placeholder

logger = logging.getLogger(__name__)

_BATNA_VALUES = {"strong", "moderate", "weak"}
_TIME_VALUES = {"urgent", "tight", "loose"}


# ------------------------------------------------------------------
# Parties — formatted from persona data, no inference
# ------------------------------------------------------------------

def _initial(name: Optional[str]) -> str:
    name = (name or "").strip()
    return name[:1].upper() if name else "?"


def _first_present(*values: Optional[str]) -> str:
    """First meaningful value, skipping blanks and placeholders like "N/A"."""
    for value in values:
        if value and value.strip() and not is_placeholder(value):
            return value.strip()
    return ""


def build_parties(detail: dict) -> Optional[dict]:
    """Build the two party cards from the negotiation's user + partner personas."""
    user_persona = detail.get("user_persona") or {}
    partners = detail.get("partners") or []
    partner = partners[0] if partners else {}

    me_name = _first_present(user_persona.get("name"), "You")
    me_summary = _first_present(
        user_persona.get("role_title"),
        user_persona.get("negotiation_strengths"),
        user_persona.get("notes"),
    )
    them_name = _first_present(partner.get("name"), "Partner")
    them_summary = _first_present(
        partner.get("role_title"),
        partner.get("company"),
        partner.get("known_interests"),
        partner.get("relationship_notes"),
    )

    return {
        "me": {"name": me_name, "initial": _initial(me_name), "summary": me_summary},
        "them": {"name": them_name, "initial": _initial(them_name), "summary": them_summary},
    }


# ------------------------------------------------------------------
# Leverage + vitals — one structured LLM call
# ------------------------------------------------------------------

_SYSTEM = """You are a negotiation analyst. Read the briefing and the transcript, then assess where leverage sits and the deal's vitals.

Respond with ONLY a JSON object, no prose, no code fence, matching exactly:
{
  "leverage": {
    "mineWeight": 0.0,
    "summary": "short read, max 6 words, e.g. 'slight tilt — you'",
    "mine": [{"label": "short edge", "confidence": "high"}],
    "theirs": [{"label": "short edge", "confidence": "high"}]
  },
  "vitals": {
    "zopa": "e.g. £42-58k, or null",
    "batna": "strong | moderate | weak | null",
    "time": "urgent | tight | loose | null"
  }
}

Rules:
- mineWeight is the user's share of leverage, 0 to 1; 0.5 is parity.
- 2 to 4 edges per side. confidence is "high" only when the transcript clearly supports it, otherwise "low".
- Use null (not a guess) for any vital the transcript gives no basis for.
- "mine" is the user; "theirs" is the counterparty."""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of an LLM reply, tolerating a
    surrounding code fence or stray prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


def _clean_edges(raw: Any) -> list[dict]:
    edges: list[dict] = []
    if not isinstance(raw, list):
        return edges
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        confidence = "low" if str(item.get("confidence", "")).lower() == "low" else "high"
        edges.append({"label": label, "confidence": confidence})
    return edges


def _enum_or_none(value: Any, allowed: set[str]) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return None


def _normalise(data: dict) -> dict:
    """Coerce a parsed LLM object into the leverage/vitals contract, filling
    safe defaults so the frontend always receives well-formed data."""
    leverage_in = data.get("leverage") or {}
    try:
        mine_weight = float(leverage_in.get("mineWeight", 0.5))
    except (TypeError, ValueError):
        mine_weight = 0.5
    mine_weight = min(1.0, max(0.0, mine_weight))

    leverage = {
        "mineWeight": round(mine_weight, 2),
        "theirsWeight": round(1.0 - mine_weight, 2),
        "summary": str(leverage_in.get("summary", "")).strip()[:60],
        "mine": _clean_edges(leverage_in.get("mine")),
        "theirs": _clean_edges(leverage_in.get("theirs")),
    }

    vitals_in = data.get("vitals") or {}
    zopa = vitals_in.get("zopa")
    zopa = str(zopa).strip() if isinstance(zopa, str) and zopa.strip().lower() not in ("", "null", "none") else None
    vitals = {
        "zopa": zopa,
        "batna": _enum_or_none(vitals_in.get("batna"), _BATNA_VALUES),
        "time": _enum_or_none(vitals_in.get("time"), _TIME_VALUES),
    }
    return {"leverage": leverage, "vitals": vitals}


def analyse_leverage_vitals(transcript: str, briefing: str, llm: Any) -> Optional[dict]:
    """Run the structured analysis. Returns {'leverage', 'vitals'} or None on a
    parse/LLM failure, so the caller can fall back gracefully."""
    user_content = f"## Briefing\n{briefing or '(none)'}\n\n## Transcript\n{transcript or '(no messages yet)'}"
    try:
        response = llm.invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ])
    except Exception as exc:
        logger.warning("Leverage/vitals analysis call failed: %s", exc)
        return None

    content = response.content if hasattr(response, "content") else str(response)
    parsed = _extract_json(content)
    if parsed is None:
        logger.warning("Leverage/vitals analysis returned unparseable output")
        return None
    return _normalise(parsed)
