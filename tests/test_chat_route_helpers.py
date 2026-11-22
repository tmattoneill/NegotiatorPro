"""Unit tests for the pure helpers in the chat route (no database)."""
import pytest
pytestmark = pytest.mark.unit

from backend.api.routes.chat import build_negotiation_briefing


def test_briefing_empty_when_no_detail():
    assert build_negotiation_briefing(None) == ""
    assert build_negotiation_briefing({}) == ""


def test_briefing_formats_parties_from_detail():
    detail = {
        "title": "Audyence contract",
        "description": "Six month advisory term",
        "user_persona": {
            "name": "Eniko",
            "role_title": "Advisor",
            "organization": "ETS Advisory",
            "notes": "N/A",
        },
        "partners": [
            {"name": "Roland", "role_title": "CEO", "company": "Audyence", "known_interests": "keep cost near benchmarks"},
        ],
    }
    briefing = build_negotiation_briefing(detail)
    assert briefing.startswith("## Negotiation Briefing")
    assert "**Title**: Audyence contract" in briefing
    assert "**Description**: Six month advisory term" in briefing
    assert "**You (Eniko)**" in briefing
    assert "- **Organisation**: ETS Advisory" in briefing
    assert "Notes" not in briefing  # placeholder dropped
    assert "**Counterparty 1: Roland**" in briefing
    assert "- **Known interests**: keep cost near benchmarks" in briefing
