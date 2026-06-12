"""Unit tests for negotiation context: parties + leverage/vitals analysis."""
import pytest
pytestmark = pytest.mark.unit

from backend.negotiation_context import (
    build_parties,
    analyse_leverage_vitals,
    _extract_json,
    _normalise,
)


class _Resp:
    def __init__(self, content):
        self.content = content


class _LLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return _Resp(self._content)


# ------------------------------------------------------------------
# build_parties
# ------------------------------------------------------------------

def test_build_parties_from_personas():
    detail = {
        "user_persona": {"name": "Matt", "role_title": "Founder"},
        "partners": [{"name": "Acme Corp", "company": "Acme", "role_title": "Procurement"}],
    }
    parties = build_parties(detail)
    assert parties["me"] == {"name": "Matt", "initial": "M", "summary": "Founder"}
    assert parties["them"]["name"] == "Acme Corp"
    assert parties["them"]["initial"] == "A"


def test_build_parties_falls_back_when_personas_missing():
    parties = build_parties({})
    assert parties["me"]["name"] == "You"
    assert parties["them"]["name"] == "Partner"
    assert parties["me"]["initial"] == "Y"


# ------------------------------------------------------------------
# _extract_json
# ------------------------------------------------------------------

def test_extract_json_from_fenced_block():
    text = 'Sure:\n```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_from_bare_object_with_prose():
    text = 'Here it is {"a": 1, "b": {"c": 2}} done.'
    assert _extract_json(text) == {"a": 1, "b": {"c": 2}}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("no json here") is None


# ------------------------------------------------------------------
# _normalise
# ------------------------------------------------------------------

def test_normalise_clamps_weight_and_derives_theirs():
    out = _normalise({"leverage": {"mineWeight": 1.4}, "vitals": {}})
    assert out["leverage"]["mineWeight"] == 1.0
    assert out["leverage"]["theirsWeight"] == 0.0


def test_normalise_coerces_vitals_enums_and_nulls():
    out = _normalise({
        "leverage": {},
        "vitals": {"zopa": "null", "batna": "Strong", "time": "whenever"},
    })
    assert out["vitals"]["zopa"] is None       # "null" string → None
    assert out["vitals"]["batna"] == "strong"  # case-folded into the enum
    assert out["vitals"]["time"] is None        # not in the allowed set


def test_normalise_caps_edges_and_defaults_confidence():
    out = _normalise({
        "leverage": {
            "mine": [{"label": f"edge {i}"} for i in range(6)],
            "theirs": [{"label": "x", "confidence": "low"}],
        },
        "vitals": {},
    })
    assert len(out["leverage"]["mine"]) == 4          # capped at 4
    assert out["leverage"]["mine"][0]["confidence"] == "high"  # default
    assert out["leverage"]["theirs"][0]["confidence"] == "low"


# ------------------------------------------------------------------
# analyse_leverage_vitals end to end (fake LLM)
# ------------------------------------------------------------------

def test_analyse_parses_fenced_json_reply():
    reply = (
        'Analysis:\n```json\n{"leverage": {"mineWeight": 0.6, "summary": "tilt you",'
        ' "mine": [{"label": "BATNA", "confidence": "high"}], "theirs": []},'
        ' "vitals": {"zopa": "£40-50k", "batna": "moderate", "time": "tight"}}\n```'
    )
    out = analyse_leverage_vitals("User: hi", "", _LLM(reply))
    assert out["leverage"]["mineWeight"] == 0.6
    assert out["leverage"]["theirsWeight"] == 0.4
    assert out["vitals"]["batna"] == "moderate"


def test_analyse_returns_none_on_unparseable_reply():
    assert analyse_leverage_vitals("x", "", _LLM("I cannot help with that.")) is None


def test_analyse_returns_none_when_llm_raises():
    class _Boom:
        def invoke(self, _m):
            raise RuntimeError("backend down")

    assert analyse_leverage_vitals("x", "", _Boom()) is None
