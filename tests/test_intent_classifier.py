"""Unit tests for the intent classifier."""

import pytest
pytestmark = pytest.mark.unit
from backend.intent_classifier import classify


# ------------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------------

def test_analysis_email_from_header():
    assert classify("From: john@example.com\nSubject: Re: Contract\n\nHi there...") == "ANALYSIS"

def test_analysis_email_subject_header():
    assert classify("Subject: Renewal proposal\nTo: alice@corp.com\n\nSee below.") == "ANALYSIS"

def test_analysis_re_header():
    # "Re:" at start of a line
    assert classify("Here is the thread:\n\nRe: Deal terms\nThey offered 80k.") == "ANALYSIS"

def test_analysis_explicit_verb_analyze():
    assert classify("Can you analyze this email thread for me?") == "ANALYSIS"

def test_analysis_explicit_verb_analyse():
    assert classify("Please analyse this negotiation transcript.") == "ANALYSIS"

def test_analysis_what_do_you_think():
    assert classify("What do you think of this counteroffer?") == "ANALYSIS"

def test_analysis_review_this():
    assert classify("Can you review this draft response before I send it?") == "ANALYSIS"

def test_analysis_long_pasted_content():
    # Long text with newlines but no first-person tactical signal.
    # "They proposed" etc. are narrative — no "should I / how do I" framing.
    transcript = (
        "Meeting notes from supplier discussion.\n"
        "We opened at 100k annual contract value.\n"
        "The supplier revised the scope mid-meeting.\n"
        "Delivery timelines were a key discussion point.\n"
        "Terms remain open on payment schedule.\n"
        "The supplier floated a 90-day payment window.\n"
        "We requested 30 days. No resolution yet.\n"
        "Further discussion needed on SLA definitions.\n"
        "Both parties agreed to reconvene next week."
    ) * 4  # make it >500 chars with newlines
    assert classify(transcript) == "ANALYSIS"


# ------------------------------------------------------------------
# TACTICAL
# ------------------------------------------------------------------

def test_tactical_they_said():
    assert classify("They said they can only go to 75k. What should I do?") == "TACTICAL"

def test_tactical_they_countered():
    assert classify("They came back with 80k, should I accept?") == "TACTICAL"

def test_tactical_how_do_i_respond():
    assert classify("How do I respond to this pushback on price?") == "TACTICAL"

def test_tactical_should_i_accept():
    assert classify("Should I accept their revised offer?") == "TACTICAL"

def test_tactical_what_should_i_say():
    assert classify("What should I say when they ask about our BATNA?") == "TACTICAL"

def test_tactical_just_got():
    assert classify("Just got off a call with the buyer. They rejected the whole deal.") == "TACTICAL"

def test_tactical_about_to_send():
    assert classify("I'm about to send the final proposal. Any last thoughts?") == "TACTICAL"

def test_tactical_beats_long_pasted_content():
    # Long paste WITH a tactical signal — should classify as TACTICAL not ANALYSIS
    long_paste = ("Background context here.\n" * 40)
    question = long_paste + "\nHow do I respond to their last offer?"
    assert classify(question) == "TACTICAL"

def test_tactical_beats_email_not_really():
    # Email headers win over tactical (email headers are checked first)
    # This is the correct behaviour: if they paste an email thread,
    # the structural analysis is more appropriate than a tactical response.
    email_with_question = (
        "From: supplier@company.com\n"
        "Subject: Re: Contract renewal\n\n"
        "Here's what they sent. Should I accept?"
    )
    # Email header fires first → ANALYSIS
    assert classify(email_with_question) == "ANALYSIS"


# ------------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------------

def test_question_what_is_batna():
    assert classify("What is BATNA?") == "QUESTION"

def test_question_what_is_zopa():
    assert classify("What is a ZOPA?") == "QUESTION"

def test_question_best_way():
    assert classify("What's the best way to start a car negotiation?") == "QUESTION"

def test_question_best_approach():
    assert classify("What's the best approach for anchoring?") == "QUESTION"

def test_question_when_should():
    assert classify("When should I make the first offer?") == "QUESTION"

def test_question_how_does_mirroring_work():
    assert classify("How does mirroring work in negotiation?") == "QUESTION"

def test_question_explain():
    assert classify("Can you explain the concept of tactical empathy?") == "QUESTION"

def test_question_walk_me_through():
    assert classify("Walk me through the principled negotiation framework.") == "QUESTION"


# ------------------------------------------------------------------
# GENERAL (fallback)
# ------------------------------------------------------------------

def test_general_open_ended():
    assert classify("Tell me about negotiation.") == "GENERAL"

def test_general_short():
    assert classify("Interesting situation here.") == "GENERAL"

def test_general_empty():
    assert classify("") == "GENERAL"

def test_general_whitespace():
    assert classify("   \n  ") == "GENERAL"

def test_general_vague():
    assert classify("I'm preparing for a tough negotiation next week.") == "GENERAL"
