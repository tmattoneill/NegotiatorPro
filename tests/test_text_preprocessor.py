"""Unit tests for the email preprocessor's footer stripping."""
import pytest
pytestmark = pytest.mark.unit

from backend.text_preprocessor import TextPreprocessor


THREAD = (
    "Analyse this negotiation:\n"
    "Sent thank you!\n\n"
    "On Fri, 20 Nov 2026 at 16:05, Matt wrote:\n"
    "Give me a day or two. They are still keen.\n\n"
    "Sent from my iPhone\n\n"
    "On 20 Nov 2026, at 15:59, Eniko wrote:\n"
    "Roland came back at 13,500 EUR per month plus 10 percent.\n"
    "He wants a decision by January.\n\n"
    "This email may contain confidential information.\n"
    "If you are not the intended recipient please delete it.\n\n"
    "Best,\nEniko\n"
)


def test_sent_from_my_strips_only_its_own_line():
    text = TextPreprocessor().remove_email_footers(THREAD)
    assert "Sent from my iPhone" not in text
    assert "13,500 EUR" in text
    assert "decision by January" in text


def test_confidentiality_lines_go_but_the_body_stays():
    text = TextPreprocessor().remove_email_footers(THREAD)
    assert "confidential information" not in text
    assert "intended recipient" not in text
    assert "Roland came back" in text


def test_lowercase_private_in_prose_survives():
    text = TextPreprocessor().remove_email_footers("We run a private beta.\nPricing is open.\n")
    assert "private beta" in text and "Pricing is open" in text


def test_full_preprocess_keeps_most_of_a_thread():
    result = TextPreprocessor().preprocess(THREAD)
    assert result["reduction_percentage"] < 50
    assert "13,500" in result["processed_text"]
