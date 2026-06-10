"""Unit tests for source title mapping and citation building."""
from backend.source_titles import display_title
from backend.rag_engine import EnhancedNegotiationRAG


class _Doc:
    """Minimal stand-in for a langchain Document."""
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


# ------------------------------------------------------------------
# display_title
# ------------------------------------------------------------------

def test_curated_title_from_bare_filename():
    assert display_title("getting-past-no.pdf") == "Ury · Getting Past No"


def test_curated_title_from_pathed_filename():
    assert display_title("negotiation/never-split-the-difference.pdf") == "Voss · Never Split the Difference"


def test_curated_title_matches_on_stem_prefix():
    # The Shell book filename carries an author suffix; prefix match still wins.
    assert display_title("bargaining_for_advantage_shell-richard-g.pdf") == "Shell · Bargaining for Advantage"


def test_unknown_file_falls_back_to_titlecased_stem():
    assert display_title("some-other-book.pdf") == "Some Other Book"


def test_empty_filename_is_safe():
    assert display_title("") == "Source"


# ------------------------------------------------------------------
# _build_source_refs
# ------------------------------------------------------------------

def test_build_refs_dedupes_by_title_and_marks_top_active():
    docs = [
        _Doc("Repeat the last three words. Then shut up.", {"source_file": "never-split-the-difference.pdf"}),
        _Doc("Separate the people from the problem.", {"source_file": "getting-to-yes.pdf"}),
        _Doc("Another Voss chunk.", {"source_file": "never-split-the-difference.pdf"}),
    ]
    refs = EnhancedNegotiationRAG._build_source_refs(docs)
    assert [r["title"] for r in refs] == [
        "Voss · Never Split the Difference",
        "Fisher & Ury · Getting to Yes",
    ]
    assert refs[0]["active"] is True
    assert refs[1]["active"] is False


def test_build_refs_truncates_long_quote():
    long_passage = "word " * 100
    refs = EnhancedNegotiationRAG._build_source_refs(
        [_Doc(long_passage, {"source_file": "getting-past-no.pdf"})]
    )
    assert refs[0]["quote"].endswith("…")
    assert len(refs[0]["quote"]) <= 161


def test_build_refs_empty_when_no_docs():
    assert EnhancedNegotiationRAG._build_source_refs([]) == []
