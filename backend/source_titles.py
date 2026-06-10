"""
Turn a retrieved chunk's source filename into a human citation title.

The vectorstore stores a source filename per chunk (e.g. "getting-past-no.pdf"
or "negotiation/never-split-the-difference.pdf"). The corpus books are known, so
a small curated map gives proper "Author · Title" citations; anything not in the
map falls back to a cleaned, title-cased filename. Pure functions — no I/O — so
the synchronous RAG engine can call them inline.
"""
from __future__ import annotations

import os
import re

# Curated display titles for the known negotiation/sales corpus, keyed by the
# normalised filename stem (lower-case, extension stripped, underscores → dashes).
# Match is by stem prefix so variants like "bargaining_for_advantage_shell-richard-g"
# still resolve. These attributions are the books' actual authors.
_CURATED: dict[str, str] = {
    "never-split-the-difference": "Voss · Never Split the Difference",
    "art-of-negotiation-voss": "Voss · The Art of Negotiation",
    "getting-to-yes": "Fisher & Ury · Getting to Yes",
    "getting-past-no": "Ury · Getting Past No",
    "power-of-a-positive-no": "Ury · The Power of a Positive No",
    "bargaining-for-advantage": "Shell · Bargaining for Advantage",
    "better-not-perfect": "Bazerman · Better, Not Perfect",
    "conflict-management-textbook-sharma": "Sharma · Conflict Management",
    "negotiation-and-conflict-management": "Negotiation and Conflict Management",
    "upward-spiral-workbook": "The Upward Spiral Workbook",
    "you-get-what-you-negotiate": "Karrass · You Get What You Negotiate",
}


def _stem(source_file: str) -> str:
    """Normalise a source path to a lower-case, dash-joined filename stem."""
    base = os.path.basename(source_file or "").strip()
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)  # drop extension
    base = base.replace("_", "-").lower()
    return re.sub(r"-{2,}", "-", base).strip("-")


def display_title(source_file: str) -> str:
    """Citation title for a source filename, curated where known."""
    stem = _stem(source_file)
    if not stem:
        return "Source"
    for key, title in _CURATED.items():
        if stem == key or stem.startswith(key):
            return title
    # Fallback: title-case the cleaned stem ("getting-past-no" → "Getting Past No").
    words = [w for w in stem.split("-") if w]
    return " ".join(word.capitalize() for word in words) or "Source"
