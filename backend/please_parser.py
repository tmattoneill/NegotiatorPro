"""
Parse the PLEASE self-assessment out of an assistant answer.

ANALYSIS-intent responses end with a machine-readable block (see the ANALYSIS
format instruction in intent_classifier.py):

    ```please
    polite=4 logical=5 empathetic=3 assertive=2 strategic=4 engaging=3
    ```

parse_and_strip pulls the six scores into a structured dict and removes the
block (and any leftover "PLEASE Self-Assessment" heading) from the answer, so
the chat bubble shows clean prose while the gutter renders the gauges.

Pure functions — no I/O, no side effects.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, TypedDict

# The six elements in canonical order, each with its single-letter display code.
# Two elements share the letter "E"; the code is only used for the weakest-spot
# label, where a repeated letter is acceptable and dedupes cleanly.
_FIELDS: list[tuple[str, str]] = [
    ("polite", "P"),
    ("logical", "L"),
    ("empathetic", "E"),
    ("assertive", "A"),
    ("strategic", "S"),
    ("engaging", "E"),
]


class PleaseScore(TypedDict):
    polite: int
    logical: int
    empathetic: int
    assertive: int
    strategic: int
    engaging: int
    total: int
    weakest: list[str]


# A fenced ```please block, capturing its inner body. Tolerant of casing and of
# a missing language tag where the body still reads as key=value pairs.
_FENCED_BLOCK = re.compile(
    r"```\s*please\s*\n(?P<body>.*?)```",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: the six key=value pairs on a line without a fence, in case the model
# drops the code fence but keeps the data.
_BARE_LINE = re.compile(
    r"(?:^|\n)[^\n]*\bpolite\s*=\s*\d[^\n]*\bengaging\s*=\s*\d[^\n]*",
    re.IGNORECASE,
)

# A trailing heading the model sometimes emits above the block ("### PLEASE
# Self-Assessment", "**PLEASE Self-Assessment**", etc.) with nothing useful
# under it once the block is removed.
_DANGLING_HEADING = re.compile(
    r"\n*\s*(?:#{1,6}\s*|\*\*)?PLEASE(?:\s+Self[-\s]?Assessment)?\s*:?\*{0,2}\s*$",
    re.IGNORECASE,
)


def _extract_scores(body: str) -> Optional[dict[str, int]]:
    """Pull each field's 1–5 score from a block body. Returns None if any of the
    six is missing or out of range — a partial block is treated as no block."""
    scores: dict[str, int] = {}
    for field, _code in _FIELDS:
        match = re.search(rf"\b{field}\s*=\s*([1-5])\b", body, re.IGNORECASE)
        if not match:
            return None
        scores[field] = int(match.group(1))
    return scores


def _weakest_codes(scores: dict[str, int]) -> list[str]:
    """The letter codes of the soft spots: the fields in the lowest two distinct
    score tiers, ordered weakest first, deduplicated by letter. Empty when every
    element scores the same — there is no relative weak spot to flag."""
    distinct = sorted({scores[f] for f, _ in _FIELDS})
    if len(distinct) == 1:
        return []
    soft_tiers = set(distinct[:2])
    codes: list[str] = []
    for value in distinct:
        if value not in soft_tiers:
            continue
        for field, code in _FIELDS:
            if scores[field] == value and code not in codes:
                codes.append(code)
    return codes


def parse_and_strip(answer: str) -> Tuple[str, Optional[PleaseScore]]:
    """Split an answer into (clean_answer, score).

    Returns the answer unchanged with None when no valid PLEASE block is found
    (e.g. TACTICAL/QUESTION/GENERAL responses, or a malformed block).
    """
    if not answer:
        return answer, None

    span: Optional[Tuple[int, int]] = None
    scores: Optional[dict[str, int]] = None

    fenced = _FENCED_BLOCK.search(answer)
    if fenced:
        scores = _extract_scores(fenced.group("body"))
        if scores is not None:
            span = fenced.span()

    if scores is None:
        bare = _BARE_LINE.search(answer)
        if bare:
            scores = _extract_scores(bare.group(0))
            if scores is not None:
                span = bare.span()

    if scores is None or span is None:
        return answer, None

    clean = (answer[: span[0]] + answer[span[1]:])
    clean = _DANGLING_HEADING.sub("", clean).rstrip()

    total = sum(scores[field] for field, _ in _FIELDS)
    please: PleaseScore = {
        **scores,  # type: ignore[typeddict-item]
        "total": total,
        "weakest": _weakest_codes(scores),
    }
    return clean, please
