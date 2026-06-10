"""
Intent classifier for user queries.

Classifies a raw query into one of four intents so the prompt manager can
select the right response format. Classification is pure regex — no LLM call,
no I/O, no side effects. Fast enough to run inline before every request.

Priority order (first match wins):
  1. ANALYSIS  — email headers present (high-specificity structural signal)
  2. ANALYSIS  — explicit analysis-request verb
  3. TACTICAL  — live-situation, first-person decision language
  4. ANALYSIS  — long pasted content (heuristic, lower priority than TACTICAL
                 so "here's the thread — how should I respond?" → TACTICAL)
  5. QUESTION  — conceptual, educational, framework-level question
  6. GENERAL   — fallback

Why this order matters: a message like "Here is the email thread (long paste) —
how should I respond?" has both ANALYSIS signals (the paste) and a TACTICAL
signal (how should I respond?). The TACTICAL intent is more specific to the
user's actual need, so it fires first.
"""
from __future__ import annotations

import re
from typing import Literal

QueryIntent = Literal["ANALYSIS", "TACTICAL", "QUESTION", "GENERAL"]

# ------------------------------------------------------------------
# Compiled patterns (module-level, evaluated once at import)
# ------------------------------------------------------------------

# ANALYSIS: email / document structure markers — very high specificity
_RE_EMAIL_HEADERS = re.compile(
    r"(?:^|\n)\s*(?:from|to|subject|re|cc|bcc|date)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# ANALYSIS: explicit request to review or analyze something
_RE_ANALYSIS_VERB = re.compile(
    r"\b(?:analys[ei]s?|analy[sz]e|what\s+do\s+you\s+think\s+(?:of|about)|"
    r"review\s+this|thoughts\s+on\s+this|feedback\s+on\s+this|"
    r"critique\s+this|assess\s+this)\b",
    re.IGNORECASE,
)

# TACTICAL: the user is in a live situation and needs an immediate move
_RE_TACTICAL = re.compile(
    r"\b(?:"
    r"they\s+(?:said|told|offered|proposed|came\s+back|pushed\s+back|rejected|"
    r"countered|responded|replied|want|asked\s+for)|"
    r"how\s+(?:do|should)\s+i\s+(?:respond|reply|handle|deal\s+with|counter)|"
    r"how\s+to\s+respond\s+to|"
    r"should\s+i\s+(?:accept|reject|counter|push\s+back|walk\s+away|agree)|"
    r"do\s+i\s+(?:accept|reject|counter|take\s+it|go\s+back)|"
    r"what\s+should\s+i\s+(?:say|do|offer|reply|respond)|"
    r"just\s+(?:got|received|heard|had\s+a\s+call|got\s+off\s+a\s+call)|"
    r"right\s+now|"
    r"in\s+the\s+middle\s+of|"
    r"on\s+a\s+call|"
    r"about\s+to\s+(?:send|call|meet|reply)"
    r")\b",
    re.IGNORECASE,
)

# QUESTION: conceptual / educational / framework-level
_RE_QUESTION = re.compile(
    r"\b(?:"
    # "what's the best way/approach/..." — \s* handles the space after the optional group
    r"what(?:\'s|\s+is|\s+are|\s+was|\s+were)\s+(?:the\s+best|a\s+good|the\s+difference|"
    r"the\s+right|a\s+)?\s*(?:way|approach|tactic|strategy|technique|method|framework|principle)|"
    r"what\s+is\s+(?:a\s+|the\s+)?(?:batna|zopa|anchoring|mirroring|labell?ing|"
    r"tactical\s+empathy|principled\s+negotiation|reservation\s+price|"
    r"best\s+alternative)|"
    r"when\s+should\s+(?:i|you|we)\s+|"
    r"why\s+(?:do|does|would|should)\s+|"
    r"how\s+does\s+\w+\s+work|"
    r"can\s+you\s+explain|"
    r"explain\s+(?:the\s+concept|how|what|why)|"
    r"what(?:\'s|\s+is)\s+the\s+(?:idea|concept|theory|logic)\s+behind|"
    r"walk\s+me\s+through"  # phrase alone is sufficient signal regardless of what follows
    r")\b",
    re.IGNORECASE,
)

# ANALYSIS heuristic: long text with line breaks (pasted content)
_PASTE_MIN_LEN = 500


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def classify(question: str) -> QueryIntent:
    """
    Classify a raw user question into one of four intent types.

    Call this with the question BEFORE the negotiation briefing is prepended —
    the briefing contains structured data (persona names, role labels) that
    would pollute the signals.
    """
    if not question or not question.strip():
        return "GENERAL"

    # 1. Email/document structure → ANALYSIS (highest specificity)
    if _RE_EMAIL_HEADERS.search(question):
        return "ANALYSIS"

    # 2. Explicit analysis verb → ANALYSIS
    if _RE_ANALYSIS_VERB.search(question):
        return "ANALYSIS"

    # 3. Live-situation language → TACTICAL
    #    (checked before the length heuristic so "here's the thread —
    #    how should I respond?" classifies as TACTICAL, not ANALYSIS)
    if _RE_TACTICAL.search(question):
        return "TACTICAL"

    # 4. Long pasted content heuristic → ANALYSIS
    if len(question) > _PASTE_MIN_LEN and "\n" in question:
        return "ANALYSIS"

    # 5. Conceptual / educational → QUESTION
    if _RE_QUESTION.search(question):
        return "QUESTION"

    # 6. Fallback
    return "GENERAL"


# ------------------------------------------------------------------
# Intent-specific format instructions (injected into context_block)
# ------------------------------------------------------------------

FORMAT_INSTRUCTIONS: dict[str, str] = {
    "ANALYSIS": """\
## Response Format

Structure your response with these sections:

### Situation Analysis
What's actually happening — stated positions, underlying interests, power dynamics, and where the user is in the deal or negotiation cycle.

### Recommended Approach
Your specific recommendation. Word-for-word language for the critical moments. If multiple paths are viable, name them and recommend one.

### Calibrated Questions
3–6 questions the user should ask the other side, each with a one-line note on what it reveals.

### Scenario Planning
Two or three likely counter-responses from the other side, with the user's next move for each.

### PLEASE Self-Assessment
Score your draft 1–5 on each of Polite, Logical, Empathetic, Assertive, Strategic, Engaging. If the total is below 20, revise the draft before you answer. Do not write the scores out in prose. Instead, end your whole response with this block, exactly, on its own lines:

```please
polite=N logical=N empathetic=N assertive=N strategic=N engaging=N
```

Replace each N with the integer score. The app reads this block and renders the scores, so it must be the last thing in your response.""",

    "TACTICAL": """\
## Response Format

This is a live situation. Skip preamble. Give:

**The move** — your specific recommendation in 1–2 sentences.

**Word-for-word** — the exact language to use, ready to copy.

**If they push back** — one contingency response.

No lengthy analysis. No PLEASE scoring.""",

    "QUESTION": """\
## Response Format

Answer the question directly:

**The principle** — the core idea, named and explained concisely.

**Why it matters** — one sentence connecting it to practice.

**Applied step** — one thing the user can do with this today.

Cite the relevant framework or author by name. End with one follow-up question only if the answer genuinely depends on context you don't have.""",

    "GENERAL": """\
## Response Format

Answer directly in a teaching style. Use headers only if the answer has genuinely distinct parts. Be concrete. End with one applied next step.""",
}

# ------------------------------------------------------------------
# Intent-specific user message templates
# ------------------------------------------------------------------

USER_TEMPLATES: dict[str, str] = {
    "ANALYSIS": (
        "Please analyse this situation and provide expert guidance:\n\n"
        "{question}\n\n"
        "Draw on the reference material and your operating principles. "
        "Follow the response format."
    ),
    "TACTICAL": (
        "I need an immediate tactical recommendation:\n\n"
        "{question}\n\n"
        "Be decisive. Give me the move and the exact words to use."
    ),
    "QUESTION": (
        "I have a question about negotiation:\n\n"
        "{question}\n\n"
        "Give me a clear, principled answer with one practical step I can act on today."
    ),
    "GENERAL": (
        "{question}\n\n"
        "Give me a direct, expert answer."
    ),
}
