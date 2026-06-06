# Intent-Aware Prompting

NegotiatorPro classifies each incoming query by intent and selects a response format to match. Rather than applying a single fixed structure to every message, the system routes each query to one of four formats based on what the user is actually asking for.

## The problem this solves

A structured situation-analysis breakdown is appropriate when you paste an email thread and ask for a read. It is not appropriate when you ask "what is BATNA?" or "they just countered at 80k — what do I say?" Those queries need a direct answer, not sections. Applying one format to all three produces responses that are either under-structured (live tactics without a clear move) or over-structured (a conceptual question buried in a scoring rubric).

## The four intent modes

### TACTICAL

Triggered when the query describes a live, in-progress situation — something that just happened or requires an immediate response.

**Signals:** "they said / offered / countered", "should I accept", "how do I respond to", "just received", "right now".

**Response shape:**
- The move (1–2 sentences — a clear, decisive recommendation)
- Word-for-word language (exact phrasing the user can use)
- If they push back (one contingency)

No scoring, no lengthy background. The user is at the table.

### ANALYSIS

Triggered when the user pastes a transcript or email, or explicitly asks for a read of a situation.

**Signals:** Email headers (From: / Subject: / Dear), "analyse this / what do you think of this / break this down", or a long paste (over 400 characters) that doesn't contain tactical triggers.

**Response shape:**
- Situation analysis — what's actually happening, positions vs. interests, power dynamics
- Recommended approach — the strategic direction
- Calibrated questions — questions to ask the other side
- Scenario planning — what to expect in 2–3 likely directions
- PLEASE self-assessment — scoring on the six dimensions (Polite, Logical, Empathetic, Assertive, Strategic, Engaging)

This is the one mode where the full PLEASE framework belongs.

### QUESTION

Triggered when the user asks about a concept, framework, or principle.

**Signals:** "what is / what's", "when should I", "why does", "how does", "what's the best way", "walk me through", named frameworks (BATNA, anchoring, ZOPA, mirroring, etc.).

**Response shape:**
- The principle (concise definition)
- Why it matters (the practical stakes)
- Applied step (one concrete thing to do with it)

Cite the source author or framework where relevant.

### GENERAL (default)

Fallback for open-ended, exploratory, or strategic queries that don't fit the other patterns.

**Response shape:** Direct teaching style. Headers only if genuinely needed. One applied step. No rigid sections.

## Architecture

### Classification

`backend/intent_classifier.py` is a pure function with no I/O:

```python
from backend.intent_classifier import classify, QueryIntent

intent: QueryIntent = classify("they countered at 80k, what do I say?")
# → "TACTICAL"
```

Classification runs on the **raw user question**, before the negotiation briefing is prepended in `chat.py`. This keeps the classifier signal clean — it reads what the user typed, not the assembled prompt.

Priority order matters for edge cases:

1. Email headers → ANALYSIS (highest priority — unambiguous)
2. Explicit analysis verbs → ANALYSIS
3. Tactical signals → TACTICAL (fires before the long-paste heuristic)
4. Long paste without tactical signals → ANALYSIS
5. Question patterns → QUESTION
6. Fallback → GENERAL

TACTICAL fires before the length heuristic so that "here's the thread — they countered, what do I say?" routes to TACTICAL rather than ANALYSIS.

### Prompt stack

The system prompt has two layers with different caching behaviour:

```
[CACHED]     amfonica-meta.md + negotiation-strategist.yaml (persona)
[NOT CACHED] ## Reference Material
             {rag_chunks}

             ## Response Format
             {FORMAT_INSTRUCTIONS[intent]}

[USER MSG]   {USER_TEMPLATES[intent].format(question=question)}
```

The cached prefix stays byte-identical regardless of intent — the static system prompt doesn't change between modes. Format instructions and the user message template vary per-request but are in the non-cached suffix. The prompt caching benefit is fully preserved.

`amfonica-meta.md` contains a single line in its output section: "Follow the response format provided with each request." It no longer contains the PLEASE structure definition, which now lives in `FORMAT_INSTRUCTIONS["ANALYSIS"]`.

### Data flow

```
chat.py
  └─ raw question captured (before briefing is prepended)
  └─ get_advice(question=enhanced_question, raw_question=raw_question)

rag_engine.py
  └─ intent = classify(raw_question)
  └─ prompt_manager.get_prompt_parts(intent=intent)
  └─ returns (answer, usage, intent)

chat.py
  └─ ChatResponse(answer, model_used, detected_intent=intent)
  └─ save_conversation_turn(..., detected_intent=intent)

frontend
  └─ Mode badge on assistant messages
  └─ detected_intent persisted to chat_messages (migration 008)
     → survives navigation and session reload
```

### Key files

| File | Role |
|------|------|
| `backend/intent_classifier.py` | Classifier, `FORMAT_INSTRUCTIONS`, `USER_TEMPLATES` |
| `backend/prompt_manager.py` | Injects format instructions into context block |
| `backend/rag_engine.py` | Calls classifier, passes intent to prompt manager, returns 3-tuple |
| `backend/api/routes/chat.py` | Passes `raw_question`, unpacks 3-tuple, persists intent |
| `prompts/amfonica-meta.md` | Static system prompt — output section delegates to per-request format |
| `migrations/008_detected_intent.sql` | Adds `detected_intent VARCHAR(20)` to `chat_messages` |

## Extending the system

### Adding a new intent type

1. Add the literal to `QueryIntent` in `intent_classifier.py`:
   ```python
   QueryIntent = Literal["ANALYSIS", "TACTICAL", "QUESTION", "GENERAL", "PREPARATION"]
   ```

2. Add classifier patterns to `classify()`:
   ```python
   _RE_PREPARATION = re.compile(
       r'\b(prepare|preparing|before\s+the\s+(meeting|call|negotiation)|research|planning\s+for)\b',
       re.I,
   )
   # Insert before the QUESTION check:
   if _RE_PREPARATION.search(q):
       return "PREPARATION"
   ```

3. Add the format instruction to `FORMAT_INSTRUCTIONS`:
   ```python
   "PREPARATION": """
   ## Response format — Preparation
   Structure your response as:
   **Their likely position** — what they'll open with and why.
   **Your BATNA** — what walk-away looks like and how strong it is.
   **Opening move** — specific anchor or framing to lead with.
   **Watch for** — two or three signals that should change your approach.
   """,
   ```

4. Add the user template to `USER_TEMPLATES`:
   ```python
   "PREPARATION": "Help me prepare for this negotiation: {question}",
   ```

No other files need to change.

### Tuning an existing format

Edit only `FORMAT_INSTRUCTIONS[intent]` in `intent_classifier.py`. The static system prompt and caching are unaffected.

### Replacing the classifier

`classify()` is a pure function — swap the internals without changing the interface. A lightweight embedding-based classifier or a small dedicated model call can replace the regex implementation and nothing else in the stack changes.

## Testing

`tests/test_intent_classifier.py` covers 30 cases including edge cases and priority ordering. Run with:

```bash
pytest tests/test_intent_classifier.py -v
```

Key cases to add when extending: ensure the new pattern fires at the right priority, and add at least one case that should NOT trigger it (to guard against over-matching).
