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
2. Explicit analysis verbs → ANALYSIS (cold start only, see below)
3. Tactical signals → TACTICAL (fires before the long-paste heuristic)
4. Long paste without tactical signals → ANALYSIS
5. Question patterns → QUESTION
6. Fallback → GENERAL

TACTICAL fires before the length heuristic so that "here's the thread — they countered, what do I say?" routes to TACTICAL rather than ANALYSIS.

### Follow-up turns

`classify(question, has_history=True)` runs when earlier turns of the conversation are being sent to the model. Once the model can see its own earlier analysis, "now give me a PLEASE framework analysis of that" is a request to build on the thread, not to start over. So on follow-ups:

- The analysis-verb rule is skipped. "Analyse", "what do you think of" and friends classify as GENERAL unless a stronger signal is present.
- The long-paste threshold rises from 500 to 1500 characters. Only a substantial new paste (a fresh email, a transcript) restarts the ANALYSIS format.
- Email headers, tactical signals and question patterns fire exactly as before.

GENERAL follow-ups go to the model as typed, without the "Give me a direct, expert answer" wrapper, and the format block is prefixed with `FOLLOWUP_NOTE`, which tells the model to answer only what was asked and not restate the situation analysis.

### Prompt stack

Everything that changes per turn sits at the tail of the message list, so the prefix the providers cache only ever grows:

```
[SYSTEM A, cached]   amfonica-meta.md + negotiation-strategist.yaml (persona)
[SYSTEM B, cached]   ## Negotiation Briefing  (parties, roles, notes)
                     ## Negotiation Memory    (folded summary, see Conversation memory)
[HISTORY]            earlier turns of this conversation, verbatim user/assistant messages
                     (Anthropic: cache breakpoint on the last one)
[FINAL USER MSG]     uploaded file text / partner_info, if any
                     ## Reference Material  {rag_chunks}
                     FOLLOWUP_NOTE (follow-ups only)
                     ## Response Format  {FORMAT_INSTRUCTIONS[intent]}
                     ---
                     {USER_TEMPLATES[intent].format(question=question)}
```

System A is byte-identical for every request in a mode. System B is identical for every turn of a negotiation until the memory refreshes. On Anthropic each of these carries a `cache_control` breakpoint, plus one on the last history message, so turn N+1 reads turn N's whole prefix from cache and pays full price only for the tail. OpenAI caches the longest matching prefix automatically, so the same layout works with plain strings. `rag_engine._build_messages` owns this layout and `tests/test_message_layout.py` pins it.

Text preprocessing applies to the user's question only. The briefing, memory and history are sent as-is.

`amfonica-meta.md` contains a single line in its output section: "Follow the response format provided with each request." It no longer contains the PLEASE structure definition, which now lives in `FORMAT_INSTRUCTIONS["ANALYSIS"]`.

### Data flow

```
chat.py
  └─ raw question captured (before briefing is prepended)
  └─ get_advice(question=enhanced_question, raw_question=raw_question)

rag_engine.py
  └─ intent = classify(raw_question, has_history=bool(history))
  └─ prompt_manager.get_prompt_parts(intent=intent, briefing=, memory=, has_history=)
  └─ _build_messages(backend, parts, history, extra_context)
  └─ returns (answer, usage, intent, sources)

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
| `backend/intent_classifier.py` | Classifier, `FORMAT_INSTRUCTIONS`, `USER_TEMPLATES`, `FOLLOWUP_NOTE` |
| `backend/prompt_manager.py` | Builds `PromptParts` (static, negotiation block, context block, user prompt) |
| `backend/rag_engine.py` | Calls classifier, builds the cache-friendly message list, returns 4-tuple |
| `backend/api/routes/chat.py` | Loads conversation context once, selects the history window, persists the turn, spawns the memory fold |
| `backend/conversation_memory.py` | History window selection, memory fold prompt and background refresh |
| `migrations/012_negotiation_memory.sql` | Adds `memory JSONB` to `negotiations` |
| `prompts/amfonica-meta.md` | Static system prompt — output section delegates to per-request format |
| `migrations/008_detected_intent.sql` | Adds `detected_intent VARCHAR(20)` to `chat_messages` |

## Conversation memory

Before this, every turn was a cold start: `get_advice` sent only the system prompt and the current question, so a follow-up like "now give me a PLEASE analysis" regenerated the whole situation analysis. Two layers now give the chat continuity.

**Verbatim window.** The newest turns of the current conversation are loaded from `chat_messages` and sent as real user/assistant messages. `conversation_memory.select_history_window` keeps whole turns within a token budget (chars/4), always at least `history_min_turns`, never more than `history_max_turns`. No model call.

**Negotiation memory.** A compact markdown summary stored in `negotiations.memory`: state of play, offers on the table, agreed facts, open questions, user instructions and preferences, next steps. It spans every conversation in the negotiation. After a turn is saved, `maybe_refresh_memory` runs in the background and folds only the turns that have left the verbatim window since the last refresh (previous memory + dropped turns → new memory). It never runs on every turn, and on the first turns of a negotiation it does nothing at all. The fold uses a fixed cheap model on the system key (`gpt-4o-mini`), so it does not depend on the user's provider having a key. A failed fold leaves the old memory in place and is retried on the next save.

Config, in `config.json` under `chat_memory` (env overrides `CHAT_MEMORY_ENABLED`, `CHAT_MEMORY_WINDOW_TOKENS`, `CHAT_MEMORY_MIN_TURNS`, `CHAT_MEMORY_MAX_TURNS`):

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Turn history and memory off entirely |
| `history_window_tokens` | `7000` | Budget for the verbatim window |
| `history_min_turns` / `history_max_turns` | `4` / `12` | Bounds on the window. The start steps by (max - min) turns rather than sliding by one, so the cached prefix stays stable for a run of turns |
| `memory_refresh_model` | `gpt-4o-mini` | OpenAI model for the fold |
| `memory_max_chars` | `4000` | Cap on the stored summary |

Inspect a negotiation's memory with `GET /api/negotiations/{id}/memory?user_id=`. Super-admin chats have no conversation and stay stateless by design.

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
