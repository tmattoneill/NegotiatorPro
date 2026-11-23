"""
Per-negotiation conversation memory.

Two layers give the chat continuity without resending the whole transcript:

  1. The verbatim window: the newest turns of the current conversation, sent
     to the model as real user/assistant messages. No model call, just a
     token budget.
  2. The negotiation memory: a compact markdown summary (state of play,
     offers, agreed facts, open questions, user instructions, next steps)
     stored on negotiations.memory. It covers every conversation in the
     negotiation and is refreshed by one cheap model call, in the background,
     only when turns have fallen out of the verbatim window since the last
     refresh. Never on every turn.

The pure helpers here have no I/O so they are unit-testable; the async
maybe_refresh_memory at the bottom is the only thing that touches the
database and the model.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from .config_loader import config

logger = logging.getLogger(__name__)

# Model for the memory fold. Deliberately a fixed cheap model on the system
# key: the fold is rare, small, and must not depend on whichever provider the
# user chats with having a key configured.
MEMORY_REFRESH_PROVIDER = "openai"
MEMORY_REFRESH_MODEL = "gpt-5-mini"

# Oldest turns past this cap are dropped from a single fold. Matches the
# transcript cap used by the leverage/vitals analysis.
_FOLD_CHAR_CAP = 12000
# Rows the memory has not folded yet are kept verbatim, but only up to this
# many turns past the normal window. If the fold is broken (no system key,
# model down) the window must not grow without limit.
_UNCOVERED_EXTRA_TURNS = 6

_DEFAULTS = {
    "enabled": True,
    "history_window_tokens": 7000,
    "history_min_turns": 4,
    "history_max_turns": 12,
    "memory_refresh_model": MEMORY_REFRESH_MODEL,
    "memory_max_chars": 4000,
}

_MEMORY_SYSTEM_PROMPT = """\
You maintain a compact working memory for an ongoing negotiation. You receive \
the negotiation briefing, the previous memory, and the newest exchanges that \
are about to leave the advisor's short-term context.

The exchanges are between the user (the person negotiating) and their AI \
advisor. The advisor never speaks to the counterparty. A figure or term the \
advisor recommends is a suggestion, not an offer, until the user says they \
have put it to the other side.

Produce the UPDATED memory as markdown with exactly these headings, each a \
short bullet list. Omit a heading only when it has nothing under it.

## State of play
## Offers on the table
(Only terms one party has actually put to the other. Say who made each offer. \
Never list the advisor's recommendations here.)
## Advisor suggestions not yet acted on
## Agreed facts
## Open questions
## User instructions and preferences
## Next steps

Rules:
- Merge, do not append. Rewrite each section so it reflects the latest state.
- Drop items that have been superseded or resolved.
- Keep numbers, names, dates and quoted terms exact.
- Under "User instructions and preferences" keep any rule the user set about \
how to advise them (for example who is or is not involved, tone, scope), \
close to their own words.
- Never invent. If something is unclear, record it as an open question.
- Stay under {max_chars} characters.
- Output the markdown only, headings included. No preamble, no closing remarks, no code fence."""


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return default


def get_memory_settings() -> dict:
    """config.json["chat_memory"] with env overrides, falling back to defaults."""
    settings = dict(_DEFAULTS)
    configured = config.get("chat_memory") or {}
    if isinstance(configured, dict):
        settings.update({k: v for k, v in configured.items() if v is not None})
    settings["enabled"] = _env_bool("CHAT_MEMORY_ENABLED", bool(settings["enabled"]))
    settings["history_window_tokens"] = _env_int(
        "CHAT_MEMORY_WINDOW_TOKENS", int(settings["history_window_tokens"])
    )
    settings["history_min_turns"] = _env_int("CHAT_MEMORY_MIN_TURNS", int(settings["history_min_turns"]))
    settings["history_max_turns"] = _env_int("CHAT_MEMORY_MAX_TURNS", int(settings["history_max_turns"]))
    return settings


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Four characters per token is close enough for
    budgeting English prose on every provider we use."""
    return len(text or "") // 4


def window_turn_count(total_turns: int, min_turns: int, max_turns: int) -> int:
    """How many of the newest turns the window holds for a conversation of
    total_turns turns.

    The window start moves in steps of (max_turns - min_turns), not one turn
    at a time. A window that slid by one turn each time would change the
    cached prefix on every turn once full, and the providers would re-bill
    the whole history each time. Stepping keeps the prefix stable for a run
    of turns, at the cost of the window briefly dropping to min_turns (the
    memory covers what fell out).

    Example with min 4 / max 12: turns 1..11 keep everything, turn 12 drops to
    the newest 4, then 5, 6, ... 11, then back to 4 at turn 20.
    """
    min_turns = max(1, min_turns)
    max_turns = max(min_turns, max_turns)
    if total_turns <= min_turns:
        return total_turns
    step = max(1, max_turns - min_turns)
    return min_turns + ((total_turns - min_turns) % step)


def select_history_window(
    rows: list[dict],
    budget_tokens: int,
    min_turns: int,
    max_turns: int,
    total_turns: Optional[int] = None,
    uncovered_rows: int = 0,
) -> list[dict]:
    """Pick the newest whole turns that fit the window and the budget.

    uncovered_rows: rows the negotiation memory has not folded yet. The
    window never drops below them, so a turn sent while a fold is still
    running keeps the verbatim turns instead of losing them for one reply.

    rows: chronological chat_messages rows (role, content), at least the
    newest max_turns turns. They are paired into (user, assistant) turns from
    the end. A trailing user row with no reply is dropped (turns are saved
    atomically, so this is defensive).

    total_turns: how many turns the whole conversation holds, which decides
    where the stepped window starts (see window_turn_count). Defaults to the
    turns present in rows.

    Within the window, the oldest turns are dropped while the token budget
    is exceeded, but never below min_turns.

    Returns [{"role": ..., "content": ...}, ...] oldest first.
    """
    if not rows or max_turns <= 0:
        return []

    turns: list[tuple[dict, dict]] = []
    i = len(rows) - 1
    while i >= 1:
        assistant, user = rows[i], rows[i - 1]
        if assistant.get("role") == "assistant" and user.get("role") == "user":
            turns.append((user, assistant))
            i -= 2
        else:
            # Out-of-pattern row (system message, orphan). Skip it.
            i -= 1
    if not turns:
        return []

    if total_turns is None or total_turns < len(turns):
        total_turns = len(turns)
    uncovered_turns = min((max(0, uncovered_rows) + 1) // 2, max_turns + _UNCOVERED_EXTRA_TURNS)
    keep = min(len(turns), max(window_turn_count(total_turns, min_turns, max_turns), uncovered_turns))
    floor_turns = max(min_turns, uncovered_turns)

    selected: list[tuple[dict, dict]] = []
    spent = 0
    for user, assistant in turns[:keep]:  # newest first
        cost = estimate_tokens(user.get("content", "")) + estimate_tokens(assistant.get("content", ""))
        if selected and len(selected) >= floor_turns and spent + cost > budget_tokens:
            break
        selected.append((user, assistant))
        spent += cost

    selected.reverse()
    window: list[dict] = []
    for user, assistant in selected:
        window.append({"role": "user", "content": user.get("content", "")})
        window.append({"role": "assistant", "content": assistant.get("content", "")})
    return window


def format_dropped_turns(rows: list[dict]) -> str:
    """Render rows as a plain transcript for the fold prompt, newest last."""
    lines = []
    # Every dropped row gets a share of the cap. Cutting the oldest rows
    # instead would mark them folded while the model never saw them.
    per_row_cap = max(800, _FOLD_CHAR_CAP // max(1, len(rows)))
    for row in rows:
        # Small models conflate advice with action, so each advisor line says
        # on its face that it went to the user, never to the counterparty.
        if row.get("role") == "user":
            speaker = "User (to advisor)"
        else:
            speaker = "Advisor (recommendation to the user, not sent to the counterparty)"
        lines.append(f"{speaker}: {_trim_middle((row.get('content') or '').strip(), per_row_cap)}")
    return "\n\n".join(lines)


def _trim_middle(text: str, cap: int) -> str:
    """Keep the head and tail of an oversize message so no turn vanishes."""
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return text[:head] + "\n[... trimmed ...]\n" + text[-tail:]


def build_memory_block(memory: Optional[dict]) -> str:
    """The memory as a prompt section, or "" when there is none yet."""
    summary = (memory or {}).get("summary_md") if isinstance(memory, dict) else None
    if not summary or not str(summary).strip():
        return ""
    return "## Negotiation Memory\n\n" + str(summary).strip()


def plan_memory_fold(total: int, covered: int, window_rows: int) -> Optional[tuple[int, int]]:
    """Decide whether a fold is due and which rows it covers.

    total: messages in the negotiation after the turn just saved.
    covered: messages the current memory already accounts for.
    window_rows: rows that are still inside the verbatim window.

    Returns (offset, limit) for get_negotiation_messages_range, or None when
    nothing has left the window since the last refresh.
    """
    fold_until = total - window_rows
    if fold_until <= covered:
        return None
    return covered, fold_until - covered


def refresh_memory(
    previous_summary: str,
    dropped_turns_text: str,
    briefing: str,
    llm: Any,
    max_chars: int = _DEFAULTS["memory_max_chars"],
) -> Optional[str]:
    """One model call that folds the dropped turns into the memory.

    Returns the new summary markdown, or None when the call fails or comes
    back empty so the caller keeps the old memory.
    """
    user_content = (
        f"## Briefing\n{briefing or '(none)'}\n\n"
        f"## Previous memory\n{previous_summary or '(none)'}\n\n"
        f"## New exchanges\n{dropped_turns_text or '(none)'}"
    )
    try:
        response = llm.invoke([
            {"role": "system", "content": _MEMORY_SYSTEM_PROMPT.format(max_chars=max_chars)},
            {"role": "user", "content": user_content},
        ])
    except Exception as exc:
        logger.warning("Memory refresh call failed: %s", exc)
        return None

    content = response.content if hasattr(response, "content") else str(response)
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("markdown"):
            content = content[len("markdown"):].strip()
    if not content:
        logger.warning("Memory refresh returned an empty summary")
        return None
    if len(content) > max_chars:
        content = content[:max_chars].rstrip()
    return content


# ------------------------------------------------------------------
# Background refresh (the only I/O in this module)
# ------------------------------------------------------------------

_inflight: set[UUID] = set()
_refresh_llm_warned = False


def get_refresh_llm(rag: Any, model: Optional[str] = None) -> Optional[Any]:
    """The fixed cheap model on the system key, or None when unavailable."""
    global _refresh_llm_warned
    try:
        return rag.model_config.create_llm(MEMORY_REFRESH_PROVIDER, model or MEMORY_REFRESH_MODEL)
    except Exception as exc:
        if not _refresh_llm_warned:
            logger.warning(
                "Memory refresh disabled: cannot create %s/%s (%s)",
                MEMORY_REFRESH_PROVIDER, model or MEMORY_REFRESH_MODEL, exc,
            )
            _refresh_llm_warned = True
        return None


async def next_window_rows(conversation_id: UUID, settings: dict) -> int:
    """How many rows the NEXT turn's verbatim window will hold.

    Computed after the turn is saved, over the same rows and budget the chat
    route will use, so the fold covers exactly what the next turn will no
    longer see. Using the window that was just sent would lag one turn.
    """
    window = await load_history_window(conversation_id, settings)
    return len(window)


async def load_history_window(
    conversation_id: UUID,
    settings: dict,
    negotiation_id: Optional[UUID] = None,
    memory: Optional[dict] = None,
) -> list[dict]:
    """The verbatim window for the next turn of a conversation, from the DB.

    With negotiation_id and memory, rows the memory has not folded yet stay in
    the window (the fold runs in the background and can lose a race with a
    quick follow-up). The fold planner calls this without them so its window
    is the plain stepped one; otherwise the extension would hide the very rows
    it should fold.
    """
    from . import db_operations as db_ops

    max_turns = int(settings["history_max_turns"])
    total_rows = await db_ops.count_conversation_messages(conversation_id)

    uncovered_rows = 0
    if negotiation_id is not None:
        covered = int((memory or {}).get("source_message_count") or 0) if isinstance(memory, dict) else 0
        negotiation_rows = await db_ops.count_negotiation_messages(negotiation_id)
        uncovered_rows = min(total_rows, max(0, negotiation_rows - covered), (max_turns + _UNCOVERED_EXTRA_TURNS) * 2)

    rows = await db_ops.get_recent_conversation_messages(
        conversation_id, limit=max(max_turns * 2, uncovered_rows)
    )
    return select_history_window(
        rows,
        budget_tokens=int(settings["history_window_tokens"]),
        min_turns=int(settings["history_min_turns"]),
        max_turns=max_turns,
        total_turns=total_rows // 2,
        uncovered_rows=uncovered_rows,
    )


async def maybe_refresh_memory(
    negotiation_id: UUID,
    conversation_id: UUID,
    briefing: str,
    rag: Any,
) -> bool:
    """Fold turns that have left the verbatim window into the negotiation memory.

    Runs after the chat response has been sent. Every failure is logged and
    swallowed: a missing key or a bad model reply must never surface as a
    chat error. Returns True when the memory was updated.
    """
    from . import db_operations as db_ops

    if negotiation_id in _inflight:
        logger.info("Memory refresh already running for %s, skipping", negotiation_id)
        return False
    _inflight.add(negotiation_id)
    try:
        settings = get_memory_settings()
        negotiation = await db_ops.get_negotiation(negotiation_id)
        if not negotiation:
            return False
        memory = negotiation.get("memory") if isinstance(negotiation.get("memory"), dict) else {}
        covered = int(memory.get("source_message_count") or 0)
        total = await db_ops.count_negotiation_messages(negotiation_id)
        window_rows = await next_window_rows(conversation_id, settings)

        plan = plan_memory_fold(total, covered, window_rows)
        if plan is None:
            return False
        offset, limit = plan

        llm = get_refresh_llm(rag, settings.get("memory_refresh_model"))
        if llm is None:
            return False

        dropped = await db_ops.get_negotiation_messages_range(negotiation_id, offset, limit)
        if not dropped:
            return False

        new_summary = await asyncio.to_thread(
            refresh_memory,
            memory.get("summary_md") or "",
            format_dropped_turns(dropped),
            briefing,
            llm,
            int(settings.get("memory_max_chars") or _DEFAULTS["memory_max_chars"]),
        )
        if not new_summary:
            return False

        # Re-read before writing so a slower fold never overwrites a newer one.
        latest = await db_ops.get_negotiation(negotiation_id)
        latest_memory = (latest or {}).get("memory") or {}
        if int(latest_memory.get("source_message_count") or 0) > covered:
            logger.info("Memory for %s moved on during refresh, discarding fold", negotiation_id)
            return False

        await db_ops.update_negotiation(
            negotiation_id,
            memory={
                "summary_md": new_summary,
                "source_message_count": offset + limit,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "model": f"{MEMORY_REFRESH_PROVIDER}/{settings.get('memory_refresh_model') or MEMORY_REFRESH_MODEL}",
            },
        )
        logger.info(
            "Memory refreshed for %s: folded %d messages (now covers %d of %d)",
            negotiation_id, limit, offset + limit, total,
        )
        return True
    except Exception as exc:
        logger.error("Memory refresh failed for %s: %s", negotiation_id, exc, exc_info=True)
        return False
    finally:
        _inflight.discard(negotiation_id)
