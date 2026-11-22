"""
Mode-aware prompt manager.

Three-layer prompt architecture for the Amfonica platform:

  Layer 1 — Meta:  prompts/amfonica-meta.md
                   Always-applied identity and output rules.
  Layer 2 — Persona (per mode):
                   prompts/sales-mentor.yaml         (mode=sales)
                   prompts/negotiation-strategist.yaml (mode=negotiation)
                   No persona is applied for mode=auto.
  Layer 3 — Negotiation block (per negotiation):
                   briefing + negotiation memory. Stable between memory
                   refreshes, so it sits in the cached prefix.
  Layer 4 — Context + Question (per request):
                   RAG-retrieved chunks, format instructions and the user's
                   question. These change every turn, so they go at the tail
                   of the final user message, after the history.

Test/ping detection and the user-template plumbing are unchanged from the
previous PromptManager — connectivity checks continue to bypass persona logic.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import yaml

from .intent_classifier import (
    FOLLOWUP_GENERAL_TEMPLATE,
    FOLLOWUP_NOTE,
    FORMAT_INSTRUCTIONS,
    USER_TEMPLATES,
    QueryIntent,
)
from .prompt_renderer import render_yaml_to_markdown

logger = logging.getLogger(__name__)

# The negotiation block shares the cached prefix with the history. Past this
# size it starts crowding out the verbatim window, so trim it (memory first,
# since the briefing is the user's own words about the parties).
NEGOTIATION_BLOCK_MAX_CHARS = 6000


class PromptParts(NamedTuple):
    """The four pieces get_advice assembles into messages.

    static_system:     meta + persona. Identical for every request in a mode.
    negotiation_block: briefing + memory. Identical for every turn of a
                       negotiation until the memory refreshes.
    context_block:     reference material + (follow-up note) + format
                       instructions. Changes every turn.
    user_prompt:       the question wrapped in the intent's user template.
    """

    static_system: str
    negotiation_block: str
    context_block: str
    user_prompt: str

# Test/ping patterns — short messages that should bypass RAG and persona load.
TEST_PROMPT_PATTERNS = [
    r"^hello\s*world[!?.]?$",
    r"^(test\s*)+[!?.]?$",
    r"^testing[\s\d]*[!?.]?$",
    r"^are\s+you\s+there[!?.]?$",
    r"^hey[!?.]?$",
    r"^hi[!?.]?$",
    r"^hello[!?.]?$",
    r"^ping[!?.]?$",
    r"^is\s+(this|it)\s+(working|on)[!?.]?$",
    r"^can\s+you\s+hear\s+me[!?.]?$",
    r"^yo[!?.]?$",
]
TEST_PROMPT_MAX_LENGTH = 200

# Layout under the project root
_DEFAULT_PROMPTS_DIR = "prompts"
_DEFAULT_CONFIG_DIR = ".config"
_META_FILENAME = "amfonica-meta.md"
_PERSONA_FILES = {
    "sales": "sales-mentor.yaml",
    "negotiation": "negotiation-strategist.yaml",
}
_VALID_MODES = {"auto", "sales", "negotiation"}

_DEFAULT_USER_TEMPLATE = (
    "Please analyse this situation and provide expert guidance:\n"
    "\n"
    "{question}\n"
    "\n"
    "Draw on the reference material above and your operating principles to give a"
    " specific, actionable response following the output structure you've been"
    " told to use."
)

_DEFAULT_META_PROMPT = (
    "# Amfonica Advisor\n"
    "\n"
    "You are an Amfonica advisor with deep experience in sales, negotiation,"
    " and deal-making. Be direct, evidence-based, and actionable. Use markdown"
    " for structured output.\n"
)


class PromptManager:
    """
    Loads and serves the three-layer prompt stack.

    - `amfonica-meta.md` is read fresh on each call so admin-UI edits take
      effect immediately without a restart. Personae are parsed and rendered
      once at init (edits require a restart, by design).
    - `get_prompts_for_chat(question, context, mode)` is the only call the
      RAG engine should need.
    """

    def __init__(
        self,
        prompts_dir: str = _DEFAULT_PROMPTS_DIR,
        config_dir: str = _DEFAULT_CONFIG_DIR,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.config_dir = Path(config_dir)
        self.meta_file = self.prompts_dir / _META_FILENAME
        self.config_file = self.config_dir / "prompts.json"

        # Ensure dirs exist (helpful in fresh checkouts and tests)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Personae are rendered once at init.
        self._persona_markdown: Dict[str, str] = {}
        self._load_personae()

        # User template lives in .config/prompts.json (legacy location).
        # Keep that for now so edits to the user-side prompt are still possible
        # without a code change.
        self._user_template = self._load_user_template()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_personae(self) -> None:
        """Read each persona YAML and render to markdown. Cache in memory."""
        for mode, filename in _PERSONA_FILES.items():
            path = self.prompts_dir / filename
            if not path.exists():
                logger.warning(
                    "Persona file missing: %s — mode=%s will fall back to meta only.",
                    path, mode,
                )
                self._persona_markdown[mode] = ""
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self._persona_markdown[mode] = render_yaml_to_markdown(data)
                logger.info(
                    "Loaded persona %s from %s (%d chars rendered).",
                    mode, path, len(self._persona_markdown[mode]),
                )
            except Exception as exc:
                logger.error("Failed to load persona %s: %s", path, exc)
                self._persona_markdown[mode] = ""

    def _load_meta(self) -> str:
        """Read the meta prompt fresh from disk (no caching). Falls back to default."""
        if self.meta_file.exists():
            try:
                return self.meta_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Could not read meta prompt %s: %s", self.meta_file, exc)
        # File missing or unreadable — write the default so admin-UI edits stick
        try:
            self.meta_file.write_text(_DEFAULT_META_PROMPT, encoding="utf-8")
            logger.info("Wrote default meta prompt to %s", self.meta_file)
        except Exception as exc:
            logger.error("Could not write default meta prompt: %s", exc)
        return _DEFAULT_META_PROMPT

    def _load_user_template(self) -> str:
        """Load the user-side template from .config/prompts.json (legacy)."""
        if self.config_file.exists():
            try:
                data = json.loads(self.config_file.read_text(encoding="utf-8"))
                template = data.get("user")
                if template:
                    return template
            except Exception as exc:
                logger.error("Could not read user template: %s", exc)
        return _DEFAULT_USER_TEMPLATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prompt_parts(
        self,
        question: str,
        context: str = "",
        mode: str = "auto",
        intent: QueryIntent = "GENERAL",
        briefing: str = "",
        memory: str = "",
        has_history: bool = False,
    ) -> PromptParts:
        """
        Return the PromptParts for cache-friendly message assembly.

        - static_system = <meta> + <persona for mode>. Byte-identical across
          requests in a given mode, so it is the cacheable prefix.
        - negotiation_block = briefing + memory. Stable per negotiation, so it
          extends the cached prefix rather than breaking it.
        - context_block = RAG material + (follow-up note) + format
          instructions. Varies per request, so get_advice puts it in the
          final user message after the history.
        - user_prompt = the intent-appropriate user template with the
          question filled. On follow-ups a GENERAL question goes as typed.

        Callers that don't care about caching can use get_prompts_for_chat,
        which simply concatenates the system-side parts.
        """
        if mode not in _VALID_MODES:
            logger.warning("Unknown mode %r — falling back to 'auto'.", mode)
            mode = "auto"

        static_parts: list[str] = [self._load_meta().rstrip()]
        persona = self._persona_markdown.get(mode, "")
        if persona:
            static_parts.append(persona.rstrip())
        static_system = "\n\n".join(static_parts).rstrip() + "\n"

        negotiation_block = self._build_negotiation_block(briefing, memory)

        context_parts: list[str] = []
        if context:
            context_parts.append("## Reference Material from Knowledge Base\n\n" + context)
        if has_history:
            context_parts.append(FOLLOWUP_NOTE)
        context_parts.append(FORMAT_INSTRUCTIONS.get(intent, FORMAT_INSTRUCTIONS["GENERAL"]))
        context_block = "\n\n".join(context_parts)

        if has_history and intent == "GENERAL":
            template = FOLLOWUP_GENERAL_TEMPLATE
        else:
            template = USER_TEMPLATES.get(intent, self._user_template)
        user_prompt = template.format(question=question)
        return PromptParts(static_system, negotiation_block, context_block, user_prompt)

    @staticmethod
    def _build_negotiation_block(briefing: str, memory: str) -> str:
        """Join briefing and memory, trimming memory first if the pair is oversized."""
        briefing = (briefing or "").strip()
        memory = (memory or "").strip()
        if not briefing and not memory:
            return ""
        if len(briefing) + len(memory) > NEGOTIATION_BLOCK_MAX_CHARS:
            room_for_memory = max(0, NEGOTIATION_BLOCK_MAX_CHARS - len(briefing))
            logger.warning(
                "Negotiation block oversized (briefing=%d, memory=%d chars); trimming memory to %d",
                len(briefing), len(memory), room_for_memory,
            )
            memory = memory[:room_for_memory].rstrip()
            if len(briefing) > NEGOTIATION_BLOCK_MAX_CHARS:
                briefing = briefing[:NEGOTIATION_BLOCK_MAX_CHARS].rstrip()
        return "\n\n".join(part for part in (briefing, memory) if part)

    def get_prompts_for_chat(
        self,
        question: str,
        context: str = "",
        mode: str = "auto",
        intent: QueryIntent = "GENERAL",
    ) -> Tuple[str, str]:
        """
        Build the per-request system and user prompts as a single concatenated
        system string (meta + persona + reference material). Kept for callers
        that don't split the prompt for provider caching.
        """
        parts = self.get_prompt_parts(question, context, mode, intent)
        if parts.context_block:
            system_prompt = f"{parts.static_system.rstrip()}\n\n{parts.context_block}"
        else:
            system_prompt = parts.static_system
        return system_prompt.rstrip() + "\n", parts.user_prompt

    def get_system_prompt(self, context: str = "", mode: str = "auto") -> str:
        """Return only the system half of the prompt pair."""
        system, _ = self.get_prompts_for_chat(question="", context=context, mode=mode)
        return system

    def get_user_prompt(self, question: str) -> str:
        """Return only the user half (template with {question} filled)."""
        return self._user_template.format(question=question)

    # ------------------------------------------------------------------
    # Admin-facing (meta prompt edits)
    # ------------------------------------------------------------------

    def get_raw_prompts(self) -> Dict[str, str]:
        """
        Used by the admin SystemPromptEditor. The 'system' key returns the
        editable meta prompt (the only system-side text designed for live
        editing); 'user' returns the user template.
        """
        return {
            "system": self._load_meta(),
            "user": self._user_template,
        }

    def update_system_prompt(self, new_prompt: str) -> None:
        """Persist a new meta prompt. Persona YAMLs are not touched."""
        self.meta_file.write_text(new_prompt, encoding="utf-8")
        logger.info("Updated meta prompt at %s", self.meta_file)

    def update_user_prompt(self, new_prompt: str) -> None:
        """Persist a new user-side template."""
        self._user_template = new_prompt
        data = {"user": new_prompt}
        if self.config_file.exists():
            try:
                existing = json.loads(self.config_file.read_text(encoding="utf-8"))
                existing["user"] = new_prompt
                # Drop the legacy 'system' key — meta lives in amfonica-meta.md now.
                existing.pop("system", None)
                data = existing
            except Exception:
                pass
        self.config_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def reload_personae(self) -> None:
        """Force a re-read of the persona YAMLs (no restart). Rarely needed."""
        self._load_personae()

    # ------------------------------------------------------------------
    # Debug / introspection
    # ------------------------------------------------------------------

    def get_prompt_info(self) -> dict:
        meta = self._load_meta()
        return {
            "meta_file": str(self.meta_file),
            "meta_exists": self.meta_file.exists(),
            "meta_length": len(meta),
            "personae": {
                mode: {
                    "file": str(self.prompts_dir / fname),
                    "loaded": bool(self._persona_markdown.get(mode)),
                    "length": len(self._persona_markdown.get(mode, "")),
                }
                for mode, fname in _PERSONA_FILES.items()
            },
            "user_template_length": len(self._user_template),
        }

    def validate_prompts(self) -> Dict[str, bool]:
        """Lightweight self-check for admin UIs."""
        return {
            "meta_present": bool(self._load_meta().strip()),
            "sales_persona_loaded": bool(self._persona_markdown.get("sales")),
            "negotiation_persona_loaded": bool(self._persona_markdown.get("negotiation")),
            "user_template_has_question": "{question}" in self._user_template,
        }

    # ------------------------------------------------------------------
    # Test/ping detection (unchanged from previous implementation)
    # ------------------------------------------------------------------

    def is_test_prompt(self, question: str) -> bool:
        stripped = question.strip()
        if len(stripped) > TEST_PROMPT_MAX_LENGTH:
            return False
        normalised = stripped.lower()
        if len(stripped) < 50 and "test" in normalised:
            logger.info("Detected test prompt (keyword): %r", stripped)
            return True
        for pattern in TEST_PROMPT_PATTERNS:
            if re.match(pattern, normalised, re.IGNORECASE):
                logger.info("Detected test prompt (pattern): %r", stripped)
                return True
        return False

    def get_test_system_prompt(self) -> str:
        return (
            "You are an Amfonica advisor. The user is testing the connection."
            " Respond briefly confirming you're online and ready to help with"
            " sales and negotiation. Keep it friendly and under 2-3 sentences."
        )

    def get_test_user_prompt(self, question: str) -> str:
        return f'User said: "{question}"\n\nConfirm the connection is working and you\'re ready to help.'
