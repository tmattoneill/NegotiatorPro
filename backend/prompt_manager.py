"""
Mode-aware prompt manager.

Three-layer prompt architecture for the Amfonica platform:

  Layer 1 — Meta:  prompts/amfonica-meta.md
                   Always-applied identity and output rules.
  Layer 2 — Persona (per mode):
                   prompts/sales-mentor.yaml         (mode=sales)
                   prompts/negotiation-strategist.yaml (mode=negotiation)
                   No persona is applied for mode=auto.
  Layer 3 — Context + Question:
                   RAG-retrieved chunks and the user's question, injected per
                   request.

Test/ping detection and the user-template plumbing are unchanged from the
previous PromptManager — connectivity checks continue to bypass persona logic.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from .prompt_renderer import render_yaml_to_markdown

logger = logging.getLogger(__name__)

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

    def get_prompts_for_chat(
        self,
        question: str,
        context: str = "",
        mode: str = "auto",
    ) -> Tuple[str, str]:
        """
        Build the per-request system and user prompts.

        Layers concatenated into the system prompt:
            <meta>
            <persona for mode, if any>
            ## Reference Material from Knowledge Base
            <context>
        """
        if mode not in _VALID_MODES:
            logger.warning("Unknown mode %r — falling back to 'auto'.", mode)
            mode = "auto"

        parts: list[str] = [self._load_meta().rstrip()]

        persona = self._persona_markdown.get(mode, "")
        if persona:
            parts.append(persona.rstrip())

        if context:
            parts.append("## Reference Material from Knowledge Base\n\n" + context)

        system_prompt = "\n\n".join(parts).rstrip() + "\n"
        user_prompt = self._user_template.format(question=question)
        return system_prompt, user_prompt

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
