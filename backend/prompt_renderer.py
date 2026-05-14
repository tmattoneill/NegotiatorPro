"""
YAML → Markdown renderer for persona prompts.

The persona YAMLs (prompts/sales-mentor.yaml, prompts/negotiation-strategist.yaml)
are structured config dicts. To feed them to an LLM as a system prompt, we walk
the structure and emit markdown:

  - dict keys become headers (depth-aware: h1 at top level, h2 one level in, ...)
  - lists become bullet points
  - leaf strings are emitted inline
  - dicts whose values are dicts/lists with a 'description', 'purpose', or 'note'
    key get rendered as a labelled block

Pure function. No I/O. The caller is responsible for loading the YAML.
"""
from typing import Any


# Markdown caps at h6; anything deeper falls back to bold-bullet.
_MAX_HEADER_DEPTH = 6


def _humanise(key: str) -> str:
    """Convert snake_case_key → Title Case Words."""
    return key.replace("_", " ").strip().title()


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _render_value(value: Any, depth: int, lines: list[str]) -> None:
    """Render a value at the given heading depth, appending to `lines`."""
    if _is_scalar(value):
        lines.append(str(value))
        lines.append("")
        return

    if isinstance(value, list):
        _render_list(value, depth, lines)
        return

    if isinstance(value, dict):
        _render_dict(value, depth, lines)
        return

    lines.append(repr(value))
    lines.append("")


def _render_list(items: list, depth: int, lines: list[str]) -> None:
    """Render a list. Scalars become bullets; dicts get sub-headers."""
    for item in items:
        if _is_scalar(item):
            lines.append(f"- {item}")
        elif isinstance(item, dict):
            # A dict inside a list: render its keys as a labelled bullet block.
            # If the dict has a recognisable 'name'/'question'/'objection' field,
            # use that as the bullet title.
            title_key = next(
                (k for k in ("name", "question", "objection", "phrase", "step")
                 if k in item),
                None,
            )
            if title_key:
                lines.append(f"- **{item[title_key]}**")
                for k, v in item.items():
                    if k == title_key:
                        continue
                    if _is_scalar(v):
                        lines.append(f"  - *{_humanise(k)}*: {v}")
                    else:
                        # Nested non-scalar inside a list item — flatten one level
                        lines.append(f"  - *{_humanise(k)}*:")
                        _render_value(v, depth + 1, lines)
            else:
                # No obvious title — emit as a sub-block
                for k, v in item.items():
                    if _is_scalar(v):
                        lines.append(f"- *{_humanise(k)}*: {v}")
                    else:
                        lines.append(f"- *{_humanise(k)}*:")
                        _render_value(v, depth + 1, lines)
        elif isinstance(item, list):
            # Nested list — render inline-ish
            for sub in item:
                if _is_scalar(sub):
                    lines.append(f"- {sub}")
    lines.append("")


def _render_dict(data: dict, depth: int, lines: list[str]) -> None:
    """Render a dict. Each key gets a header at this depth."""
    for key, value in data.items():
        header_level = min(depth, _MAX_HEADER_DEPTH)
        if header_level <= _MAX_HEADER_DEPTH:
            lines.append(f"{'#' * header_level} {_humanise(key)}")
            lines.append("")

        _render_value(value, depth + 1, lines)


def render_yaml_to_markdown(data: dict) -> str:
    """
    Render a parsed YAML dict into a markdown string suitable for use as a
    system prompt.

    Args:
        data: Parsed YAML (e.g. the dict from `yaml.safe_load(open(path))`).
              May be wrapped in a single top-level key (the persona identifier)
              or flat. If wrapped and the top-level has exactly one key, the
              top key is used as h1 and its body is rendered beneath.

    Returns:
        Markdown string. Always ends with no trailing whitespace.
    """
    if not isinstance(data, dict) or not data:
        return ""

    lines: list[str] = []

    # If the dict has a single top-level key, treat that as the document title.
    if len(data) == 1:
        only_key = next(iter(data))
        body = data[only_key]
        lines.append(f"# {_humanise(only_key)}")
        lines.append("")
        if isinstance(body, dict):
            _render_dict(body, depth=2, lines=lines)
        else:
            _render_value(body, depth=2, lines=lines)
    else:
        _render_dict(data, depth=1, lines=lines)

    # Collapse runs of blank lines and trim
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).rstrip() + "\n"
