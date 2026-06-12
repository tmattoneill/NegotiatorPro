#!/usr/bin/env python3
"""PostToolUse hook: mirror every new devctx todo into Linear.

Fires on `mcp__devctx__devctx_todo_add`. Reads the hook JSON on stdin, then
creates a Linear issue in the "Amfonic (NegotiatorPro)" project (devctx's own
Linear sync ignores projectId and dumps issues in the team root, so we create
them directly here with the project pinned).

Non-blocking by contract: always exits 0, never raises out. On success it prints
a small {"systemMessage": ...} so the user sees the issue id; on any failure it
stays silent (a missing API key or network blip must not disrupt the todo add).

Note: with this hook active, do NOT also run `devctx_linear_sync push` — it would
create a second issue for the same todo (devctx never learns the hook-made id).
"""
import sys
import os
import re
import json
import urllib.request

TEAM_ID = "937975d8-e666-42fd-ac60-f282f26fd81c"          # teemo-personal-projects (TEEPER)
PROJECT_ID = "9c798975-04fa-41b7-9e53-80bc68c52d0f"        # Amfonic (NegotiatorPro)
PRIORITY_MAP = {"critical": 1, "high": 2, "medium": 3, "low": 4}


def emit(obj):
    """Write a hook JSON response and exit cleanly."""
    try:
        sys.stdout.write(json.dumps(obj))
    except Exception:
        pass
    sys.exit(0)


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        emit({})

    if data.get("tool_name") != "mcp__devctx__devctx_todo_add":
        emit({})

    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        emit({})  # silently skip when the key isn't in the environment

    tool_input = data.get("tool_input") or {}
    text = (tool_input.get("text") or "").strip()
    if not text:
        emit({})

    priority = PRIORITY_MAP.get(str(tool_input.get("priority") or "medium").lower(), 3)
    tags = tool_input.get("tags") or []

    # Recover the devctx todo id from the tool response (shape varies, so stringify).
    resp = data.get("tool_response")
    resp_str = resp if isinstance(resp, str) else json.dumps(resp)
    m = re.search(r"todo_[0-9a-f]{6,}", resp_str or "")
    todo_id = m.group(0) if m else None

    title = text if len(text) <= 250 else text[:247] + "..."
    description = text
    footer = []
    if todo_id:
        footer.append(f"devctx todo: `{todo_id}`")
    if tags:
        footer.append("tags: " + ", ".join(str(t) for t in tags))
    if footer:
        description += "\n\n---\n" + "\n".join(footer)

    mutation = (
        "mutation IssueCreate($input: IssueCreateInput!) {"
        "  issueCreate(input: $input) { success issue { identifier url } } }"
    )
    payload = {
        "query": mutation,
        "variables": {
            "input": {
                "teamId": TEAM_ID,
                "projectId": PROJECT_ID,
                "title": title,
                "description": description,
                "priority": priority,
            }
        },
    }

    try:
        req = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            out = json.load(r)
        issue = (((out or {}).get("data") or {}).get("issueCreate") or {}).get("issue")
        if issue:
            emit({"systemMessage": f"Linear: created {issue['identifier']} in Amfonic (NegotiatorPro)"})
    except Exception:
        pass  # best-effort; never disrupt the todo add

    emit({})


if __name__ == "__main__":
    main()
