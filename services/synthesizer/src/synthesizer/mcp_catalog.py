"""Declarative catalog of MCP servers + functions the runner can call.

This module is the single source of truth for *which* MCP-tier execution
hints the synthesizer is allowed to emit. The catalog is hand-curated for
the launch set of integrations (Gmail, Google Calendar, Google Drive,
Slack, Notion, Linear, GitHub via Atlassian/Asana/etc.) so the LLM never
hallucinates a function name that doesn't exist on the wire.

The catalog has two consumers:

* :func:`format_for_prompt` — emits a markdown-style reference block that
  the synth's draft prompt embeds verbatim, so the model sees exactly
  what the runner sees.
* :func:`validate_hint` — verifies a synth-emitted ``execution_hint``
  whose ``tier == "mcp"`` against the catalog, rejecting unknown servers,
  unknown functions, or arguments outside the declared signature. This is
  belt-and-suspenders against prompt drift.

Adding a new server is a one-spot change: drop a new key into
:data:`MCP_CATALOG` with the function signatures and a one-line
description per function. Bump the schema version (top of the dict) when
making a breaking change so the runner can pick the right interpreter.

The catalog deliberately stays narrower than the *full* set of functions
each MCP exposes — only the ones that map cleanly onto the kinds of
workflows Trace synthesizes (read, draft, send, schedule, status). If a
workflow needs a function that isn't in the catalog, the synthesizer
falls back to ``browser_dom`` or ``computer_use`` tiers; we'd rather have
a runnable fallback than a broken MCP call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "CATALOG_VERSION",
    "MCP_CATALOG",
    "FunctionSpec",
    "ServerSpec",
    "format_for_prompt",
    "lookup_function",
    "validate_hint",
]

CATALOG_VERSION = 1


# A function signature: every required arg is declared with its primitive
# type ("string"/"integer"/"boolean") and a one-line description. Optional
# args go in ``optional`` with the same shape. Keep descriptions short —
# the prompt cost adds up fast.
FunctionSpec = Mapping[str, Any]
ServerSpec = Mapping[str, FunctionSpec]


MCP_CATALOG: dict[str, dict[str, FunctionSpec]] = {
    # ----------------------------------------------------------------- gmail
    "gmail": {
        "search_threads": {
            "description": "Find email threads matching a query. Use to locate "
            "the email a workflow is acting on.",
            "args": {"query": "string"},
            "optional": {"max_results": "integer"},
        },
        "get_thread": {
            "description": "Fetch the full content of a specific email thread.",
            "args": {"thread_id": "string"},
        },
        "create_draft": {
            "description": "Create a new draft email (does NOT send).",
            "args": {"to": "string", "subject": "string", "body": "string"},
            "optional": {"cc": "string", "bcc": "string"},
        },
        "send_draft": {
            "description": "Send a previously-created draft. Destructive: "
            "irreversible once invoked.",
            "args": {"draft_id": "string"},
        },
        "reply_to_thread": {
            "description": "Reply to an existing email thread. Destructive when "
            "send=true.",
            "args": {"thread_id": "string", "body": "string"},
            "optional": {"send": "boolean"},
        },
    },
    # ----------------------------------------------------- google calendar
    "google_calendar": {
        "list_calendars": {
            "description": "Return the user's available calendars.",
            "args": {},
        },
        "list_events": {
            "description": "List events on a calendar between two timestamps.",
            "args": {
                "calendar_id": "string",
                "time_min": "string",
                "time_max": "string",
            },
        },
        "create_event": {
            "description": "Create a new calendar event. Destructive when the "
            "user has invitees.",
            "args": {
                "calendar_id": "string",
                "summary": "string",
                "start": "string",
                "end": "string",
            },
            "optional": {"description": "string", "attendees": "string"},
        },
        "update_event": {
            "description": "Update fields of an existing event.",
            "args": {"calendar_id": "string", "event_id": "string"},
            "optional": {"summary": "string", "start": "string", "end": "string"},
        },
        "delete_event": {
            "description": "Delete a calendar event. Destructive: irreversible.",
            "args": {"calendar_id": "string", "event_id": "string"},
        },
    },
    # ------------------------------------------------------------- slack
    "slack": {
        "post_message": {
            "description": "Post a message to a Slack channel or DM. "
            "Destructive: visible to the channel immediately.",
            "args": {"channel": "string", "text": "string"},
            "optional": {"thread_ts": "string"},
        },
        "set_status": {
            "description": "Set the user's Slack status (text + emoji + expiry).",
            "args": {"text": "string", "emoji": "string"},
            "optional": {"expiration_minutes": "integer"},
        },
        "search_messages": {
            "description": "Search messages by query.",
            "args": {"query": "string"},
        },
    },
    # ------------------------------------------------------------- notion
    "notion": {
        "search": {
            "description": "Search Notion pages and databases by query.",
            "args": {"query": "string"},
        },
        "create_page": {
            "description": "Create a new page in a parent (workspace, page, or "
            "database).",
            "args": {"parent_id": "string", "title": "string"},
            "optional": {"content_markdown": "string"},
        },
        "append_to_page": {
            "description": "Append blocks to an existing page.",
            "args": {"page_id": "string", "content_markdown": "string"},
        },
    },
    # ------------------------------------------------- google drive (lite)
    "google_drive": {
        "search_files": {
            "description": "Search Drive files by name or content.",
            "args": {"query": "string"},
        },
        "create_file": {
            "description": "Create a new file in Drive (typically a Doc).",
            "args": {"name": "string", "mime_type": "string"},
            "optional": {"parent_folder_id": "string", "content": "string"},
        },
    },
}


def lookup_function(server: str, function: str) -> FunctionSpec | None:
    """Return the function spec, or ``None`` if either name is unknown."""
    server_spec = MCP_CATALOG.get(server)
    if server_spec is None:
        return None
    return server_spec.get(function)


def validate_hint(
    *, server: str, function: str, arguments: Mapping[str, Any]
) -> str | None:
    """Verify a single ``tier == "mcp"`` hint against the catalog.

    Returns ``None`` on success, or a short human-readable error message
    suitable for feeding back to the LLM as a corrective user turn.

    Checks (in order):

    1. ``server`` is a known catalog key.
    2. ``function`` is declared on that server.
    3. Every required arg from the function spec is present in
       ``arguments``.
    4. No unknown args appear (typo guard).
    """
    spec = lookup_function(server, function)
    if spec is None:
        known_servers = ", ".join(sorted(MCP_CATALOG.keys()))
        if server not in MCP_CATALOG:
            return (
                f"unknown MCP server {server!r}. "
                f"Catalog servers: {known_servers}."
            )
        known_fns = ", ".join(sorted(MCP_CATALOG[server].keys()))
        return (
            f"server {server!r} has no function {function!r}. "
            f"Available functions: {known_fns}."
        )

    required = dict(spec.get("args", {}))
    optional = dict(spec.get("optional", {}))
    declared = set(required) | set(optional)

    missing = [k for k in required if k not in arguments]
    if missing:
        return (
            f"{server}.{function} is missing required argument(s): "
            f"{', '.join(missing)}. Required: {sorted(required)}."
        )

    extra = [k for k in arguments if k not in declared]
    if extra:
        return (
            f"{server}.{function} got unknown argument(s): "
            f"{', '.join(extra)}. Declared args: {sorted(declared)}."
        )

    return None


def format_for_prompt() -> str:
    """Return a markdown-flavoured catalog reference for the system prompt.

    Format is stable so the prompt-engineering work upstream stays
    reproducible. Each server gets a level-3 heading; each function gets
    a bulleted entry with required and optional arg lists.
    """
    lines: list[str] = ["## Available MCP servers", ""]
    for server in sorted(MCP_CATALOG):
        lines.append(f"### {server}")
        for fn_name in sorted(MCP_CATALOG[server]):
            spec = MCP_CATALOG[server][fn_name]
            required = dict(spec.get("args", {}))
            optional = dict(spec.get("optional", {}))
            arg_str = ", ".join(f"{k}: {v}" for k, v in required.items()) or "(none)"
            opt_str = (
                ", ".join(f"{k}?: {v}" for k, v in optional.items())
                if optional
                else ""
            )
            sig_tail = f"; optional: {opt_str}" if opt_str else ""
            lines.append(f"- **{fn_name}**({arg_str}{sig_tail})")
            lines.append(f"  - {spec.get('description', '').strip()}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
