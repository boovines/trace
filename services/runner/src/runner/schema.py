"""Run metadata schema + pydantic model.

The JSON schema lives at ``contracts/run-metadata.schema.json`` and is the
locked on-disk contract consumed by the UI and by post-run analysis. This
module exposes three affordances:

* ``load_run_metadata_schema()`` — read the JSON schema into a dict.
* ``validate_run_metadata(obj)`` — raise ``jsonschema.ValidationError`` on
  drift between a dict and the schema.
* ``RunMetadata`` — a pydantic model mirroring the schema for ergonomic
  in-memory use, with ``from_dict`` / ``to_dict`` round-trip and
  ``is_terminal`` / ``is_waiting`` status helpers.

Keeping both a JSON schema AND a pydantic model is deliberate: disk artifacts
are validated against the schema (so hand-edited files fail loudly), while
runtime code uses the typed model so mypy can catch field-name typos.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal[
    "pending",
    "running",
    "awaiting_confirmation",
    "succeeded",
    "failed",
    "aborted",
    "budget_exceeded",
    "rate_limited",
]

RunMode = Literal["execute", "dry_run"]

TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"succeeded", "failed", "aborted", "budget_exceeded", "rate_limited"}
)
WAITING_STATUS: Final[str] = "awaiting_confirmation"

_SCHEMA_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "contracts" / "run-metadata.schema.json"
)


@lru_cache(maxsize=1)
def load_run_metadata_schema() -> dict[str, Any]:
    """Return the parsed JSON schema for run_metadata.json.

    Cached because the schema is immutable at runtime and read from disk on
    every artifact write would add needless I/O.
    """
    schema: dict[str, Any] = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_run_metadata_schema()
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def validate_run_metadata(obj: object) -> None:
    """Validate a dict against the run-metadata schema.

    Raises ``jsonschema.ValidationError`` (or ``jsonschema.SchemaError`` for a
    malformed schema, which shouldn't happen in prod since the schema is
    version-controlled). The caller is expected to catch ValidationError and
    surface it with context about which file/run failed.
    """
    _validator().validate(obj)


class RunMetadata(BaseModel):
    """In-memory representation of run_metadata.json.

    ``extra="forbid"`` mirrors the schema's ``additionalProperties: false`` so
    a typo like ``cost_usd`` instead of ``total_cost_usd`` fails loudly at
    model-construction time rather than silently dropping the field.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    skill_slug: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime | None
    status: RunStatus
    mode: RunMode

    parameters: dict[str, str] | None = None
    final_step_reached: int | None = Field(default=None, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0)
    abort_reason: str | None = None
    input_tokens_total: int | None = Field(default=None, ge=0)
    output_tokens_total: int | None = Field(default=None, ge=0)
    error_message: str | None = None
    confirmation_count: int | None = Field(default=None, ge=0)
    destructive_actions_executed: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        """Construct a RunMetadata from a dict.

        Raises ``pydantic.ValidationError`` on any structural problem:
        missing required fields, wrong types, unknown fields, bad UUID, bad
        status enum, etc.
        """
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a schema-valid JSON-ready dict.

        Only fields that were explicitly set (either at construction or
        mutation) are emitted. This preserves round-trip fidelity: fields the
        caller never supplied do not re-appear as ``null`` on the way out.
        """
        return self.model_dump(mode="json", exclude_unset=True)

    def is_terminal(self) -> bool:
        """Return True if ``status`` is one of the five terminal states.

        Used by the RunWriter to decide when to finalize ``ended_at`` and by
        the API layer to refuse further actions on a completed run.
        """
        return self.status in TERMINAL_STATUSES

    def is_waiting(self) -> bool:
        """Return True iff the run is paused awaiting human confirmation."""
        return self.status == WAITING_STATUS


__all__ = [
    "TERMINAL_STATUSES",
    "WAITING_STATUS",
    "RunMetadata",
    "RunMode",
    "RunStatus",
    "load_run_metadata_schema",
    "validate_run_metadata",
]
