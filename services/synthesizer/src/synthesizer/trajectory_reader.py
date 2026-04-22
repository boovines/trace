"""Read and validate a Recorder-produced trajectory directory.

The trajectory format is the locked contract between the Recorder (Module 1)
and the Synthesizer (Module 2), defined at ``contracts/trajectory.schema.json``
at the repo root. This module is the synthesizer's single entry point for
loading one:

* :class:`TrajectoryReader` — validates ``metadata.json`` and every line of
  ``events.jsonl`` against the JSON schema, then exposes iteration and
  filtering helpers plus a :meth:`TrajectoryReader.summary` method for UI and
  prompt-preprocessing use.
* :class:`Event` — a pydantic v2 model wrapping a single validated event.
  ``extra='allow'`` keeps unknown fields so forward-compatible additions to
  the contract don't break old readers.
* :exc:`TrajectoryReadError` — raised for missing files, unreadable JSON, or
  schema violations. Messages include the line number (and event ``seq``
  where available) so failures are actionable.

Read-only semantics: every file is opened with mode ``'r'`` (or ``'rb'`` for
screenshots — not opened by this module). No trajectory files are ever
mutated, created, or deleted by the reader.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
    ValidationError,
)
from pydantic import BaseModel, ConfigDict, Field

from synthesizer.schema import CONTRACTS_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "TRAJECTORY_SCHEMA_PATH",
    "Event",
    "TrajectoryReadError",
    "TrajectoryReader",
]


TRAJECTORY_SCHEMA_PATH: Path = CONTRACTS_DIR / "trajectory.schema.json"


class TrajectoryReadError(Exception):
    """Raised when a trajectory directory fails to load or validate."""


class Event(BaseModel):
    """One validated event from ``events.jsonl``.

    Fields mirror the ``definitions.event`` block of the trajectory schema.
    ``extra='allow'`` lets forward-compatible additions flow through without a
    reader bump; ``frozen=True`` mirrors :class:`~synthesizer.skill_doc.ParsedSkill`
    so downstream stages can hash / cache / compare events safely.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    seq: int = Field(..., ge=1)
    t: str
    kind: str
    bundle_id: str | None = None
    target: dict[str, str] | None = None
    screenshot_ref: str | None = None
    text: str | None = None
    x: float | None = None
    y: float | None = None
    button: str | None = None
    key: str | None = None
    modifiers: list[str] | None = None
    note: str | None = None


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    with TRAJECTORY_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


@lru_cache(maxsize=1)
def _metadata_validator() -> Draft202012Validator:
    return Draft202012Validator(
        _load_schema()["definitions"]["metadata"], format_checker=FormatChecker()
    )


@lru_cache(maxsize=1)
def _event_validator() -> Draft202012Validator:
    return Draft202012Validator(
        _load_schema()["definitions"]["event"], format_checker=FormatChecker()
    )


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` as UTC."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


class TrajectoryReader:
    """Read-only accessor for a trajectory directory on disk.

    Instantiation performs full validation so downstream stages operate on
    already-validated data:

    * ``metadata.json`` must exist and satisfy the trajectory metadata schema.
    * ``events.jsonl`` must exist and every non-blank line must be valid JSON
      that satisfies the event schema.
    * Event ``seq`` values are not required to be contiguous or ordered on
      disk — :meth:`iter_events` sorts by ``seq`` — but duplicates are
      rejected since they would make :meth:`get_screenshot_path` ambiguous.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory: Path = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(
                f"Trajectory directory does not exist: {self.directory}. "
                "Expected a directory produced by the Recorder."
            )
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"Trajectory path is not a directory: {self.directory}"
            )

        self.metadata: dict[str, Any] = self._load_metadata()
        self._events: list[Event] = self._load_events()
        self._events.sort(key=lambda e: e.seq)
        self._check_unique_seqs()

    # --- loaders --------------------------------------------------------

    def _load_metadata(self) -> dict[str, Any]:
        path = self.directory / "metadata.json"
        if not path.is_file():
            raise TrajectoryReadError(
                f"metadata.json not found in {self.directory}"
            )
        with path.open("r", encoding="utf-8") as f:
            try:
                meta: dict[str, Any] = json.load(f)
            except json.JSONDecodeError as e:
                raise TrajectoryReadError(
                    f"metadata.json is not valid JSON: {e.msg} (line {e.lineno}, col {e.colno})"
                ) from e
        try:
            _metadata_validator().validate(meta)
        except ValidationError as e:
            pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
            raise TrajectoryReadError(
                f"metadata.json failed schema validation at {pointer}: {e.message}"
            ) from e
        return meta

    def _load_events(self) -> list[Event]:
        path = self.directory / "events.jsonl"
        if not path.is_file():
            raise TrajectoryReadError(
                f"events.jsonl not found in {self.directory}"
            )
        events: list[Event] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TrajectoryReadError(
                        f"events.jsonl line {line_num}: invalid JSON ({e.msg})"
                    ) from e
                seq_value = data.get("seq", "<missing>")
                try:
                    _event_validator().validate(data)
                except ValidationError as e:
                    pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
                    raise TrajectoryReadError(
                        f"events.jsonl line {line_num} (seq={seq_value}): "
                        f"schema validation failed at {pointer}: {e.message}"
                    ) from e
                try:
                    events.append(Event(**data))
                except Exception as e:  # pydantic ValidationError, TypeError, etc.
                    raise TrajectoryReadError(
                        f"events.jsonl line {line_num} (seq={seq_value}): "
                        f"failed to construct Event: {e}"
                    ) from e
        return events

    def _check_unique_seqs(self) -> None:
        seen: set[int] = set()
        for event in self._events:
            if event.seq in seen:
                raise TrajectoryReadError(
                    f"events.jsonl contains duplicate seq={event.seq}; "
                    "seq values must be unique within a trajectory"
                )
            seen.add(event.seq)

    # --- public accessors -----------------------------------------------

    def iter_events(self) -> Iterator[Event]:
        """Yield every validated event in ``seq`` order."""
        return iter(self._events)

    def iter_events_by_type(self, kind: str) -> Iterator[Event]:
        """Yield events whose ``kind`` equals ``kind``, in ``seq`` order."""
        return (e for e in self._events if e.kind == kind)

    def get_screenshot_path(self, seq: int) -> Path | None:
        """Return the on-disk path for the screenshot referenced by event ``seq``.

        Returns ``None`` when (a) no event has that seq, (b) the event has no
        ``screenshot_ref``, or (c) the referenced file is missing (in which
        case a warning is logged so a partial trajectory doesn't crash the
        synthesizer — it just proceeds without that keyframe).
        """
        for event in self._events:
            if event.seq != seq:
                continue
            if not event.screenshot_ref:
                return None
            path = self.directory / event.screenshot_ref
            if not path.is_file():
                logger.warning(
                    "Trajectory %s: screenshot %s referenced by seq=%d is missing on disk",
                    self.directory,
                    event.screenshot_ref,
                    seq,
                )
                return None
            return path
        return None

    def summary(self) -> dict[str, Any]:
        """Return a dict of aggregate counts and timing for prompt/UI use.

        Keys: ``event_count``, ``duration_ms``, ``app_focus_history`` (as
        recorded in metadata), ``click_count``, ``text_input_count``,
        ``app_switch_count``, ``keyframe_count``.
        """
        kind_counts: dict[str, int] = {}
        keyframe_count = 0
        for event in self._events:
            kind_counts[event.kind] = kind_counts.get(event.kind, 0) + 1
            if event.screenshot_ref:
                keyframe_count += 1

        duration_ms = int(
            (
                _parse_iso(self.metadata["stopped_at"])
                - _parse_iso(self.metadata["started_at"])
            ).total_seconds()
            * 1000
        )

        return {
            "event_count": len(self._events),
            "duration_ms": duration_ms,
            "app_focus_history": list(self.metadata.get("app_focus_history", [])),
            "click_count": kind_counts.get("click", 0),
            "text_input_count": kind_counts.get("text_input", 0),
            "app_switch_count": kind_counts.get("app_switch", 0),
            "keyframe_count": keyframe_count,
        }
