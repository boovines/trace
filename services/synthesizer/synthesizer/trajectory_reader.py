"""Trajectory reader — extracts semantic events from a Recorder trajectory.

The synthesizer consumes trajectory directories (metadata.json + events.jsonl +
screenshots/) produced by the Recorder module. Downstream consumers need a
semantic view of the trajectory — not every low-level ``mouse_down`` /
``mouse_up`` pair, but the user-visible "clicked the Send button", "typed the
reply body" events. The runner's replay-correctness snapshot test (X-022) uses
this module to compare a fake-mode run's dispatched actions against the
recorded trajectory's semantic events.

Semantic event shape
--------------------
* **click** — one mouse press (down-up pair collapsed to a single event) at a
  display-point coordinate. The trajectory's ``note`` field on the
  ``mouse_down`` event carries human-authored annotations:
  ``step=<N>;target=<label>[;destructive=true]``. These map into
  :class:`SemanticEvent` fields (``step_number``, ``target_label``,
  ``destructive``).
* **text_input** — one ``text_input`` event (the Recorder coalesces rapid
  keystrokes into one logical text entry).

Events without a ``step=`` annotation are kept but carry ``step_number=None``.
The reader does not attempt to infer step numbers from timing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class TrajectoryReadError(ValueError):
    """Raised when a trajectory directory cannot be read or is malformed."""


SemanticEventKind = Literal["click", "text_input"]


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    """A high-level semantic event extracted from a trajectory.

    ``x`` / ``y`` are display-point coordinates and are ``None`` for
    ``text_input`` events. ``text`` is populated for ``text_input`` only.
    ``step_number``, ``target_label`` and ``destructive`` come from the
    trajectory's per-event ``note`` annotation and may be ``None`` / ``False``
    when the recorder did not mark them.
    """

    kind: SemanticEventKind
    x: float | None
    y: float | None
    text: str | None
    target_label: str | None
    step_number: int | None
    destructive: bool


@dataclass(frozen=True, slots=True)
class ReadTrajectory:
    """A parsed trajectory with its semantic event list."""

    trajectory_id: str
    metadata: dict[str, Any]
    semantic_events: tuple[SemanticEvent, ...]

    @property
    def clicks(self) -> tuple[SemanticEvent, ...]:
        return tuple(e for e in self.semantic_events if e.kind == "click")

    @property
    def text_inputs(self) -> tuple[SemanticEvent, ...]:
        return tuple(e for e in self.semantic_events if e.kind == "text_input")

    @property
    def destructive_clicks(self) -> tuple[SemanticEvent, ...]:
        return tuple(e for e in self.clicks if e.destructive)

    @property
    def non_destructive_clicks(self) -> tuple[SemanticEvent, ...]:
        return tuple(e for e in self.clicks if not e.destructive)


def _parse_note(note: str | None) -> dict[str, str]:
    """Parse ``step=2;target=Send button;destructive=true`` into a dict."""
    if not note:
        return {}
    out: dict[str, str] = {}
    for chunk in note.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        out[key.strip()] = value.strip()
    return out


def _event_to_semantic(event: dict[str, Any]) -> SemanticEvent | None:
    kind = event.get("kind")
    annotations = _parse_note(event.get("note"))
    step_raw = annotations.get("step")
    try:
        step_number = int(step_raw) if step_raw else None
    except ValueError:
        step_number = None
    target_label = annotations.get("target") or None
    destructive = annotations.get("destructive", "").lower() == "true"

    if kind == "mouse_down":
        x = event.get("x")
        y = event.get("y")
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            return None
        return SemanticEvent(
            kind="click",
            x=float(x),
            y=float(y),
            text=None,
            target_label=target_label,
            step_number=step_number,
            destructive=destructive,
        )
    if kind == "text_input":
        text = event.get("text")
        if not isinstance(text, str):
            return None
        return SemanticEvent(
            kind="text_input",
            x=None,
            y=None,
            text=text,
            target_label=target_label,
            step_number=step_number,
            destructive=destructive,
        )
    return None


def load_trajectory(
    trajectory_id: str, trajectories_root: Path
) -> ReadTrajectory:
    """Load a trajectory directory and return its semantic event stream.

    Raises :class:`TrajectoryReadError` if the directory, metadata, or events
    file is missing or unparseable.
    """
    traj_dir = trajectories_root / trajectory_id
    if not traj_dir.is_dir():
        raise TrajectoryReadError(f"Trajectory directory not found: {traj_dir}")

    metadata_path = traj_dir / "metadata.json"
    events_path = traj_dir / "events.jsonl"
    if not metadata_path.is_file():
        raise TrajectoryReadError(f"metadata.json missing in {traj_dir}")
    if not events_path.is_file():
        raise TrajectoryReadError(f"events.jsonl missing in {traj_dir}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrajectoryReadError(f"metadata.json is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise TrajectoryReadError("metadata.json must be an object")

    semantic_events: list[SemanticEvent] = []
    for line_no, raw_line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise TrajectoryReadError(
                f"events.jsonl line {line_no} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise TrajectoryReadError(
                f"events.jsonl line {line_no} is not an object"
            )
        semantic = _event_to_semantic(event)
        if semantic is not None:
            semantic_events.append(semantic)

    return ReadTrajectory(
        trajectory_id=trajectory_id,
        metadata=metadata,
        semantic_events=tuple(semantic_events),
    )


__all__ = [
    "ReadTrajectory",
    "SemanticEvent",
    "SemanticEventKind",
    "TrajectoryReadError",
    "load_trajectory",
]
