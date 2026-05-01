"""Read and validate a Recorder-produced trajectory directory.

The trajectory format is the locked contract between the Recorder (Module 1)
and the Synthesizer (Module 2), defined at ``contracts/trajectory.schema.json``
at the repo root. This module is the synthesizer's single entry point for
loading one:

* :class:`TrajectoryReader` — validates ``metadata.json`` and every line of
  ``events.jsonl`` against the JSON schema, then exposes iteration and
  filtering helpers plus a :meth:`TrajectoryReader.summary` method for UI and
  prompt-preprocessing use.
* :class:`Event` — a pydantic v2 model wrapping a single validated event in
  the synthesizer's *internal* (legacy-flat) shape. The reader translates
  canonical-shape recorder output to this shape at load time so downstream
  preprocessing / drafting code stays unchanged.
* :exc:`TrajectoryReadError` — raised for missing files, unreadable JSON, or
  schema violations. Messages include the line number (and event ``seq``
  where available) so failures are actionable.

Two on-disk shapes are supported:

* **Canonical** — what the Recorder actually emits per
  ``contracts/trajectory.schema.json`` (events have nested
  ``app: {bundle_id,name,pid}``, ``payload: {…}``, ``timestamp_ms``,
  ``type``; metadata has ``stopped_at`` and ``display_info``).
* **Legacy** — the flat shape synth's older fixtures use (events have flat
  ``t``, ``kind``, ``bundle_id``, ``text``, ``x``, ``y``, …; metadata may
  omit ``display_info`` and use ``at``/``title`` in ``app_focus_history``).

The reader detects the shape per file and normalises both to the same
on-disk-canonical form for schema validation, then back to the legacy
:class:`Event` shape for the rest of the synthesizer pipeline. Real
recordings produced by `feat/recorder` (canonical) and pre-existing
synthesizer fixtures (legacy) both Just Work.

Read-only semantics: every file is opened with mode ``'r'`` (or ``'rb'`` for
screenshots — not opened by this module). No trajectory files are ever
mutated, created, or deleted by the reader.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import (
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

# Mapping of canonical event ``type`` → synth-internal ``kind``. Anything not
# listed passes through unchanged so a future canonical addition still parses
# (the synth's downstream logic falls into the "unknown kind" branch).
_TYPE_TO_KIND: dict[str, str] = {
    "click": "click",
    "keypress": "key_down",  # synth's downstream treats key_down/key_up; recorder only emits down
    "scroll": "scroll",
    "app_switch": "app_switch",
    "window_focus": "app_focus",
    "text_input": "text_input",
    "keyframe": "screenshot",
    "tap_reenabled": "annotation",
}

# Reverse map for legacy → canonical translation when validating.
_KIND_TO_TYPE: dict[str, str] = {
    "click": "click",
    "key_down": "keypress",
    "key_up": "keypress",
    "scroll": "scroll",
    "app_switch": "app_switch",
    "app_focus": "window_focus",
    "text_input": "text_input",
    "screenshot": "keyframe",
    "annotation": "tap_reenabled",
}

# Legacy synth fixtures sometimes carry low-level mouse-motion events that
# the canonical recorder never emits and that synth's preprocess pipeline
# filters out anyway. Drop them at load time rather than fail the canonical
# schema validator on an unknown ``type`` enum value.
_LEGACY_NOISE_KINDS: frozenset[str] = frozenset(
    {"mouse_move", "mouse_down", "mouse_up"}
)


class TrajectoryReadError(Exception):
    """Raised when a trajectory directory fails to load or validate."""


class Event(BaseModel):
    """One validated event from ``events.jsonl`` in synth-internal flat shape.

    The reader translates canonical recorder events into this shape so
    downstream preprocessing / drafting code can stay simple. ``extra='allow'``
    lets forward-compatible additions flow through without a reader bump;
    ``frozen=True`` so downstream stages can hash / cache / compare events
    safely.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    seq: int = Field(..., ge=1)
    t: str
    kind: str
    bundle_id: str | None = None
    target: dict[str, Any] | None = None
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


def _schema_defs() -> dict[str, Any]:
    """Return the schema's named definitions block.

    Recorder's canonical schema uses the JSON Schema 2020-12 ``$defs``
    keyword; older synth-side schemas may have used ``definitions``. Accept
    either so the reader works against any committed snapshot of the
    contract.
    """
    schema = _load_schema()
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(defs, dict):
        return {}
    return defs


def _wrap_with_defs(ref_path: str) -> dict[str, Any]:
    """Build a thin wrapper schema that ``$ref``s into the full schema's defs.

    The metadata and event sub-schemas use ``$ref: '#/$defs/...'`` to
    reference shared definitions (``isoDateTime``, ``app``, ``frame``, …).
    A validator whose schema is just the sub-schema can't resolve those
    refs because ``$defs`` lives at the root. Wrapping the sub-schema in a
    new root that re-publishes the full ``$defs`` block keeps every ref
    resolvable while letting us still validate just metadata or just one
    event.
    """
    schema = _load_schema()
    defs = _schema_defs()
    defs_keyword = "$defs" if "$defs" in schema else "definitions"
    return {"$ref": ref_path, defs_keyword: defs}


@lru_cache(maxsize=1)
def _metadata_validator() -> Draft202012Validator:
    schema = _load_schema()
    ref = "#/$defs/metadata" if "$defs" in schema else "#/definitions/metadata"
    return Draft202012Validator(
        _wrap_with_defs(ref), format_checker=FormatChecker()
    )


@lru_cache(maxsize=1)
def _event_validator() -> Draft202012Validator:
    schema = _load_schema()
    ref = "#/$defs/event" if "$defs" in schema else "#/definitions/event"
    return Draft202012Validator(
        _wrap_with_defs(ref), format_checker=FormatChecker()
    )


# --------------------------------------------------------------------------
# canonical ⇄ legacy translation helpers
# --------------------------------------------------------------------------


def _is_canonical_event(ev: dict[str, Any]) -> bool:
    """Heuristic: canonical events carry ``type`` + ``payload``; legacy uses ``kind``."""
    return "type" in ev and "payload" in ev


def _is_canonical_metadata(meta: dict[str, Any]) -> bool:
    """Heuristic: canonical metadata carries ``stopped_at`` + a fully-shaped
    ``display_info`` (i.e. with ``scale_factor``, not the legacy ``scale``)
    + ``app_focus_history`` entries with ``entered_at`` (not legacy ``at``)."""
    if "stopped_at" not in meta:
        return False
    di = meta.get("display_info")
    if not isinstance(di, dict) or "scale_factor" not in di:
        return False
    history = meta.get("app_focus_history") or []
    return not (
        history
        and isinstance(history[0], dict)
        and "entered_at" not in history[0]
    )


def _ms_to_iso(ms: int | float) -> str:
    """Unix milliseconds → ISO-8601 with millisecond precision (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="milliseconds")


def _iso_to_ms(value: str) -> int:
    """ISO-8601 string → unix milliseconds; tolerant of trailing ``Z``."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _legacy_to_canonical_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Lossy upgrade of a legacy-flat event into canonical-nested shape.

    Used only to feed the schema validator — the legacy dict is also kept
    for direct construction of :class:`Event`. The "lossy" bit: canonical
    requires ``app.{name,pid}`` and ``target.frame`` which legacy doesn't
    carry, so we synthesise empty/zero defaults that satisfy the schema
    without contaminating downstream logic.
    """
    kind = ev.get("kind") or ""
    payload: dict[str, Any] = {}
    if kind == "click":
        if ev.get("button") is not None:
            payload["button"] = ev["button"]
        if ev.get("modifiers") is not None:
            payload["modifiers"] = ev["modifiers"]
    elif kind in ("key_down", "key_up"):
        keys: list[str] = []
        if ev.get("modifiers"):
            keys.extend(ev["modifiers"])
        if ev.get("key"):
            keys.append(ev["key"])
        if keys:
            payload["keys"] = keys
        if ev.get("modifiers") is not None:
            payload["modifiers"] = ev["modifiers"]
    elif kind == "scroll":
        # Legacy stores scroll deltas in ``x``/``y``; canonical uses
        # ``direction`` + magnitude ``delta``.
        dy = ev.get("y") or 0.0
        dx = ev.get("x") or 0.0
        if abs(dx) > abs(dy):
            payload["direction"] = "right" if dx > 0 else "left"
            payload["delta"] = float(abs(dx))
        else:
            payload["direction"] = "up" if dy >= 0 else "down"
            payload["delta"] = float(abs(dy))
    elif kind == "text_input":
        payload["text"] = ev.get("text", "")
    elif kind == "app_switch":
        payload["to_bundle_id"] = ev.get("bundle_id") or ""
    elif kind == "app_focus":
        payload["window_title"] = (ev.get("target") or {}).get("label")
    elif kind == "screenshot":
        payload["reason"] = "periodic"
    elif kind == "annotation":
        payload["cause"] = "user_input"

    target = ev.get("target")
    canonical_target: dict[str, Any] | None
    if isinstance(target, dict):
        canonical_target = {
            "role": target.get("role"),
            "label": target.get("label"),
            "description": target.get("description"),
            "frame": {
                "x": float(target.get("x", 0.0)),
                "y": float(target.get("y", 0.0)),
                "w": float(target.get("w", 0.0)),
                "h": float(target.get("h", 0.0)),
            },
            "ax_identifier": target.get("ax_identifier"),
        }
    else:
        canonical_target = None

    out: dict[str, Any] = {
        "seq": ev["seq"],
        "timestamp_ms": _iso_to_ms(ev["t"]) if "t" in ev else ev.get("timestamp_ms", 0),
        "type": _KIND_TO_TYPE.get(kind, kind),
        "screenshot_ref": ev.get("screenshot_ref"),
        "app": {
            "bundle_id": ev.get("bundle_id") or "",
            "name": ev.get("bundle_id") or "",
            "pid": 0,
        },
        "target": canonical_target,
        "payload": payload,
    }
    return out


def _canonical_to_legacy_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Flatten a canonical recorder event into synth's legacy ``Event`` shape.

    The translation is lossy where it has to be (e.g. recorder's keypress
    payload merges modifiers into ``keys``; we surface the last as ``key``
    and keep the rest as ``modifiers``). The fields :class:`Event` doesn't
    declare are still preserved by ``extra='allow'``.
    """
    payload = ev.get("payload") or {}
    target = ev.get("target")
    legacy_target: dict[str, Any] | None = None
    if isinstance(target, dict):
        legacy_target = {
            k: v
            for k, v in target.items()
            if k in ("role", "label", "description", "ax_identifier")
            and v is not None
        }
        frame = target.get("frame") or {}
        if isinstance(frame, dict):
            for axis in ("x", "y", "w", "h"):
                if axis in frame:
                    legacy_target[axis] = frame[axis]

    canonical_type = ev.get("type") or ""
    kind = _TYPE_TO_KIND.get(canonical_type, canonical_type)

    text: str | None = None
    button: str | None = None
    key: str | None = None
    modifiers: list[str] | None = None
    note: str | None = None
    x: float | None = None
    y: float | None = None

    if canonical_type == "click":
        button = payload.get("button")
        modifiers = payload.get("modifiers")
        if legacy_target is not None:
            x = legacy_target.get("x")
            y = legacy_target.get("y")
    elif canonical_type == "keypress":
        keys = payload.get("keys") or []
        if isinstance(keys, list) and keys:
            key = keys[-1]
            if len(keys) > 1:
                modifiers = list(keys[:-1])
        if modifiers is None:
            modifiers = payload.get("modifiers")
    elif canonical_type == "scroll":
        delta = float(payload.get("delta") or 0.0)
        direction = payload.get("direction") or "down"
        if direction == "up":
            y = delta
        elif direction == "down":
            y = -delta
        elif direction == "right":
            x = delta
        elif direction == "left":
            x = -delta
    elif canonical_type == "text_input":
        text = payload.get("text")
    elif canonical_type == "tap_reenabled":
        note = f"tap_reenabled cause={payload.get('cause', 'unknown')}"

    timestamp_ms = ev.get("timestamp_ms")
    t = _ms_to_iso(timestamp_ms) if timestamp_ms is not None else ""

    bundle_id: str | None = None
    app = ev.get("app")
    if isinstance(app, dict):
        bundle_id = app.get("bundle_id") or None

    legacy: dict[str, Any] = {
        "seq": ev["seq"],
        "t": t,
        "kind": kind,
        "bundle_id": bundle_id,
        "screenshot_ref": ev.get("screenshot_ref"),
        "target": legacy_target,
        "text": text,
        "x": x,
        "y": y,
        "button": button,
        "key": key,
        "modifiers": modifiers,
        "note": note,
    }
    return legacy


def _legacy_to_canonical_app_focus(entry: dict[str, Any]) -> dict[str, Any]:
    """Translate a single ``app_focus_history`` entry to canonical shape."""
    if "entered_at" in entry and "name" in entry:
        return entry
    return {
        "bundle_id": entry.get("bundle_id", ""),
        "name": entry.get("name") or entry.get("title") or entry.get("bundle_id", ""),
        "entered_at": entry.get("entered_at") or entry.get("at"),
        "exited_at": entry.get("exited_at"),
    }


def _legacy_to_canonical_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Lossy upgrade of legacy-shape metadata to canonical for validation.

    Defaults for missing fields:

    * ``stopped_at`` defaults to ``started_at`` (zero-duration recording).
    * ``display_info`` defaults to a non-Retina 1440x900 placeholder so the
      schema's required field is satisfied even when the legacy fixture
      didn't carry one.
    * ``app_focus_history`` entries get ``at`` → ``entered_at`` and
      ``title`` → ``name`` translation.
    """
    out = dict(meta)
    if "stopped_at" not in out:
        out["stopped_at"] = out.get("started_at")
    di = out.get("display_info")
    if not isinstance(di, dict):
        out["display_info"] = {"width": 1440, "height": 900, "scale_factor": 2.0}
    else:
        new_di = dict(di)
        # Legacy synth fixtures used ``scale``; canonical schema requires
        # ``scale_factor``.
        if "scale_factor" not in new_di and "scale" in new_di:
            new_di["scale_factor"] = new_di.pop("scale")
        new_di.setdefault("width", 1440)
        new_di.setdefault("height", 900)
        new_di.setdefault("scale_factor", 1.0)
        out["display_info"] = new_di
    history = out.get("app_focus_history") or []
    out["app_focus_history"] = [_legacy_to_canonical_app_focus(e) for e in history]
    return out


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


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
        canonical_meta = (
            meta if _is_canonical_metadata(meta) else _legacy_to_canonical_metadata(meta)
        )
        # Forward compat: the recorder may add new optional fields in
        # future schema versions; the canonical schema is strict
        # (``additionalProperties: false``) but synth should keep loading.
        # Validate a stripped copy that only carries known top-level keys,
        # then hand back the full dict (with extras intact) to the caller.
        _CANONICAL_META_KEYS = {
            "id",
            "label",
            "started_at",
            "stopped_at",
            "display_info",
            "app_focus_history",
        }
        for_validation = {
            k: v for k, v in canonical_meta.items() if k in _CANONICAL_META_KEYS
        }
        try:
            _metadata_validator().validate(for_validation)
        except ValidationError as e:
            pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
            raise TrajectoryReadError(
                f"metadata.json failed schema validation at {pointer}: {e.message}"
            ) from e
        # Return the canonical form so summary() and downstream consumers
        # can rely on a single shape (started_at + stopped_at always
        # present, app_focus_history entries with bundle_id+name).
        return canonical_meta

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
                # Drop legacy mouse-motion noise: the canonical recorder
                # never emits these and synth's downstream pipeline filters
                # them anyway. Translating mouse_move into the canonical
                # schema would require a fake type, so we just skip.
                if not _is_canonical_event(data) and data.get("kind") in _LEGACY_NOISE_KINDS:
                    continue
                # Validate against the canonical schema. Legacy-shape events
                # are upgraded first so the schema only ever sees the
                # contract's canonical form.
                canonical = (
                    data
                    if _is_canonical_event(data)
                    else _legacy_to_canonical_event(data)
                )
                try:
                    _event_validator().validate(canonical)
                except ValidationError as e:
                    pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
                    raise TrajectoryReadError(
                        f"events.jsonl line {line_num} (seq={seq_value}): "
                        f"schema validation failed at {pointer}: {e.message}"
                    ) from e
                # Construct the synth-internal Event from the legacy-flat
                # representation: if the on-disk data was already legacy we
                # use it directly; if it was canonical we flatten it.
                legacy = (
                    data if not _is_canonical_event(data) else _canonical_to_legacy_event(data)
                )
                try:
                    events.append(Event(**legacy))
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

        started_iso = self.metadata["started_at"]
        stopped_iso = self.metadata.get("stopped_at") or started_iso
        duration_ms = _iso_to_ms(stopped_iso) - _iso_to_ms(started_iso)

        return {
            "event_count": len(self._events),
            "duration_ms": duration_ms,
            "app_focus_history": list(self.metadata.get("app_focus_history", [])),
            "click_count": kind_counts.get("click", 0),
            "text_input_count": kind_counts.get("text_input", 0),
            "app_switch_count": kind_counts.get("app_switch", 0),
            "keyframe_count": keyframe_count,
        }
