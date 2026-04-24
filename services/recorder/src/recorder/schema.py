"""Trajectory event schema loading and validation.

Thin wrapper around the `jsonschema` package that locates
``contracts/trajectory.schema.json`` at the repo root and exposes helpers
for validating individual events and full metadata objects.

The schema itself is the locked contract between Recorder, Synthesizer, and
Runner — see contracts/trajectory.schema.json.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

__all__ = [
    "ValidationError",
    "find_schema_path",
    "load_event_schema",
    "load_metadata_schema",
    "load_trajectory_schema",
    "validate_event",
    "validate_metadata",
]

_SCHEMA_FILENAME = "trajectory.schema.json"


def find_schema_path() -> Path:
    """Locate ``contracts/trajectory.schema.json`` by walking up from this file.

    Works for editable installs (the uv workspace layout Trace uses). Raises
    ``FileNotFoundError`` if no ``contracts/`` directory is found along the
    ancestry chain.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "contracts" / _SCHEMA_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate contracts/{_SCHEMA_FILENAME} above {here}"
    )


@lru_cache(maxsize=1)
def load_trajectory_schema() -> dict[str, Any]:
    """Return the parsed top-level trajectory schema document (with $defs)."""
    path = find_schema_path()
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _subschema(fragment: str) -> dict[str, Any]:
    root = load_trajectory_schema()
    # A purely local fragment $ref avoids the jsonschema resolver attempting
    # to fetch the schema's $id over the network at validate time.
    return {
        "$schema": root.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$ref": f"#/$defs/{fragment}",
        "$defs": root["$defs"],
    }


def load_event_schema() -> dict[str, Any]:
    """Return a standalone JSON Schema that validates a single event dict."""
    return _subschema("event")


def load_metadata_schema() -> dict[str, Any]:
    """Return a standalone JSON Schema that validates a metadata.json dict."""
    return _subschema("metadata")


@lru_cache(maxsize=1)
def _event_validator() -> Draft202012Validator:
    return Draft202012Validator(
        load_event_schema(),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


@lru_cache(maxsize=1)
def _metadata_validator() -> Draft202012Validator:
    return Draft202012Validator(
        load_metadata_schema(),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def validate_event(event: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if the event is not schema-valid."""
    _event_validator().validate(event)


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if the metadata dict is not valid."""
    _metadata_validator().validate(metadata)
