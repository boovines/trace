"""Sanity-check that the shipped JSON schemas in contracts/ are themselves valid
JSON Schema documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"


@pytest.mark.parametrize(
    "name",
    ["trajectory.schema.json", "skill-meta.schema.json", "run-metadata.schema.json"],
)
def test_schema_is_well_formed(name: str) -> None:
    with (CONTRACTS_DIR / name).open() as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
