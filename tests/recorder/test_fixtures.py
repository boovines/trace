"""Validate the committed reference-workflow trajectory fixtures.

The five fixtures under ``fixtures/trajectories/<slug>/`` are the
acceptance bar shared across the three Python modules. These tests only
exercise *structural* properties — schema validity, per-fixture minimums
(one click with an AX target, one text_input with a field_label, one
app_switch), and PNG validity for every ``screenshot_ref``. They do not
assert on any real app behaviour; a human recording replaces these
fixtures during smoke testing.

See ``scripts/generate_synthetic_fixtures.py`` (synthesizer) and
``scripts/regenerate_fixtures.sh`` (human process).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from recorder.schema import validate_event, validate_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "fixtures" / "trajectories"
REFERENCE_SLUGS = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_FIXTURES_BYTES = 5 * 1024 * 1024  # 5 MB cap per PRD R-013.


def _load_events(slug: str) -> list[dict[str, Any]]:
    path = FIXTURES_ROOT / slug / "events.jsonl"
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_metadata(slug: str) -> dict[str, Any]:
    path = FIXTURES_ROOT / slug / "metadata.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_directory_exists(slug: str) -> None:
    fixture_dir = FIXTURES_ROOT / slug
    assert fixture_dir.is_dir(), f"Missing fixtures/trajectories/{slug}/"
    assert (fixture_dir / "metadata.json").is_file()
    assert (fixture_dir / "events.jsonl").is_file()
    assert (fixture_dir / "screenshots").is_dir()


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_metadata_validates(slug: str) -> None:
    validate_metadata(_load_metadata(slug))


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_every_event_validates(slug: str) -> None:
    events = _load_events(slug)
    assert events, f"{slug}/events.jsonl is empty"
    for ev in events:
        validate_event(ev)


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_seq_is_monotonic_and_one_indexed(slug: str) -> None:
    events = _load_events(slug)
    seqs = [ev["seq"] for ev in events]
    assert seqs[0] == 1, f"{slug}: first seq must be 1, got {seqs[0]}"
    for prev, curr in itertools.pairwise(seqs):
        assert curr == prev + 1, f"{slug}: seq gap {prev} -> {curr}"


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_has_minimum_event_coverage(slug: str) -> None:
    """Each fixture must have at least one click-with-target, one text_input
    with a field_label, and one app_switch — the R-013 minimums.
    """
    events = _load_events(slug)

    click_with_target = [
        ev
        for ev in events
        if ev["type"] == "click"
        and ev.get("target") is not None
        and ev["target"].get("role")
    ]
    assert click_with_target, f"{slug}: needs >=1 click with a resolved AX target"

    text_inputs = [
        ev
        for ev in events
        if ev["type"] == "text_input"
        and ev["payload"].get("field_label") not in (None, "")
    ]
    assert text_inputs, f"{slug}: needs >=1 text_input with a field_label"

    switches = [ev for ev in events if ev["type"] == "app_switch"]
    assert switches, f"{slug}: needs >=1 app_switch"


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_screenshot_refs_resolve_to_valid_png(slug: str) -> None:
    fixture_dir = FIXTURES_ROOT / slug
    events = _load_events(slug)
    refs = {ev["screenshot_ref"] for ev in events if ev.get("screenshot_ref")}
    assert refs, f"{slug}: needs at least one event with a screenshot_ref"
    for ref in refs:
        path = fixture_dir / ref
        assert path.is_file(), f"{slug}: screenshot_ref {ref!r} missing on disk"
        assert path.read_bytes().startswith(PNG_MAGIC), (
            f"{slug}: {ref} is not a valid PNG"
        )


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_fixture_metadata_id_matches_directory_contents(slug: str) -> None:
    meta = _load_metadata(slug)
    assert meta["label"], f"{slug}: metadata.label must be non-empty"
    assert meta["display_info"]["width"] >= 1
    assert meta["display_info"]["height"] >= 1


def test_fixtures_total_size_under_cap() -> None:
    """PRD R-013: fixtures/ tree must stay under 5 MB."""
    total = 0
    for path in FIXTURES_ROOT.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    assert total < MAX_FIXTURES_BYTES, (
        f"fixtures/trajectories/ is {total} bytes, over the {MAX_FIXTURES_BYTES}-byte cap"
    )
