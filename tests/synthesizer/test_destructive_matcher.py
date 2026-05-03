"""Tests for ``synthesizer.destructive_matcher``.

Covers the PRD S-008 acceptance bar:

* Every one of the 14 destructive keywords matches as a word boundary.
* ``sender`` does NOT match ``send`` (no substring matches).
* LLM flag missed → matcher adds the flag; ``report.added_flags`` is populated.
* LLM flag correct → matcher leaves it alone; ``report.unchanged`` is populated.
* Step text containing a keyword is NOT sufficient — the click ``target.label``
  drives matching.
* Non-click steps (kbd shortcut, scroll) are never flagged by the matcher.
* ``Envoyer`` (French) is NOT matched — English-only v1, documented limit.
* Integration with :func:`synthesizer.draft.generate_draft`: after the draft
  call succeeds, the matcher runs and the returned ``DraftResult`` reflects
  the combined LLM + matcher destructive set with a consistent markdown +
  meta.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from synthesizer.destructive_matcher import (
    DESTRUCTIVE_KEYWORDS,
    apply_destructive_matcher,
    label_has_destructive_keyword,
)
from synthesizer.skill_doc import DESTRUCTIVE_MARKER, ParsedSkill, Step
from synthesizer.trajectory_reader import TrajectoryReader

# --- trajectory fixture helper --------------------------------------------


def _iso(seconds_offset: float, base: datetime | None = None) -> str:
    base = base or datetime(2026, 4, 22, 14, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=seconds_offset)).isoformat().replace(
        "+00:00", "Z"
    )


def _make_trajectory(
    tmp_path: Path,
    events: list[dict[str, Any]],
    *,
    started_at: str = "2026-04-22T14:00:00Z",
    stopped_at: str | None = None,
    bundle_id: str = "com.google.Chrome",
) -> TrajectoryReader:
    if stopped_at is None:
        if events:
            last_t = events[-1]["t"]
            parsed = last_t.replace("Z", "+00:00")
            dt = datetime.fromisoformat(parsed) + timedelta(seconds=1)
            stopped_at = dt.isoformat().replace("+00:00", "Z")
        else:
            stopped_at = started_at

    traj_dir = tmp_path / f"traj-{uuid.uuid4().hex[:8]}"
    traj_dir.mkdir()
    metadata = {
        "id": str(uuid.uuid4()),
        "started_at": started_at,
        "stopped_at": stopped_at,
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": started_at, "bundle_id": bundle_id, "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return TrajectoryReader(traj_dir)


def _click_event(
    seq: int,
    label: str,
    *,
    seconds_offset: float,
    bundle_id: str = "com.google.Chrome",
    role: str = "button",
) -> dict[str, Any]:
    return {
        "seq": seq,
        "t": _iso(seconds_offset),
        "kind": "click",
        "x": 100.0,
        "y": 100.0,
        "button": "left",
        "bundle_id": bundle_id,
        "target": {"label": label, "role": role, "bundle_id": bundle_id},
    }


def _parsed_skill(
    step_specs: list[tuple[str, bool]],
    *,
    title: str = "Test Skill",
    description: str = "Test workflow.",
) -> ParsedSkill:
    steps = [
        Step(number=i + 1, text=text, destructive=destructive)
        for i, (text, destructive) in enumerate(step_specs)
    ]
    return ParsedSkill(
        title=title,
        description=description,
        parameters=[],
        preconditions=[],
        steps=steps,
        expected_outcome="Workflow completes successfully.",
        notes=None,
    )


# --- keyword word-boundary behavior ---------------------------------------


@pytest.mark.parametrize("keyword", DESTRUCTIVE_KEYWORDS)
def test_every_keyword_matches_as_word(keyword: str) -> None:
    # Exact match, case-insensitive variants, surrounded by punctuation.
    assert label_has_destructive_keyword(keyword) is True
    assert label_has_destructive_keyword(keyword.upper()) is True
    assert label_has_destructive_keyword(keyword.capitalize()) is True
    assert label_has_destructive_keyword(f'"{keyword}"') is True
    assert label_has_destructive_keyword(f"{keyword} now") is True


def test_keyword_does_not_substring_match() -> None:
    # 'sender' must NOT match 'send'; 'posted' must NOT match 'post'.
    assert label_has_destructive_keyword("sender") is False
    assert label_has_destructive_keyword("senders") is False
    assert label_has_destructive_keyword("posted") is False
    assert label_has_destructive_keyword("sharepoint") is False
    assert label_has_destructive_keyword("approved_by") is False
    # Hyphens are non-word chars, so 'share-link' DOES match 'share'. This is
    # intentional — the label 'share-link' is destructive.
    assert label_has_destructive_keyword("share-link") is True


def test_empty_and_none_labels_never_match() -> None:
    assert label_has_destructive_keyword(None) is False
    assert label_has_destructive_keyword("") is False
    assert label_has_destructive_keyword("   ") is False


def test_non_english_keywords_not_matched() -> None:
    # v1 documented limitation — English-only.
    assert label_has_destructive_keyword("Envoyer") is False  # French 'send'
    assert label_has_destructive_keyword("Löschen") is False  # German 'delete'
    assert label_has_destructive_keyword("送信") is False  # Japanese 'send'


# --- matcher behavior: adds missing flag ----------------------------------


def test_matcher_adds_flag_when_llm_misses_send(tmp_path: Path) -> None:
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Send", seconds_offset=0.0)],
    )
    parsed = _parsed_skill([("Click the Send button.", False)])

    result = apply_destructive_matcher(parsed, reader)

    assert result.parsed.steps[0].destructive is True
    assert result.report.added_flags == [1]
    assert result.report.unchanged == []
    assert result.report.llm_flags == []


def test_matcher_leaves_existing_flag_alone(tmp_path: Path) -> None:
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Delete", seconds_offset=0.0)],
    )
    parsed = _parsed_skill([("Click Delete.", True)])

    result = apply_destructive_matcher(parsed, reader)

    assert result.parsed.steps[0].destructive is True
    assert result.report.added_flags == []
    assert result.report.unchanged == [1]
    assert result.report.llm_flags == [1]


def test_matcher_is_additive_when_llm_flagged_but_label_does_not_match(
    tmp_path: Path,
) -> None:
    # LLM flagged the step destructive; click label is harmless ('Open'). The
    # matcher must not un-flag it — additive only.
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Open", seconds_offset=0.0)],
    )
    parsed = _parsed_skill([("Open the menu.", True)])

    result = apply_destructive_matcher(parsed, reader)

    assert result.parsed.steps[0].destructive is True
    assert result.report.added_flags == []
    assert result.report.unchanged == [1]
    assert result.report.llm_flags == [1]


# --- matcher behavior: step text vs click label ---------------------------


def test_step_text_keyword_does_not_flag_when_label_is_safe(
    tmp_path: Path,
) -> None:
    # Step text contains 'Submit', but the click target label is 'Ticket form'
    # (no keyword). The matcher must NOT flag — label, not step text, drives
    # matching.
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Ticket form", seconds_offset=0.0)],
    )
    parsed = _parsed_skill([("Submit a ticket via the form.", False)])

    result = apply_destructive_matcher(parsed, reader)

    assert result.parsed.steps[0].destructive is False
    assert result.report.added_flags == []


# --- matcher behavior: non-click steps never flagged ----------------------


def test_non_click_steps_never_flagged_by_matcher(tmp_path: Path) -> None:
    # Trajectory has zero clicks; the skill has 3 steps corresponding to
    # a keyboard shortcut, a scroll, and a typed input. No step can be
    # flagged because no click labels exist.
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "key_down",
            "bundle_id": "com.google.Chrome",
            "key": "s",
            "modifiers": ["cmd"],
        },
        {
            "seq": 2,
            "t": _iso(1.0),
            "kind": "scroll",
            "bundle_id": "com.google.Chrome",
            "y": -50.0,
        },
        {
            "seq": 3,
            "t": _iso(2.0),
            "kind": "text_input",
            "bundle_id": "com.google.Chrome",
            "text": "notes",
        },
    ]
    reader = _make_trajectory(tmp_path, events)
    parsed = _parsed_skill(
        [
            # Even with keyword-shaped wording, no click means no label to
            # match against.
            ("Press Cmd+S to send the file.", False),
            ("Scroll down to the submit section.", False),
            ("Type 'delete me' into the notes field.", False),
        ]
    )

    result = apply_destructive_matcher(parsed, reader)

    assert [s.destructive for s in result.parsed.steps] == [False, False, False]
    assert result.report.added_flags == []
    assert result.report.unchanged == []
    assert result.report.llm_flags == []


# --- matcher behavior: multi-step positional mapping ----------------------


def test_matcher_positional_mapping_over_multiple_clicks(tmp_path: Path) -> None:
    # Step 1 ↔ click 1 (Open — safe), step 2 ↔ click 2 (Send — destructive),
    # step 3 ↔ click 3 (Cancel — safe). Only step 2 should be flagged.
    reader = _make_trajectory(
        tmp_path,
        [
            _click_event(1, "Open", seconds_offset=0.0),
            _click_event(2, "Send", seconds_offset=1.0),
            _click_event(3, "Cancel", seconds_offset=2.0),
        ],
    )
    parsed = _parsed_skill(
        [
            ("Click Open.", False),
            ("Click Send.", False),
            ("Click Cancel.", False),
        ]
    )

    result = apply_destructive_matcher(parsed, reader)

    assert [s.destructive for s in result.parsed.steps] == [False, True, False]
    assert result.report.added_flags == [2]


def test_matcher_skips_steps_beyond_click_count(tmp_path: Path) -> None:
    # Only 1 click event; the skill has 3 steps. Steps 2 and 3 have no click
    # mapping and are never flagged by the matcher.
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Send", seconds_offset=0.0)],
    )
    parsed = _parsed_skill(
        [
            ("Click Send.", False),
            ("Wait for confirmation.", False),
            ("Close the dialog.", False),
        ]
    )

    result = apply_destructive_matcher(parsed, reader)

    assert [s.destructive for s in result.parsed.steps] == [True, False, False]
    assert result.report.added_flags == [1]


def test_matcher_returns_new_parsed_skill_and_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    reader = _make_trajectory(
        tmp_path,
        [_click_event(1, "Send", seconds_offset=0.0)],
    )
    original = _parsed_skill([("Click Send.", False)])

    result = apply_destructive_matcher(original, reader)

    # Original frozen object untouched.
    assert original.steps[0].destructive is False
    # Matcher result distinct and carries the new flag.
    assert result.parsed is not original
    assert result.parsed.steps[0].destructive is True


def test_click_with_no_target_label_is_not_flagged(tmp_path: Path) -> None:
    # Some clicks lack accessibility labels (e.g., canvas clicks). The matcher
    # must gracefully skip — absence of a label is not a destructive hint.
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "click",
            "x": 100.0,
            "y": 100.0,
            "button": "left",
            "bundle_id": "com.google.Chrome",
            # no target
        },
    ]
    reader = _make_trajectory(tmp_path, events)
    parsed = _parsed_skill([("Click somewhere in the canvas.", False)])

    result = apply_destructive_matcher(parsed, reader)

    assert result.parsed.steps[0].destructive is False
    assert result.report.added_flags == []


# --- integration with S-007 generate_draft --------------------------------


def test_generate_draft_applies_matcher_after_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the LLM returns a draft missing the ⚠️ on a Send click, the final
    DraftResult must reflect the matcher's added flag in markdown + meta +
    parsed skill — all three in sync."""
    import json as _json

    from synthesizer.draft import build_user_content, generate_draft
    from synthesizer.draft_prompt import DRAFT_SYSTEM_PROMPT
    from synthesizer.llm_client import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL,
        LLMClient,
        compute_request_hash,
        save_fake_response,
    )
    from synthesizer.preprocess import preprocess_trajectory

    # Reuse the committed gmail_reply fixture — it has a Send click at the
    # end. The LLM response we register OMITS the destructive flag; the
    # matcher must add it.
    here = Path(__file__).resolve()
    repo_root = next(
        p for p in (here, *here.parents) if (p / "fixtures" / "trajectories").is_dir()
    )
    fixture_dir = repo_root / "fixtures" / "trajectories" / "gmail_reply"
    reader = TrajectoryReader(fixture_dir)
    preprocessed = preprocess_trajectory(reader)

    # Isolate fake responses + cost logs.
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    monkeypatch.setenv("TRACE_FAKE_RESPONSES_DIR", str(fakes))
    monkeypatch.setenv("TRACE_DATA_DIR", str(tmp_path / "data"))

    step_entries = [
        e for e in preprocessed.digest if e.kind not in ("scroll_run", "idle")
    ]
    # The matcher binds each destructive click label to the earliest step
    # whose text case-insensitively contains that label substring. The
    # gmail_reply digest produces step summaries like
    # ``Clicked button labeled "Send" in Chrome`` — so the Send click
    # binds to the step whose summary contains "Send".
    flagged_step = next(
        i
        for i, entry in enumerate(step_entries, start=1)
        if "send" in entry.summary_text.lower()
    )

    # Build a markdown where destructive_steps is empty — LLM missed it.
    steps_md: list[str] = []
    for idx, entry in enumerate(step_entries, start=1):
        summary = entry.summary_text.replace("\n", " ").replace("\r", " ")
        steps_md.append(f"{idx}. {summary}.")
    markdown = (
        "# Gmail Reply\n"
        "\n"
        "Reply to the most recent email.\n"
        "\n"
        "## Parameters\n"
        "\n"
        "_None._\n"
        "\n"
        "## Preconditions\n"
        "\n"
        "_None._\n"
        "\n"
        "## Steps\n"
        "\n"
        + "\n".join(steps_md)
        + "\n"
        "\n"
        "## Expected outcome\n"
        "\n"
        "The reply is sent.\n"
    )
    meta = {
        "slug": "gmail_reply",
        "name": "Gmail Reply",
        "trajectory_id": "00000000-0000-0000-0000-000000000000",
        "created_at": "2026-04-22T00:00:00+00:00",
        "parameters": [],
        "destructive_steps": [],  # deliberately missing the Send flag
        "preconditions": [],
        "step_count": len(step_entries),
    }
    response_text = _json.dumps(
        {"markdown": markdown, "meta": meta, "questions": []}
    )

    messages = [{"role": "user", "content": build_user_content(preprocessed, reader)}]
    request_hash = compute_request_hash(
        system=DRAFT_SYSTEM_PROMPT,
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=None,
    )
    save_fake_response(
        request_hash=request_hash,
        text=response_text,
        input_tokens=1000,
        output_tokens=300,
        directory=fakes,
    )

    client = LLMClient()
    result = generate_draft(preprocessed, client, reader=reader)

    # Matcher-added flag visible everywhere.
    assert flagged_step in result.meta["destructive_steps"]
    flagged = [s for s in result.parsed.steps if s.destructive]
    assert [s.number for s in flagged] == [flagged_step]
    assert DESTRUCTIVE_MARKER in result.markdown
    # Telemetry report reflects the matcher's action.
    assert flagged_step in result.matcher_report.added_flags
    assert result.matcher_report.llm_flags == []
