"""Tests for :mod:`synthesizer.writer` — S-010 SkillWriter.

Each test uses ``tmp_path`` for the skills root and overrides
``TRACE_DATA_DIR`` so the SQLite index is written to a per-test temp dir.
No fixture files are ever mutated.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import uuid
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError  # type: ignore[import-untyped]
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.skill_doc import Parameter, ParsedSkill, Step, parse_skill_md
from synthesizer.trajectory_reader import TrajectoryReader
from synthesizer.writer import (
    MAX_PREVIEW_SCREENSHOTS,
    SKILL_DIR_MODE,
    SkillAlreadyExistsError,
    SkillWriteError,
    SkillWriter,
    _pick_preview_indices,
    index_db_path,
)

# --- fixtures & helpers ----------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Every writer test gets its own TRACE_DATA_DIR so index.db is isolated."""
    data_dir = tmp_path / "trace_data"
    data_dir.mkdir()
    monkeypatch.setenv("TRACE_DATA_DIR", str(data_dir))
    return data_dir


def _iso(seconds_offset: float = 0.0) -> str:
    base = datetime(2026, 4, 22, 14, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=seconds_offset)).isoformat().replace(
        "+00:00", "Z"
    )


def _one_pixel_png() -> bytes:
    """Minimal valid 1x1 grayscale PNG (no Pillow dep)."""
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    raw = b"\x00\x00"
    idat = zlib.compress(raw, 9)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _make_trajectory(
    tmp_path: Path,
    *,
    keyframe_count: int = 3,
    trajectory_id: str | None = None,
    dir_name: str | None = None,
) -> TrajectoryReader:
    """Build a trajectory on disk with ``keyframe_count`` click keyframes."""
    trajectory_id = trajectory_id or str(uuid.uuid4())
    name = dir_name or f"traj-{uuid.uuid4().hex[:8]}"
    traj_dir = tmp_path / name
    traj_dir.mkdir()

    ss_dir = traj_dir / "screenshots"
    ss_dir.mkdir()
    events: list[dict[str, Any]] = []

    # First event is always an app_switch with screenshot 0001.
    events.append(
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "app_switch",
            "bundle_id": "com.google.Chrome",
            "screenshot_ref": "screenshots/0001.png",
        }
    )
    (ss_dir / "0001.png").write_bytes(_one_pixel_png())

    # Then click events, each with their own screenshot.
    for i in range(1, keyframe_count):
        seq = i + 1
        filename = f"{seq:04d}.png"
        (ss_dir / filename).write_bytes(_one_pixel_png())
        events.append(
            {
                "seq": seq,
                "t": _iso(i * 2.0),
                "kind": "click",
                "x": 100 + i,
                "y": 200 + i,
                "button": "left",
                "bundle_id": "com.google.Chrome",
                "target": {
                    "label": f"Button {i}",
                    "role": "button",
                    "bundle_id": "com.google.Chrome",
                },
                "screenshot_ref": f"screenshots/{filename}",
            }
        )

    metadata = {
        "id": trajectory_id,
        "started_at": _iso(0.0),
        "stopped_at": _iso(max(keyframe_count, 1) * 2.0),
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": _iso(0.0), "bundle_id": "com.google.Chrome", "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return TrajectoryReader(traj_dir)


def _make_skill(
    *,
    slug: str = "gmail_reply",
    trajectory_id: str,
    destructive_step: int | None = 2,
) -> tuple[ParsedSkill, dict[str, Any]]:
    steps = [
        Step(number=1, text="Open Chrome and load Gmail.", destructive=False),
        Step(number=2, text='Click "Send" to deliver the reply.', destructive=False),
    ]
    if destructive_step == 2:
        steps[1] = Step(
            number=2, text='Click "Send" to deliver the reply.', destructive=True
        )
    parsed = ParsedSkill(
        title="Reply to the newest Gmail message",
        description="Reply to the most recent unread message in Gmail with a short "
        "template and send it.",
        parameters=[
            Parameter(
                name="message_body",
                type="string",
                required=True,
                default=None,
                description="Body text of the reply.",
            )
        ],
        preconditions=["Chrome is open", "User is signed into Gmail"],
        steps=steps,
        expected_outcome="The reply is sent successfully.",
        notes=None,
    )
    # The skill body references {message_body} so meta.parameters must include
    # it for validate_meta_against_markdown to pass. Rewrite step 1 to embed
    # the reference.
    parsed = parsed.model_copy(
        update={
            "steps": [
                Step(
                    number=1,
                    text="Type {message_body} into the reply composer.",
                    destructive=False,
                ),
                steps[1],
            ]
        }
    )
    meta: dict[str, Any] = {
        "slug": slug,
        "name": "Reply to Gmail",
        "trajectory_id": trajectory_id,
        "created_at": _iso(0.0).replace("Z", "+00:00"),
        "parameters": [
            {"name": "message_body", "type": "string", "required": True}
        ],
        "destructive_steps": [2] if destructive_step == 2 else [],
        "preconditions": ["Chrome is open", "User is signed into Gmail"],
        "step_count": 2,
    }
    return parsed, meta


# --- write() happy path ----------------------------------------------------


def test_write_produces_parseable_skill_md(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=3)
    skills_root = tmp_path / "skills"
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])

    written = SkillWriter().write(parsed, meta, reader, skills_root)

    assert written.slug == "gmail_reply"
    assert written.path == skills_root / "gmail_reply"
    assert written.path.is_dir()

    skill_md = (written.path / "SKILL.md").read_text()
    reparsed = parse_skill_md(skill_md)
    assert reparsed == parsed

    meta_on_disk = json.loads((written.path / "skill.meta.json").read_text())
    assert meta_on_disk == meta
    validate_meta(meta_on_disk)
    validate_meta_against_markdown(meta_on_disk, skill_md)


def test_write_returns_cost_total_usd(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(
        parsed, meta, reader, tmp_path / "skills", cost_total_usd=0.1234
    )
    assert written.cost_total_usd == 0.1234


# --- collision / idempotency ----------------------------------------------


def test_write_raises_on_existing_slug(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    skills_root = tmp_path / "skills"

    SkillWriter().write(parsed, meta, reader, skills_root)
    with pytest.raises(SkillAlreadyExistsError):
        SkillWriter().write(parsed, meta, reader, skills_root)


def test_write_raises_when_index_row_exists_but_dir_does_not(
    tmp_path: Path,
) -> None:
    reader = _make_trajectory(tmp_path)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    skills_root = tmp_path / "skills"

    # Seed the index but not the directory to simulate stale state. The writer
    # must still refuse.
    db_path = index_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS skills ("
            "slug TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "trajectory_id TEXT NOT NULL, created_at TEXT NOT NULL, "
            "step_count INTEGER NOT NULL, "
            "destructive_step_count INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO skills VALUES (?,?,?,?,?,?)",
            ("gmail_reply", "x", str(uuid.uuid4()), _iso(), 1, 0),
        )
        conn.commit()

    with pytest.raises(SkillAlreadyExistsError):
        SkillWriter().write(parsed, meta, reader, skills_root)


# --- preview screenshots ---------------------------------------------------


def test_preview_screenshots_are_copies_not_symlinks(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=3)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(parsed, meta, reader, tmp_path / "skills")

    assert len(written.preview_paths) == 3
    for p in written.preview_paths:
        assert p.is_file()
        assert not p.is_symlink()
        assert p.name in {"01.png", "02.png", "03.png"}


def test_preview_cap_at_max_five(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=12)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(parsed, meta, reader, tmp_path / "skills")

    assert len(written.preview_paths) == MAX_PREVIEW_SCREENSHOTS
    assert [p.name for p in written.preview_paths] == [
        "01.png",
        "02.png",
        "03.png",
        "04.png",
        "05.png",
    ]


def test_preview_no_duplicate_padding_when_few_keyframes(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=2)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(parsed, meta, reader, tmp_path / "skills")
    # AC: 2 keyframes → 2 screenshots, NOT 5 with duplicates.
    assert len(written.preview_paths) == 2


def test_preview_dir_absent_when_no_keyframes(tmp_path: Path) -> None:
    # Build a trajectory with zero screenshot_ref events.
    traj_dir = tmp_path / "traj-noss"
    traj_dir.mkdir()
    metadata = {
        "id": str(uuid.uuid4()),
        "started_at": _iso(0.0),
        "stopped_at": _iso(1.0),
        "label": "test",
        "display_info": {"width": 100, "height": 100, "scale": 1.0},
        "app_focus_history": [
            {"at": _iso(0.0), "bundle_id": "com.example.App", "title": "X"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    (traj_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "seq": 1,
                "t": _iso(0.0),
                "kind": "click",
                "x": 1,
                "y": 1,
                "button": "left",
                "bundle_id": "com.example.App",
                "target": {
                    "label": "OK",
                    "role": "button",
                    "bundle_id": "com.example.App",
                },
            }
        )
        + "\n"
    )
    reader = TrajectoryReader(traj_dir)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(parsed, meta, reader, tmp_path / "skills")

    assert written.preview_paths == []
    assert not (written.path / "preview").exists()


# --- directory permissions -------------------------------------------------


def test_skill_dir_and_preview_have_0700_perms(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=3)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    written = SkillWriter().write(parsed, meta, reader, tmp_path / "skills")

    mode = os.stat(written.path).st_mode & 0o777
    assert mode == SKILL_DIR_MODE, oct(mode)
    preview_mode = os.stat(written.path / "preview").st_mode & 0o777
    assert preview_mode == SKILL_DIR_MODE, oct(preview_mode)


# --- atomicity / crash simulation -----------------------------------------


def test_fsync_failure_leaves_no_skill_dir_and_no_index_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=3)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    skills_root = tmp_path / "skills"

    call_count = {"n": 0}
    real_fsync = os.fsync

    def flaky_fsync(fd: int) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated fsync failure on meta.json.tmp")
        real_fsync(fd)

    monkeypatch.setattr("synthesizer.writer.os.fsync", flaky_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        SkillWriter().write(parsed, meta, reader, skills_root)

    assert not (skills_root / meta["slug"]).exists()

    db = index_db_path()
    if db.exists():
        with sqlite3.connect(db) as conn:
            try:
                rows = conn.execute(
                    "SELECT slug FROM skills WHERE slug = ?", (meta["slug"],)
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            assert rows == []


# --- SQLite index ---------------------------------------------------------


def test_index_row_contents(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path, keyframe_count=3)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    SkillWriter().write(parsed, meta, reader, tmp_path / "skills")

    with sqlite3.connect(index_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM skills WHERE slug = ?", (meta["slug"],)
        ).fetchone()
    assert row is not None
    assert row["slug"] == meta["slug"]
    assert row["name"] == meta["name"]
    assert row["trajectory_id"] == meta["trajectory_id"]
    assert row["step_count"] == 2
    assert row["destructive_step_count"] == 1


# --- validation pre-flight -------------------------------------------------


def test_write_rejects_invalid_slug_in_meta(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    meta["slug"] = "BadSlug"  # uppercase disallowed
    with pytest.raises(SkillWriteError):
        SkillWriter().write(parsed, meta, reader, tmp_path / "skills")


def test_write_rejects_markdown_meta_mismatch(tmp_path: Path) -> None:
    reader = _make_trajectory(tmp_path)
    parsed, meta = _make_skill(trajectory_id=reader.metadata["id"])
    # Drop destructive_steps but markdown still has the marker → cross-check fails.
    meta["destructive_steps"] = []
    with pytest.raises(ValidationError):
        SkillWriter().write(parsed, meta, reader, tmp_path / "skills")
    assert not (tmp_path / "skills" / meta["slug"]).exists()


# --- preview index math ---------------------------------------------------


@pytest.mark.parametrize(
    "n,cap,expected",
    [
        (0, 5, []),
        (1, 5, [0]),
        (2, 5, [0, 1]),
        (5, 5, [0, 1, 2, 3, 4]),
        (6, 5, [0, 1, 2, 4, 5]),
        (12, 5, [0, 3, 6, 8, 11]),
        (100, 5, [0, 25, 50, 74, 99]),
    ],
)
def test_pick_preview_indices(n: int, cap: int, expected: list[int]) -> None:
    assert _pick_preview_indices(n, cap) == expected
