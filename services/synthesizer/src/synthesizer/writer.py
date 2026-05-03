"""Atomic on-disk writer for synthesized skills.

Given a final :class:`~synthesizer.skill_doc.ParsedSkill` + validated meta dict
+ the source :class:`~synthesizer.trajectory_reader.TrajectoryReader`, the
writer:

1. Runs every last-chance validation gate (round-trip parse, schema,
   cross-check, slug regex, unique slug).
2. Writes ``SKILL.md`` and ``skill.meta.json`` atomically (``.tmp`` +
   ``fsync`` + ``rename``) so a crash mid-write never leaves a half-written
   skill on disk.
3. Copies up to 5 preview screenshots from the trajectory into ``preview/``
   (renumbered ``01.png`` … ``05.png`` so downstream consumers never need to
   know the original seq numbers).
4. Commits an entry to the shared SQLite index at
   ``<trace_data_dir>/index.db`` as the very last step — "atomic
   write then register" means a partial write never leaves a ghost row.

The writer is the only module in the synthesizer that is allowed to create
files under ``skills/`` or touch ``index.db``. Every callsite — session
approval, tests, any future CLI — must go through :class:`SkillWriter` to
keep on-disk state and index state in lock-step.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthesizer.llm_client import trace_data_dir
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.skill_doc import (
    ParsedSkill,
    parse_skill_md,
    render_skill_md,
)
from synthesizer.slug import SLUG_REGEX
from synthesizer.trajectory_reader import TrajectoryReader

__all__ = [
    "INDEX_DB_FILENAME",
    "MAX_PREVIEW_SCREENSHOTS",
    "SKILL_DIR_MODE",
    "SkillAlreadyExistsError",
    "SkillWriteError",
    "SkillWriter",
    "WrittenSkill",
    "index_db_path",
]


MAX_PREVIEW_SCREENSHOTS: int = 5
SKILL_DIR_MODE: int = 0o700
INDEX_DB_FILENAME: str = "index.db"

_SKILL_FILENAME = "SKILL.md"
_META_FILENAME = "skill.meta.json"
_PREVIEW_DIRNAME = "preview"

_INDEX_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS skills (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    step_count INTEGER NOT NULL,
    destructive_step_count INTEGER NOT NULL
)
"""


class SkillWriteError(RuntimeError):
    """Raised when a skill fails a pre-write validation or a write step."""


class SkillAlreadyExistsError(SkillWriteError):
    """Raised when the target slug already has a directory or an index row.

    The caller is responsible for resolving the slug beforehand (see
    :func:`synthesizer.slug.resolve_unique_slug`); the writer is strict by
    design so silent overwrites are impossible.
    """


@dataclass(frozen=True)
class WrittenSkill:
    """Return value of :meth:`SkillWriter.write`.

    ``path`` is the absolute path to the newly-created skill directory;
    ``preview_paths`` lists the PNGs written under ``<path>/preview/`` in the
    order they were selected from the trajectory. ``cost_total_usd`` mirrors
    whatever the session accumulated before :meth:`write` was called — it is
    passed through unchanged so callers can display a final "this synthesis
    cost $X" line without re-reading ``costs.jsonl``.
    """

    slug: str
    path: Path
    preview_paths: list[Path]
    cost_total_usd: float


def index_db_path() -> Path:
    """Absolute path to the SQLite skills index for the active profile."""
    return trace_data_dir() / INDEX_DB_FILENAME


class SkillWriter:
    """Atomic writer for synthesized skills.

    Stateless — holds no configuration beyond what
    :func:`synthesizer.llm_client.trace_data_dir` already resolves. Instantiate
    once per session (or per request) for tidiness; constructing a new writer
    per call is cheap.
    """

    def write(
        self,
        parsed: ParsedSkill,
        meta: dict[str, Any],
        reader: TrajectoryReader,
        skills_root: Path,
        *,
        cost_total_usd: float = 0.0,
    ) -> WrittenSkill:
        """Validate, write, and index a skill on disk.

        Raises :class:`SkillAlreadyExistsError` if the slug already has an
        on-disk directory or an index row. Raises :class:`SkillWriteError` on
        any other validation or write failure; on failure the writer guarantees
        no partial skill directory is left behind and no index row is created.
        """
        slug = meta.get("slug")
        if not isinstance(slug, str) or not SLUG_REGEX.match(slug):
            raise SkillWriteError(
                f"meta['slug'] must match {SLUG_REGEX.pattern}, got {slug!r}"
            )

        # Final pre-flight: cross-check parsed vs meta and prove round-trip
        # stability. Any drift here would otherwise surface to the Runner as a
        # mystery parser failure weeks later.
        self._validate_round_trip(parsed)
        rendered = render_skill_md(parsed)
        validate_meta(meta)
        validate_meta_against_markdown(meta, rendered)

        skills_root = Path(skills_root)
        skills_root.mkdir(parents=True, exist_ok=True, mode=SKILL_DIR_MODE)
        skill_dir = skills_root / slug

        # Existence check covers both on-disk collisions and index rows — they
        # must stay in sync. Resolving the slug is the caller's job.
        if skill_dir.exists():
            raise SkillAlreadyExistsError(
                f"Skill directory already exists: {skill_dir}. "
                "Call resolve_unique_slug() before write()."
            )
        if self._index_has_slug(slug):
            raise SkillAlreadyExistsError(
                f"index.db already has a row for slug {slug!r}. "
                "Call resolve_unique_slug() before write()."
            )

        preview_paths: list[Path] = []
        try:
            skill_dir.mkdir(mode=SKILL_DIR_MODE)
            self._atomic_write_text(skill_dir / _SKILL_FILENAME, rendered)
            self._atomic_write_text(
                skill_dir / _META_FILENAME,
                json.dumps(meta, indent=2, sort_keys=True) + "\n",
            )
            preview_paths = self._copy_preview_screenshots(reader, skill_dir)
        except Exception:
            # Roll back the entire skill directory. Index has NOT been touched
            # yet so there's nothing to undo there.
            shutil.rmtree(skill_dir, ignore_errors=True)
            raise

        # Commit phase: any failure here is treated as terminal — the files
        # are already on disk, so if the insert fails we rip the dir back out
        # to preserve the "either fully registered or fully absent" invariant.
        try:
            self._insert_index_row(slug, parsed, meta)
        except Exception:
            shutil.rmtree(skill_dir, ignore_errors=True)
            raise

        return WrittenSkill(
            slug=slug,
            path=skill_dir,
            preview_paths=preview_paths,
            cost_total_usd=float(cost_total_usd),
        )

    # --- validation helpers -------------------------------------------------

    @staticmethod
    def _validate_round_trip(parsed: ParsedSkill) -> None:
        rendered = render_skill_md(parsed)
        reparsed = parse_skill_md(rendered)
        if reparsed != parsed:
            raise SkillWriteError(
                "ParsedSkill failed the render→parse round-trip check. "
                "This indicates a bug in skill_doc or a ParsedSkill shape "
                "that the renderer cannot reproduce faithfully."
            )

    # --- atomic file writes -------------------------------------------------

    @staticmethod
    def _atomic_write_text(target: Path, content: str) -> None:
        """Write ``content`` to ``target`` atomically.

        Sequence: open ``<target>.tmp`` for writing → ``write`` → ``flush`` →
        ``os.fsync`` → ``os.rename`` over ``target``. An exception anywhere
        removes the ``.tmp`` before re-raising so a crash mid-fsync does not
        leave stray temp files for the next writer to trip over.
        """
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, target)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    # --- preview screenshots ------------------------------------------------

    def _copy_preview_screenshots(
        self, reader: TrajectoryReader, skill_dir: Path
    ) -> list[Path]:
        """Copy up to :data:`MAX_PREVIEW_SCREENSHOTS` keyframes into ``preview/``.

        Selection: keyframes in seq order, thinned to at most 5 spread across
        the timeline (first, ~25%, ~50%, ~75%, last). Fewer than 5 keyframes
        returns everything available; zero keyframes returns an empty list and
        no ``preview/`` directory is created.
        """
        keyframes: list[Path] = []
        for event in reader.iter_events():
            if not event.screenshot_ref:
                continue
            path = reader.directory / event.screenshot_ref
            if path.is_file():
                keyframes.append(path)

        if not keyframes:
            return []

        selected = _pick_preview_indices(len(keyframes), MAX_PREVIEW_SCREENSHOTS)
        sources = [keyframes[i] for i in selected]

        preview_dir = skill_dir / _PREVIEW_DIRNAME
        preview_dir.mkdir(mode=SKILL_DIR_MODE)
        written: list[Path] = []
        for index, src in enumerate(sources, start=1):
            dst = preview_dir / f"{index:02d}.png"
            # shutil.copy2 preserves mtime/permissions but resolves symlinks —
            # we want a real copy so the skill is self-contained (the PRD AC
            # is explicit: copies, not symlinks).
            shutil.copy2(src, dst)
            written.append(dst)
        return written

    # --- SQLite index -------------------------------------------------------

    def _index_has_slug(self, slug: str) -> bool:
        path = index_db_path()
        if not path.exists():
            return False
        with sqlite3.connect(path) as conn:
            _ensure_schema(conn)
            cur = conn.execute("SELECT 1 FROM skills WHERE slug = ?", (slug,))
            return cur.fetchone() is not None

    def _insert_index_row(
        self, slug: str, parsed: ParsedSkill, meta: dict[str, Any]
    ) -> None:
        path = index_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        destructive_count = len(meta.get("destructive_steps", []))
        with sqlite3.connect(path) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO skills "
                "(slug, name, trajectory_id, created_at, step_count, "
                "destructive_step_count) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    slug,
                    meta["name"],
                    meta["trajectory_id"],
                    meta.get("created_at") or _now_iso(),
                    len(parsed.steps),
                    destructive_count,
                ),
            )
            conn.commit()


# --- helpers ---------------------------------------------------------------


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_INDEX_SCHEMA_SQL)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _pick_preview_indices(n: int, cap: int) -> list[int]:
    """Return indices into a list of length ``n``, capped at ``cap`` entries.

    When ``n <= cap`` returns ``[0, 1, ..., n-1]``. Otherwise returns indices
    spread across the timeline: first, ~25%, ~50%, ~75%, last (or as many as
    ``cap`` allows) — deduplicated while preserving order so small ``n`` that
    still exceed ``cap`` don't double-count the same frame.
    """
    if n <= cap:
        return list(range(n))
    # cap is always 5 in practice; compute fractional positions in [0, n-1].
    if cap == 1:
        return [0]
    raw: list[int] = []
    for i in range(cap):
        # Spread evenly across [0, n-1] inclusive.
        pos = round(i * (n - 1) / (cap - 1))
        raw.append(pos)
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in raw:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered
