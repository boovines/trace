"""Write run artifacts to disk atomically and incrementally.

A run produces four kinds of on-disk state inside ``<runs_root>/<run_id>/``:

* ``run_metadata.json`` — rewritten atomically on every status change.
  Callers must see either the full old file or the full new file, never a
  partial write. We achieve this with the classic write-tmp / fsync / rename
  idiom. The metadata is also idempotent: writing the same values twice is a
  no-op on disk content (though it still syncs).
* ``events.jsonl`` — append-only human-readable trace, one JSON line per
  event. Flushed on every write so tail-f works from another process.
* ``transcript.jsonl`` — append-only record of every agent turn, same
  append+flush discipline as events.
* ``screenshots/NNNN.png`` — zero-padded 4-digit filename. Written whole
  (no partial-file concern since PNGs are self-contained and we open with
  ``"wb"``). Magic-byte validated before hitting disk so we never persist
  corrupt PNGs into a run directory.

Thread-safety: a single ``RLock`` guards every public method. The runner's
execution loop and a hypothetical supervisor thread (e.g. the kill-switch
handler) can both call update_status concurrently without corrupting the
metadata file.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from runner.run_index import RunIndex
from runner.schema import RunMetadata

_PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC: Final[bytes] = b"\xff\xd8\xff"
_DIR_PERMS: Final[int] = 0o700
_SCREENSHOT_DIGITS: Final[int] = 4
_DOM_FRAME_DIGITS: Final[int] = 4


class RunWriter:
    """Per-run artifact writer with atomic metadata updates.

    The writer owns the run directory exclusively for its lifetime. It does
    NOT assume the directory exists on entry — the constructor creates it with
    mode ``0700`` (matching the safety invariant that all per-user data lives
    under a restrictive-permissioned profile root).
    """

    def __init__(
        self,
        *,
        run_id: str,
        skill_slug: str,
        mode: str,
        runs_root: Path,
        run_index: RunIndex | None = None,
    ) -> None:
        self._run_id = run_id
        self._skill_slug = skill_slug
        self._mode = mode
        self._run_dir = runs_root / run_id
        self._screenshots_dir = self._run_dir / "screenshots"
        # ``dom_frames/`` is created lazily on the first ``append_dom_frame``
        # call so a run that never uses tier=browser_dom never grows it.
        self._dom_frames_dir = self._run_dir / "dom_frames"
        self._dom_frames_dir_created = False
        self._dom_frame_seq = 0
        self._metadata_path = self._run_dir / "run_metadata.json"
        self._events_path = self._run_dir / "events.jsonl"
        self._transcript_path = self._run_dir / "transcript.jsonl"
        self._lock = threading.RLock()
        self._closed = False
        self._last_metadata_bytes: bytes | None = None
        self._run_index = run_index

        self._run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._run_dir, _DIR_PERMS)
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._screenshots_dir, _DIR_PERMS)

        if run_index is not None:
            run_index.upsert(
                run_id=run_id,
                skill_slug=skill_slug,
                status="pending",
                mode=mode,
            )

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def run_id(self) -> str:
        return self._run_id

    def write_metadata(self, metadata: RunMetadata) -> None:
        """Write ``run_metadata.json`` atomically.

        The write sequence is: serialize → write to ``.tmp`` sibling → fsync
        → rename over the destination. If any step raises, the ``.tmp`` file
        is removed before the exception propagates so a crashed process never
        leaves stale scratch files behind to confuse the UI.

        Idempotent: if the serialized bytes are identical to the last
        successful write, the file contents on disk are unchanged.
        """
        if metadata.run_id.hex != _hex_or_dashed(self._run_id):
            raise ValueError(
                f"RunMetadata.run_id={metadata.run_id} does not match "
                f"writer's run_id={self._run_id}"
            )
        if metadata.skill_slug != self._skill_slug:
            raise ValueError(
                f"RunMetadata.skill_slug={metadata.skill_slug!r} does not "
                f"match writer's skill_slug={self._skill_slug!r}"
            )
        if metadata.mode != self._mode:
            raise ValueError(
                f"RunMetadata.mode={metadata.mode!r} does not match "
                f"writer's mode={self._mode!r}"
            )

        payload = metadata.to_dict()
        serialized = (
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

        with self._lock:
            if serialized == self._last_metadata_bytes:
                return
            _atomic_write_bytes(self._metadata_path, serialized)
            self._last_metadata_bytes = serialized
            if self._run_index is not None:
                self._run_index.upsert_from_metadata(metadata)

    def update_status(self, metadata: RunMetadata, status: str, **fields: Any) -> RunMetadata:
        """Return a metadata object with ``status`` (and any fields) updated, writing it.

        Rebuilding the pydantic model is deliberate: it re-runs field
        validators (so ``update_status("banana")`` fails loudly) and keeps the
        on-disk file schema-valid no matter what the caller passed.
        """
        with self._lock:
            merged = metadata.model_dump() | {"status": status, **fields}
            updated = RunMetadata.model_validate(merged)
            self.write_metadata(updated)
            return updated

    def append_event(
        self,
        *,
        seq: int,
        event_type: str,
        message: str,
        step_number: int | None = None,
        screenshot_seq: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append one human-readable event to ``events.jsonl``."""
        record: dict[str, Any] = {
            "seq": seq,
            "type": event_type,
            "message": message,
            "step_number": step_number,
            "screenshot_ref": (
                _screenshot_filename(screenshot_seq) if screenshot_seq is not None else None
            ),
        }
        if timestamp_ms is not None:
            record["timestamp_ms"] = timestamp_ms
        self._append_jsonl(self._events_path, record)

    def append_transcript(
        self,
        *,
        turn: int,
        role: str,
        content: list[Any],
        input_tokens: int,
        output_tokens: int,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append one agent turn to ``transcript.jsonl``."""
        record: dict[str, Any] = {
            "turn": turn,
            "role": role,
            "content": content,
            "input_tokens_this_turn": input_tokens,
            "output_tokens_this_turn": output_tokens,
        }
        if timestamp_ms is not None:
            record["timestamp_ms"] = timestamp_ms
        self._append_jsonl(self._transcript_path, record)

    def write_screenshot(self, seq: int, png_bytes: bytes) -> Path:
        """Write ``screenshots/NNNN.png`` after validating PNG magic bytes.

        Screenshots are large enough that a rejected write is cheaper than a
        silently-corrupt run directory. Anything that does not start with the
        PNG signature is rejected.
        """
        if not isinstance(png_bytes, (bytes, bytearray)):  # pragma: no cover
            raise TypeError("png_bytes must be bytes")
        if not png_bytes.startswith(_PNG_MAGIC):
            raise ValueError(
                "screenshot bytes do not start with the PNG magic signature "
                "(89 50 4E 47 0D 0A 1A 0A); refusing to persist corrupt image"
            )
        if seq < 0:
            raise ValueError(f"screenshot seq must be non-negative, got {seq}")
        dest = self._screenshots_dir / _screenshot_filename(seq)
        with self._lock, dest.open("wb") as fh:
            fh.write(png_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        return dest

    def append_dom_frame(self, jpg_bytes: bytes) -> tuple[int, Path]:
        """Append a JPEG frame to ``dom_frames/`` and return ``(seq, path)``.

        Used by the browser_dom tier's live-stream observability path:
        the dispatcher captures a frame after every successful action,
        the executor / observing writer feeds the bytes here, and the
        gateway serves the resulting file via
        ``GET /run/{run_id}/dom_frames/{filename}``.

        Sequence numbers are assigned monotonically per writer
        instance, so the dashboard can render a deterministic ordering
        without consulting filesystem timestamps. The ``dom_frames/``
        subdirectory is created lazily on the first frame so runs
        that never use the browser_dom tier never accumulate the
        empty dir.
        """
        if not isinstance(jpg_bytes, (bytes, bytearray)):  # pragma: no cover
            raise TypeError("jpg_bytes must be bytes")
        if not jpg_bytes.startswith(_JPEG_MAGIC):
            raise ValueError(
                "dom frame bytes do not start with the JPEG magic signature "
                "(FF D8 FF); refusing to persist corrupt image"
            )
        with self._lock:
            if not self._dom_frames_dir_created:
                self._dom_frames_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(self._dom_frames_dir, _DIR_PERMS)
                self._dom_frames_dir_created = True
            seq = self._dom_frame_seq
            self._dom_frame_seq += 1
            dest = self._dom_frames_dir / _dom_frame_filename(seq)
            with dest.open("wb") as fh:
                fh.write(jpg_bytes)
                fh.flush()
                os.fsync(fh.fileno())
        return seq, dest

    def close(self) -> None:
        """Mark the writer closed. Idempotent."""
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")
        with self._lock, path.open("ab") as fh:
            fh.write(encoded)
            fh.flush()


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically via tmp + fsync + rename.

    On any exception the ``.tmp`` file is removed before the error
    propagates. ``dest`` itself is never touched until the final rename, so a
    prior successful write remains intact.
    """
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
        raise


def _screenshot_filename(seq: int) -> str:
    return f"{seq:0{_SCREENSHOT_DIGITS}d}.png"


def _dom_frame_filename(seq: int) -> str:
    return f"{seq:0{_DOM_FRAME_DIGITS}d}.jpg"


def dom_frame_filename(seq: int) -> str:
    """Public: ``NNNN.jpg`` filename for a DOM frame seq.

    Exposed so the observing writer's WS broadcast and the gateway's
    file-serving route can construct the same canonical filename
    without duplicating the format string.
    """
    return _dom_frame_filename(seq)


def _hex_or_dashed(run_id: str) -> str:
    """Normalize a run_id string to a UUID hex (lowercase, no dashes).

    Our writer accepts the ``run_id`` as a plain string at construction time
    but ``RunMetadata.run_id`` is a ``uuid.UUID``. This helper lets us compare
    them without forcing callers to care about dashed vs hex.
    """
    return run_id.replace("-", "").lower()


def get_run_dir_perms(path: Path) -> int:
    """Return the permission bits (lowest 9) for ``path``.

    Tiny helper mostly for tests — keeps ``stat`` / bitmask fiddling out of
    the test body.
    """
    return stat.S_IMODE(path.stat().st_mode)


__all__ = [
    "RunWriter",
    "get_run_dir_perms",
]
