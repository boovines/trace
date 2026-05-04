"""ObservingRunWriter.append_dom_frame tests — Step 4.3 commit A.

Narrow scope: confirms the dom_frame WS event matches what the
dashboard expects (run_id, seq, filename, url) and that the underlying
parent ``append_dom_frame`` still writes the file. The base
:class:`RunWriter.append_dom_frame` behaviour (lazy mkdir, JPEG magic
validation, monotonic seq) is covered separately in
``test_run_writer.py``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from runner.event_stream import EventBroadcaster
from runner.observing_writer import ObservingRunWriter

_MIN_JPEG: bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00" + b"\x00" * 80


class _CollectingBroadcaster(EventBroadcaster):
    """Captures every published payload so tests can inspect them."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, dict]] = []  # type: ignore[type-arg]

    def publish(self, run_id: str, payload: dict) -> None:  # type: ignore[type-arg, override]
        self.published.append((run_id, payload))
        super().publish(run_id, payload)


def _make(tmp_path: Path) -> tuple[ObservingRunWriter, _CollectingBroadcaster, str]:
    run_id = str(uuid.uuid4())
    broadcaster = _CollectingBroadcaster()
    writer = ObservingRunWriter(
        run_id=run_id,
        skill_slug="gmail_reply",
        mode="dry_run",
        runs_root=tmp_path,
        broadcaster=broadcaster,
    )
    return writer, broadcaster, run_id


def test_append_dom_frame_broadcasts_ws_event(tmp_path: Path) -> None:
    writer, broadcaster, run_id = _make(tmp_path)
    seq, dest = writer.append_dom_frame(_MIN_JPEG)

    assert seq == 0
    assert dest.name == "0000.jpg"
    assert dest.read_bytes() == _MIN_JPEG

    # The broadcaster should have a single dom_frame payload.
    assert len(broadcaster.published) == 1
    pub_run_id, payload = broadcaster.published[0]
    assert pub_run_id == run_id
    assert payload == {
        "type": "dom_frame",
        "run_id": run_id,
        "seq": 0,
        "filename": "0000.jpg",
        "url": f"/run/{run_id}/dom_frames/0000.jpg",
    }


def test_append_dom_frame_publishes_one_event_per_frame(tmp_path: Path) -> None:
    writer, broadcaster, _run_id = _make(tmp_path)
    writer.append_dom_frame(_MIN_JPEG)
    writer.append_dom_frame(_MIN_JPEG)
    writer.append_dom_frame(_MIN_JPEG)
    seqs = [p[1]["seq"] for p in broadcaster.published]
    assert seqs == [0, 1, 2]
    filenames = [p[1]["filename"] for p in broadcaster.published]
    assert filenames == ["0000.jpg", "0001.jpg", "0002.jpg"]
