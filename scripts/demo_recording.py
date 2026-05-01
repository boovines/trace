"""Live demo of the Recorder: capture → summarize → open the trajectory.

Usage (from the repo root):

    uv run python scripts/demo_recording.py
    # or, with TRACE_DEV_MODE=1 already exported, just
    .venv/bin/python scripts/demo_recording.py

The script:
  1. Starts a real ``RecordingSession`` writing to
     ``~/Library/Application Support/Trace[-dev]/trajectories/<uuid>/``.
  2. Tails ``events.jsonl`` and prints a one-line live summary every
     second so you can see clicks / keystrokes / app-switches as they
     happen.
  3. Stops on Enter (or Ctrl+C), prints a full breakdown of what was
     captured, and opens the trajectory directory in Finder so you can
     poke at the screenshots and the raw event log.

Permissions: needs Accessibility, Screen Recording, and Input Monitoring
granted to whatever python binary is running this script (see the
recorder smoke walkthrough). If any are missing the script aborts with
the structured error the HTTP API would have returned.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Make the recorder package importable when running directly without
# `uv run`. Harmless when uv has already installed it editable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "services" / "recorder" / "src"))

from recorder.session import (  # noqa: E402  (sys.path tweak above)
    PermissionsMissingError,
    RecordingSession,
    SessionSummary,
)
from recorder.storage import (  # noqa: E402
    default_trajectories_root,
    ensure_trajectories_root,
)


def _print_permissions_error(err: PermissionsMissingError) -> None:
    payload = err.error
    print("\n  permissions missing — recorder cannot start.")
    for name in payload["permissions"]:
        print(f"   • {name}: {payload['how_to_grant'][name]}")
    print(
        "\n  TIP: prompts auto-fire only on the first call from a binary; "
        "see tests/recorder/smoke.md for the manual System Settings path.\n"
    )


def _live_tail(events_path: Path, stop_event: threading.Event) -> None:
    """Print a one-line live counter while the recording is running."""
    counts: collections.Counter[str] = collections.Counter()
    seen_bytes = 0
    started = time.monotonic()
    while not stop_event.is_set():
        if events_path.exists():
            with events_path.open("rb") as fh:
                fh.seek(seen_bytes)
                chunk = fh.read()
                seen_bytes += len(chunk)
            if chunk:
                for raw in chunk.splitlines():
                    if not raw.strip():
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    counts[ev.get("type", "?")] += 1
        elapsed = time.monotonic() - started
        line = (
            f"  {elapsed:5.1f}s  "
            + "  ".join(f"{k}={v}" for k, v in counts.most_common())
            + " " * 8
        )
        sys.stdout.write("\r" + line[:120])
        sys.stdout.flush()
        stop_event.wait(0.5)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _summarize(traj_dir: Path) -> None:
    events_path = traj_dir / "events.jsonl"
    metadata_path = traj_dir / "metadata.json"
    screenshots_dir = traj_dir / "screenshots"

    metadata = json.loads(metadata_path.read_text())
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    counts = collections.Counter(e["type"] for e in events)
    text_inputs = [e for e in events if e["type"] == "text_input"]
    app_switches = [e for e in events if e["type"] == "app_switch"]
    clicks = [e for e in events if e["type"] == "click"]
    screenshots = sorted(screenshots_dir.glob("*.png"))
    total_size = sum(p.stat().st_size for p in screenshots) if screenshots else 0

    print("\n" + "=" * 60)
    print(f"trajectory: {traj_dir}")
    print("=" * 60)
    print(f"  events    : {len(events)} total — {dict(counts)}")
    print(f"  duration  : started {metadata['started_at']}")
    print(f"            : stopped {metadata['stopped_at']}")
    print(f"  apps seen : {len(metadata['app_focus_history'])} entries in app_focus_history")
    for entry in metadata["app_focus_history"]:
        print(f"             - {entry['name']:20s}  {entry['bundle_id']}")
    print(f"  shots     : {len(screenshots)} keyframes, {total_size / 1024:.0f} KB total")

    if app_switches:
        print("\n  app_switch events:")
        for e in app_switches[:8]:
            p = e.get("payload", {})
            print(
                f"    seq={e['seq']:>3}  "
                f"{p.get('from_name') or '<start>'} → {p.get('to_name')}"
            )
        if len(app_switches) > 8:
            print(f"    ... ({len(app_switches) - 8} more)")

    if text_inputs:
        print("\n  text_input events (what you typed):")
        for e in text_inputs[:8]:
            p = e.get("payload", {})
            text_preview = p.get("text", "")[:60].replace("\n", "\\n").replace("\r", "\\r")
            label = p.get("field_label") or "<no field label>"
            app = (e.get("app") or {}).get("name", "?")
            print(f"    seq={e['seq']:>3}  [{app}] {label:<24s}  '{text_preview}'")
        if len(text_inputs) > 8:
            print(f"    ... ({len(text_inputs) - 8} more)")

    if clicks:
        print(f"\n  click events: {len(clicks)} total")
        for e in clicks[:5]:
            t = e.get("target") or {}
            app = (e.get("app") or {}).get("name", "?")
            print(
                f"    seq={e['seq']:>3}  [{app:<14s}] role={t.get('role')!s:<14s} "
                f"label={t.get('label')!r}"
            )
        if len(clicks) > 5:
            print(f"    ... ({len(clicks) - 5} more)")

    print("\n  → opening trajectory dir in Finder")
    print("=" * 60 + "\n")


def main() -> int:
    # Default to the dev profile so a demo run never pollutes the prod
    # trajectories dir. Override by exporting TRACE_DEV_MODE=0 explicitly.
    os.environ.setdefault("TRACE_DEV_MODE", "1")

    root = ensure_trajectories_root(default_trajectories_root())
    print(f"  recording into: {root}")
    print("  press Enter to stop (or Ctrl+C)\n")

    session = RecordingSession(root)
    try:
        traj_id = session.start(label="demo")
    except PermissionsMissingError as e:
        _print_permissions_error(e)
        return 1

    traj_dir = root / traj_id
    events_path = traj_dir / "events.jsonl"
    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_live_tail, args=(events_path, stop_event), daemon=True
    )
    tail_thread.start()

    with contextlib.suppress(KeyboardInterrupt, EOFError):
        # Block until the user hits Enter; Ctrl+C also stops cleanly.
        input()

    stop_event.set()
    tail_thread.join(timeout=1.0)

    summary: SessionSummary = session.stop()
    print(
        f"\n  stopped: {summary['event_count']} events in "
        f"{summary['duration_ms'] / 1000:.1f}s"
    )

    _summarize(traj_dir)
    subprocess.run(["open", str(traj_dir)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
