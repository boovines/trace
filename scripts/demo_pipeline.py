"""End-to-end demo: trajectory on disk → preprocessed digest → SKILL.md draft.

Usage::

    # Real LLM call (requires ANTHROPIC_API_KEY in the environment):
    uv run python scripts/demo_pipeline.py

    # Pin a specific trajectory id:
    uv run python scripts/demo_pipeline.py --trajectory 34205da8-1856-...

    # Force the bundled gmail_reply fixture instead of latest dev recording:
    uv run python scripts/demo_pipeline.py --fixture gmail_reply

The script:

1. Locates a trajectory (the most recent in
   ``~/Library/Application Support/Trace-dev/trajectories/``, the user-
   supplied id, or one of the five reference fixtures).
2. Loads it via :class:`TrajectoryReader`, prints a summary
   (event-type counts, app_focus_history, duration).
3. Runs :func:`preprocess_trajectory` and prints the digest stats
   (number of entries, screenshots included, estimated input tokens).
4. If ``ANTHROPIC_API_KEY`` is set, calls
   :func:`generate_draft` against the real API and prints the resulting
   ``SKILL.md`` plus parsed metadata. Otherwise prints a clear
   "set-the-key" message and exits.

The point of the demo is to show the recorder → synthesizer
contract working end-to-end on real artifacts produced by the
recorder, not to replace the FastAPI surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "services" / "recorder" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "services" / "synthesizer" / "src"))

# Recorder-side imports (used to find the dev profile dir).
from recorder.storage import default_trajectories_root  # noqa: E402

# Synthesizer-side imports.
from synthesizer.draft import build_user_content, generate_draft  # noqa: E402
from synthesizer.llm_client import LLMClient  # noqa: E402
from synthesizer.preprocess import preprocess_trajectory  # noqa: E402
from synthesizer.trajectory_reader import TrajectoryReader  # noqa: E402

REFERENCE_SLUGS = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


def _latest_trajectory_dir(root: Path) -> Path | None:
    """Return the most recent *complete* recording, or None.

    Filters out interrupted recordings — a trajectory dir is only usable
    when both ``metadata.json`` and ``events.jsonl`` exist on disk.
    """
    candidates = [
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "metadata.json").is_file()
        and (d / "events.jsonl").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _resolve_trajectory(args: argparse.Namespace) -> Path:
    if args.fixture:
        if args.fixture not in REFERENCE_SLUGS:
            raise SystemExit(f"unknown fixture {args.fixture!r}; pick one of {REFERENCE_SLUGS}")
        return _REPO_ROOT / "fixtures" / "trajectories" / args.fixture

    # Default to dev profile so a typical demo run stays out of the prod dir.
    os.environ.setdefault("TRACE_DEV_MODE", "1")
    root = default_trajectories_root()

    if args.trajectory:
        path = root / args.trajectory
        if not path.is_dir():
            raise SystemExit(f"trajectory {args.trajectory!r} not found under {root}")
        return path

    latest = _latest_trajectory_dir(root) if root.is_dir() else None
    if latest is None:
        # Fall back to a bundled fixture so the demo always has *something*.
        print(
            f"  no recordings found under {root}; "
            "falling back to fixtures/trajectories/gmail_reply"
        )
        return _REPO_ROOT / "fixtures" / "trajectories" / "gmail_reply"
    return latest


def _print_recorder_side(reader: TrajectoryReader) -> None:
    summary = reader.summary()
    meta = reader.metadata
    print("=" * 72)
    print(f"  trajectory: {reader.directory}")
    print(f"  label:      {meta.get('label')}")
    print(f"  duration:   {summary['duration_ms'] / 1000:.1f}s "
          f"({meta.get('started_at')} → {meta.get('stopped_at')})")
    print(f"  events:     {summary['event_count']} total — "
          f"clicks={summary['click_count']}, "
          f"text_inputs={summary['text_input_count']}, "
          f"app_switches={summary['app_switch_count']}, "
          f"keyframes={summary['keyframe_count']}")
    history = summary.get("app_focus_history") or []
    print(f"  apps seen:  {len(history)}")
    for entry in history[:8]:
        name = entry.get("name") or entry.get("title") or entry.get("bundle_id")
        print(f"              - {name}")
    print("=" * 72)


def _print_preprocess_side(preprocessed: object, reader: TrajectoryReader) -> None:
    digest = preprocessed.digest  # type: ignore[attr-defined]
    print("\n[preprocess]")
    print(f"  digest entries:        {preprocessed.digest_entry_count}")  # type: ignore[attr-defined]
    print(f"  screenshots included:  {preprocessed.screenshots_included}")  # type: ignore[attr-defined]
    print(f"  est. input tokens:     {preprocessed.estimated_input_tokens:,}")  # type: ignore[attr-defined]
    if digest:
        print("  first 6 digest entries:")
        for entry in digest[:6]:
            print(f"    seq={entry.seq:>3}  {entry.kind:<14s}  {entry.summary_text[:80]}")
    user_content = build_user_content(preprocessed, reader)
    image_blocks = sum(1 for b in user_content if b.get("type") == "image")
    text_blocks = [b for b in user_content if b.get("type") == "text"]
    text_chars = sum(len(b.get("text", "")) for b in text_blocks)
    print(
        f"  user_content shape:    {len(text_blocks)} text block(s), "
        f"{image_blocks} image block(s), {text_chars:,} text chars total"
    )


def _print_skill(draft: object) -> None:
    print("\n" + "=" * 72)
    print("  SKILL.md")
    print("=" * 72)
    parsed = draft.parsed  # type: ignore[attr-defined]
    print(f"  title:       {parsed.title}")
    print(f"  description: {parsed.description}")
    print(f"  parameters:  {[p.name for p in parsed.parameters]}")
    print(f"  steps:       {len(parsed.steps)} (destructive: "
          f"{[s.number for s in parsed.steps if s.destructive]})")
    print("\n  --- markdown ---")
    print(parsed.raw_markdown)
    print("  --- meta.json ---")
    print(json.dumps(draft.meta, indent=2))  # type: ignore[attr-defined]
    questions = draft.questions  # type: ignore[attr-defined]
    if questions:
        print(f"\n  follow-up questions: {len(questions)}")
        for q in questions:
            print(f"    [{q.id}] {q.question}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory", help="trajectory id under the dev profile dir (default: latest)"
    )
    parser.add_argument(
        "--fixture", help=f"name of a bundled reference fixture: {', '.join(REFERENCE_SLUGS)}"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="stop after preprocess; do not call the Anthropic API",
    )
    args = parser.parse_args()

    traj_dir = _resolve_trajectory(args)
    print(f"  using trajectory: {traj_dir}\n")
    reader = TrajectoryReader(traj_dir)
    _print_recorder_side(reader)

    preprocessed = preprocess_trajectory(reader)
    _print_preprocess_side(preprocessed, reader)

    if args.no_llm:
        print("\n  --no-llm passed; stopping before LLM call.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ANTHROPIC_API_KEY not set — skipping LLM call.")
        print("  Set it (e.g. `export ANTHROPIC_API_KEY=sk-ant-...`) and re-run")
        print("  to generate a real SKILL.md draft.")
        return 0

    print("\n[generating draft via Claude - this takes ~10-30s]")
    client = LLMClient()
    try:
        draft = generate_draft(preprocessed, client, reader=reader)
    except Exception as exc:  # surface the failure but exit cleanly
        print(f"\n  draft generation failed: {exc}")
        return 1

    _print_skill(draft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
