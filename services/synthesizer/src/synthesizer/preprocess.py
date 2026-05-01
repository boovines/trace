"""Pre-LLM trajectory preprocessing: collapse, summarize, and pick keyframes.

The raw output of the Recorder can contain tens of thousands of events for a
multi-minute recording — far more than we want to push into a Claude prompt.
This module reduces a :class:`~synthesizer.trajectory_reader.TrajectoryReader`
to a :class:`PreprocessedTrajectory`: an ordered list of
:class:`DigestEntry` objects plus per-call statistics.

The reduction has three deterministic stages (no LLM calls):

1. **Noise filter** — drops ``mouse_move``, ``mouse_down``, ``mouse_up`` (clicks
   already capture intent; raw mouse movement is pure noise).
2. **Collapse** — consecutive same-app ``scroll`` events within
   :data:`SCROLL_COLLAPSE_GAP_MS` fold into a synthetic ``scroll_run`` entry
   carrying ``{total_delta, duration_ms}``. Gaps of ≥
   :data:`IDLE_THRESHOLD_MS` between retained events inject a synthetic
   ``idle`` entry with ``{duration_ms}``.
3. **Keyframe selection** — at most :data:`MAX_KEYFRAMES` (20) of the
   surviving entries retain a ``screenshot_ref`` to include in the prompt.
   Selection priority: (1) every ``app_switch`` with a screenshot, (2) the
   screenshot immediately preceding each distinct click cluster, (3) uniform
   time-spaced fillers from the remaining candidates.

Invariant: every ``click``, ``text_input``, and ``app_switch`` from the
source trajectory is preserved unmodified in the digest. Only scrolls are
collapsed, and only mouse-motion events are dropped. Downstream prompts can
treat the digest as the semantic backbone of the workflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from synthesizer.trajectory_reader import Event, TrajectoryReader

__all__ = [
    "APP_NAMES",
    "CLICK_CLUSTER_GAP_MS",
    "IDLE_THRESHOLD_MS",
    "MAX_KEYFRAMES",
    "SCROLL_COLLAPSE_GAP_MS",
    "TOKENS_PER_IMAGE",
    "DigestEntry",
    "PreprocessedTrajectory",
    "preprocess_trajectory",
]


MAX_KEYFRAMES: int = 20
SCROLL_COLLAPSE_GAP_MS: int = 3_000
IDLE_THRESHOLD_MS: int = 5_000
CLICK_CLUSTER_GAP_MS: int = 2_000
TOKENS_PER_IMAGE: int = 1_500
_CHARS_PER_TOKEN: float = 4.0

# Best-effort bundle_id → human name map. Unknown bundle_ids fall through
# to the bundle_id itself, which the LLM can still make sense of.
APP_NAMES: dict[str, str] = {
    "com.google.Chrome": "Chrome",
    "com.apple.Safari": "Safari",
    "com.apple.finder": "Finder",
    "com.apple.mail": "Mail",
    "com.apple.Notes": "Notes",
    "com.apple.iCal": "Calendar",
    "com.apple.Terminal": "Terminal",
    "com.tinyspeck.slackmacgap": "Slack",
    "com.microsoft.VSCode": "VS Code",
}


class DigestEntry(BaseModel):
    """One entry in the digest fed to Claude during draft generation.

    ``seq`` is the source event's seq for real events. For synthetic
    ``scroll_run`` entries it is the seq of the first collapsed scroll. For
    synthetic ``idle`` entries it is ``0`` (no source event; 0 is never a
    valid event seq per the trajectory schema).

    ``screenshot_ref`` is populated only for keyframes selected by the
    prompt-fitting pass; non-keyframe entries carry ``None``.

    ``payload`` carries kind-specific extras:

    * ``scroll_run``: ``{total_delta: float, duration_ms: int}``
    * ``idle``: ``{duration_ms: int}``
    * all others: empty
    """

    model_config = ConfigDict(frozen=True)

    seq: int = Field(..., ge=0)
    timestamp_ms: int = Field(..., ge=0)
    kind: str
    summary_text: str
    screenshot_ref: str | None = None
    app_bundle_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PreprocessedTrajectory(BaseModel):
    """Result of :func:`preprocess_trajectory`: ordered digest plus stats."""

    model_config = ConfigDict(frozen=True)

    digest: list[DigestEntry]
    original_event_count: int
    digest_entry_count: int
    screenshots_included: int
    estimated_input_tokens: int


# --- helpers --------------------------------------------------------------


def _parse_iso_ms(value: str) -> float:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).timestamp() * 1_000


def _app_name(bundle_id: str | None) -> str:
    if not bundle_id:
        return "the active app"
    return APP_NAMES.get(bundle_id, bundle_id)


_NOISE_KINDS: frozenset[str] = frozenset({"mouse_move", "mouse_down", "mouse_up"})


def _summary_for_event(event: Event) -> str:
    """Deterministic 1-sentence description — no LLM involved."""
    app = _app_name(event.bundle_id)
    kind = event.kind
    if kind == "click":
        target = event.target or {}
        label = target.get("label", "").strip()
        role = target.get("role", "").strip() or "element"
        if label:
            return f'Clicked {role} labeled "{label}" in {app}'
        return f"Clicked in {app}"
    if kind == "text_input":
        text = event.text or ""
        if len(text) > 60:
            text = text[:60] + "…"
        return f'Typed "{text}" in {app}'
    if kind == "app_switch":
        return f"Switched to {app}"
    if kind == "app_focus":
        return f"Focused {app}"
    if kind == "scroll":
        return f"Scrolled in {app}"
    if kind in ("key_down", "key_up"):
        mods = event.modifiers or []
        key = event.key or ""
        combo = "+".join([*mods, key]) if mods else key or "key"
        verb = "Pressed" if kind == "key_down" else "Released"
        return f"{verb} {combo} in {app}"
    if kind == "screenshot":
        return f"Screenshot in {app}"
    if kind == "annotation":
        return (event.note or "").strip() or f"Annotation in {app}"
    return f"{kind} event in {app}"


def _scroll_delta(event: Event) -> float:
    """Extract a scalar delta from a scroll event.

    The trajectory schema exposes numeric ``x``/``y`` on every event; for
    scrolls we treat ``y`` as the vertical delta (horizontal scrolls
    contribute via ``x``). Returns 0.0 when neither is present.
    """
    y = event.y if event.y is not None else 0.0
    x = event.x if event.x is not None else 0.0
    return y + x


def _collapse_stream(
    events: list[Event], t0_ms: float
) -> list[DigestEntry]:
    """First-pass collapse: drop noise, fold scroll runs, insert idle gaps."""
    entries: list[DigestEntry] = []

    # scroll-run accumulator
    scroll_first: Event | None = None
    scroll_last: Event | None = None
    scroll_total_delta: float = 0.0

    def flush_scroll() -> None:
        nonlocal scroll_first, scroll_last, scroll_total_delta
        if scroll_first is None or scroll_last is None:
            return
        start_ms = _parse_iso_ms(scroll_first.t)
        end_ms = _parse_iso_ms(scroll_last.t)
        entries.append(
            DigestEntry(
                seq=scroll_first.seq,
                timestamp_ms=int(start_ms - t0_ms),
                kind="scroll_run",
                summary_text=(
                    f"Scrolled in {_app_name(scroll_first.bundle_id)} "
                    f"(Δ={scroll_total_delta:.0f}, "
                    f"{int(end_ms - start_ms)}ms)"
                ),
                screenshot_ref=None,
                app_bundle_id=scroll_first.bundle_id,
                payload={
                    "total_delta": scroll_total_delta,
                    "duration_ms": int(end_ms - start_ms),
                },
            )
        )
        scroll_first = None
        scroll_last = None
        scroll_total_delta = 0.0

    last_retained_ms: float | None = None

    for event in events:
        if event.kind in _NOISE_KINDS:
            continue

        event_ms = _parse_iso_ms(event.t)

        # Scroll accumulation or flush
        if event.kind == "scroll":
            if scroll_first is not None and scroll_last is not None:
                same_app = event.bundle_id == scroll_first.bundle_id
                within_gap = (
                    event_ms - _parse_iso_ms(scroll_last.t) <= SCROLL_COLLAPSE_GAP_MS
                )
                if same_app and within_gap:
                    scroll_last = event
                    scroll_total_delta += _scroll_delta(event)
                    continue
                flush_scroll()
            scroll_first = event
            scroll_last = event
            scroll_total_delta = _scroll_delta(event)
            continue

        # Non-scroll event: flush any open scroll run first
        flush_scroll()

        # Idle injection: gap since the previous retained event
        if last_retained_ms is not None and event_ms - last_retained_ms >= IDLE_THRESHOLD_MS:
            gap_ms = int(event_ms - last_retained_ms)
            entries.append(
                DigestEntry(
                    seq=0,
                    timestamp_ms=int(last_retained_ms - t0_ms),
                    kind="idle",
                    summary_text=f"Idle for {gap_ms // 1000}s",
                    screenshot_ref=None,
                    app_bundle_id=None,
                    payload={"duration_ms": gap_ms},
                )
            )

        entries.append(
            DigestEntry(
                seq=event.seq,
                timestamp_ms=int(event_ms - t0_ms),
                kind=event.kind,
                summary_text=_summary_for_event(event),
                # keyframe selection assigns screenshot_ref in a later pass;
                # stash the source ref on app_bundle_id-independent entries
                # via payload so selection can reach it without re-querying
                # the reader.
                screenshot_ref=None,
                app_bundle_id=event.bundle_id,
                payload=(
                    {"source_screenshot_ref": event.screenshot_ref}
                    if event.screenshot_ref
                    else {}
                ),
            )
        )
        last_retained_ms = event_ms

    # Flush trailing scroll run (if any)
    flush_scroll()

    return entries


def _find_click_cluster_heads(entries: list[DigestEntry]) -> list[int]:
    """Return indices of entries that are the FIRST click in a cluster.

    Two clicks belong to the same cluster when the gap between them is
    under :data:`CLICK_CLUSTER_GAP_MS` AND no app_switch interrupts them.
    """
    heads: list[int] = []
    prev_click_idx: int | None = None
    for i, entry in enumerate(entries):
        if entry.kind == "app_switch":
            # an app_switch breaks any cluster in progress
            prev_click_idx = None
            continue
        if entry.kind != "click":
            continue
        if prev_click_idx is None:
            heads.append(i)
        else:
            gap = entry.timestamp_ms - entries[prev_click_idx].timestamp_ms
            if gap > CLICK_CLUSTER_GAP_MS:
                heads.append(i)
        prev_click_idx = i
    return heads


def _select_keyframes(entries: list[DigestEntry]) -> set[int]:
    """Return the set of entry indices that should carry a screenshot.

    Candidates are entries whose source event had a ``screenshot_ref``
    (stashed in ``payload.source_screenshot_ref`` by :func:`_collapse_stream`).
    """
    candidates: list[int] = [
        i
        for i, e in enumerate(entries)
        if e.payload.get("source_screenshot_ref")
    ]
    if len(candidates) <= MAX_KEYFRAMES:
        return set(candidates)

    selected: set[int] = set()

    # Priority 1: every app_switch with a screenshot (up to the cap)
    for i in candidates:
        if entries[i].kind == "app_switch":
            selected.add(i)
            if len(selected) >= MAX_KEYFRAMES:
                return selected

    # Priority 2: the candidate immediately at-or-before each click cluster head
    for head_idx in _find_click_cluster_heads(entries):
        best: int | None = None
        for c in candidates:
            if c <= head_idx:
                best = c
            else:
                break
        if best is not None and best not in selected:
            selected.add(best)
            if len(selected) >= MAX_KEYFRAMES:
                return selected

    # Priority 3: fill via uniform sampling across remaining candidates
    remaining = [i for i in candidates if i not in selected]
    needed = MAX_KEYFRAMES - len(selected)
    if needed > 0 and remaining:
        if needed >= len(remaining):
            selected.update(remaining)
        else:
            step = len(remaining) / needed
            for k in range(needed):
                idx = int(k * step)
                selected.add(remaining[idx])

    return selected


def _apply_keyframe_selection(
    entries: list[DigestEntry], selected: set[int]
) -> list[DigestEntry]:
    """Rebuild entries with ``screenshot_ref`` set on selected indices only.

    Also scrubs the internal ``source_screenshot_ref`` stash from every
    entry's payload so the digest that leaves this module carries only
    public fields.
    """
    rebuilt: list[DigestEntry] = []
    for i, entry in enumerate(entries):
        source_ref = entry.payload.get("source_screenshot_ref")
        clean_payload = {
            k: v for k, v in entry.payload.items() if k != "source_screenshot_ref"
        }
        new_ref: str | None = source_ref if (i in selected and source_ref) else None
        rebuilt.append(
            entry.model_copy(
                update={"screenshot_ref": new_ref, "payload": clean_payload}
            )
        )
    return rebuilt


def _estimate_tokens(entries: list[DigestEntry]) -> int:
    """Rough heuristic: 4 chars/token for text + fixed per-image cost.

    Purpose is prompt-fit planning, not billing accuracy. Target accuracy is
    within ±20% of a tiktoken count on synthesizer-shaped text, and screenshot
    cost dominates anyway.
    """
    text_chars = 0
    image_count = 0
    for entry in entries:
        # Structural framing (kind + timestamp + newline separators) — use
        # the summary length + a small constant per entry.
        text_chars += len(entry.summary_text) + 16
        if entry.screenshot_ref:
            image_count += 1
    return int(text_chars / _CHARS_PER_TOKEN) + image_count * TOKENS_PER_IMAGE


# --- public entry point ---------------------------------------------------


def preprocess_trajectory(reader: TrajectoryReader) -> PreprocessedTrajectory:
    """Collapse, digest, and keyframe-select a trajectory for LLM consumption."""
    events: list[Event] = list(reader.iter_events())
    original_event_count = len(events)

    if not events:
        return PreprocessedTrajectory(
            digest=[],
            original_event_count=0,
            digest_entry_count=0,
            screenshots_included=0,
            estimated_input_tokens=0,
        )

    # Anchor relative timestamps at ``min(metadata.started_at, first_event.t)``
    # so synthetic / regenerated fixtures whose metadata clock disagrees with
    # the events' clock still produce non-negative deltas. Real recordings
    # always satisfy ``started_at <= first_event.t`` and this is a no-op for
    # them.
    metadata_t0_ms = _parse_iso_ms(reader.metadata["started_at"])
    first_event_t0_ms = _parse_iso_ms(events[0].t) if events else metadata_t0_ms
    t0_ms = min(metadata_t0_ms, first_event_t0_ms)
    collapsed = _collapse_stream(events, t0_ms)
    selected = _select_keyframes(collapsed)
    digest = _apply_keyframe_selection(collapsed, selected)

    screenshots_included = sum(1 for e in digest if e.screenshot_ref is not None)
    estimated_tokens = _estimate_tokens(digest)

    return PreprocessedTrajectory(
        digest=digest,
        original_event_count=original_event_count,
        digest_entry_count=len(digest),
        screenshots_included=screenshots_included,
        estimated_input_tokens=estimated_tokens,
    )
