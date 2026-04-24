#!/usr/bin/env bash
# Regenerate the five reference-workflow trajectory fixtures.
#
# WHO RUNS THIS: a human on a real Mac, during smoke testing. This script
# is NOT run by Ralph or CI. The Ralph loop ships synthetic fixtures (see
# scripts/generate_synthetic_fixtures.py) that pass the JSON schema but
# don't reflect real app behaviour. A human tester replaces them with
# real recordings before the module is considered shipped.
#
# PREREQS:
#   * macOS with Accessibility, Screen Recording, and Input Monitoring
#     permissions granted to the terminal running this script.
#   * The recorder service running locally on 127.0.0.1:8765 (e.g.
#     `uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8765`).
#   * `jq` and `curl` on PATH.
#
# WORKFLOWS (record each on a real machine — slug: description):
#   gmail_reply      — Chrome → Gmail: find most recent unread from a
#                      sender, reply with a template, send.
#   calendar_block   — Chrome → Google Calendar: create a 30-minute focus
#                      block tomorrow at 2pm.
#   finder_organize  — Finder: move every .pdf in ~/Downloads older than
#                      7 days into ~/Documents/Archive.
#   slack_status     — Slack: set status to "🎯 heads down" with a
#                      2-hour expiry.
#   notes_daily      — Apple Notes: create a new note titled with today's
#                      date, paste the fixed template.
#
# PER WORKFLOW (repeat for each slug above):
#   1) POST /recorder/start with {"label": "<slug>"} and capture the
#      returned trajectory_id.
#   2) Perform the workflow on your machine — clicks, typing, scrolling
#      as appropriate. Aim for 1–3 minutes total.
#   3) POST /recorder/stop.
#   4) Copy the trajectory directory out of
#      ~/Library/Application\ Support/Trace-dev/trajectories/<id>/
#      into fixtures/trajectories/<slug>/  (overwriting the synthetic
#      fixture that was there).
#   5) Strip the screenshots/*.png down in size (ImageMagick:
#      `mogrify -resize 320x -quality 70 screenshots/*.png`) so the
#      whole fixtures/ tree stays under 5MB.
#   6) Re-run `uv run pytest tests/recorder/test_fixtures.py` to confirm
#      the new fixture still validates.
#
# FALLBACK (Ralph / no-real-machine path):
#   Run `uv run python scripts/generate_synthetic_fixtures.py` from the
#   repo root to regenerate the synthetic JSON + tiny PNG fixtures. The
#   synthetic fixtures exist purely so downstream module test suites
#   (Synthesizer, Runner) can run without requiring a human recording.

set -euo pipefail

cat <<'BANNER'
This script is a human-only checklist.

To regenerate the synthetic fallback fixtures (Ralph / CI), run:

    uv run python scripts/generate_synthetic_fixtures.py

To regenerate REAL fixtures, follow the numbered steps at the top of this
file on a real Mac with the recorder service running.
BANNER
