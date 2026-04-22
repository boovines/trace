# fixtures

Shared acceptance fixtures for the five reference workflows listed in CLAUDE.md:
`gmail_reply`, `calendar_block`, `finder_organize`, `slack_status`, `notes_daily`.

- `trajectories/<slug>/` — produced by the Recorder, consumed by the Synthesizer.
- `skills/<slug>/` — produced by the Synthesizer, consumed by the Runner.

Regenerate via `scripts/regenerate_fixtures.sh` when the recorder materially changes.
