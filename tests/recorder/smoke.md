# Recorder Human Smoke Test

This checklist is the final acceptance gate for the `feat/recorder` branch. A
human tester runs it on a real Mac after the Ralph loop has driven every
automated story to `passes: true`. Unit/integration tests run green inside
Ralph's sandbox, but the recorder touches macOS Accessibility, Screen
Recording, and Input Monitoring APIs that have no hermetic equivalent — so a
real machine is the only place we can prove it actually works.

The Ralph loop **must not** output the `RECORDER_DONE` / `COMPLETE` promise
until the [Sign-off](#sign-off) section at the bottom of this file is filled
in. See `scripts/ralph/CLAUDE.md` for the pre-promise guard.

---

## Pre-flight

Run these once before starting any workflow checklist.

- [ ] `uv sync --extra dev` completes cleanly on the test machine
- [ ] `uv run uvicorn gateway.main:app --host 127.0.0.1 --port 8765` starts
      without error
- [ ] `TRACE_DEV_MODE=1` is exported in the shell driving the test
      (trajectories land under `~/Library/Application Support/Trace-dev/`, not
      `Trace/`)
- [ ] macOS version recorded below in [Sign-off](#sign-off)
- [ ] Main display resolution and scale factor noted (Retina vs. non-Retina
      matters for the storage-sanity check)

### Permissions grant

System Settings → Privacy & Security → grant the `uv`/`python` process (or
Terminal / iTerm hosting it) the three permissions below, then restart the
gateway.

- [ ] Accessibility
- [ ] Screen Recording
- [ ] Input Monitoring

`GET /recorder/status` should return `{"recording": false, ...}` with no
permission error. `POST /recorder/start` against a fresh session should
succeed — no 403.

---

## Reference-workflow checklists

For each workflow, expected signals are minimums. "N clicks with resolved AX
targets" means events of `type == "click"` whose `target` is non-null and has
a non-empty `role`.

Use `python -m recorder.tools.trajectory_summary <id>` or a manual
`jq`-through-`events.jsonl` script to count event types. (If no CLI exists
yet, counting by piping `events.jsonl` through `jq -r '.type' | sort | uniq -c`
is fine.)

Storage sanity applies per workflow: directory size must stay under 30 MB for
a recording of no more than three minutes.

### 1. `gmail_reply`

Open Gmail in Chrome, find the most recent unread from a known sender, reply
with a template, and send.

- [ ] Permissions granted (see Pre-flight)
- [ ] Started recording via `POST /recorder/start` with
      `{"label": "gmail_reply"}`, captured `trajectory_id`
- [ ] Performed the workflow on a real machine (end-to-end, send included)
- [ ] Stopped recording via `POST /recorder/stop`
- [ ] `GET /trajectories/{id}` returns sensible data (non-zero event_count,
      non-null stopped_at, screenshots served by the static mount)

Expected signals:

- [ ] Event count in the range 40 – 400
- [ ] ≥ 5 `click` events with resolved AX targets (`target.role` non-empty —
      think inbox row, reply button, compose body, send button, message thread)
- [ ] ≥ 1 `text_input` event with a non-null `field_label` (the reply body)
- [ ] ≥ 1 `app_switch` event (launcher/finder → Chrome, or similar)
- [ ] ≥ 5 keyframe screenshots (periodic + pre/post-click + app_switch mix)
- [ ] Trajectory directory size < 30 MB

### 2. `calendar_block`

Open Google Calendar, create a 30-minute focus block tomorrow at 2 PM.

- [ ] Permissions granted
- [ ] Started recording via `POST /recorder/start` with
      `{"label": "calendar_block"}`, captured `trajectory_id`
- [ ] Performed the workflow on a real machine
- [ ] Stopped recording
- [ ] `GET /trajectories/{id}` returns sensible data

Expected signals:

- [ ] Event count in the range 25 – 250
- [ ] ≥ 3 `click` events with resolved AX targets (time slot, create button,
      save button)
- [ ] ≥ 1 `text_input` event (event title)
- [ ] ≥ 1 `app_switch` event (browser focus change at minimum)
- [ ] ≥ 5 keyframe screenshots
- [ ] Trajectory directory size < 30 MB

### 3. `finder_organize`

In Finder, move every `.pdf` in `~/Downloads` older than 7 days into
`~/Documents/Archive`. (Seed at least three test PDFs with matching mtimes
before recording.)

- [ ] Permissions granted
- [ ] Started recording via `POST /recorder/start` with
      `{"label": "finder_organize"}`, captured `trajectory_id`
- [ ] Performed the workflow on a real machine
- [ ] Stopped recording
- [ ] `GET /trajectories/{id}` returns sensible data

Expected signals:

- [ ] Event count in the range 30 – 300
- [ ] ≥ 4 `click` events with resolved AX targets (Finder sidebar / toolbar /
      file rows)
- [ ] ≥ 1 `text_input` event (search box or new-folder rename — acceptable to
      be an empty-buffer no-op only if you performed NO typing; flag as a gap
      otherwise)
- [ ] ≥ 1 `app_switch` event (focus into Finder)
- [ ] ≥ 5 keyframe screenshots
- [ ] Trajectory directory size < 30 MB
- [ ] Note any drag-and-drop gaps in the Notes section — v1 represents drags
      as click-pairs and this workflow is the canary (see PRD `openQuestions`)

### 4. `slack_status`

Open Slack, set status to `🎯 heads down` with a 2-hour expiry.

- [ ] Permissions granted
- [ ] Started recording via `POST /recorder/start` with
      `{"label": "slack_status"}`, captured `trajectory_id`
- [ ] Performed the workflow on a real machine
- [ ] Stopped recording
- [ ] `GET /trajectories/{id}` returns sensible data

Expected signals:

- [ ] Event count in the range 20 – 200
- [ ] ≥ 3 `click` events with resolved AX targets (avatar menu, status item,
      duration picker, save)
- [ ] ≥ 1 `text_input` event (status message text)
- [ ] ≥ 1 `app_switch` event (into Slack)
- [ ] ≥ 5 keyframe screenshots
- [ ] Trajectory directory size < 30 MB
- [ ] Electron AX shallowness check — if any Slack click resolves to a
      generic `AXGroup` with no label, confirm the ancestor walk surfaced a
      meaningful parent label; otherwise flag in Notes

### 5. `notes_daily`

Open Apple Notes, create a new note titled with today's date, paste a fixed
template.

- [ ] Permissions granted
- [ ] Started recording via `POST /recorder/start` with
      `{"label": "notes_daily"}`, captured `trajectory_id`
- [ ] Performed the workflow on a real machine
- [ ] Stopped recording
- [ ] `GET /trajectories/{id}` returns sensible data

Expected signals:

- [ ] Event count in the range 15 – 200
- [ ] ≥ 2 `click` events with resolved AX targets (new-note button, note body)
- [ ] ≥ 1 `text_input` event (the title or body typed after paste)
- [ ] ≥ 1 `app_switch` event (into Notes)
- [ ] ≥ 5 keyframe screenshots
- [ ] Trajectory directory size < 30 MB

---

## Resilience tests

### Kill-tap test

CGEventTap can get disabled under CPU load. The recorder must re-arm it and
emit a `tap_reenabled` event.

- [ ] Start a recording, label `kill_tap_test`
- [ ] Run a CPU-hogging command for ~10 seconds while recording (e.g.
      `yes > /dev/null &` across all cores, or `stress --cpu $(sysctl -n hw.ncpu)`)
- [ ] Stop the CPU load, continue the recording for another 5 seconds, then
      stop the recording
- [ ] At least one event in `events.jsonl` has `type == "tap_reenabled"` —
      inspect with `grep '"tap_reenabled"' .../events.jsonl`
- [ ] The recording captured user events both before and after the reenable
      (i.e. the tap actually recovered, not just emitted the reenable marker)

### Permission-denial test

The API must refuse to start a session when any required permission is
missing, and return a structured error.

- [ ] Revoke Accessibility for the gateway's host process in System Settings
      (the python/uv/terminal bundle as appropriate)
- [ ] Restart the gateway (permissions are cached at process start)
- [ ] `POST /recorder/start` returns **HTTP 403**
- [ ] Response body matches
      `{"error": "missing_permission", "permissions": [...], "how_to_grant": {...}}`
- [ ] `permissions` array includes `"accessibility"`
- [ ] Re-grant Accessibility, restart the gateway, confirm `POST
      /recorder/start` returns 200 again

---

## Storage sanity (aggregate)

- [ ] Total `~/Library/Application Support/Trace-dev/trajectories/` size
      after all 5 workflow recordings is < 150 MB (30 MB × 5 budget)
- [ ] `index.db` at `~/Library/Application Support/Trace-dev/index.db` lists
      every trajectory you created above; row counts match on-disk dirs
- [ ] `DELETE /trajectories/{id}` on one trajectory removes both the
      directory and its index row

---

## Notes

Free-form section for anything the tester observed that doesn't fit a
checkbox — drag-and-drop fidelity complaints, surprising AX roles, laggy
workflows, missing events, etc. These feed the v2 backlog.

- …
- …

---

## Sign-off

The Ralph loop checks for **non-placeholder** values in both `Date` and
`macOS version` below. Leave the `_____` placeholders in place to signal
"not yet signed off" — the pre-promise guard in `scripts/ralph/CLAUDE.md`
relies on that literal string.

- Date: `_____`
- macOS version: `_____`
- Hardware (chip + display): `_____`
- Tester name / initials: `_____`
- All checklist items above marked complete: [ ]
- All 5 reference trajectories present under
      `~/Library/Application Support/Trace-dev/trajectories/` at sign-off
      time: [ ]
- Any blocking regressions filed as issues on the tracker (list IDs here or
      write "none"): `_____`
