#!/usr/bin/env bash
# End-to-end cross-module contract and fixture consistency check.
#
# Checks performed:
#   1. Every *.schema.json under contracts/ is valid JSON Schema draft 2020-12.
#   2. Every trajectory under fixtures/trajectories/<dir>/ passes the trajectory
#      schema (metadata.json + every line of events.jsonl).
#   3. Every golden skill under fixtures/skills/<dir>/ passes schema + round-trip
#      + markdown/meta cross-check (same suite as scripts/check_fixtures.sh).
#   4. Every golden skill's meta.trajectory_id resolves to a real trajectory
#      fixture whose metadata.json.id matches.
#
# Exits non-zero (and prints the specific offending file + reason) on any
# failure. Finishes in well under ten seconds on the full fixture set.
#
# Works from the repo root regardless of which branch is checked out — the
# script cd's into the repo root before delegating so running it from another
# working directory (or via an absolute path) produces the same output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

exec uv run --package trace-synthesizer python -m synthesizer.check_contracts "$@"
