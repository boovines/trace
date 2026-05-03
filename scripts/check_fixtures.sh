#!/usr/bin/env bash
# Validate every hand-crafted golden skill fixture under fixtures/skills/.
#
# Checks performed for each fixture directory:
#   1. skill.meta.json conforms to contracts/skill-meta.schema.json.
#   2. SKILL.md round-trips: parse_skill_md(render_skill_md(parse_skill_md(f))) == parse_skill_md(f).
#   3. validate_meta_against_markdown cross-check passes (step count, destructive
#      flags, parameter refs all agree between markdown and meta).
#   4. meta.trajectory_id refers to an existing fixtures/trajectories/<dir>
#      whose metadata.json carries the same id.
#
# Exits non-zero (and prints the specific offending file + reason) on any
# failure. Must finish in under 5 seconds on the full fixture set.
#
# IMPORTANT: the golden fixtures under fixtures/skills/ are HAND-CRAFTED
# ground truth for the synthesizer's snapshot similarity test. Do NOT let a
# Ralph iteration regenerate them automatically — see fixtures/skills/README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

exec uv run --package trace-synthesizer python -m synthesizer.check_fixtures "$@"
