#!/usr/bin/env bash
# Verify repo-level contracts that cannot be expressed inside the JSON Schemas
# themselves. Run locally before changing anything in contracts/ and in CI.
#
# Current checks:
#   * contracts/destructive_keywords.json exists, is a JSON array, has exactly
#     14 entries, and every entry is a lowercase non-empty string. The list is
#     imported by both the runner (harness-layer gate) and the synthesizer
#     (authoring-time ⚠️ flagger); expanding it without coordination is a bug.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KEYWORDS_FILE="${REPO_ROOT}/contracts/destructive_keywords.json"

if [[ ! -f "${KEYWORDS_FILE}" ]]; then
  echo "error: ${KEYWORDS_FILE} is missing" >&2
  exit 1
fi

python3 - "${KEYWORDS_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    sys.exit(f"error: {path} must be a JSON array, got {type(data).__name__}")

if len(data) != 14:
    sys.exit(
        f"error: {path} must have exactly 14 entries (got {len(data)}); "
        "expanding this list requires coordinated updates in both the "
        "runner and synthesizer branches"
    )

for i, word in enumerate(data):
    if not isinstance(word, str):
        sys.exit(f"error: {path}[{i}] is not a string: {word!r}")
    if not word:
        sys.exit(f"error: {path}[{i}] is empty")
    if word != word.lower():
        sys.exit(f"error: {path}[{i}] must be lowercase: {word!r}")

print(f"ok: {path} has 14 lowercase string entries")
PY
