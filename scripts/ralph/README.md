# scripts/ralph

Ralph loop scripts and the shared prompt. Each feature branch adds its own
`prd.json` at the repo root and its own per-branch driver here.

Conventions (see root CLAUDE.md):
- Max 40–50 iterations per branch.
- Never set `TRACE_ALLOW_LIVE`.
- Use a dedicated Anthropic API key with a monthly cap.
- Write to `Trace-dev/`, never the prod profile.
- Append `{timestamp, iteration, module, input_tokens, output_tokens, cost_estimate}`
  to `costs.jsonl` on every iteration; report cumulative cost at completion.
- Append learnings to root `progress.txt`.
