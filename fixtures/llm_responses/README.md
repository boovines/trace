# Canned LLM responses (fake mode)

When `TRACE_LLM_FAKE_MODE=1` is set, `synthesizer.llm_client.LLMClient.complete`
loads a JSON file from this directory whose name is the SHA-256 hash of the
request inputs.

Compute the hash with `synthesizer.llm_client.compute_request_hash(...)` and
write a JSON file with at least the `text` key (and optionally `stop_reason`,
`input_tokens`, `output_tokens`).

Tests typically register canned responses dynamically by setting
`TRACE_FAKE_RESPONSES_DIR` to a temp directory and calling
`synthesizer.llm_client.save_fake_response(...)` — only fixtures that need to
be checked into the repo (e.g. for downstream stories' integration tests) live
here.
