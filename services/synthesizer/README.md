# services/synthesizer

Module 2. Developed on branch `feat/synthesizer`. Completion promise: `SYNTHESIZER_DONE`.

Converts a recorded trajectory into a `skill/` directory (SKILL.md + skill.meta.json)
using Claude Sonnet 4.5 with multimodal calls over keyframe screenshots, plus a
user Q&A pass to fill gaps.

Contracts: reads [trajectory.schema.json](../../contracts/trajectory.schema.json),
writes per [skill-meta.schema.json](../../contracts/skill-meta.schema.json).
