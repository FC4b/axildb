# Code-recall fixture

Tiny, deterministic fixture used by `scripts/code-recall-gate.sh` and the
in-tree code-recall regression gate. Files are deliberately small and
self-contained — they are NOT part of the workspace and are never
compiled.

## Sections

The Markdown headings here exist so the section-proxy splitter has
something to chunk against.

### Login flow

Credential check happens in `src/auth.rs::login`. Tokens are validated by
`validate_token`.

### Scoring

Vector + FTS + recency blending lives in `src/scoring.rs::rank`.
