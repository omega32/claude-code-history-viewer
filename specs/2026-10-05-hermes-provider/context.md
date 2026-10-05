# Context

Installed Hermes source at revision `e473f5a9c9` retains native sessions in `state.db`; the current `HERMES_HOME` is a Windows local-app-data directory. Read-only inspection found today's desktop conversation with four user messages, native model `gpt-6.1-sol` and billing provider `openai-codex`. The database has a live WAL, so copying only the main file would omit recent committed records.

Authoritative local storage behavior is in `hermes_state.py`, `hermes_state_messages.py`, `hermes_state_compression.py` and `hermes_constants.py`. The public [session storage documentation](https://hermes-agent.nousresearch.com/docs/developer-guide/session-storage/) describes the owning store. Native display projection prefers active representatives of `display_order` groups, includes compaction archives, excludes rewind and model-only rows, and walks only verified compression ancestry. Multimodal content uses the literal NUL-prefixed `json:` sentinel; ordinary JSON-looking authored text is not structured content.

The harness-kit plugin is unavailable; these files follow the repository's documented specification structure directly. Tests must use temporary stores and keep Rust tests single-threaded. The existing provider-generated synthetic selectors and complete snapshot fallback are the compatibility boundary; Hermes receives no optimized cursor or stable-record certification.

## Verification

The final implementation passed 13 temporary-store Hermes tests and 42 headless tests with a single test thread, `cargo fmt --check`, and all-target/all-feature Clippy with warnings denied. Frontend verification passed the TypeScript build, targeted ESLint, i18n validation, 20 provider registry tests and the production Vite build. A release binary was built after the frontend artifacts completed. Independent change verification found no remaining blockers.

The compatible new provider capability advances the base version once from 1.26.0 to 1.27.0. The configured ccmsg integration verified the real Windows desktop session and left its database and WAL bytes unchanged; integration evidence is canonical in ccmsg's `docs/providers/hermes-support.md`. No rendered GUI or macOS/Linux execution was performed.
