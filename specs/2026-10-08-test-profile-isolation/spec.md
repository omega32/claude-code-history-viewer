# Test profile isolation

## Problem and evidence

The previous Codex activity work repaired the Windows settings and archive tests after a test run overwrote real profile files. Those repairs remain committed in `16a5bbc`. A follow-up audit confirmed that their directory shim was opt-in at ten module imports, MCP and unified preset writers still used native directories, and integration tests compiled the production library and scanned live providers. Codex session loading also writes derived caches; its basic project scan does not, so the earlier broader cache-write attribution to that integration scan was unsupported. This specification refines the isolation finding recorded in [the original investigation](../2026-10-08-codex-subagent-activity/context.md); data recovery is complete and is not part of this change.

## Required behavior

- Route application profile directory and environment access through one enforced Rust boundary. Unit tests and isolated integration builds must never fall back to native home, config, data, Windows KnownFolders or WSL discovery when fixture context is missing.
- Keep fixture overrides scoped, nested and panic-safe; concurrent test contexts must not borrow each other's paths or mutate the process environment. Unpropagated worker threads must fail closed.
- Cover settings, archives, Claude presets, MCP presets, unified presets and provider discovery through meaningful regression tests. Preserve normal native directory and environment behavior in desktop, headless, debug and release builds.
- Separate live-profile diagnostics from default tests. Preserve the diagnostics behind an explicit opt-in command and disclose that provider session loading can write derived caches.
- Provide one supported cross-platform runner that creates a fresh temporary profile, rejects unsafe arguments, scrubs inherited provider overrides, propagates isolated integration compilation, and enforces serial libtest execution. Route local commands and CI through it; nextest keeps its process-per-test model.
- Enforce the boundary through compiler/lint and runner checks, and add Windows CI alongside the existing Linux and macOS coverage.

## Scope and limits

This is an application profile-resolution boundary, not an operating-system filesystem sandbox. Tests that deliberately use explicit external absolute paths, new native APIs, subprocesses or third-party libraries require separate review. Compilation may use the existing Cargo and Rustup toolchain/cache locations. No user profile recovery, commits, release publication or live-provider diagnostics are authorized by this task.

## Acceptance

Focused negative regressions must fail safely before the implementation. Final checks must exercise actual preset/settings/archive writers and integration fixture execution, verify nested/concurrent/missing-context behavior, reject bypasses and unsafe runner invocations, pass Rust format and all-feature Clippy, and record protected-profile hashes before and after the safe suite. Document the exact checks performed and distinguish local Windows results from CI configurations that have not run yet.
