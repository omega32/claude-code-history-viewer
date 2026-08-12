# Spec: Codex authorship differential audit

- Status: Complete
- Created: 2026-08-12

## Problem and goal

The Codex rollout parser now classifies authored input and provider-injected context through strict structural evidence, but its captured-shape regression is not an independent oracle. Add an opt-in, read-only headless audit that compares one normalized rollout with a separately captured Codex app-server `thread/read` response, and report unresolved authorship or protocol drift without changing ordinary dump behavior.

## Scope

- Add `--audit-codex-authorship <absolute-rollout-path> --app-server-response <absolute-json-path> [--format json] [--output <file>]`.
- Accept a schema-versioned capture envelope containing the `thread/read` result, Codex CLI version, thread id, and the exact rollout byte length and SHA-256. A bare result or JSON-RPC response may be inspected, but is unbound and therefore always inconclusive.
- Require a matching non-empty thread id, explicit `itemsView:"full"` plus a concrete items array for every turn, unique non-empty turn ids, and unique non-empty authored `clientId` values before treating the response as an oracle.
- Compare ordered authored client identities and physical turn ids against `authored_user` / `steer` normalized messages.
- Report differences through opaque identity digests, `authorship_unknown` rows, and unrecognized app-server item types without serializing provider-controlled identity strings, transcript content, or private input paths.
- Treat the envelope as an integrity binding rather than a capture-time attestation; require the documented trusted collector workflow to fingerprint the rollout immediately before and after `thread/read`.
- Create `--output` atomically and only when the target does not exist, so neither input nor any other existing evidence can be overwritten.
- Return exit 0 only for an exact, drift-free audit; return exit 1 for mismatches, unresolved authorship, drift, or invalid input; return exit 2 for invalid command usage.
- Advertise the additive command through headless capabilities and document the capture/audit boundary.

## Out of scope

- Starting or authenticating Codex app-server from the viewer.
- Running a live network/service dependency in the normal test suite.
- Changing ordinary `--dump-session`, snapshot, UI, or authorship classification behavior.
- Treating diagnostic output as permission to hide an unresolved record.

## Acceptance

- A captured `thread/read` response for the affected real session independently agrees with the base parser's 18 authored inputs, 15 physical turns, and 3 steers.
- Missing, duplicate, cross-turn, reordered, incomplete, unknown, unbound, or source-mismatched oracle evidence cannot produce a passing audit.
- An `authorship_unknown` normalized row is reported with message identity, optional provider turn id, and source line when available, while its content remains absent from the audit output.
- Focused tests, serial Codex tests, formatter, Clippy, and the full serial Rust suite are run according to repository guidance.
