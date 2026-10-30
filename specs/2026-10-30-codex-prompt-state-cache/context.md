# Context and decisions

The [completed in-memory prototype](../2026-10-30-codex-incremental-prompts/context.md#measured-native-acceptance) preserved exact candidate/projection parity and reduced native new-prompt acquisition by 94.20% normally and 93.20% below-normal. It did not retain state across separate CLI processes or establish the whole-ccmsg goal. This slice starts from clean base e646a11, package 1.31.0, and parent c8121549.

The parent approved complete proven state protected at rest with current-user DPAPI rather than an additional worker/service. Existing app caches are plaintext metadata hints and explicitly do not certify snapshot continuity; no existing native protected-cache helper was found. Fresh source hashes bind bytes but cannot independently authenticate a forged semantic checkpoint/ledger, so caller-supplied or merely checksummed saved state is never trusted.

Microsoft documents [current-user DPAPI protection and its MAC](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata) and warns that [unprotection can sometimes succeed with corrupted output](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptunprotectdata). This motivates both DPAPI and inside-envelope integrity/shape checks. UI is forbidden, the prompt structure is null, machine-wide protection is unused, and sensitive native output is cleared before LocalFree. Same-user malicious actors able to call DPAPI or modify authoritative sources remain outside executable/state-authentication claims.

The exact approved wire is one bare `--prompt-state-cache` on the existing Codex candidate command plus Windows feature `codex-prompt-state-cache-v1`, with unchanged bare-array output and no operational state in stdout. The parent owns TypeScript integration and whole-CLI measurements; native tests alone cannot establish the terminal-level speed goal. The production benchmark will pass owned absolute `CODEX_HOME` and `CCHV_PROMPT_STATE_CACHE_DIR` through the existing sanitized helper environment because native Windows KnownFolder cache lookup does not follow ambient LOCALAPPDATA overrides.

Canonical instructions require the Rust architect design/test-first workflow and independent verification. The concrete direction, bounds, override and version decision were approved before implementation. Harness-kit tools are unavailable; this follows the established three-file specification structure. No Cargo operation begins without parent coordination; release replacement remains coordinated separately.

The cache has a 24-hour reuse TTL, checked against the protected update timestamp. Maintenance runs on opt-in cache access and opportunistically prunes expired files; no daemon/timer promises physical deletion at exactly 24 hours. Aggregate ciphertext accounting deliberately includes the old slot plus the pending encrypted replacement until atomic publication completes, so temporary coexistence remains within the 128 MiB bound.

## Native validation and release

The native test-first flag regression failed before implementation, as expected. The implemented focused gate passed 33 tests including real current-user DPAPI roundtrip/tampering, fresh engine ownership/new-prompt restoration, binding/accounting/TTL refusal, source rewrite/collision restoration, lineage twin drift, nonblocking locks, bounded entry maintenance and failed atomic replacement. The final synchronized 1.32.0 source passed mandatory serial isolated `pnpm test:rust`: 1,190 library and 10 integration tests, zero failures, one explicitly ignored finite probe and nine explicitly ignored live-profile diagnostics. Rebuild time was 7m 39s; library execution was 18.65s.

Both `pnpm test:rust clippy` and native `cargo clippy --all-targets --all-features -- -D warnings` passed (33.37s and 33.80s); `cargo fmt --all -- --check`, version parity and diff checks passed. The independent verifier approved the source and narrow lint/MSRV refinements without remaining findings. No lint was suppressed. Package 1.32.0 was synchronized exactly once through `node scripts/sync-version.cjs`; Cargo, lock and Tauri versions agree.

After frontend preparation and parent gate staging, the standard configured `cargo build --release --bin claude-code-history-viewer` succeeded in 2m 42s without killing/stopping consumers. The configured release artifact is `src-tauri/target/release/claude-code-history-viewer.exe`, 32,921,600 bytes; an explicit redirected capability process returned exit 0, empty stderr, API 1, version 1.32.0 and `codex-prompt-state-cache-v1`. An initial implicit PowerShell capture raced the GUI-subsystem process and closed its stdout pipe; that was a probe-harness failure, corrected without runtime changes. Exact native receipts are under ignored `scratch/codex-prompt-state-cache-{rust,clippy-isolated,clippy-native,fmt,release}.{stdout,exit}` and `codex-prompt-state-cache-release-artifact.json`.

## Production acceptance

The coordinator kept ordinary ccmsg on baseline 11.58.1 until the paired whole-command gate passed, then published the exact validated 11.58.2 bundle. Baseline and candidate used the same configured 1.32.0 native executable and three copied carriers totaling 265,397,652 stored bytes. A fixture-only one-row SQLite index pinned the selected carrier, ordinary targeted metadata independently confirmed that selection, and timed source-before/source-after byte receipts matched each replayed extent. No live database was copied or modified for that fixture.

Two genuinely new ordinary prompt steps were measured per priority, with the queried future prompts absent at seed and each canonical input appended before its completion. All four new-prompt comparisons were available and matched private text, stable record identity and status; observed priorities and parser-image stability checks matched, and all owned private fixtures were removed.

| Whole-command priority | Baseline mean (ms) | Cached mean (ms) | Paired-mean reduction |
| --- | --- | --- | --- |
| Normal | 4,332.44080 | 2,102.09605 | 51.48010% |
| Below-normal | 6,368.71505 | 3,041.05970 | 52.25003% |

Both predeclared per-priority paired-mean at-least-50% gates passed. This is not an every-sample guarantee: the second individual pairs improved 39.34% normally and 39.85% below-normal. The sample is two new prompts per priority, with uncontrolled OS file-cache state and held replay boundaries; no universal or continuous-writer latency guarantee is established. Cold seed costs were 6,432.5828 and 7,069.0755 ms and both seed queries were pending as expected. Encrypted state sizes were 556,198–557,878 bytes.

A separate live ordinary-CLI smoke compared the existing prompt through 11.58.1 and published 11.58.2: both were available, exited 0 with empty stderr, and matched private text/record identity. The native default encrypted cache had one 556,822-byte entry. This smoke is distinct from the genuinely-new-prompt speed measurements. No new provider prompt, Notifier change, hook, model turn or service was introduced.

The [owning measured investigation](../../../../My%20Tools/My%20Commit%20Message%20Generator/docs/investigations/prompt-read-performance.md#production-state-acceptance-2026-10-30) and daily `prompt-read-performance-2026-10-30.json.productionStateCache` record own the retained metadata and interpretation. Source/runtime bytes remained frozen throughout acceptance; the public record excludes prompt content, source identities and protected state contents.
