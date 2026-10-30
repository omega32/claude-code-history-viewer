# Context and evidence

The [preceding candidate transport slice](../2026-10-30-prompt-candidate-transport/context.md) still performs complete normalization. Temporary native attribution, subsequently removed byte-for-byte, established 519,997,980 accepted logical bytes across three carriers, 83,171 physical rows and 14,560 normalized records for the representative paginated workload. Lineage acquisition and parsing dominate; discovery and generic finalization are small. The [owning ccmsg performance investigation](../../../../My%20Tools/My%20Commit%20Message%20Generator/docs/investigations/prompt-read-performance.md) retains the original measurements and their limits. This prototype tests reuse of already-verified structural state rather than another cold decoder or output-projection optimization.

Canonical base instructions require the Rust architecture review, tests-first backend work, specifications, the isolated serial Rust runner and independent verification. The read-only architect review approved a cfg(test)-only prototype and pre-clear capture at snapshot callsites, with an explicit no-residual-user-tool-result gate. It also identified two source-proof requirements beyond naively retaining finalized snapshots: re-resolve authoritative lineage carrier selection on every refresh, and treat a seeded paginated cursor as provisional until fresh accepted-prefix validation succeeds. Existing first-new asynchronous reply fallback remains necessary.

The parent owns real source capture, immutable isolated ancestors/current input under ignored scratch, the TypeScript projection parity consumer and public metadata-only reporting. The native probe owns only its initially absent working selected carrier under that fixture root. There is no authorization to mutate live stores, source settings, hooks, services or Notifier code. Harness-kit tooling is unavailable; this follows the repository's existing specification structure directly. Base starts clean at 9c4dc73 and version 1.31.0; the root consumer starts at 1a6280c4.

## Reproduce the optimized isolated development artifact

Run `pnpm exec tsc -p scripts/tsconfig.json` from the base repository root, then save and invoke the following exact build-only script at `scratch/build-codex-prompt-prototype.mjs` with `node scratch/build-codex-prompt-prototype.mjs`. It reuses the supported test boundary and sanitized isolation environment without expanding the public runner argument policy. The fixed Node ESM recipe loads the already-compiled runner helpers while deriving its checked types from their authoritative TypeScript source; it avoids introducing another maintained command or build layer. Verify the recipe with `pnpm exec tsc --allowJs --checkJs --strict --noEmit --target ES2023 --module NodeNext --moduleResolution NodeNext --types node --skipLibCheck scratch/build-codex-prompt-prototype.mjs`. The fixed Cargo vector builds only the optimized isolated library-test executable; the configured application image is untouched.

```javascript
// @ts-check
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
/** @type {typeof import('../scripts/test-rust.js')} */
const { buildInvocation, checkTestBoundary } = await import(new URL('./test-tools/test-rust.js', import.meta.url).href);

const repositoryRoot = resolve('.');
checkTestBoundary(repositoryRoot);
const fixtureRoot = mkdtempSync(join(tmpdir(), 'cchv-prototype-build-'));
for (const directory of ['home', 'config', 'data', 'cache', 'tmp']) mkdirSync(join(fixtureRoot, directory), { recursive: true });
/** @type {NodeJS.ProcessEnv} */
const inherited = { ...process.env };
inherited.CARGO_HOME ||= join(homedir(), '.cargo');
inherited.RUSTUP_HOME ||= join(homedir(), '.rustup');
const invocation = buildInvocation(['test', '--lib'], inherited, fixtureRoot, repositoryRoot);
const args = ['test', '--release', '--lib', '--no-run', '--message-format=json'];
const result = spawnSync(invocation.command, args, {
  cwd: invocation.cwd, env: invocation.env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  windowsHide: true,
});
writeFileSync('scratch/codex-prompt-prototype-optimized.stdout', result.stdout ?? '');
writeFileSync('scratch/codex-prompt-prototype-optimized.stderr', result.stderr ?? '');
writeFileSync('scratch/codex-prompt-prototype-optimized.exit', String(result.status ?? 1));
if (result.error || result.status !== 0) process.exit(1);
const artifacts = result.stdout.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)).filter(item =>
  item.reason === 'compiler-artifact' && item.target.name === 'claude_code_history_viewer_lib'
  && item.profile.test === true && typeof item.executable === 'string');
if (artifacts.length !== 1) throw new Error('Expected exactly one optimized isolated library-test artifact.');
const artifact = artifacts[0].executable;
const { createHash } = await import('node:crypto');
const sha256 = createHash('sha256').update(readFileSync(artifact)).digest('hex');
const metadata = { artifact, sha256, fixedArgs: args, cargoTargetDir: invocation.env.CARGO_TARGET_DIR };
writeFileSync('scratch/codex-prompt-prototype-optimized-artifact.json', JSON.stringify(metadata, null, 2));
console.log(JSON.stringify(metadata));
```

Invoke the resulting artifact using exactly `--ignored --exact providers::codex::prompt_prototype::tests::private_batch_probe --test-threads=1 --nocapture`, with the same sanitized `buildInvocation(["test", "--lib"], ...)` environment, one bounded stdin object and a private parent pipe. The parent must set a 180-second deadline, terminate only its exact owned child on cancellation/deadline, validate fatal UTF-8 and all fixed frames, and never display raw stdout/stderr or persist candidate payloads.

The metric reason allowlist is `initial`, `append`, `unchanged`, `provider-full`, `seed-unconfirmed`, `missing-cursor`, `capture-missing`, `merge-required`, `state-budget`, `coordinate-mismatch`, `parser-drift`, `path-rotation`, `selection-drift`, `source-refused`. `normalizedRecords` and `acceptedLogicalBytes` are optional typed retained-state receipts; absent state yields null. Stored compressed length is never presented as decoded or accepted logical bytes.

The parent fixture selector names only the known Codex session. Choosing its latest ordinary canonical task cutpoints is internal fixture preparation scoped to that session; it is not a provider-wide prompt discovery API or a Notifier lookup. Raw selected capture and cloned ancestors remain in the owned ignored fixture; the public evidence retains aggregate sizes, timing, modes and parity verdicts only.

## Validation and interpretation

The initial tests-first stub deliberately returned a complete result on every read; the first-new-prompt regression failed because it required a Replace result, while the pre-clear ledger capture test passed. Subsequent synthetic setup failures picked a session metadata row rather than the authored row and omitted a required paginated-parent header flag. Correcting those fixtures confirms the existing normalizer contracts; those failures are not evidence of new provider defects. The merge regression now constructs a matching assistant tool call and mixed user text/result record, demonstrating content mutation despite equal record count before checking the conservative retention gate.

A seed is a fresh-process complete normalization plus accepted-prefix confirmation. A warm measurement keeps only the bounded in-memory state in that same finite process and appends a genuinely new retained ordinary prompt. Operating-system file-cache state is uncontrolled; this is not an OS-cold storage benchmark or an end-to-end production ccmsg latency claim. Persistence, process lifetime and integration remain deferred until the native acquisition gain and exact production projection parity are measured.

The final lint-corrected source passed `pnpm test:rust`: 1,177 library and 10 integration tests, zero failures, with the finite private probe and nine live-profile diagnostics explicitly ignored by the default gate. The final rebuild took 2m 32s and library execution took 15.00s. Both `pnpm test:rust clippy` and native `cargo clippy --all-targets --all-features -- -D warnings` passed (23.21s and 27.62s), as did `cargo fmt --all -- --check`. The fixed Node build recipe passed strict TypeScript `checkJs`. Independent source review approved the capture/ledger/proof/fixture design, the real independent seed control and the narrow RAII/lint corrections. Exact outputs and exit receipts are under ignored `scratch/codex-prompt-prototype-{rust,clippy-isolated,clippy-native,fmt}.{stdout,exit}`.

The seed frame uses a real ordinary full-normalizer control outside `seedMs`, reported separately as `seedControlMs`; both observed sets must be empty/nonrefused. The private consumer independently requires that receipt and checks native arrays/refusal and production projection separately, so an older fabricated-seed artifact cannot satisfy acceptance. Optimized artifact and real-fixture benchmark evidence are recorded after completion below and owned by the [incremental Codex prototype investigation](../../../../My%20Tools/My%20Commit%20Message%20Generator/docs/investigations/prompt-read-performance.md#incremental-codex-prototype).

## Measured native acceptance

The fixed optimized isolated library-test build succeeded in 4m 54s. Its artifact and hash are recorded privately in ignored `scratch/codex-prompt-prototype-optimized-artifact.json`, with compiler outputs in `scratch/codex-prompt-prototype-optimized.{stdout,stderr,exit}`. The configured production release image remained byte-identical; package, Cargo, lock and Tauri versions remain 1.31.0. No production executable replacement, CLI capability, saved cache, worker, hook, service or provider-store change belongs to this slice.

The coordinator replayed the actual copied three-source workload with two genuinely new ordinary prompts per priority. All six frames matched exact native candidate arrays/refusal and existing production TypeScript projection: both independent seed observations were empty/pending, and all four new-prompt steps were available. Every warm result used `replace` with reason `append`; observed native child priorities 0 and 10 matched the requested normal and below-normal classes.

| Native priority | Full control steps (ms) | Warm new-prompt steps (ms) | Full mean (ms) | Warm mean (ms) | Mean reduction |
| --- | --- | --- | --- | --- | --- |
| Normal | 2,389.15 / 3,810.82 | 114.39 / 245.19 | 3,099.98 | 179.79 | 94.20% |
| Below-normal | 3,619.16 / 3,426.98 | 240.21 / 238.82 | 3,523.07 | 239.52 | 93.20% |

Every individual warm step also exceeded the at-least-50% native component gate. Engine seed cost was 2,601.85 ms normal and 4,010.41 ms below-normal; the separately measured real seed controls were 2,514.74 and 4,063.48 ms. Retained candidates grew from 369 to 371, sparse references from 481 to 484, and the accounted byte estimate from 568,683 to 570,439. The raw normalized extent grew from 14,517 to 14,552. These are logical/serialized/reference accounting receipts, not measured RSS or total heap/seed peak.

Typed accepted logical prefix length was 516,749,750 bytes at seed, then 516,756,540 and 518,711,833 bytes at the two new-prompt boundaries. This is a bounded replay near the approximately 520 MB logical workload, stopping after each actual new canonical input rather than claiming that the entire final carrier was accepted. Configured and development parser images stayed stable, and the coordinator removed both owned source fixtures after acceptance. The public evidence retains aggregate metadata only; prompt content, selectors and source identities are omitted.

The [owning investigation](../../../../My%20Tools/My%20Commit%20Message%20Generator/docs/investigations/prompt-read-performance.md#incremental-codex-prototype) and its daily `prompt-read-performance-2026-10-30.json` record own the retained benchmark evidence. The result validates the in-memory native mechanism and supports a later storage/process-lifetime decision. It does not establish the later whole-ccmsg at-least-50% enablement goal: CLI startup, outer metadata/carrier/parser-image proofs and integration costs were excluded. No production path is enabled, no live model turn was created and operating-system file-cache coldness was not controlled.
