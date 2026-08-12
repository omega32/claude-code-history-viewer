# Context: Codex authorship differential audit

## Grounding

- Codex CLI 0.146.0 `thread/read` with `includeTurns:true` returns `thread.turns[].items[]`; authored input is `type:"userMessage"` with provider-generated item `id` and optional `clientId`.
- The affected real session returns 15 turns and 18 user messages. All 18 have unique non-empty `clientId` values; the three additional inputs share their active physical turn.
- App-server item ids such as `item-1` are not rollout response-item ids, so exact comparison must use `clientId` plus the containing app-server turn id.
- The base parser already emits the same event client id as `data.clientMessageId`, the physical turn as `data.providerTurnId`, and unresolved rows as `subtype:"authorship_unknown"`.
- The existing `load_messages_matches_captured_app_server_authorship_shape` test is intentionally synthetic and must not be described as a live differential oracle.

## Security and compatibility decisions

- The viewer consumes an explicit captured response rather than launching app-server, keeping authentication, process lifecycle, and service availability outside the normal parser and test path. Only a `schemaVersion:1` capture whose `rollout` object matches the exact thread id, byte length, and SHA-256 of the selected rollout can return `status:"match"`; bare responses remain useful for inspection but return `status:"inconclusive"`.
- Both inputs must be absolute regular non-symlink files; the response has a bounded size. Existing Codex rollout confinement remains authoritative for the session path.
- Audit output contains counts, opaque SHA-256 identity and grammar-validated CLI-version digests, diagnostic kinds, and source line numbers only. It never emits provider-controlled strings, message content, raw payloads, capture hashes, or the private input paths.
- The wrapper cannot prove when an independently supplied response was captured. A trusted collector must hash the rollout immediately before and after `thread/read` and create the envelope only when both fingerprints agree; the audit proves that the envelope's asserted fingerprint matches its stable read, not that an untrusted collector followed that procedure.
- Missing client identities, missing/malformed/non-full `itemsView`, duplicate identities, ambiguous envelopes, and mismatched thread ids are unverifiable input rather than evidence of parity. This deliberately requires stronger completeness evidence than the general `thread/read` contract and may yield a safe false-negative when a server omits `itemsView` despite returning all items.
- Physical turns come from normalized `task_started` progress records, including turns with no authored input. Authored messages are then compared by ordered `(turn id, client id, subtype)` identity so missing zero-input turns and steer placement cannot disappear from the denominator.
