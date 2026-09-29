# Codex supplemental history export

## Goal

Add an opt-in, capability-gated headless command that returns the unchanged ordinary Codex projection as `primary` plus independently certified post-cutoff tails from superseded paginated carriers as `supplemental`.

## Requirements

- Advertise `dump-session-history` and the provider-discriminated `supplemental-history-v1` envelope through `--capabilities`; only Codex implements it in this release.
- Accept only the Codex provider and preserve the existing `--dump-session`, `--dump-session-snapshot`, listing, search, and offline-backup contracts.
- Produce `primary` from the same normalized lineage bytes and provider-neutral finalization used by the ordinary dump.
- Return one supplemental group per non-empty same-thread post-cutoff physical carrier tail in oldest-to-newest lineage order; copied cross-thread fork ancestry remains primary-only.
- Bind each group to the immutable rollout id, exact accepted-prefix and tail byte/ordinal bounds, a BLAKE3 fingerprint of the complete decoded carrier, and certified primary record references on both sides of the insertion boundary.
- Normalize a tail only when it is independently append-only relative to the accepted projection; cross-boundary normalization effects reject the history export.
- Reject the complete history export on missing or ambiguous carriers, cycles, non-paginated or mismatched metadata, split record boundaries, malformed tail JSON, non-contiguous tail ordinals, changing sources, unsafe parser references, or duplicate certified `recordRef` identities across either lane.
- Bound captured decoded carrier data, independently replayed bytes, and group count so adversarial lineages cannot amplify the opt-in proof path without limit.
- Keep superseded carriers absent from ordinary session listings and their tails absent from ordinary dump, snapshot, and search results.

## Non-goals

- No ccmsg consumption, legacy-ordinal allocation, turn reconstruction, selectors, Match integration, UI, or supplemental search.
- No recovery from deleted carriers or from evidence that does not satisfy the existing `history_base` lineage contract.
- No change to the headless API v1 ordinary command shapes.

## Output contract

`--dump-session-history <session-id|session-path> --provider codex` emits a schema-versioned object with the logical provider/session identity, `primary`, and `supplemental`. Every supplemental group carries `reason:"superseded-history-base-tail"`, its carrier evidence, accepted and tail bounds, decoded-byte source fingerprint, required certified adjacent primary record references, and normalized `messages`.
