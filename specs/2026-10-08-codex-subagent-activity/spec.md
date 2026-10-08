# Codex subagent activity

Provide an additive, read-only `codex-subagent-activity-v1` headless envelope that separates proven inherited context from native child tasks and routed inter-agent messages. Preserve ordinary dump, snapshot, listing, search, export, numbered-turn and certified-record contracts exactly. Public task selectors and task-based transcript export are outside this slice.

The first child `session_meta` owns session identity, spawn-parent identity, agent path and `subagent_history_start_ordinal`. Require contiguous physical ordinals, consistent parent metadata and a boundary inside the accepted carrier. Never infer inherited provenance from timestamps or quoted prose. Invalid or unsupported proof yields an unavailable boundary and retains ordinary records as unassigned. Paginated lineage without a complete boundary remains unavailable.

Reuse the existing Codex conversation parser and provider-neutral finalizer. Track source contributions separately without changing any normalized record. Do not split merged tool results across history/task boundaries. Group by exact native `task_started`, `task_complete` and `turn_aborted` IDs; retain ambiguous ownership as unassigned with diagnostics. Read routing and readable content from native `agent_message` items, represent encrypted payload presence only as an unavailable Boolean, and bind `triggerTurn` only to adjacent native communication metadata. Never synthesize authored user input or expose ciphertext.

Bound decoded bytes, native records, normalized records, tasks and routed messages. Reject changed reads, oversized input and duplicate message identities. Malformed provenance makes the boundary unavailable; duplicate/overlapping tasks receive diagnostics and no guessed attribution. Every normalized record retains its ordinary child session and provider binding.

Verify redacted fixtures representing the five reported sessions, invalid boundaries, missing metadata, task reactivation/interruption, routing, encrypted text, overlap, merged tool results and append refresh. Run single-thread Rust tests, full-feature Clippy, formatting and a release binary end-to-end against the five supplied read-only rollouts.

## Wire contract

The command requires explicit `--provider codex`; `--format` accepts only `json`. Successful stdout contains exactly one JSON document. `--output <new-file>` uses the existing read-only audit emitter's atomic create-new semantics. Existing output paths are refused.

```json
{
  "schemaVersion": 1,
  "provider": "codex",
  "sessionId": "native-child-id",
  "parentSessionId": "native-spawn-parent-id",
  "agentPath": "/root/child",
  "boundary": { "status": "verified", "startOrdinal": 12 },
  "inherited": [],
  "tasks": [{
    "taskId": "native-task-id",
    "status": "complete",
    "startedAt": "2026-10-07T20:22:45Z",
    "completedAt": "2026-10-07T20:27:27Z",
    "startOrdinal": 13,
    "endOrdinalExclusive": 130,
    "records": [],
    "messages": [{
      "id": "native-agent-message-id",
      "ordinal": 18,
      "timestamp": "2026-10-07T20:22:45Z",
      "author": "/root",
      "recipient": "/root/child",
      "direction": "inbound",
      "triggerTurn": true,
      "text": "Readable assignment header",
      "encryptedPayloadUnavailable": true
    }]
  }],
  "unassigned": [],
  "unassignedMessages": [],
  "diagnostics": []
}
```

Optional properties are omitted, never serialized as null. An unavailable boundary is exactly `{ "status": "unavailable", "reason": "typed-reason-code" }`, with empty `inherited` and `tasks`, the complete ordinary normalized projection in `unassigned`, and all readable routed evidence in `unassignedMessages`. Routed-message `ordinal` always denotes its zero-based physical record position; this equals the native ordinal under verified contiguous proof, while unavailable proof makes no native identity or task claim. A verified `startOrdinal` is the first child-local native record, including bookkeeping before task start. A task's `endOrdinalExclusive` is its exact terminal ordinal plus one; active tasks omit the end and completion timestamp. Status is `active`, `complete` or `interrupted`. Record arrays contain the unchanged provider-neutral finalized `ClaudeMessage` records stamped with the selected child session and Codex provider. This is an additive activity surface, not a new integer turn namespace.

The classifier retains every normalized record once. Source spans include merged tool results, canonical user correlation, usage/terminal enrichment and confirmed fork-boundary enrichment. A span crossing inherited/local provenance or distinct task ownership remains unassigned. Every task involved in overlapping activity or repeated native task IDs is omitted, with its records and routed messages retained as unassigned. Unidentified or contradictory lifecycle records quarantine active and subsequent task lanes instead of reusing a prior active lane. Unknown explicit task IDs never fall back to the sole active lane. Diagnostics carry only a typed `code` and optional physical `ordinal`; they do not copy raw provider payloads.

Routed-message direction uses exact equality against the child's authoritative agent path; missing or conflicting routing is `unknown`. Only immediately preceding `inter_agent_communication_metadata.payload.trigger_turn` supplies an optional Boolean. Readable `input_text`, `output_text` and `text` blocks are concatenated in source order. Encrypted blocks supply only the unavailable Boolean. Ciphertext, encrypted keys and arbitrary metadata are never copied into these message descriptors. With verified boundary proof, the v1 routed-message lane is child-local only; known inherited routed headers are excluded rather than labeled as unassigned child activity. An explicit inherited routed-message lane is deferred. Unavailable proof retains every routed descriptor as unassigned because no inherited/local split is claimed.

Limits are 128 MiB decoded input and emitted pretty JSON including stdout's final newline, 250,000 physical and normalized records, 4,096 tasks and 16,384 routed messages. Duplicate normalized record UUIDs and routed-message IDs reject the command. Carrier size/mtime and an independent BLAKE3 re-read must remain stable. Paginated history remains unavailable for a boundary split, but its complete ordinary projection and routed evidence use the existing bounded lineage capture with an aggregate 128 MiB budget, normalized identity/count rechecks and exact revalidation of every ancestor carrier. Ordinary normalized messages and snapshot cursor version 22 remain unchanged; the new independently advertised capability warrants the compatible minor version transition from 1.28.0 to 1.29.0.
