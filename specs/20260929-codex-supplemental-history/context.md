# Context

Codex pagination stores a continuation's preceding immutable rollout id and accepted `end_byte_offset` / `end_ordinal_exclusive` in `session_meta.payload.history_base`. The ordinary loader recursively concatenates only those accepted prefixes and the selected carrier. Bytes after an ancestor cutoff can contain superseded turns but are intentionally ignored by ordinary dump, snapshot, listing, and search behavior.

ccmsg now has stable provider record references, derived turn references, and a private append-only legacy-ordinal registry. Its staged design requires provider-owned supplemental proof before it can reconstruct hidden history without shifting established integer turn references. This slice supplies only that provider wire and keeps every existing projection unchanged.
