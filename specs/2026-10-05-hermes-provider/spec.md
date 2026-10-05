# Hermes session provider

Add read-only native Hermes discovery and normalized conversation loading for ccmsg, including sessions whose inference backend is the Codex subscription. Hermes remains the host provider; a subscription backend does not change conversation ownership.

Read `state.db` from `HERMES_HOME`, otherwise the platform's Hermes home, and discover retained named profiles. Preserve native session IDs, titles, recorded working directories, surface, model and supported inference settings. Sessions without a recorded directory use a synthetic Hermes project. Never infer a repository from prompt or tool text.

Use SQLite read-only transactions that include committed WAL data. Preserve display history across in-place compaction, exclude rewound and model-only rows, pair tools with native occurrence UIDs or fenced wire aliases while retaining native IDs alongside normalized pairing identities, decode Hermes's sentinel-prefixed multimodal content, and include only verified compression ancestors. Do not read credentials, run Hermes, migrate its schema, mutate lifecycle state, or certify unsupported stable record identities.

Expose the provider through the existing listing, dump, search, reconstruction, attribution, TUI and Companion paths. Native rename, deletion, archive writes, resume execution, optimized cursors and backup creation remain unsupported until separately implemented.
