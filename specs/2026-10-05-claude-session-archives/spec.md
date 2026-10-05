# Claude session archives

Update the headless Claude lifecycle contract to match Claude Code's current VS Code archive state. Membership in `Anthropic.claude-code.hiddenSessionIds` reports `is_archived:true`; retain `is_hidden` as a legacy wire field for older consumers. Listing and targeted metadata must agree, including teleported rows.

Advertise `claude-session-archive-v1` and extend the existing archive/unarchive commands to an exact native Claude session ID. Archive retains the transcript. Unarchive removes the ID and atomically writes the native `sessionUnarchivedAt` grace timestamp, retaining only timestamps strictly newer than the 14-day cutoff. Preserve unrelated extension state, refuse malformed state, and report unavailable stores explicitly. Keep the Copilot command behavior and legacy Claude hide response compatible.

Validate with temporary SQLite stores and read-only native metadata; never mutate real editor or session state during verification. The additive lifecycle capability receives a minor version transition from 1.25.0 to 1.26.0.
