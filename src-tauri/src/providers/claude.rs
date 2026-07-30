use super::{ProviderInfo, SessionSnapshotLoad};
use crate::commands::multi_provider::finalize_loaded_messages;
use crate::commands::session::{load_session_messages_sync, parse_visible_message_line};
use crate::models::ClaudeMessage;
use base64::prelude::{Engine as _, BASE64_URL_SAFE_NO_PAD};
use memchr::memchr_iter;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

const SNAPSHOT_CURSOR_VERSION: u32 = 1;

/// Detect Claude Code installation
pub fn detect() -> Option<ProviderInfo> {
    let home = dirs::home_dir()?;
    let claude_path = home.join(".claude");
    let projects_path = claude_path.join("projects");

    Some(ProviderInfo {
        id: "claude".to_string(),
        display_name: "Claude Code".to_string(),
        base_path: claude_path.to_string_lossy().to_string(),
        is_available: projects_path.exists() && projects_path.is_dir(),
    })
}

/// Get the Claude base path (~/.claude)
pub fn get_base_path() -> Option<String> {
    let home = dirs::home_dir()?;
    let claude_path = home.join(".claude");
    if claude_path.exists() {
        Some(claude_path.to_string_lossy().to_string())
    } else {
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ClaudeParserCheckpoint {
    byte_offset: u64,
    line_number: usize,
    replace_from: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClaudeSnapshotCursor {
    version: u32,
    provider: String,
    canonical_path: String,
    accepted_len: u64,
    accepted_digest: String,
    checkpoint: ClaudeParserCheckpoint,
    checkpoint_digest: String,
}

#[derive(Clone)]
struct SnapshotRecord {
    byte_offset: usize,
    line_number: usize,
    message: Option<ClaudeMessage>,
    uuid: Option<String>,
    parent_uuid: Option<String>,
    tool_use_ids: HashSet<String>,
    tool_result_ids: HashSet<String>,
    stable_identity: bool,
}

fn digest_bytes(bytes: &[u8]) -> String {
    BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(bytes))
}

fn encode_snapshot_cursor(cursor: &ClaudeSnapshotCursor) -> Result<String, String> {
    serde_json::to_vec(cursor)
        .map(|bytes| BASE64_URL_SAFE_NO_PAD.encode(bytes))
        .map_err(|error| format!("Failed to encode Claude snapshot cursor: {error}"))
}

fn decode_snapshot_cursor(encoded: &str) -> Result<ClaudeSnapshotCursor, String> {
    const MAX_CURSOR_BYTES: usize = 64 * 1024;
    if encoded.len() > MAX_CURSOR_BYTES {
        return Err("Claude snapshot cursor is too large".to_string());
    }
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|error| format!("Invalid Claude snapshot cursor encoding: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid Claude snapshot cursor payload: {error}"))
}

/** Authoritative normalized replacement boundary carried by a provider cursor. */
#[cfg(test)]
pub(crate) fn snapshot_cursor_replace_from(encoded: &str) -> Result<usize, String> {
    Ok(decode_snapshot_cursor(encoded)?.checkpoint.replace_from)
}

fn canonical_session_path(path: &Path, original: &str) -> Result<PathBuf, String> {
    if !path.exists() {
        return Err(format!("Session file not found: {original}"));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("Failed to resolve Claude session path: {error}"))?;
    if !canonical.is_file() {
        return Err(format!("Claude session path is not a file: {original}"));
    }
    Ok(canonical)
}

fn is_plain_jsonl(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
}

#[allow(unsafe_code)] // Required for the same read-only mmap strategy as the complete loader.
fn map_session(path: &Path) -> Result<(Mmap, std::fs::Metadata), String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    // SAFETY: The file is opened read-only and the mapping is only read.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|error| error.to_string())?;
    Ok((mmap, metadata))
}

fn source_stayed_stable(
    before: &std::fs::Metadata,
    after: &std::fs::Metadata,
    mapped_len: usize,
) -> bool {
    let (Ok(before_modified), Ok(after_modified)) = (before.modified(), after.modified()) else {
        return false;
    };
    before.len() == u64::try_from(mapped_len).unwrap_or(u64::MAX)
        && after.len() == before.len()
        && after_modified == before_modified
}

fn line_accepted_len(bytes: &[u8]) -> usize {
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return bytes.len();
    }

    let start = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |position| position + 1);
    if serde_json::from_slice::<Value>(&bytes[start..]).is_ok() {
        bytes.len()
    } else {
        start
    }
}

fn collect_tool_ids(raw: &Value, kind: &str, key: &str) -> HashSet<String> {
    let content = raw
        .get("message")
        .and_then(|message| message.get("content"))
        .or_else(|| raw.get("content"));
    let mut output = HashSet::new();
    if let Some(blocks) = content.and_then(Value::as_array) {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some(kind) {
                if let Some(id) = block.get(key).and_then(Value::as_str) {
                    output.insert(id.to_string());
                }
            }
        }
    }
    output
}

fn parse_snapshot_records(
    bytes: &[u8],
    absolute_offset: usize,
    first_line_number: usize,
) -> Result<Vec<SnapshotRecord>, ()> {
    let mut records = Vec::new();
    let mut start = 0;

    for (line_number, end) in (first_line_number..).zip(
        memchr_iter(b'\n', bytes)
            .map(Some)
            .chain((!bytes.is_empty() && bytes.last() != Some(&b'\n')).then_some(None)),
    ) {
        let end = end.unwrap_or(bytes.len());
        let raw_line = &bytes[start..end];
        if !raw_line
            .iter()
            .all(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
        {
            let raw: Value = serde_json::from_slice(raw_line).map_err(|_| ())?;
            let mut mutable_line = raw_line.to_vec();
            let mut message = parse_visible_message_line(line_number, &mut mutable_line);
            if let Some(message) = &mut message {
                message.provider.get_or_insert_with(|| "claude".to_string());
            }

            let uuid = raw
                .get("uuid")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let parent_uuid = raw
                .get("parentUuid")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let stable_identity = message.is_none()
                || (uuid.is_some()
                    && raw
                        .get("sessionId")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                    && raw
                        .get("timestamp")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty()));
            let tool_use_ids = collect_tool_ids(&raw, "tool_use", "id");
            let tool_result_ids = collect_tool_ids(&raw, "tool_result", "tool_use_id");
            records.push(SnapshotRecord {
                byte_offset: absolute_offset + start,
                line_number,
                message,
                uuid,
                parent_uuid,
                tool_use_ids,
                tool_result_ids,
                stable_identity,
            });
        }
        start = end.saturating_add(1);
    }

    Ok(records)
}

fn finalized_messages(records: &[SnapshotRecord]) -> Vec<ClaudeMessage> {
    finalize_loaded_messages(
        records
            .iter()
            .filter_map(|record| record.message.clone())
            .collect(),
    )
}

fn is_authored_user_boundary(record: &SnapshotRecord) -> bool {
    let Some(message) = record.message.as_ref() else {
        return false;
    };
    if message.message_type != "user" || message.role.as_deref() != Some("user") {
        return false;
    }
    if matches!(
        message.subtype.as_deref(),
        Some("queued_command" | "task_notification" | "compact_summary" | "local_command")
    ) {
        return false;
    }
    record.tool_result_ids.is_empty()
}

fn boundary_is_safe(records: &[SnapshotRecord], boundary: usize) -> bool {
    let prefix_uuids: HashSet<&str> = records[..boundary]
        .iter()
        .filter_map(|record| record.uuid.as_deref())
        .collect();
    let prefix_tool_uses: HashSet<&str> = records[..boundary]
        .iter()
        .flat_map(|record| record.tool_use_ids.iter().map(String::as_str))
        .collect();

    records[boundary..]
        .iter()
        .enumerate()
        .all(|(tail_index, record)| {
            let parent_is_safe = tail_index == 0
                || record
                    .parent_uuid
                    .as_deref()
                    .map_or(true, |parent| !prefix_uuids.contains(parent));
            let tools_are_safe = record
                .tool_result_ids
                .iter()
                .all(|id| !prefix_tool_uses.contains(id.as_str()));
            parent_is_safe && tools_are_safe
        })
}

fn select_checkpoint(
    records: &[SnapshotRecord],
    base_replace_from: usize,
    fallback_byte_offset: usize,
    fallback_line_number: usize,
) -> ClaudeParserCheckpoint {
    let boundary = records
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, record)| {
            (is_authored_user_boundary(record) && boundary_is_safe(records, index)).then_some(index)
        });

    let Some(boundary) = boundary else {
        return ClaudeParserCheckpoint {
            byte_offset: u64::try_from(fallback_byte_offset).unwrap_or(0),
            line_number: fallback_line_number,
            replace_from: base_replace_from,
        };
    };
    let prefix_len = finalized_messages(&records[..boundary]).len();
    ClaudeParserCheckpoint {
        byte_offset: u64::try_from(records[boundary].byte_offset).unwrap_or(0),
        line_number: records[boundary].line_number,
        replace_from: base_replace_from + prefix_len,
    }
}

fn cursor_for(
    canonical_path: &Path,
    bytes: &[u8],
    checkpoint: ClaudeParserCheckpoint,
) -> Result<String, String> {
    let accepted_len = u64::try_from(bytes.len())
        .map_err(|_| "Claude session is too large to cursor".to_string())?;
    let accepted_digest = digest_bytes(bytes);
    let checkpoint_digest =
        checkpoint_digest(canonical_path, accepted_len, &accepted_digest, &checkpoint)?;
    encode_snapshot_cursor(&ClaudeSnapshotCursor {
        version: SNAPSHOT_CURSOR_VERSION,
        provider: "claude".to_string(),
        canonical_path: canonical_path.to_string_lossy().into_owned(),
        accepted_len,
        accepted_digest,
        checkpoint,
        checkpoint_digest,
    })
}

fn checkpoint_digest(
    canonical_path: &Path,
    accepted_len: u64,
    accepted_digest: &str,
    checkpoint: &ClaudeParserCheckpoint,
) -> Result<String, String> {
    let proof = serde_json::to_vec(&(
        "ccmsg-claude-snapshot-checkpoint-v1",
        canonical_path.to_string_lossy(),
        accepted_len,
        accepted_digest,
        checkpoint,
    ))
    .map_err(|error| format!("Failed to encode Claude checkpoint proof: {error}"))?;
    Ok(digest_bytes(&proof))
}

fn load_complete_messages(path: &Path) -> Result<Vec<ClaudeMessage>, String> {
    let mut messages = load_session_messages_sync(&path.to_string_lossy())?;
    for message in &mut messages {
        message.provider.get_or_insert_with(|| "claude".to_string());
    }
    Ok(finalize_loaded_messages(messages))
}

fn complete_snapshot_from_path(
    canonical_path: &Path,
    reason: impl Into<String>,
) -> Result<SessionSnapshotLoad, String> {
    let reason = reason.into();
    if !is_plain_jsonl(canonical_path) {
        return Ok(SessionSnapshotLoad::Full {
            reason,
            messages: load_complete_messages(canonical_path)?,
            cursor: None,
            cursor_replace_from: None,
        });
    }

    let (mmap, before) = map_session(canonical_path)?;
    let accepted_len = line_accepted_len(&mmap);
    let records = parse_snapshot_records(&mmap[..accepted_len], 0, 0);
    let after = fs::metadata(canonical_path).map_err(|error| error.to_string())?;
    let stable = source_stayed_stable(&before, &after, mmap.len());
    let eligible = records
        .as_ref()
        .is_ok_and(|records| records.iter().all(|record| record.stable_identity));

    if !stable || !eligible || accepted_len != mmap.len() {
        return Ok(SessionSnapshotLoad::Full {
            reason,
            messages: load_complete_messages(canonical_path)?,
            cursor: None,
            cursor_replace_from: None,
        });
    }

    let records = records.expect("eligible records");
    let messages = finalized_messages(&records);
    let checkpoint = select_checkpoint(&records, 0, 0, 0);
    let cursor_replace_from = checkpoint.replace_from;
    let cursor = cursor_for(canonical_path, &mmap[..accepted_len], checkpoint)?;
    Ok(SessionSnapshotLoad::Full {
        reason,
        messages,
        cursor: Some(cursor),
        cursor_replace_from: Some(cursor_replace_from),
    })
}

fn checkpoint_is_valid(cursor: &ClaudeSnapshotCursor, bytes: &[u8]) -> bool {
    let Ok(offset) = usize::try_from(cursor.checkpoint.byte_offset) else {
        return false;
    };
    let Ok(accepted_len) = usize::try_from(cursor.accepted_len) else {
        return false;
    };
    if offset > accepted_len || accepted_len > bytes.len() {
        return false;
    }
    let checkpoint_digest = checkpoint_digest(
        Path::new(&cursor.canonical_path),
        cursor.accepted_len,
        &cursor.accepted_digest,
        &cursor.checkpoint,
    );
    if !checkpoint_digest.is_ok_and(|digest| digest == cursor.checkpoint_digest) {
        return false;
    }

    let line_number = memchr_iter(b'\n', &bytes[..offset]).count();
    if line_number != cursor.checkpoint.line_number || cursor.checkpoint.replace_from > line_number
    {
        return false;
    }
    if offset == 0 {
        return cursor.checkpoint.line_number == 0 && cursor.checkpoint.replace_from == 0;
    }
    if offset >= accepted_len || bytes.get(offset.wrapping_sub(1)) != Some(&b'\n') {
        return false;
    }

    let end = memchr::memchr(b'\n', &bytes[offset..accepted_len])
        .map_or(accepted_len, |relative| offset + relative);
    let Ok(records) =
        parse_snapshot_records(&bytes[offset..end], offset, cursor.checkpoint.line_number)
    else {
        return false;
    };
    records.first().is_some_and(|record| {
        record.byte_offset == offset && record.stable_identity && is_authored_user_boundary(record)
    })
}

fn prefix_contains_dependencies(
    prefix: &[u8],
    parent_candidates: &HashSet<&str>,
    tool_candidates: &HashSet<&str>,
) -> bool {
    if parent_candidates.is_empty() && tool_candidates.is_empty() {
        return false;
    }
    let Ok(records) = parse_snapshot_records(prefix, 0, 0) else {
        return true;
    };
    records.iter().any(|record| {
        record
            .uuid
            .as_deref()
            .is_some_and(|uuid| parent_candidates.contains(uuid))
            || record
                .tool_use_ids
                .iter()
                .any(|id| tool_candidates.contains(id.as_str()))
    })
}

/// Load a cursor-aware normalized Claude snapshot.
///
/// The cursor proves the accepted byte prefix and replays from a safe authored
/// user boundary. Cross-boundary causal or tool-result dependencies conservatively
/// fall back to the same complete normalized load used by the GUI.
pub(crate) fn load_session_snapshot(
    session_path: &str,
    encoded_cursor: Option<&str>,
) -> Result<SessionSnapshotLoad, String> {
    let canonical_path = canonical_session_path(Path::new(session_path), session_path)?;
    let Some(encoded_cursor) = encoded_cursor else {
        return complete_snapshot_from_path(&canonical_path, "initial");
    };
    if !is_plain_jsonl(&canonical_path) {
        return complete_snapshot_from_path(&canonical_path, "unsupported-source");
    }

    let cursor = match decode_snapshot_cursor(encoded_cursor) {
        Ok(cursor) => cursor,
        Err(_) => return complete_snapshot_from_path(&canonical_path, "invalid-cursor"),
    };
    if cursor.version != SNAPSHOT_CURSOR_VERSION
        || cursor.provider != "claude"
        || cursor.canonical_path != canonical_path.to_string_lossy()
    {
        return complete_snapshot_from_path(&canonical_path, "incompatible-cursor");
    }

    let (mmap, before) = map_session(&canonical_path)?;
    let Ok(previous_accepted_len) = usize::try_from(cursor.accepted_len) else {
        return complete_snapshot_from_path(&canonical_path, "invalid-cursor");
    };
    if previous_accepted_len > mmap.len() {
        return complete_snapshot_from_path(&canonical_path, "source-shrank");
    }
    if !checkpoint_is_valid(&cursor, &mmap) {
        return complete_snapshot_from_path(&canonical_path, "invalid-checkpoint");
    }
    if digest_bytes(&mmap[..previous_accepted_len]) != cursor.accepted_digest {
        return complete_snapshot_from_path(&canonical_path, "prefix-mismatch");
    }

    let accepted_len = line_accepted_len(&mmap);
    let after = fs::metadata(&canonical_path).map_err(|error| error.to_string())?;
    if !source_stayed_stable(&before, &after, mmap.len()) {
        return complete_snapshot_from_path(&canonical_path, "source-changed-during-read");
    }
    if accepted_len < previous_accepted_len {
        return complete_snapshot_from_path(&canonical_path, "accepted-prefix-shrank");
    }
    if accepted_len == previous_accepted_len {
        return Ok(SessionSnapshotLoad::Unchanged {
            cursor: encoded_cursor.to_string(),
        });
    }

    let checkpoint_offset = usize::try_from(cursor.checkpoint.byte_offset)
        .map_err(|_| "Invalid Claude snapshot checkpoint".to_string())?;
    let tail = &mmap[checkpoint_offset..accepted_len];
    let records =
        match parse_snapshot_records(tail, checkpoint_offset, cursor.checkpoint.line_number) {
            Ok(records) if records.iter().all(|record| record.stable_identity) => records,
            _ => return complete_snapshot_from_path(&canonical_path, "unstable-record"),
        };

    let tail_uuids: HashSet<&str> = records
        .iter()
        .filter_map(|record| record.uuid.as_deref())
        .collect();
    let tail_tool_uses: HashSet<&str> = records
        .iter()
        .flat_map(|record| record.tool_use_ids.iter().map(String::as_str))
        .collect();
    let new_visible_record_count = records
        .iter()
        .filter(|record| record.byte_offset >= previous_accepted_len && record.message.is_some())
        .count();
    if new_visible_record_count == 0 {
        let next_cursor = cursor_for(&canonical_path, &mmap[..accepted_len], cursor.checkpoint)?;
        let after_parse = fs::metadata(&canonical_path).map_err(|error| error.to_string())?;
        if !source_stayed_stable(&before, &after_parse, mmap.len()) {
            return complete_snapshot_from_path(&canonical_path, "source-changed-during-read");
        }
        return Ok(SessionSnapshotLoad::Unchanged {
            cursor: next_cursor,
        });
    }
    let new_records = records
        .iter()
        .filter(|record| record.byte_offset >= previous_accepted_len);
    let mut parent_candidates = HashSet::new();
    let mut tool_candidates = HashSet::new();
    for record in new_records {
        if record.message.is_none() {
            continue;
        }
        if let Some(parent) = record.parent_uuid.as_deref() {
            if !tail_uuids.contains(parent) {
                parent_candidates.insert(parent);
            }
        }
        for tool_result_id in &record.tool_result_ids {
            if !tail_tool_uses.contains(tool_result_id.as_str()) {
                tool_candidates.insert(tool_result_id.as_str());
            }
        }
    }
    if prefix_contains_dependencies(
        &mmap[..checkpoint_offset],
        &parent_candidates,
        &tool_candidates,
    ) {
        return complete_snapshot_from_path(&canonical_path, "unsafe-backward-reference");
    }

    let messages = finalized_messages(&records);
    let next_checkpoint = select_checkpoint(
        &records,
        cursor.checkpoint.replace_from,
        checkpoint_offset,
        cursor.checkpoint.line_number,
    );
    let cursor_replace_from = next_checkpoint.replace_from;
    let next_cursor = cursor_for(&canonical_path, &mmap[..accepted_len], next_checkpoint)?;
    let after_parse = fs::metadata(&canonical_path).map_err(|error| error.to_string())?;
    if !source_stayed_stable(&before, &after_parse, mmap.len()) {
        return complete_snapshot_from_path(&canonical_path, "source-changed-during-read");
    }
    Ok(SessionSnapshotLoad::Replace {
        replace_from: cursor.checkpoint.replace_from,
        messages,
        cursor: next_cursor,
        cursor_replace_from,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::SessionSnapshotLoad;
    use std::io::Write;

    fn user(uuid: &str, parent: Option<&str>, text: &str) -> String {
        serde_json::json!({
            "uuid": uuid,
            "parentUuid": parent,
            "sessionId": "session-1",
            "timestamp": "2026-07-29T10:00:00Z",
            "type": "user",
            "message": {"role": "user", "content": text}
        })
        .to_string()
    }

    fn assistant(uuid: &str, parent: &str, content: serde_json::Value) -> String {
        serde_json::json!({
            "uuid": uuid,
            "parentUuid": parent,
            "sessionId": "session-1",
            "timestamp": "2026-07-29T10:00:01Z",
            "type": "assistant",
            "message": {"role": "assistant", "content": content}
        })
        .to_string()
    }

    fn write_lines(lines: &[String]) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".jsonl")
            .tempfile()
            .expect("temp session");
        for line in lines {
            writeln!(file, "{line}").expect("write fixture");
        }
        file.flush().expect("flush fixture");
        file
    }

    fn complete_messages(path: &str) -> Vec<crate::models::ClaudeMessage> {
        match load_session_snapshot(path, None).expect("complete snapshot") {
            SessionSnapshotLoad::Full { messages, .. } => messages,
            _ => panic!("initial load must be complete"),
        }
    }

    #[test]
    fn snapshot_reuses_a_safe_authored_user_boundary_for_append() {
        let mut file = write_lines(&[
            user("u1", None, "one"),
            assistant("a1", "u1", serde_json::json!("answer one")),
            user("u2", Some("a1"), "two"),
            assistant("a2", "u2", serde_json::json!("answer two")),
        ]);
        let path = file.path().to_string_lossy().into_owned();

        let (initial, cursor) = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                messages,
                cursor: Some(cursor),
                ..
            } => (messages, cursor),
            _ => panic!("stable JSONL must produce a cursor"),
        };
        assert_eq!(
            serde_json::to_value(&initial).expect("snapshot messages"),
            serde_json::to_value(load_complete_messages(file.path()).expect("ordinary messages"))
                .expect("ordinary messages")
        );
        assert_eq!(snapshot_cursor_replace_from(&cursor).expect("boundary"), 2);
        assert!(matches!(
            load_session_snapshot(&path, Some(&cursor)).expect("unchanged"),
            SessionSnapshotLoad::Unchanged { .. }
        ));

        writeln!(file, "{}", user("u3", Some("a2"), "three")).expect("append user");
        writeln!(
            file,
            "{}",
            assistant("a3", "u3", serde_json::json!("answer three"))
        )
        .expect("append assistant");
        file.flush().expect("flush append");

        let (replace_from, replacement, next_cursor) =
            match load_session_snapshot(&path, Some(&cursor)).expect("delta") {
                SessionSnapshotLoad::Replace {
                    replace_from,
                    messages,
                    cursor,
                    ..
                } => (replace_from, messages, cursor),
                _ => panic!("safe append must produce replacement"),
            };
        let fresh = complete_messages(&path);
        let mut spliced = initial[..replace_from].to_vec();
        spliced.extend(replacement);
        assert_eq!(
            serde_json::to_value(spliced).expect("spliced"),
            serde_json::to_value(fresh).expect("fresh")
        );
        assert_eq!(
            snapshot_cursor_replace_from(&next_cursor).expect("next boundary"),
            4
        );
    }

    #[test]
    fn snapshot_falls_back_when_a_late_tool_result_reaches_before_the_boundary() {
        let mut file = write_lines(&[
            user("u1", None, "one"),
            assistant(
                "a1",
                "u1",
                serde_json::json!([{"type":"tool_use","id":"old-call","name":"Read","input":{}}]),
            ),
            user("u2", Some("a1"), "two"),
            assistant("a2", "u2", serde_json::json!("answer two")),
        ]);
        let path = file.path().to_string_lossy().into_owned();
        let cursor = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };

        writeln!(
            file,
            "{}",
            serde_json::json!({
                "uuid": "result",
                "parentUuid": "a2",
                "sessionId": "session-1",
                "timestamp": "2026-07-29T10:00:02Z",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type":"tool_result","tool_use_id":"old-call","content":"late"}]
                }
            })
        )
        .expect("append result");
        file.flush().expect("flush append");

        match load_session_snapshot(&path, Some(&cursor)).expect("safe fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "unsafe-backward-reference");
            }
            _ => panic!("late result must force a complete load"),
        }
    }

    #[test]
    fn snapshot_accepts_tail_local_tool_results_and_metadata_only_appends() {
        let mut file = write_lines(&[
            user("u1", None, "one"),
            assistant("a1", "u1", serde_json::json!("answer one")),
            user("u2", Some("a1"), "two"),
            assistant(
                "a2",
                "u2",
                serde_json::json!([{"type":"tool_use","id":"tail-call","name":"Read","input":{}}]),
            ),
        ]);
        let path = file.path().to_string_lossy().into_owned();
        let (initial, cursor) = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                messages,
                cursor: Some(cursor),
                ..
            } => (messages, cursor),
            _ => panic!("cursor expected"),
        };

        writeln!(
            file,
            "{}",
            serde_json::json!({
                "uuid": "result",
                "parentUuid": "a2",
                "sessionId": "session-1",
                "timestamp": "2026-07-29T10:00:02Z",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type":"tool_result","tool_use_id":"tail-call","content":"ok"}]
                }
            })
        )
        .expect("append result");
        file.flush().expect("flush result");
        let (replace_from, replacement, cursor) =
            match load_session_snapshot(&path, Some(&cursor)).expect("tail result") {
                SessionSnapshotLoad::Replace {
                    replace_from,
                    messages,
                    cursor,
                    ..
                } => (replace_from, messages, cursor),
                _ => panic!("tail-local result must remain incremental"),
            };
        let mut spliced = initial[..replace_from].to_vec();
        spliced.extend(replacement);
        assert_eq!(
            serde_json::to_value(spliced).expect("spliced"),
            serde_json::to_value(complete_messages(&path)).expect("fresh")
        );

        writeln!(
            file,
            "{}",
            serde_json::json!({
                "type": "custom-title",
                "sessionId": "session-1",
                "customTitle": "metadata only"
            })
        )
        .expect("append metadata");
        file.flush().expect("flush metadata");
        let advanced_cursor =
            match load_session_snapshot(&path, Some(&cursor)).expect("metadata append") {
                SessionSnapshotLoad::Unchanged { cursor: next } => next,
                _ => panic!("metadata-only append must not replace messages"),
            };
        assert_ne!(advanced_cursor, cursor);
    }

    #[test]
    fn snapshot_falls_back_when_a_new_parent_targets_the_retained_prefix() {
        let mut file = write_lines(&[
            user("u1", None, "one"),
            assistant("a1", "u1", serde_json::json!("answer one")),
            user("u2", Some("a1"), "two"),
            assistant("a2", "u2", serde_json::json!("answer two")),
        ]);
        let path = file.path().to_string_lossy().into_owned();
        let cursor = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };
        writeln!(
            file,
            "{}",
            user("branch", Some("a1"), "branch from retained prefix")
        )
        .expect("append branch");
        file.flush().expect("flush branch");

        match load_session_snapshot(&path, Some(&cursor)).expect("backward fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "unsafe-backward-reference");
            }
            _ => panic!("backward parent must force a complete load"),
        }
    }

    #[test]
    fn snapshot_accepts_complete_unterminated_lines_but_not_incomplete_tails() {
        let mut file = tempfile::Builder::new()
            .suffix(".jsonl")
            .tempfile()
            .expect("temp session");
        write!(file, "{}", user("u1", None, "one")).expect("write complete line");
        file.flush().expect("flush complete line");
        let path = file.path().to_string_lossy().into_owned();
        let cursor = match load_session_snapshot(&path, None).expect("unterminated complete") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("complete JSON is a stable accepted record"),
        };

        write!(file, "\n{{\"type\":\"user\"").expect("append incomplete line");
        file.flush().expect("flush incomplete line");
        match load_session_snapshot(&path, Some(&cursor)).expect("incomplete tail") {
            SessionSnapshotLoad::Unchanged { cursor: advanced } => {
                assert_ne!(advanced, cursor);
            }
            _ => panic!("incomplete tail must remain outside the accepted prefix"),
        }
    }

    #[test]
    fn snapshot_falls_back_for_rewrite_shrink_and_unstable_visible_records() {
        let mut file = write_lines(&[
            user("u1", None, "one"),
            assistant("a1", "u1", serde_json::json!("answer one")),
        ]);
        let path = file.path().to_string_lossy().into_owned();
        let cursor = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };

        file.as_file_mut().set_len(0).expect("shrink");
        writeln!(file, "{}", user("rewritten", None, "rewritten")).expect("rewrite");
        file.flush().expect("flush rewrite");
        match load_session_snapshot(&path, Some(&cursor)).expect("rewrite fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert!(matches!(
                    reason.as_str(),
                    "source-shrank" | "prefix-mismatch"
                ));
            }
            _ => panic!("rewrite must force a complete load"),
        }

        let unstable = write_lines(&[serde_json::json!({
            "sessionId": "session-1",
            "timestamp": "2026-07-29T10:00:00Z",
            "type": "user",
            "message": {"role": "user", "content": "missing uuid"}
        })
        .to_string()]);
        match load_session_snapshot(&unstable.path().to_string_lossy(), None)
            .expect("unstable complete")
        {
            SessionSnapshotLoad::Full { cursor, .. } => assert!(cursor.is_none()),
            _ => panic!("unstable initial load must be complete"),
        }
    }

    #[test]
    fn snapshot_rejects_a_semantically_corrupted_checkpoint() {
        let file = write_lines(&[
            user("u1", None, "one"),
            assistant("a1", "u1", serde_json::json!("answer one")),
            user("u2", Some("a1"), "two"),
            assistant("a2", "u2", serde_json::json!("answer two")),
        ]);
        let path = file.path().to_string_lossy().into_owned();
        let cursor = match load_session_snapshot(&path, None).expect("initial") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };
        let mut decoded = decode_snapshot_cursor(&cursor).expect("decode cursor");
        assert_eq!(decoded.checkpoint.replace_from, 2);
        decoded.checkpoint.replace_from = 1;
        let corrupted = encode_snapshot_cursor(&decoded).expect("encode corruption");

        match load_session_snapshot(&path, Some(&corrupted)).expect("invalid fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "invalid-checkpoint");
            }
            _ => panic!("corrupted checkpoint must force a complete load"),
        }
    }
}
