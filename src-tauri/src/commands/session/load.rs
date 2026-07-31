//! Session loading functions

use crate::models::{ClaudeMessage, ClaudeSession, MessagePage, RawLogEntry};
use crate::utils::{
    extract_project_name, find_line_ranges, find_line_starts, prompt_attachment_name,
    prompt_attachments_data,
};
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use uuid::Uuid;
use walkdir::WalkDir;

/// Cache entry for a single session file (supports incremental parsing)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct CachedSessionMetadata {
    /// File modification time (as Unix timestamp)
    modified_time: u64,
    /// File size in bytes (for detecting append-only changes)
    file_size: u64,
    /// Last byte offset processed (for incremental parsing)
    last_byte_offset: u64,
    /// Cached session data (None if file had no valid messages)
    session: Option<ClaudeSession>,
    /// Number of sidechain messages (for filtering adjustment)
    sidechain_count: usize,
    /// Whether `tool_use` was detected (for incremental updates)
    has_tool_use: bool,
    /// Whether errors were detected (for incremental updates)
    has_errors: bool,
    /// First user content (for multi-tier fallback)
    #[serde(default)]
    first_user_content: Option<String>,
    /// Last user content (for multi-tier fallback)
    #[serde(default)]
    last_user_content: Option<String>,
    /// First assistant text (for multi-tier fallback)
    #[serde(default)]
    first_assistant_text: Option<String>,
    /// Rename name from /rename command
    #[serde(default)]
    rename_name: Option<String>,
}

/// Session metadata cache file structure
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SessionMetadataCache {
    /// Version for cache invalidation on format changes
    version: u32,
    /// Map of file path -> cached metadata
    entries: HashMap<String, CachedSessionMetadata>,
}

// Bumped 8 -> 9: ClaudeSession gained the `entrypoint` field, so stale caches
// must be invalidated to force a reparse that populates it.
// Bumped 9 -> 10: rename_name is now also extracted from `/branch` custom-title
// events (not just `/rename`), AND Claude project names are derived from the
// JSONL `cwd` when available; stale caches must be invalidated to pick both up.
// Bumped 10 -> 11: a verifiable folder name now takes priority over the JSONL
// `cwd` for the project name (handles sessions moved between project folders);
// stale caches must be invalidated to recompute project_name.
// Bumped 11 -> 12: message_count is now computed from an allowlist (user +
// assistant + `queued_command` attachments) instead of an exclusion denylist, so
// it no longer counts non-conversational metadata records (`ai-title`, `mode`,
// `permission-mode`, and any future type); stale caches hold the old inflated
// count, so they must be invalidated to recompute.
// Bumped 12 -> 13: Claude local slash-command envelopes and their stdout echoes
// are no longer counted as authored conversation.
// Bumped 13 -> 14: agent-invoking `/init` envelopes now count as conversational
// boundaries instead of generic local-command plumbing.
// Bumped 14 -> 15: provider-generated task-notification user records no longer
// count as authored conversation or participate in title fallbacks.
const CACHE_VERSION: u32 = 15;
const DEFAULT_SESSION_PAGE_LIMIT: usize = 250;
const MAX_SESSION_PAGE_LIMIT: usize = 500;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPage {
    pub sessions: Vec<ClaudeSession>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
    pub next_offset: usize,
    pub has_more: bool,
}
/// Get the cache file path for a project
fn get_cache_path(project_path: &str) -> PathBuf {
    PathBuf::from(project_path).join(".session_cache.json")
}

/// Load cache from disk
fn load_cache(project_path: &str) -> SessionMetadataCache {
    let cache_path = get_cache_path(project_path);
    if let Ok(content) = fs::read_to_string(&cache_path) {
        if let Ok(cache) = serde_json::from_str::<SessionMetadataCache>(&content) {
            if cache.version == CACHE_VERSION {
                return cache;
            }
        }
    }
    SessionMetadataCache::default()
}

/// Save cache to disk atomically (best effort, errors are ignored)
fn save_cache(project_path: &str, cache: &SessionMetadataCache) {
    let cache_path = get_cache_path(project_path);
    if let Ok(content) = serde_json::to_string(cache) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp_path = cache_path.with_extension(format!("json.{nonce}.tmp"));
        if fs::write(&tmp_path, content.as_bytes()).is_ok() {
            // On Windows, fs::rename fails if the destination already exists
            #[cfg(target_os = "windows")]
            {
                if cache_path.exists() {
                    let _ = fs::remove_file(&cache_path);
                }
            }
            let _ = fs::rename(&tmp_path, &cache_path);
        }
    }
}

/// Get file modification time as Unix timestamp
fn get_modified_time(path: &PathBuf) -> Option<u64> {
    path.metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// Get file size in bytes
fn get_file_size(path: &PathBuf) -> Option<u64> {
    path.metadata().ok().map(|m| m.len())
}

/// Data needed for incremental parsing continuation
#[derive(Clone)]
struct IncrementalParseState {
    /// Byte offset to start reading from
    start_offset: u64,
    /// Previous message count
    message_count: usize,
    /// Previous sidechain count
    sidechain_count: usize,
    /// Previous last timestamp
    last_timestamp: Option<String>,
    /// Already detected `tool_use`
    has_tool_use: bool,
    /// Already detected errors
    has_errors: bool,
    /// Session ID (already known)
    session_id: Option<String>,
    /// First timestamp (already known)
    first_timestamp: Option<String>,
    /// Summary (already known)
    summary: Option<String>,
    /// First user content (already known)
    first_user_content: Option<String>,
    /// Last user content (already known, for fallback)
    last_user_content: Option<String>,
    /// First assistant text (already known, for fallback)
    first_assistant_text: Option<String>,
    /// Rename name from /rename command (already known)
    rename_name: Option<String>,
    /// Originating client entrypoint (already known)
    entrypoint: Option<String>,
    /// Project display name (already known)
    project_name: Option<String>,
}

/// Minimal struct for fast line classification (avoids full parsing)
#[derive(serde::Deserialize)]
struct LineClassifier {
    #[serde(rename = "type")]
    message_type: String,
    subtype: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    #[serde(rename = "isMeta")]
    is_meta: Option<bool>,
}

/// Minimal struct for extracting session metadata without full message parsing
#[derive(serde::Deserialize)]
struct SessionMetadataEntry {
    #[serde(rename = "type")]
    message_type: String,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    timestamp: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    #[serde(rename = "isMeta")]
    is_meta: Option<bool>,
    summary: Option<String>,
    subtype: Option<String>,
    content: Option<serde_json::Value>,
    #[serde(rename = "toolUse")]
    tool_use: Option<serde_json::Value>,
    #[serde(rename = "toolUseResult")]
    tool_use_result: Option<serde_json::Value>,
    entrypoint: Option<String>,
    cwd: Option<String>,
    message: Option<SessionMetadataMessage>,
    #[serde(rename = "customTitle")]
    custom_title: Option<String>,
    attachment: Option<serde_json::Value>,
    origin: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct SessionMetadataMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

/// Minimal classifier for fast line counting (smaller than `SessionMetadataEntry`)
#[derive(serde::Deserialize)]
struct QuickLineClassifier {
    #[serde(rename = "type")]
    message_type: String,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    timestamp: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    #[serde(rename = "isMeta")]
    is_meta: Option<bool>,
    entrypoint: Option<String>,
    #[serde(rename = "customTitle")]
    custom_title: Option<String>,
    attachment: Option<serde_json::Value>,
    message: Option<SessionMetadataMessage>,
    origin: Option<serde_json::Value>,
}

/// Fast session metadata extraction result
struct SessionExtractionResult {
    session: ClaudeSession,
    sidechain_count: usize,
    /// Final byte offset after parsing (for incremental updates)
    final_byte_offset: u64,
    /// Whether `tool_use` was detected
    has_tool_use: bool,
    /// Whether errors were detected
    has_errors: bool,
    /// First user content (for incremental caching)
    first_user_content: Option<String>,
    /// Last user content (for incremental caching)
    last_user_content: Option<String>,
    /// First assistant text (for incremental caching)
    first_assistant_text: Option<String>,
    /// Rename name from /rename command (for caching)
    rename_name: Option<String>,
}

/// Fast session metadata extraction with two-phase parsing:
/// Phase 1: Extract essential metadata from first ~50 lines
/// Phase 2: Count remaining messages with minimal parsing
/// Always extracts total count (without sidechain filtering) for caching purposes
fn extract_session_metadata_from_file(file_path: &PathBuf) -> Option<SessionExtractionResult> {
    extract_session_metadata_internal(file_path, None)
}

/// Incremental session metadata extraction - only parses new content from given offset
fn extract_session_metadata_incremental(
    file_path: &PathBuf,
    state: IncrementalParseState,
) -> Option<SessionExtractionResult> {
    extract_session_metadata_internal(file_path, Some(state))
}

/// Internal extraction function that supports both full and incremental parsing
fn extract_session_metadata_internal(
    file_path: &PathBuf,
    incremental_state: Option<IncrementalParseState>,
) -> Option<SessionExtractionResult> {
    let metadata = file_path.metadata().ok();
    let file_size = metadata.as_ref().map_or(0, std::fs::Metadata::len);
    let last_modified = metadata
        .as_ref()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let dt: DateTime<Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    let mut file = fs::File::open(file_path).ok()?;
    let file_path_str = file_path.to_string_lossy().to_string();

    // Initialize from incremental state or start fresh
    let (
        start_offset,
        mut message_count,
        mut sidechain_count,
        mut first_timestamp,
        mut last_timestamp,
        mut actual_session_id,
        mut session_summary,
        mut has_tool_use,
        mut has_errors,
        mut first_user_content,
        mut last_user_content,
        mut first_assistant_text,
        mut rename_name,
        mut entrypoint,
        mut session_cwd,
        incremental_project_name,
    ) = if let Some(ref state) = incremental_state {
        (
            state.start_offset,
            state.message_count,
            state.sidechain_count,
            state.first_timestamp.clone(),
            state.last_timestamp.clone(),
            state.session_id.clone(),
            state.summary.clone(),
            state.has_tool_use,
            state.has_errors,
            state.first_user_content.clone(),
            state.last_user_content.clone(),
            state.first_assistant_text.clone(),
            state.rename_name.clone(),
            state.entrypoint.clone(),
            None,
            state.project_name.clone(),
        )
    } else {
        (
            0u64, 0usize, 0usize, None, None, None, None, false, false, None, None, None, None,
            None, None, None,
        )
    };

    // Seek to start position for incremental parsing
    if start_offset > 0 && file.seek(SeekFrom::Start(start_offset)).is_err() {
        return None;
    }

    // Use larger buffer for better I/O performance on large files
    let reader = BufReader::with_capacity(64 * 1024, file);

    // For incremental parsing, we skip the metadata collection phase
    // since we already have it from the previous parse
    let is_incremental = incremental_state.is_some();
    let mut metadata_complete = is_incremental;
    let mut lines_processed = 0usize;
    const METADATA_PHASE_LINES: usize = 100; // Full parse first N lines

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };

        if line.trim().is_empty() {
            continue;
        }

        lines_processed += 1;

        // Phase 1: Full metadata extraction for first N lines (skip if incremental)
        if !metadata_complete && lines_processed <= METADATA_PHASE_LINES {
            if let Ok(entry) = serde_json::from_str::<SessionMetadataEntry>(&line) {
                if session_cwd.is_none() {
                    if let Some(ref cwd) = entry.cwd {
                        let trimmed = cwd.trim();
                        if !trimmed.is_empty() {
                            session_cwd = Some(trimmed.to_string());
                        }
                    }
                }

                // Handle summary messages
                if entry.message_type == "summary" {
                    if session_summary.is_none() {
                        session_summary = entry.summary;
                    }
                    continue;
                }

                // Extract rename name from system/local_command messages before skipping
                if entry.message_type == "system" {
                    if let Some(name) = try_extract_rename(&entry) {
                        rename_name = Some(name);
                    }
                    continue;
                }

                // Extract rename name from /branch custom-title events before skipping
                if entry.message_type == "custom-title" {
                    if let Some(name) =
                        try_extract_custom_title(&entry.message_type, entry.custom_title.as_deref())
                    {
                        rename_name = Some(name);
                    }
                    continue;
                }

                // Count only genuine conversational messages, via an ALLOWLIST: a
                // `user` or `assistant` record, or a `queued_command` attachment (an
                // authored user message sent while the agent was generating, which the
                // dump path normalizes into a user message). An allowlist — rather than
                // excluding a denylist of metadata types — keeps `message_count` immune
                // to the non-conversational record types Claude Code keeps adding
                // (`ai-title`, `mode`, `permission-mode`, …): anything not on the list is
                // simply not counted, with no per-type maintenance. (`summary` / `system`
                // / `custom-title` are handled above for title extraction, so they never
                // reach here. The viewer's own denylist — `EXCLUDED_MESSAGE_TYPES` via
                // `is_system_message_type` — is intentionally left untouched.)
                let command_subtype = entry.message.as_ref().and_then(|message| {
                    local_command_subtype(
                        &entry.message_type,
                        message.role.as_deref(),
                        message.content.as_ref(),
                    )
                });
                let is_conversational = (entry.message_type == "user"
                    && command_subtype != Some("local_command")
                    && !is_task_notification_origin(entry.origin.as_ref()))
                    || entry.message_type == "assistant"
                    || (entry.message_type == "attachment"
                        && queued_command_prompt(entry.attachment.as_ref()).is_some());
                if !is_conversational {
                    continue;
                }

                // Need timestamp or session_id to be valid
                if entry.session_id.is_none() && entry.timestamp.is_none() {
                    continue;
                }

                // Skip meta messages (internal/command-related messages)
                if entry.is_meta.unwrap_or(false) {
                    continue;
                }

                // Track sidechain messages separately
                let is_sidechain = entry.is_sidechain.unwrap_or(false);
                if is_sidechain {
                    sidechain_count += 1;
                }
                message_count += 1;

                // Track timestamps
                if let Some(ref ts) = entry.timestamp {
                    if first_timestamp.is_none() {
                        first_timestamp = Some(ts.clone());
                    }
                    last_timestamp = Some(ts.clone());
                }

                // Track session ID
                if actual_session_id.is_none() {
                    if let Some(ref sid) = entry.session_id {
                        actual_session_id = Some(sid.clone());
                    }
                }

                // Track originating client entrypoint (first hit wins)
                if entrypoint.is_none() {
                    if let Some(ref ep) = entry.entrypoint {
                        entrypoint = Some(ep.clone());
                    }
                }

                // Check for tool use
                if !has_tool_use {
                    if entry.tool_use.is_some() || entry.tool_use_result.is_some() {
                        has_tool_use = true;
                    } else if entry.message_type == "assistant" {
                        if let Some(ref msg) = entry.message {
                            if let Some(ref content) = msg.content {
                                if let Some(arr) = content.as_array() {
                                    for item in arr {
                                        if item.get("type").and_then(|v| v.as_str())
                                            == Some("tool_use")
                                        {
                                            has_tool_use = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check for errors
                if !has_errors {
                    if let Some(ref result) = entry.tool_use_result {
                        if let Some(stderr) = result.get("stderr") {
                            if !stderr.as_str().unwrap_or("").is_empty() {
                                has_errors = true;
                            }
                        }
                    }
                }

                // Extract first user message for summary fallback
                // Note: last_user_content is tracked only within METADATA_PHASE_LINES (first 100 lines).
                // For longer sessions, the actual last user message may be beyond this limit.
                if entry.message_type == "user"
                    && !is_task_notification_origin(entry.origin.as_ref())
                {
                    if let Some(ref msg) = entry.message {
                        if let Some(ref content) = msg.content {
                            let user_text = extract_user_text(content);
                            if first_user_content.is_none() {
                                // Only store genuine user text (skip command displays like "/init")
                                let is_command = matches!(content, serde_json::Value::String(text) if !is_genuine_user_text(text));
                                if !is_command {
                                    first_user_content.clone_from(&user_text);
                                }
                            }
                            if let Some(text) = user_text {
                                last_user_content = Some(text);
                            }
                        }
                    }
                }

                // Extract first assistant text for fallback (resume summaries, etc.)
                if first_assistant_text.is_none() && entry.message_type == "assistant" {
                    if let Some(ref msg) = entry.message {
                        if let Some(ref content) = msg.content {
                            first_assistant_text = extract_assistant_text(content);
                        }
                    }
                }

                // Check if we have all essential metadata
                if actual_session_id.is_some()
                    && first_timestamp.is_some()
                    && (first_user_content.is_some() || session_summary.is_some())
                {
                    metadata_complete = true;
                }
            }
            continue;
        }

        // Phase 2: Fast counting with minimal parsing
        if let Ok(classifier) = serde_json::from_str::<QuickLineClassifier>(&line) {
            // Skip summary
            if classifier.message_type == "summary" {
                // Still capture summary if we don't have one
                if session_summary.is_none() {
                    if let Ok(entry) = serde_json::from_str::<SessionMetadataEntry>(&line) {
                        session_summary = entry.summary;
                    }
                }
                continue;
            }

            // Extract rename from system messages (using fast string check before full parse)
            if classifier.message_type == "system" {
                if line.contains("Session renamed to: ") {
                    if let Ok(entry) = serde_json::from_str::<SessionMetadataEntry>(&line) {
                        if let Some(name) = try_extract_rename(&entry) {
                            rename_name = Some(name);
                        }
                    }
                }
                continue;
            }

            // Extract rename from /branch custom-title events (already parsed above)
            if classifier.message_type == "custom-title" {
                if let Some(name) = try_extract_custom_title(
                    &classifier.message_type,
                    classifier.custom_title.as_deref(),
                ) {
                    rename_name = Some(name);
                }
                continue;
            }

            // Same allowlist as Phase 1: count only `user` / `assistant` records and
            // `queued_command` attachments; everything else (metadata, plumbing, and
            // any future non-conversational type) is simply not counted.
            let command_subtype = classifier.message.as_ref().and_then(|message| {
                local_command_subtype(
                    &classifier.message_type,
                    message.role.as_deref(),
                    message.content.as_ref(),
                )
            });
            let is_conversational = (classifier.message_type == "user"
                && command_subtype != Some("local_command")
                && !is_task_notification_origin(classifier.origin.as_ref()))
                || classifier.message_type == "assistant"
                || (classifier.message_type == "attachment"
                    && queued_command_prompt(classifier.attachment.as_ref()).is_some());
            if !is_conversational {
                continue;
            }

            // Need timestamp or session_id to be valid
            if classifier.session_id.is_none() && classifier.timestamp.is_none() {
                continue;
            }

            // Skip meta messages (internal/command-related messages)
            if classifier.is_meta.unwrap_or(false) {
                continue;
            }

            // Track sidechain messages separately
            let is_sidechain = classifier.is_sidechain.unwrap_or(false);
            if is_sidechain {
                sidechain_count += 1;
            }
            message_count += 1;

            // Update last timestamp
            if let Some(ts) = classifier.timestamp {
                last_timestamp = Some(ts);
            }

            // Track originating client entrypoint (first hit wins)
            if entrypoint.is_none() {
                if let Some(ep) = classifier.entrypoint {
                    entrypoint = Some(ep);
                }
            }

            // Quick tool_use check via string search (faster than full parse)
            if !has_tool_use
                && (line.contains("\"toolUse\"")
                    || line.contains("\"toolUseResult\"")
                    || line.contains("\"tool_use\""))
            {
                has_tool_use = true;
            }

            // Quick error check via string search
            if !has_errors && line.contains("\"stderr\"") && !line.contains("\"stderr\":\"\"") {
                has_errors = true;
            }
        }
    }

    if message_count == 0 {
        return None;
    }

    let raw_project_name = file_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("Unknown")
        .to_string();

    // A verifiable folder name is authoritative (handles sessions moved between
    // project folders, whose embedded `cwd` is stale); otherwise fall back to
    // the JSONL `cwd`, then the cached value, then a lossy folder-name decode.
    let verified_project_name = file_path
        .parent()
        .and_then(|p| p.to_str())
        .and_then(crate::utils::decode_project_path_verified)
        .as_deref()
        .and_then(project_display_name_from_path);
    let project_name = verified_project_name
        .or_else(|| {
            session_cwd
                .as_deref()
                .and_then(project_display_name_from_path)
        })
        .or(incremental_project_name)
        .unwrap_or_else(|| extract_project_name(&raw_project_name));
    // Rename name takes highest priority, then existing summary fallback chain
    let final_summary = rename_name
        .clone()
        .or(session_summary)
        .or(first_user_content.clone())
        .or(first_assistant_text.clone())
        .or(last_user_content.clone());

    Some(SessionExtractionResult {
        session: ClaudeSession {
            session_id: file_path_str.clone(),
            actual_session_id: actual_session_id.unwrap_or_else(|| "unknown-session".to_string()),
            file_path: file_path_str,
            project_name,
            message_count,
            first_message_time: first_timestamp.unwrap_or_else(|| Utc::now().to_rfc3339()),
            last_message_time: last_timestamp
                .clone()
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            last_modified,
            has_tool_use,
            has_errors,
            summary: final_summary,
            is_renamed: rename_name.is_some(),
            provider: None,
            storage_type: None,
            entrypoint,
            forked_from_id: None,
        },
        sidechain_count,
        final_byte_offset: file_size,
        has_tool_use,
        has_errors,
        first_user_content,
        last_user_content,
        first_assistant_text,
        rename_name,
    })
}

/// Non-conversational record types that should never become normalized messages.
///
/// The shared visibility checks also use this list for the GUI, pagination,
/// counts, and offsets so every message surface describes the same sequence.
const EXCLUDED_MESSAGE_TYPES: [&str; 12] = [
    "progress",
    "queue-operation",
    "file-history-snapshot",
    // Incremental editor backup metadata. It has no conversational payload or
    // stable message identity, so normalizing it would invent a UUID.
    "file-history-delta",
    "last-prompt",
    "pr-link",
    // `/branch` naming metadata is consumed by the session-list scanner. It has
    // no stable message UUID/timestamp and must not enter normalized dumps.
    "custom-title",
    // Emitted alongside "custom-title" by the `/branch` command; redundant with it
    // (same name), so it's excluded from the viewer rather than used as a rename source.
    "agent-name",
    // Per-turn AI-generated session title (`{type:"ai-title",aiTitle:…}`) — non-
    // conversational metadata, hidden from the viewer.
    "ai-title",
    // `/mode`-change marker (`{type:"mode",mode:…,sessionId:…}`) — non-conversational
    // metadata, hidden from the viewer.
    "mode",
    // Permission selection and project-relocation state are session metadata,
    // not authored conversation. Observed records have no UUID or timestamp.
    "permission-mode",
    "relocated",
    // Note: session-list `message_count` is computed from a separate conversational
    // allowlist in `extract_session_metadata_internal`, so this denylist does not
    // affect cached metadata or require a cache-version bump.
];

/// System subtypes that are internal metadata (excluded from the viewer).
/// Subtypes NOT in this list (`local_command`, `compact_boundary`, `api_error`, etc.)
/// are shown to the user via `SystemMessageRenderer`.
const HIDDEN_SYSTEM_SUBTYPES: [&str; 2] = ["stop_hook_summary", "turn_duration"];

/// Check if a message should be excluded from the viewer.
/// For "system" type, only specific subtypes are hidden; others are shown.
#[inline]
fn is_system_message_type(message_type: &str) -> bool {
    EXCLUDED_MESSAGE_TYPES.contains(&message_type)
}

/// Check if a system message should be hidden based on its subtype
#[inline]
fn is_hidden_system_subtype(subtype: Option<&str>) -> bool {
    match subtype {
        Some(st) => HIDDEN_SYSTEM_SUBTYPES.contains(&st),
        None => true, // system messages without subtype are internal metadata
    }
}

/// Extract session rename name from a `system/local_command` message content.
/// Matches the pattern: `<local-command-stdout>Session renamed to: {name}</local-command-stdout>`
/// Returns None if the content doesn't match the rename pattern or the name is empty.
fn extract_rename_from_content(content: &serde_json::Value) -> Option<String> {
    let text = content.as_str()?;
    const PREFIX: &str = "<local-command-stdout>Session renamed to: ";
    const SUFFIX: &str = "</local-command-stdout>";
    let rest = text.strip_prefix(PREFIX)?;
    let name = rest.strip_suffix(SUFFIX)?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

/// Try to extract a rename name from a `SessionMetadataEntry`.
/// Returns `Some(name)` if the entry is a `system/local_command` rename message.
fn try_extract_rename(entry: &SessionMetadataEntry) -> Option<String> {
    if entry.message_type != "system" {
        return None;
    }
    if entry.subtype.as_deref() != Some("local_command") {
        return None;
    }
    entry.content.as_ref().and_then(extract_rename_from_content)
}

/// Try to extract a rename name from a top-level `custom-title` event, emitted by the
/// `/branch` command (a newer alternative to `/rename` for naming a session).
fn try_extract_custom_title(message_type: &str, custom_title: Option<&str>) -> Option<String> {
    if message_type != "custom-title" {
        return None;
    }
    let name = custom_title?.trim();
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

/// Derive a project display name from a real working-directory path (its leaf).
fn project_display_name_from_path(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    Path::new(trimmed)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

/// Fast classification of a line without full parsing
/// Returns true if the line should be counted as a valid message
#[inline]
#[allow(dead_code)] // Keep for fallback and tests
fn classify_line(line: &str, exclude_sidechain: bool) -> bool {
    if line.trim().is_empty() {
        return false;
    }

    // Fast path: try to extract just the type field
    if let Ok(classifier) = serde_json::from_str::<LineClassifier>(line) {
        // Exclude summary messages
        if classifier.message_type == "summary" {
            return false;
        }
        // Exclude system message types (progress, queue-operation, file-history-snapshot, system)
        if is_system_message_type(&classifier.message_type) {
            return false;
        }
        // Exclude meta messages (internal/command-related messages)
        if classifier.is_meta.unwrap_or(false) {
            return false;
        }
        if exclude_sidechain && classifier.is_sidechain.unwrap_or(false) {
            return false;
        }
        return true;
    }
    false
}

// Helper to check if text is a genuine user message (not system-generated)
fn is_genuine_user_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Skip XML/HTML-like tags (system messages)
    if trimmed.starts_with('<') {
        return false;
    }
    // Skip known system messages
    const SYSTEM_PHRASES: [&str; 4] = [
        "Session Cleared",
        "session cleared",
        "Caveat:",
        "Tool execution",
    ];
    for phrase in &SYSTEM_PHRASES {
        if trimmed.starts_with(phrase) {
            return false;
        }
    }
    true
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars).collect();
        format!("{truncated}...")
    } else {
        text.to_string()
    }
}

fn parse_timestamp_sort_key(timestamp: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|dt| dt.timestamp())
}

fn session_sort_key(session: &ClaudeSession) -> i64 {
    parse_timestamp_sort_key(&session.last_message_time)
        .or_else(|| parse_timestamp_sort_key(&session.last_modified))
        .unwrap_or(0)
}

fn metadata_sort_snapshot(path: &Path) -> (Option<u64>, u64, i64) {
    let Ok(metadata) = path.metadata() else {
        return (None, 0, 0);
    };

    let modified_time = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let sort_key = modified_time
        .and_then(|t| i64::try_from(t).ok())
        .unwrap_or(0);

    (modified_time, metadata.len(), sort_key)
}

fn session_with_sidechain_filter(
    mut session: ClaudeSession,
    sidechain_count: usize,
    exclude: bool,
) -> Option<ClaudeSession> {
    if exclude {
        session.message_count = session.message_count.saturating_sub(sidechain_count);
        if session.message_count == 0 {
            return None;
        }
    }
    Some(session)
}

fn propagate_session_summaries(sessions: &mut [ClaudeSession]) {
    let mut summary_map: HashMap<String, String> = HashMap::new();

    for session in sessions.iter() {
        if let Some(ref summary) = session.summary {
            if !summary.is_empty() {
                summary_map.insert(session.actual_session_id.clone(), summary.clone());
            }
        }
    }

    for session in sessions.iter_mut() {
        if session.summary.is_none()
            || session
                .summary
                .as_ref()
                .is_some_and(std::string::String::is_empty)
        {
            if let Some(summary) = summary_map.get(&session.actual_session_id) {
                session.summary = Some(summary.clone());
            }
        }
    }
}

// Extract text from message content, filtering out system messages
// Falls back to extracting command name + args for command messages
fn extract_user_text(content: &serde_json::Value) -> Option<String> {
    match content {
        serde_json::Value::String(text) => {
            if is_genuine_user_text(text) {
                Some(truncate_text(text, 100))
            } else {
                // Fallback: extract command display (e.g., "/clear", "/research args")
                extract_command_display(text)
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if item_type == "text" {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            if is_genuine_user_text(text) {
                                return Some(truncate_text(text, 100));
                            }
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract command name + args from command message XML tags
/// e.g., "<command-name>/research</command-name><command-args>query</command-args>"
///   → "/research query"
fn extract_command_name(text: &str) -> Option<&str> {
    let start = text.find("<command-name>")?;
    let after = &text[start + 14..];
    let end = after.find("</command-name>")?;
    let command = after[..end].trim();
    (!command.is_empty()).then_some(command)
}

fn extract_command_display(text: &str) -> Option<String> {
    let mut parts = Vec::new();

    // Extract command name
    if let Some(command) = extract_command_name(text) {
        parts.push(command.to_string());
    }

    // Extract command args
    if let Some(start) = text.find("<command-args>") {
        let after = &text[start + 14..];
        if let Some(end) = after.find("</command-args>") {
            let args = after[..end].trim();
            if !args.is_empty() {
                parts.push(args.to_string());
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(truncate_text(&parts.join(" "), 100))
    }
}

/// Extract text from assistant message content for summary fallback
fn extract_assistant_text(content: &serde_json::Value) -> Option<String> {
    match content {
        serde_json::Value::String(text) => {
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed.len() > 10 {
                Some(truncate_text(trimmed, 100))
            } else {
                None
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if item_type == "text" {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() && trimmed.len() > 10 {
                                return Some(truncate_text(trimmed, 100));
                            }
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Categorization of how to handle a file
enum FileParseStrategy {
    /// Use cached data as-is (file unchanged)
    UseCached(ClaudeSession, usize), // (session, sidechain_count)
    /// File grew - use incremental parsing from offset
    Incremental(PathBuf, IncrementalParseState),
    /// Full reparse needed (new file or file shrunk/modified in place)
    FullParse(PathBuf),
}

#[derive(Clone)]
enum SessionPageCandidateSource {
    Cached(Box<ClaudeSession>, usize),
    Parse(PathBuf),
    KnownDropped,
}

#[derive(Clone)]
struct SessionPageCandidate {
    sort_key: i64,
    source: SessionPageCandidateSource,
}

/// In-memory record of candidate files that produced no valid session at all,
/// keyed by project path, then file path -> (mtime, size) captured when the
/// parse failed. `save_cache` is best-effort (a full disk or read-only project
/// dir silently drops it), so later pages cannot rely on the on-disk cache
/// alone to know how many candidates were already dropped by earlier pages.
/// This registry keeps `SessionPage::total` cumulative across page calls
/// within one app run; entries are ignored (and replaced) once the file's
/// mtime or size changes.
#[allow(clippy::type_complexity)]
static PAGE_DROPPED_CANDIDATES: once_cell::sync::Lazy<
    std::sync::Mutex<HashMap<String, HashMap<String, (u64, u64)>>>,
> = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

fn is_recorded_dropped_candidate(
    project_path: &str,
    path_str: &str,
    mtime: Option<u64>,
    size: u64,
) -> bool {
    let registry = PAGE_DROPPED_CANDIDATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry
        .get(project_path)
        .and_then(|files| files.get(path_str))
        .is_some_and(|&(recorded_mtime, recorded_size)| {
            Some(recorded_mtime) == mtime && recorded_size == size
        })
}

fn record_dropped_candidate(project_path: &str, path_str: &str, mtime: Option<u64>, size: u64) {
    let Some(mtime) = mtime else {
        // Without a stable mtime the record could never be validated later.
        return;
    };
    let mut registry = PAGE_DROPPED_CANDIDATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry
        .entry(project_path.to_string())
        .or_default()
        .insert(path_str.to_string(), (mtime, size));
}

fn clear_dropped_candidate(project_path: &str, path_str: &str) {
    let mut registry = PAGE_DROPPED_CANDIDATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(files) = registry.get_mut(project_path) {
        files.remove(path_str);
        if files.is_empty() {
            registry.remove(project_path);
        }
    }
}

#[tauri::command]
pub async fn load_project_sessions_page(
    project_path: String,
    exclude_sidechain: Option<bool>,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<SessionPage, String> {
    let project_path = project_path.trim().to_string();
    if project_path.is_empty() {
        return Err("project_path is required".to_string());
    }

    let project_root = Path::new(&project_path);
    if !project_root.is_absolute() {
        return Err("project_path must be an absolute path".to_string());
    }

    // The canonical root is only used to reject symlinked files that escape
    // the project directory. Cache keys and the cache file location keep the
    // caller-provided path form so they match the entries written by
    // `load_project_sessions` (canonicalizing here would orphan that cache,
    // e.g. `/var` vs `/private/var` on macOS).
    let canonical_project_root = project_root
        .canonicalize()
        .map_err(|error| format!("Failed to resolve project_path: {error}"))?;

    let exclude = exclude_sidechain.unwrap_or(false);
    let offset = offset.unwrap_or(0);
    let limit = limit
        .unwrap_or(DEFAULT_SESSION_PAGE_LIMIT)
        .clamp(1, MAX_SESSION_PAGE_LIMIT);

    let mut cache = load_cache(&project_path);
    let mut cache_updated = false;

    let mut candidates: Vec<SessionPageCandidate> = Vec::new();
    let mut known_dropped_candidates = 0usize;

    for entry in WalkDir::new(project_root)
        .follow_links(false)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
    {
        if entry.file_type().is_symlink() {
            continue;
        }

        let Ok(canonical_path) = entry.path().canonicalize() else {
            continue;
        };
        if !canonical_path.starts_with(&canonical_project_root) {
            continue;
        }

        let path = entry.path().to_path_buf();
        let path_str = path.to_string_lossy().to_string();
        let (current_mtime, current_size, modified_sort_key) = metadata_sort_snapshot(&path);

        if let Some(cached) = cache.entries.get(&path_str) {
            if Some(cached.modified_time) == current_mtime && cached.file_size == current_size {
                let (sort_key, source) = if let Some(session) = cached.session.clone() {
                    let sort_key = session_sort_key(&session);
                    if session_with_sidechain_filter(
                        session.clone(),
                        cached.sidechain_count,
                        exclude,
                    )
                    .is_some()
                    {
                        (
                            sort_key,
                            SessionPageCandidateSource::Cached(
                                Box::new(session),
                                cached.sidechain_count,
                            ),
                        )
                    } else {
                        known_dropped_candidates = known_dropped_candidates.saturating_add(1);
                        (sort_key, SessionPageCandidateSource::KnownDropped)
                    }
                } else {
                    known_dropped_candidates = known_dropped_candidates.saturating_add(1);
                    (modified_sort_key, SessionPageCandidateSource::KnownDropped)
                };

                candidates.push(SessionPageCandidate { sort_key, source });
                continue;
            }

            if current_size > cached.file_size {
                if let Some(session) = cached.session.as_ref() {
                    candidates.push(SessionPageCandidate {
                        sort_key: modified_sort_key.max(session_sort_key(session)),
                        source: SessionPageCandidateSource::Parse(path),
                    });
                    continue;
                }
            }
        }

        // No usable on-disk cache entry: fall back to the in-memory record of
        // candidates an earlier page already parsed and dropped, so `total`
        // stays cumulative even when the cache file could not be written.
        if is_recorded_dropped_candidate(&project_path, &path_str, current_mtime, current_size) {
            known_dropped_candidates = known_dropped_candidates.saturating_add(1);
            candidates.push(SessionPageCandidate {
                sort_key: modified_sort_key,
                source: SessionPageCandidateSource::KnownDropped,
            });
            continue;
        }

        candidates.push(SessionPageCandidate {
            sort_key: modified_sort_key,
            source: SessionPageCandidateSource::Parse(path),
        });
    }

    candidates.sort_by_key(|candidate| Reverse(candidate.sort_key));

    let candidate_total = candidates.len();
    let page_start_offset = offset.min(candidate_total);
    let mut next_offset = page_start_offset;
    let mut newly_dropped_candidates = 0usize;
    let mut sessions: Vec<ClaudeSession> = Vec::with_capacity(limit);

    while sessions.len() < limit && next_offset < candidate_total {
        let remaining_slots = limit - sessions.len();
        let batch_end = next_offset
            .saturating_add(remaining_slots)
            .min(candidate_total);
        let page_candidates: Vec<SessionPageCandidate> =
            candidates[next_offset..batch_end].to_vec();
        next_offset = batch_end;

        let results: Vec<(SessionPageCandidateSource, Option<SessionExtractionResult>)> =
            page_candidates
                .into_par_iter()
                .map(|candidate| match candidate.source {
                    SessionPageCandidateSource::Cached(session, sidechain_count) => (
                        SessionPageCandidateSource::Cached(session, sidechain_count),
                        None,
                    ),
                    SessionPageCandidateSource::Parse(path) => {
                        let result = extract_session_metadata_from_file(&path);
                        (SessionPageCandidateSource::Parse(path), result)
                    }
                    SessionPageCandidateSource::KnownDropped => {
                        (SessionPageCandidateSource::KnownDropped, None)
                    }
                })
                .collect();

        for (source, result_opt) in results {
            match source {
                SessionPageCandidateSource::Cached(session, sidechain_count) => {
                    if let Some(session) =
                        session_with_sidechain_filter(*session, sidechain_count, exclude)
                    {
                        sessions.push(session);
                    } else {
                        newly_dropped_candidates = newly_dropped_candidates.saturating_add(1);
                    }
                }
                SessionPageCandidateSource::KnownDropped => {}
                SessionPageCandidateSource::Parse(path) => {
                    let path_str = path.to_string_lossy().to_string();
                    let (modified_time, file_size, _) = metadata_sort_snapshot(&path);

                    let (
                        session_for_cache,
                        sidechain_count,
                        byte_offset,
                        has_tool_use,
                        has_errors,
                        first_user_content,
                        last_user_content,
                        first_assistant_text,
                        cached_rename_name,
                    ) = match &result_opt {
                        Some(result) => (
                            Some(result.session.clone()),
                            result.sidechain_count,
                            result.final_byte_offset,
                            result.has_tool_use,
                            result.has_errors,
                            result.first_user_content.clone(),
                            result.last_user_content.clone(),
                            result.first_assistant_text.clone(),
                            result.rename_name.clone(),
                        ),
                        None => (None, 0, 0, false, false, None, None, None, None),
                    };

                    cache.entries.insert(
                        path_str.clone(),
                        CachedSessionMetadata {
                            modified_time: modified_time.unwrap_or(0),
                            file_size,
                            last_byte_offset: byte_offset,
                            session: session_for_cache,
                            sidechain_count,
                            has_tool_use,
                            has_errors,
                            first_user_content,
                            last_user_content,
                            first_assistant_text,
                            rename_name: cached_rename_name,
                        },
                    );
                    cache_updated = true;

                    if let Some(result) = result_opt {
                        clear_dropped_candidate(&project_path, &path_str);
                        if let Some(session) = session_with_sidechain_filter(
                            result.session,
                            result.sidechain_count,
                            exclude,
                        ) {
                            sessions.push(session);
                        } else {
                            newly_dropped_candidates = newly_dropped_candidates.saturating_add(1);
                        }
                    } else {
                        record_dropped_candidate(
                            &project_path,
                            &path_str,
                            modified_time,
                            file_size,
                        );
                        newly_dropped_candidates = newly_dropped_candidates.saturating_add(1);
                    }
                }
            }
        }
    }

    let total = candidate_total
        .saturating_sub(known_dropped_candidates.saturating_add(newly_dropped_candidates));
    let has_more = candidates
        .iter()
        .skip(next_offset)
        .any(|candidate| !matches!(candidate.source, SessionPageCandidateSource::KnownDropped));

    sessions.sort_by_key(|session| Reverse(session_sort_key(session)));
    propagate_session_summaries(&mut sessions);

    if cache_updated {
        cache.version = CACHE_VERSION;
        save_cache(&project_path, &cache);
    }

    Ok(SessionPage {
        sessions,
        total,
        offset,
        limit,
        next_offset,
        has_more,
    })
}

#[tauri::command]
pub async fn load_project_sessions(
    project_path: String,
    exclude_sidechain: Option<bool>,
) -> Result<Vec<ClaudeSession>, String> {
    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();

    let exclude = exclude_sidechain.unwrap_or(false);

    // 1. Load existing cache
    let mut cache = load_cache(&project_path);
    let mut cache_updated = false;

    // 2. Collect all JSONL file paths
    let file_paths: Vec<PathBuf> = WalkDir::new(&project_path)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .map(|e| e.path().to_path_buf())
        .collect();

    #[cfg(debug_assertions)]
    eprintln!(
        "🔍 load_project_sessions: processing {} files",
        file_paths.len()
    );

    // 3. Categorize files into: cached, incremental, full parse
    let mut strategies: Vec<FileParseStrategy> = Vec::with_capacity(file_paths.len());
    #[cfg(debug_assertions)]
    let mut cache_hit_count = 0usize;
    #[cfg(debug_assertions)]
    let mut incremental_count = 0usize;
    #[cfg(debug_assertions)]
    let mut full_parse_count = 0usize;

    for path in &file_paths {
        let path_str = path.to_string_lossy().to_string();
        let current_size = get_file_size(path).unwrap_or(0);
        let current_mtime = get_modified_time(path);

        if let Some(cached) = cache.entries.get(&path_str) {
            // Check if file hasn't changed at all
            if Some(cached.modified_time) == current_mtime && cached.file_size == current_size {
                if let Some(ref session) = cached.session {
                    #[cfg(debug_assertions)]
                    {
                        cache_hit_count += 1;
                    }
                    strategies.push(FileParseStrategy::UseCached(
                        session.clone(),
                        cached.sidechain_count,
                    ));
                    continue;
                }
            }

            // Check if file grew (append-only) - use incremental parsing
            if current_size > cached.file_size {
                if let Some(session) = cached.session.as_ref() {
                    #[cfg(debug_assertions)]
                    {
                        incremental_count += 1;
                    }
                    strategies.push(FileParseStrategy::Incremental(
                        path.clone(),
                        IncrementalParseState {
                            start_offset: cached.last_byte_offset,
                            message_count: session.message_count,
                            sidechain_count: cached.sidechain_count,
                            last_timestamp: Some(session.last_message_time.clone()),
                            has_tool_use: cached.has_tool_use,
                            has_errors: cached.has_errors,
                            session_id: Some(session.actual_session_id.clone()),
                            first_timestamp: Some(session.first_message_time.clone()),
                            summary: session.summary.clone(),
                            first_user_content: cached.first_user_content.clone(),
                            last_user_content: cached.last_user_content.clone(),
                            first_assistant_text: cached.first_assistant_text.clone(),
                            rename_name: cached.rename_name.clone(),
                            entrypoint: session.entrypoint.clone(),
                            project_name: Some(session.project_name.clone()),
                        },
                    ));
                    continue;
                }
            }
        }

        // New file or file was modified (not just appended) - full parse
        #[cfg(debug_assertions)]
        {
            full_parse_count += 1;
        }
        strategies.push(FileParseStrategy::FullParse(path.clone()));
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "📦 Cache hits: {cache_hit_count}, incremental parsing: {incremental_count}, full parsing: {full_parse_count}"
    );

    // 4. Process strategies in parallel
    let results: Vec<(FileParseStrategy, Option<SessionExtractionResult>)> = strategies
        .into_par_iter()
        .map(|strategy| match &strategy {
            FileParseStrategy::UseCached(_, _) => (strategy, None),
            FileParseStrategy::Incremental(path, state) => {
                let result = extract_session_metadata_incremental(path, state.clone());
                (strategy, result)
            }
            FileParseStrategy::FullParse(path) => {
                let result = extract_session_metadata_from_file(path);
                (strategy, result)
            }
        })
        .collect();

    // 5. Process results and update cache
    let mut sessions: Vec<ClaudeSession> = Vec::with_capacity(results.len());

    for (strategy, result_opt) in results {
        match strategy {
            FileParseStrategy::UseCached(session, sidechain_count) => {
                let mut session_clone = session;
                if exclude {
                    session_clone.message_count =
                        session_clone.message_count.saturating_sub(sidechain_count);
                    if session_clone.message_count == 0 {
                        continue;
                    }
                }
                sessions.push(session_clone);
            }
            FileParseStrategy::Incremental(path, _) | FileParseStrategy::FullParse(path) => {
                let path_str = path.to_string_lossy().to_string();
                let mtime = get_modified_time(&path).unwrap_or(0);
                let file_size = get_file_size(&path).unwrap_or(0);

                let (
                    session_for_cache,
                    sidechain_count,
                    byte_offset,
                    has_tool_use,
                    has_errors,
                    first_user_content,
                    last_user_content,
                    first_assistant_text,
                    cached_rename_name,
                ) = match &result_opt {
                    Some(result) => (
                        Some(result.session.clone()),
                        result.sidechain_count,
                        result.final_byte_offset,
                        result.has_tool_use,
                        result.has_errors,
                        result.first_user_content.clone(),
                        result.last_user_content.clone(),
                        result.first_assistant_text.clone(),
                        result.rename_name.clone(),
                    ),
                    None => (None, 0, 0, false, false, None, None, None, None),
                };

                cache.entries.insert(
                    path_str,
                    CachedSessionMetadata {
                        modified_time: mtime,
                        file_size,
                        last_byte_offset: byte_offset,
                        session: session_for_cache,
                        sidechain_count,
                        has_tool_use,
                        has_errors,
                        first_user_content,
                        last_user_content,
                        first_assistant_text,
                        rename_name: cached_rename_name,
                    },
                );
                cache_updated = true;

                if let Some(result) = result_opt {
                    let mut session = result.session;
                    if exclude {
                        session.message_count =
                            session.message_count.saturating_sub(result.sidechain_count);
                        if session.message_count == 0 {
                            continue;
                        }
                    }
                    sessions.push(session);
                }
            }
        }
    }

    // 6. Sort by last message time (conversation time) instead of filesystem modification time
    sessions.sort_by(|a, b| b.last_message_time.cmp(&a.last_message_time));

    // 8. Summary propagation
    propagate_session_summaries(&mut sessions);

    // 9. Save updated cache
    if cache_updated {
        cache.version = CACHE_VERSION;
        save_cache(&project_path, &cache);
    }

    #[cfg(debug_assertions)]
    {
        let elapsed = start_time.elapsed();
        println!(
            "📊 load_project_sessions performance: {} sessions, {}ms elapsed",
            sessions.len(),
            elapsed.as_millis()
        );
    }

    Ok(sessions)
}

/// Parse a single line into `ClaudeMessage` (with line number)
#[allow(dead_code)] // Keep for fallback and tests
fn parse_line_to_message(
    line_num: usize,
    line: &str,
    include_summary: bool,
) -> Option<ClaudeMessage> {
    if line.trim().is_empty() {
        return None;
    }

    let log_entry: RawLogEntry = serde_json::from_str(line).ok()?;

    // Skip meta messages (internal/command-related messages)
    if log_entry.is_meta.unwrap_or(false) {
        return None;
    }

    if is_system_message_type(&log_entry.message_type) {
        return None;
    }

    if log_entry.message_type == "summary" {
        if !include_summary {
            return None;
        }
        let summary_text = log_entry.summary?;
        let uuid = log_entry.uuid.unwrap_or_else(|| Uuid::new_v4().to_string());

        return Some(ClaudeMessage {
            uuid,
            parent_uuid: log_entry.leaf_uuid,
            session_id: log_entry
                .session_id
                .unwrap_or_else(|| "unknown-session".to_string()),
            timestamp: log_entry
                .timestamp
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            message_type: "summary".to_string(),
            content: Some(serde_json::Value::String(summary_text)),
            project_name: None,
            tool_use: None,
            tool_use_result: None,
            is_sidechain: None,
            usage: None,
            role: None,
            model: None,
            inference: None,
            stop_reason: None,
            cost_usd: None,
            duration_ms: None,
            message_id: None,
            snapshot: None,
            is_snapshot_update: None,
            data: None,
            tool_use_id: None,
            parent_tool_use_id: None,
            operation: None,
            subtype: None,
            level: None,
            hook_count: None,
            hook_infos: None,
            stop_reason_system: None,
            prevented_continuation: None,
            compact_metadata: None,
            microcompact_metadata: None,
            provider: None,
        });
    }

    // Skip entries without session_id and timestamp
    if log_entry.session_id.is_none() && log_entry.timestamp.is_none() {
        return None;
    }

    let uuid = log_entry
        .uuid
        .unwrap_or_else(|| format!("{}-line-{}", Uuid::new_v4(), line_num + 1));

    let (role, message_id, model, stop_reason, usage) = if let Some(ref msg) = log_entry.message {
        (
            Some(msg.role.clone()),
            msg.id.clone(),
            msg.model.clone(),
            msg.stop_reason.clone(),
            msg.usage.clone(),
        )
    } else {
        (None, None, None, None, None)
    };

    let is_task_notification = is_task_notification_origin(log_entry.origin.as_ref());

    Some(ClaudeMessage {
        uuid,
        parent_uuid: log_entry.parent_uuid,
        session_id: log_entry
            .session_id
            .unwrap_or_else(|| "unknown-session".to_string()),
        timestamp: log_entry
            .timestamp
            .unwrap_or_else(|| Utc::now().to_rfc3339()),
        message_type: log_entry.message_type,
        content: log_entry.message.map(|m| m.content).or(log_entry.content),
        project_name: None,
        tool_use: log_entry.tool_use,
        tool_use_result: log_entry.tool_use_result,
        is_sidechain: log_entry.is_sidechain,
        usage,
        role,
        model,
        inference: inference_from_effort(log_entry.effort),
        stop_reason,
        cost_usd: log_entry.cost_usd,
        duration_ms: log_entry.duration_ms,
        message_id: message_id.or(log_entry.message_id),
        snapshot: log_entry.snapshot,
        is_snapshot_update: log_entry.is_snapshot_update,
        data: log_entry.data,
        tool_use_id: log_entry.tool_use_id,
        parent_tool_use_id: log_entry.parent_tool_use_id,
        operation: log_entry.operation,
        // A compaction summary is recorded as a `type: "user"` record flagged
        // `isCompactSummary`; stamp a provenance subtype (like `queued_command`)
        // so consumers can tell it apart from an authored user turn. Content and
        // type stay as-is — this only tags "what this record is".
        subtype: if is_task_notification {
            Some("task_notification".to_string())
        } else if log_entry.is_compact_summary.unwrap_or(false) {
            Some("compact_summary".to_string())
        } else {
            log_entry.subtype
        },
        level: log_entry.level,
        hook_count: log_entry.hook_count,
        hook_infos: log_entry.hook_infos,
        stop_reason_system: log_entry.stop_reason_system,
        prevented_continuation: log_entry.prevented_continuation,
        compact_metadata: log_entry.compact_metadata,
        microcompact_metadata: log_entry.microcompact_metadata,
        provider: None,
    })
}

/// Parse a single line using simd-json for faster parsing
/// Returns None if the line is empty or fails to parse
/// If a raw `attachment` record is a `queued_command` — a message the user sent
/// while the agent was still generating (queued, then injected) — return its
/// prompt content so it can be normalized into a user message. The text lives in
/// `attachment.prompt`, already shaped as a content-block array
/// (`[{ "type": "text", "text": … }]`), which is passed through unchanged.
/// Returns `None` for every other attachment subtype (todo reminders, hook
/// context, listing deltas, file refs, … — UI/plumbing, not authored content).
fn queued_command_prompt(attachment: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let att = attachment?;
    if att.get("type").and_then(|v| v.as_str()) != Some("queued_command") {
        return None;
    }
    // Claude also transports background-task completions through a
    // `queued_command` attachment, but marks them structurally. They are SDK
    // plumbing, not text typed by the user, even if a future version stores the
    // prompt in the usual content-block array shape.
    if att.get("commandMode").and_then(|v| v.as_str()) == Some("task-notification") {
        return None;
    }
    att.get("prompt").filter(|p| p.is_array()).cloned()
}

/// Claude Code persists completed background-task notifications as synthetic
/// user records. Trust its provider-owned origin metadata rather than matching
/// the XML-shaped message text, which a user could legitimately quote.
fn is_task_notification_origin(origin: Option<&serde_json::Value>) -> bool {
    origin
        .and_then(|value| value.get("kind"))
        .and_then(serde_json::Value::as_str)
        == Some("task-notification")
}

/// Preserve the display name of a file the user explicitly attached to a
/// prompt, without exposing its absolute path or embedded file contents. Claude
/// records IDE-selected context separately as `edited_text_file`; accepting only
/// the exact `file` subtype keeps those two concepts distinct.
fn prompt_file_attachment_data(
    attachment: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let attachment = attachment?;
    if attachment.get("type").and_then(serde_json::Value::as_str) != Some("file") {
        return None;
    }

    let display_name = ["displayPath", "filename"]
        .into_iter()
        .filter_map(|key| attachment.get(key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .find(|value| !value.is_empty())?;
    prompt_attachments_data([prompt_attachment_name(display_name)?])
}

/// Claude Code persists local slash-command UI traffic as ordinary user
/// messages. Recognize only a complete reserved envelope: quoted tags or a
/// wrapper followed by authored prose must remain ordinary user content.
fn is_local_command_plumbing_text(text: &str) -> bool {
    let text = text.trim();
    const STDOUT_OPEN: &str = "<local-command-stdout>";
    const STDOUT_CLOSE: &str = "</local-command-stdout>";
    if text.starts_with(STDOUT_OPEN) && text.ends_with(STDOUT_CLOSE) {
        return text[STDOUT_OPEN.len()..]
            .find(STDOUT_CLOSE)
            .is_some_and(|at| STDOUT_OPEN.len() + at == text.len() - STDOUT_CLOSE.len());
    }

    let mut rest = text;
    let mut seen = 0_u8;
    while !rest.is_empty() {
        let Some((close, bit, after_open)) = [
            ("<command-name>", "</command-name>", 1_u8),
            ("<command-message>", "</command-message>", 2_u8),
            ("<command-args>", "</command-args>", 4_u8),
        ]
        .into_iter()
        .find_map(|(open, close, bit)| rest.strip_prefix(open).map(|after| (close, bit, after))) else {
            return false;
        };
        if seen & bit != 0 {
            return false;
        }
        seen |= bit;
        let Some(close_at) = after_open.find(close) else {
            return false;
        };
        rest = after_open[close_at + close.len()..].trim_start();
    }
    seen & 1 != 0
}

fn is_local_command_plumbing(
    message_type: &str,
    role: Option<&str>,
    content: Option<&serde_json::Value>,
) -> bool {
    message_type == "user"
        && role == Some("user")
        && content
            .and_then(serde_json::Value::as_str)
            .is_some_and(is_local_command_plumbing_text)
}

/// Local command envelopes normally describe UI-only plumbing (`/model`,
/// `/rename`, and their stdout echoes). `/init` is different: Claude expands it
/// into an `isMeta` prompt and then produces a normal assistant response. The
/// expanded prompt is intentionally excluded, so preserve the authored command
/// envelope as the conversational boundary that owns that response.
fn local_command_subtype(
    message_type: &str,
    role: Option<&str>,
    content: Option<&serde_json::Value>,
) -> Option<&'static str> {
    if !is_local_command_plumbing(message_type, role, content) {
        return None;
    }
    let command_name = content
        .and_then(serde_json::Value::as_str)
        .and_then(extract_command_name);
    if command_name == Some("/init") {
        Some("prompt_command")
    } else {
        Some("local_command")
    }
}

/// Seed the normalized inference object from Claude Code's top-level `effort`
/// field (the reasoning-effort level stamped on assistant records since
/// ~v2.1.214). The provider-loading boundary's `hydrate_inference` later folds
/// the legacy top-level model/stop/usage fields into the same object while
/// preserving this provider-set value; records without `effort` stay `None`.
fn inference_from_effort(effort: Option<String>) -> Option<crate::models::InferenceMetadata> {
    effort.map(|effort| crate::models::InferenceMetadata {
        reasoning_effort: Some(effort),
        ..Default::default()
    })
}

fn parse_line_simd(
    line_num: usize,
    line: &mut [u8],
    include_summary: bool,
) -> Option<ClaudeMessage> {
    if line
        .iter()
        .all(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
    {
        return None;
    }

    // Use simd_json for faster parsing
    let log_entry: RawLogEntry = simd_json::serde::from_slice(line).ok()?;

    // Skip meta messages
    if log_entry.is_meta.unwrap_or(false) {
        return None;
    }

    if is_system_message_type(&log_entry.message_type) {
        return None;
    }

    if log_entry.message_type == "summary" {
        if !include_summary {
            return None;
        }
        let summary_text = log_entry.summary?;
        let uuid = log_entry.uuid.unwrap_or_else(|| Uuid::new_v4().to_string());

        return Some(ClaudeMessage {
            uuid,
            parent_uuid: log_entry.leaf_uuid,
            session_id: log_entry
                .session_id
                .unwrap_or_else(|| "unknown-session".to_string()),
            timestamp: log_entry
                .timestamp
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            message_type: "summary".to_string(),
            content: Some(serde_json::Value::String(summary_text)),
            project_name: None,
            tool_use: None,
            tool_use_result: None,
            is_sidechain: None,
            usage: None,
            role: None,
            model: None,
            inference: None,
            stop_reason: None,
            cost_usd: None,
            duration_ms: None,
            message_id: None,
            snapshot: None,
            is_snapshot_update: None,
            data: None,
            tool_use_id: None,
            parent_tool_use_id: None,
            operation: None,
            subtype: None,
            level: None,
            hook_count: None,
            hook_infos: None,
            stop_reason_system: None,
            prevented_continuation: None,
            compact_metadata: None,
            microcompact_metadata: None,
            provider: None,
        });
    }

    // Skip entries without session_id and timestamp
    if log_entry.session_id.is_none() && log_entry.timestamp.is_none() {
        return None;
    }

    let uuid = log_entry
        .uuid
        .unwrap_or_else(|| format!("{}-line-{}", Uuid::new_v4(), line_num + 1));

    // A message sent while the agent was still generating is persisted as a
    // `queued_command` attachment, not a `user` record — but it is authored user
    // text. Normalize it into a user message (keeping its uuid/parentUuid, so the
    // causal chain and any reply still link to it); otherwise it would surface
    // with null content and be dropped downstream.
    if log_entry.message_type == "attachment" {
        if let Some(content) = queued_command_prompt(log_entry.attachment.as_ref()) {
            return Some(ClaudeMessage {
                uuid,
                parent_uuid: log_entry.parent_uuid,
                session_id: log_entry
                    .session_id
                    .unwrap_or_else(|| "unknown-session".to_string()),
                timestamp: log_entry
                    .timestamp
                    .unwrap_or_else(|| Utc::now().to_rfc3339()),
                message_type: "user".to_string(),
                content: Some(content),
                project_name: None,
                tool_use: None,
                tool_use_result: None,
                is_sidechain: log_entry.is_sidechain,
                usage: None,
                role: Some("user".to_string()),
                model: None,
                inference: None,
                stop_reason: None,
                cost_usd: None,
                duration_ms: None,
                message_id: None,
                snapshot: None,
                is_snapshot_update: None,
                data: None,
                tool_use_id: None,
                parent_tool_use_id: None,
                operation: None,
                // Mark provenance so consumers can present a queued/mid-response
                // message distinctly from a normal user turn. The content is a
                // normal user message; this is only a "how it was delivered" tag.
                subtype: Some("queued_command".to_string()),
                level: None,
                hook_count: None,
                hook_infos: None,
                stop_reason_system: None,
                prevented_continuation: None,
                compact_metadata: None,
                microcompact_metadata: None,
                provider: None,
            });
        }
    }

    let is_task_notification = is_task_notification_origin(log_entry.origin.as_ref());
    let command_subtype = log_entry.message.as_ref().and_then(|message| {
        local_command_subtype(
            &log_entry.message_type,
            Some(message.role.as_str()),
            Some(&message.content),
        )
    });

    let (role, message_id, model, stop_reason, usage, extracted_tool_use) =
        if let Some(ref msg) = log_entry.message {
            // Try to extract tool_use from content array if not present at top level
            let extracted = if log_entry.tool_use.is_none() {
                msg.content.as_array().and_then(|arr| {
                    arr.iter()
                        .find(|item| item.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
                        .cloned()
                })
            } else {
                None
            };

            (
                Some(msg.role.clone()),
                msg.id.clone(),
                msg.model.clone(),
                msg.stop_reason.clone(),
                msg.usage.clone(),
                extracted,
            )
        } else {
            (None, None, None, None, None, None)
        };

    let prompt_attachment_data = prompt_file_attachment_data(log_entry.attachment.as_ref());

    Some(ClaudeMessage {
        uuid,
        parent_uuid: log_entry.parent_uuid,
        session_id: log_entry
            .session_id
            .unwrap_or_else(|| "unknown-session".to_string()),
        timestamp: log_entry
            .timestamp
            .unwrap_or_else(|| Utc::now().to_rfc3339()),
        message_type: log_entry.message_type,
        content: log_entry.message.map(|m| m.content).or(log_entry.content),
        project_name: None,
        tool_use: log_entry.tool_use.or(extracted_tool_use),
        tool_use_result: log_entry.tool_use_result,
        is_sidechain: log_entry.is_sidechain,
        usage,
        role,
        model,
        inference: inference_from_effort(log_entry.effort),
        stop_reason,
        cost_usd: log_entry.cost_usd,
        duration_ms: log_entry.duration_ms,
        message_id: message_id.or(log_entry.message_id),
        snapshot: log_entry.snapshot,
        is_snapshot_update: log_entry.is_snapshot_update,
        data: prompt_attachment_data.or(log_entry.data),
        tool_use_id: log_entry.tool_use_id,
        parent_tool_use_id: log_entry.parent_tool_use_id,
        operation: log_entry.operation,
        // A compaction summary is recorded as a `type: "user"` record flagged
        // `isCompactSummary`; stamp a provenance subtype (like `queued_command`)
        // so consumers can tell it apart from an authored user turn. Content and
        // type stay as-is — this only tags "what this record is".
        subtype: if is_task_notification {
            Some("task_notification".to_string())
        } else if log_entry.is_compact_summary.unwrap_or(false) {
            Some("compact_summary".to_string())
        } else if let Some(subtype) = command_subtype {
            Some(subtype.to_string())
        } else {
            log_entry.subtype
        },
        level: log_entry.level,
        hook_count: log_entry.hook_count,
        hook_infos: log_entry.hook_infos,
        stop_reason_system: log_entry.stop_reason_system,
        prevented_continuation: log_entry.prevented_continuation,
        compact_metadata: log_entry.compact_metadata,
        microcompact_metadata: log_entry.microcompact_metadata,
        provider: None,
    })
}

/// Parse one Claude JSONL record through the same visibility rules used by the
/// complete mmap loader.
pub(crate) fn parse_visible_message_line(
    line_num: usize,
    line: &mut [u8],
) -> Option<ClaudeMessage> {
    parse_line_simd(line_num, line, false).filter(|message| {
        if is_system_message_type(&message.message_type) {
            return false;
        }
        if message.message_type == "system" {
            return !is_hidden_system_subtype(message.subtype.as_deref());
        }
        true
    })
}

#[allow(unsafe_code)] // Required for mmap performance optimization
pub(crate) fn load_session_messages_sync(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();

    // Use memory-mapped file for faster I/O
    let file =
        fs::File::open(session_path).map_err(|e| format!("Failed to open session file: {e}"))?;

    // SAFETY: We're only reading the file, and the file handle is kept open
    // for the duration of the mmap's lifetime. No concurrent modifications expected
    // as session files are append-only by Claude.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("Failed to memory-map session file: {e}"))?;

    // Find line boundaries efficiently using SIMD-accelerated memchr
    let line_starts = find_line_starts(&mmap);

    // Parse lines in parallel using simd-json
    let mut messages: Vec<(usize, ClaudeMessage)> = line_starts
        .par_iter()
        .enumerate()
        .filter_map(|(line_num, &start)| {
            let end = line_starts.get(line_num + 1).map_or(mmap.len(), |&e| e - 1);
            if start >= end {
                return None;
            }

            // Create a mutable copy for simd-json (it requires mutable slice)
            let mut line_bytes = mmap[start..end].to_vec();

            parse_visible_message_line(line_num, &mut line_bytes).map(|message| (line_num, message))
        })
        .collect();

    // Sort by line number to maintain original order
    messages.sort_by_key(|(line_num, _)| *line_num);
    let messages: Vec<ClaudeMessage> = messages.into_iter().map(|(_, msg)| msg).collect();

    #[cfg(debug_assertions)]
    {
        let elapsed = start_time.elapsed();
        eprintln!(
            "📤 [load_session_messages] {} messages, {}ms elapsed (simd-json + mmap optimized)",
            messages.len(),
            elapsed.as_millis()
        );
    }

    Ok(messages)
}

#[tauri::command]
pub async fn load_session_messages(session_path: String) -> Result<Vec<ClaudeMessage>, String> {
    load_session_messages_sync(&session_path)
}

// ============================================================================
// SubAgent Session Discovery
// ============================================================================

/// Metadata for a single subagent conversation file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubagentSession {
    pub agent_id: String,
    pub file_path: String,
    pub message_count: usize,
    pub file_size: u64,
    pub first_message_time: Option<String>,
    pub last_message_time: Option<String>,
    pub summary: Option<String>,
    /// Task `tool_use` id that spawned this subagent, read from the sibling
    /// `agent-<id>.meta.json` (newer Claude Code format). `None` for older sessions
    /// that have no meta file; the frontend then falls back to progress messages.
    pub tool_use_id: Option<String>,
    /// Workflow run this agent belongs to (`wf_…`, the directory name under
    /// `subagents/workflows/`). `None` for regular flat subagents. Workflow
    /// agents have no `toolUseId` in their meta.json; the frontend anchors
    /// them to the spawning `Workflow` tool call via this run id instead
    /// (the `tool_result` text contains the run's transcript dir) — #449.
    pub workflow_run_id: Option<String>,
}

/// Derive the workflow run id (`wf_…`) for a subagent transcript path.
/// Returns `Some` only for the `…/subagents/workflows/<run>/agent-*.jsonl`
/// layout; flat subagents return `None`.
fn workflow_run_id_for(sa_path: &Path) -> Option<String> {
    let run_dir = sa_path.parent()?;
    let workflows_dir = run_dir.parent()?;
    if workflows_dir.file_name().and_then(|n| n.to_str()) != Some("workflows") {
        return None;
    }
    run_dir.file_name().map(|n| n.to_string_lossy().to_string())
}

/// Returns subagent sessions for a given parent session file.
#[tauri::command]
pub async fn get_session_subagents(session_path: String) -> Result<Vec<SubagentSession>, String> {
    use crate::utils::find_subagent_files;

    let path = PathBuf::from(&session_path);
    if !path.is_absolute() {
        return Err("session_path must be an absolute path".to_string());
    }
    let subagent_files = find_subagent_files(&path);

    if subagent_files.is_empty() {
        return Ok(Vec::new());
    }

    let mut sessions: Vec<SubagentSession> = Vec::new();
    for sa_path in subagent_files {
        let file_name = sa_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        // Extract agent_id: "agent-acompact-35a20cf6" → "acompact-35a20cf6"
        let agent_id = file_name
            .strip_prefix("agent-")
            .unwrap_or(&file_name)
            .to_string();

        let file_size = fs::metadata(&sa_path).map(|m| m.len()).unwrap_or(0);

        // Quick scan: first and last lines + line count
        let (message_count, first_time, last_time, summary) = extract_subagent_metadata(&sa_path);

        // Newer Claude Code persists the spawning Task tool_use id in a sibling
        // `agent-<id>.meta.json`. Read it so multi-subagent sessions can map a click
        // to the right file (#288); older sessions have no meta file -> None.
        let meta_path = sa_path.with_file_name(format!("{file_name}.meta.json"));
        let tool_use_id = read_subagent_tool_use_id(&meta_path);
        let workflow_run_id = workflow_run_id_for(&sa_path);

        sessions.push(SubagentSession {
            agent_id,
            file_path: sa_path.to_string_lossy().to_string(),
            message_count,
            file_size,
            first_message_time: first_time,
            last_message_time: last_time,
            summary,
            tool_use_id,
            workflow_run_id,
        });
    }

    // Sort by first_message_time ascending
    sessions.sort_by(|a, b| a.first_message_time.cmp(&b.first_message_time));

    Ok(sessions)
}

/// Read the `toolUseId` from a subagent's sibling `agent-<id>.meta.json`.
/// Returns `None` if the file is missing, a symlink, unreadable, or has no
/// non-empty `toolUseId` — all non-fatal, so a subagent without a meta file still
/// lists. The symlink guard mirrors the session-path hardening.
fn read_subagent_tool_use_id(meta_path: &std::path::Path) -> Option<String> {
    let meta = std::fs::symlink_metadata(meta_path).ok()?;
    if meta.file_type().is_symlink() {
        return None;
    }
    let content = std::fs::read_to_string(meta_path).ok()?;

    #[derive(serde::Deserialize)]
    struct SubagentMeta {
        #[serde(rename = "toolUseId")]
        tool_use_id: Option<String>,
    }

    let parsed: SubagentMeta = serde_json::from_str(&content).ok()?;
    parsed.tool_use_id.filter(|s| !s.is_empty())
}

/// Metadata extraction from a subagent JSONL file.
///
/// Scans all lines to count messages, extract first/last timestamps, and find
/// the first user message content as a summary.
fn extract_subagent_metadata(
    path: &PathBuf,
) -> (usize, Option<String>, Option<String>, Option<String>) {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return (0, None, None, None),
    };
    let reader = std::io::BufReader::new(file);
    let mut line_count: usize = 0;
    let mut first_time: Option<String> = None;
    let mut last_time: Option<String> = None;
    let mut summary: Option<String> = None;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { continue };
        if line.trim().is_empty() {
            continue;
        }
        line_count += 1;

        // Parse only the fields we need with a minimal struct
        #[derive(serde::Deserialize)]
        struct MinimalEntry {
            timestamp: Option<String>,
            #[serde(rename = "type")]
            msg_type: Option<String>,
            message: Option<MinimalMessage>,
        }
        #[derive(serde::Deserialize)]
        struct MinimalMessage {
            role: Option<String>,
            content: Option<serde_json::Value>,
        }

        if let Ok(entry) = serde_json::from_str::<MinimalEntry>(&line) {
            if let Some(ref ts) = entry.timestamp {
                if first_time.is_none() {
                    first_time = Some(ts.clone());
                }
                last_time = Some(ts.clone());
            }

            // Extract summary from first user message content
            if summary.is_none() && entry.msg_type.as_deref() == Some("user") {
                if let Some(msg) = &entry.message {
                    if msg.role.as_deref() == Some("user") {
                        let text = match &msg.content {
                            Some(serde_json::Value::String(s)) => Some(s.clone()),
                            Some(serde_json::Value::Array(arr)) => arr.iter().find_map(|item| {
                                item.get("text").and_then(|t| t.as_str()).map(String::from)
                            }),
                            _ => None,
                        };
                        if let Some(t) = text {
                            let truncated: String = t.chars().take(100).collect();
                            summary = Some(truncated);
                        }
                    }
                }
            }
        }
    }

    (line_count, first_time, last_time, summary)
}

/// Shared viewer-visibility rules for a classified JSONL line.
/// Must stay in sync with the filtering in `load_session_messages`.
fn is_viewer_visible_line(
    message_type: &str,
    subtype: Option<&str>,
    is_sidechain: Option<bool>,
    is_meta: Option<bool>,
    exclude_sidechain: bool,
) -> bool {
    if message_type == "summary" {
        return false;
    }
    if is_system_message_type(message_type) {
        return false;
    }
    if message_type == "system" && is_hidden_system_subtype(subtype) {
        return false;
    }
    if is_meta.unwrap_or(false) {
        return false;
    }
    if exclude_sidechain && is_sidechain.unwrap_or(false) {
        return false;
    }
    true
}

/// Fast line classifier for simd-json (mutable slice)
fn classify_line_fast(line: &[u8], exclude_sidechain: bool) -> bool {
    if line
        .iter()
        .all(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
    {
        return false;
    }

    // Try fast simd-json parsing with minimal struct
    let mut line_copy = line.to_vec();
    if let Ok(classifier) = simd_json::serde::from_slice::<LineClassifier>(&mut line_copy) {
        return is_viewer_visible_line(
            &classifier.message_type,
            classifier.subtype.as_deref(),
            classifier.is_sidechain,
            classifier.is_meta,
            exclude_sidechain,
        );
    }
    false
}

/// Minimal classifier that also captures the message uuid.
/// Used only by `get_session_message_offset` — keeping it separate from
/// `LineClassifier` avoids a per-line String allocation on the hot
/// classification path of the paginated loader.
#[derive(serde::Deserialize)]
struct LineUuidClassifier {
    #[serde(rename = "type")]
    message_type: String,
    subtype: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    #[serde(rename = "isMeta")]
    is_meta: Option<bool>,
    uuid: Option<String>,
}

/// Find how far from the NEWEST viewer-visible message a uuid sits.
/// Returns `Some(0)` for the newest visible message, `Some(n)` when `n`
/// visible messages are newer than it, and `None` when the uuid is absent.
/// Loading a chat-style window with `offset = 0, limit = n + 1` therefore
/// guarantees the message is inside the window.
///
/// Iterates newest → oldest so deep links to recent messages exit early.
#[allow(unsafe_code)] // Required for mmap performance optimization
pub fn get_session_message_offset(
    session_path: String,
    message_uuid: String,
    exclude_sidechain: Option<bool>,
) -> Result<Option<usize>, String> {
    let file =
        fs::File::open(&session_path).map_err(|e| format!("Failed to open session file: {e}"))?;

    // SAFETY: We're only reading the file, and the file handle is kept open
    // for the duration of the mmap's lifetime. No concurrent modifications expected
    // as session files are append-only by Claude.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("Failed to memory-map session file: {e}"))?;

    let exclude = exclude_sidechain.unwrap_or(false);
    let line_ranges = find_line_ranges(&mmap);

    let mut newer_visible = 0usize;
    for &(start, end) in line_ranges.iter().rev() {
        let line = &mmap[start..end];
        if line
            .iter()
            .all(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
        {
            continue;
        }

        let mut line_copy = line.to_vec();
        let Ok(classifier) = simd_json::serde::from_slice::<LineUuidClassifier>(&mut line_copy)
        else {
            continue;
        };
        if !is_viewer_visible_line(
            &classifier.message_type,
            classifier.subtype.as_deref(),
            classifier.is_sidechain,
            classifier.is_meta,
            exclude,
        ) {
            continue;
        }

        if classifier.uuid.as_deref() == Some(message_uuid.as_str()) {
            return Ok(Some(newer_visible));
        }
        newer_visible += 1;
    }

    Ok(None)
}

#[tauri::command]
#[allow(unsafe_code)] // Required for mmap performance optimization
pub async fn load_session_messages_paginated(
    session_path: String,
    offset: usize,
    limit: usize,
    exclude_sidechain: Option<bool>,
) -> Result<MessagePage, String> {
    #[cfg(debug_assertions)]
    let start_time = std::time::Instant::now();

    // Use memory-mapped file for faster I/O
    let file =
        fs::File::open(&session_path).map_err(|e| format!("Failed to open session file: {e}"))?;

    // SAFETY: We're only reading the file, and the file handle is kept open
    // for the duration of the mmap's lifetime. No concurrent modifications expected
    // as session files are append-only by Claude.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("Failed to memory-map session file: {e}"))?;

    let exclude = exclude_sidechain.unwrap_or(false);

    // Find line boundaries efficiently using SIMD-accelerated memchr
    let line_ranges = find_line_ranges(&mmap);

    // Phase 1: Build valid line indices (fast classification)
    let valid_indices: Vec<usize> = line_ranges
        .iter()
        .enumerate()
        .filter(|(_, &(start, end))| {
            let line = &mmap[start..end];
            classify_line_fast(line, exclude)
        })
        .map(|(idx, _)| idx)
        .collect();

    let total_count = valid_indices.len();

    // Chat-style pagination: offset=0 means newest messages (at the end)
    if total_count == 0 {
        return Ok(MessagePage {
            messages: vec![],
            total_count: 0,
            has_more: false,
            next_offset: 0,
        });
    }

    let already_loaded = offset;
    let remaining_messages = total_count.saturating_sub(already_loaded);
    let messages_to_load = std::cmp::min(limit, remaining_messages);

    let (start_idx, end_idx) = if remaining_messages == 0 {
        (0, 0)
    } else {
        let start = total_count - already_loaded - messages_to_load;
        let end = total_count - already_loaded;
        (start, end)
    };

    // Phase 2: Parse only the target lines (parallel with simd-json)
    let target_indices = &valid_indices[start_idx..end_idx];
    let mut parsed: Vec<(usize, ClaudeMessage)> = target_indices
        .par_iter()
        .filter_map(|&range_idx| {
            let (start, end) = line_ranges[range_idx];
            let mut line_bytes = mmap[start..end].to_vec();
            let msg = parse_line_simd(range_idx, &mut line_bytes, false)?;
            Some((range_idx, msg))
        })
        .collect();

    // Sort by line number to maintain original order
    parsed.sort_by_key(|(line_num, _)| *line_num);
    let messages: Vec<ClaudeMessage> = parsed.into_iter().map(|(_, msg)| msg).collect();

    let has_more = start_idx > 0;
    let next_offset = offset + messages.len();

    #[cfg(debug_assertions)]
    {
        let elapsed = start_time.elapsed();
        eprintln!("📊 load_session_messages_paginated performance: {}/{} messages, {}ms elapsed (simd-json + mmap)",
                 messages.len(), total_count, elapsed.as_millis());
    }

    Ok(MessagePage {
        messages,
        total_count,
        has_more,
        next_offset,
    })
}

#[tauri::command]
#[allow(unsafe_code)] // Required for mmap performance optimization
pub async fn get_session_message_count(
    session_path: String,
    exclude_sidechain: Option<bool>,
) -> Result<usize, String> {
    // Use memory-mapped file for faster I/O
    let file =
        fs::File::open(&session_path).map_err(|e| format!("Failed to open session file: {e}"))?;

    // SAFETY: We're only reading the file, and the file handle is kept open
    // for the duration of the mmap's lifetime. No concurrent modifications expected
    // as session files are append-only by Claude.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("Failed to memory-map session file: {e}"))?;

    let exclude = exclude_sidechain.unwrap_or(false);

    // Find line boundaries and count valid lines using SIMD-accelerated memchr
    let line_ranges = find_line_ranges(&mmap);

    // Parallel counting with fast classification
    let count: usize = line_ranges
        .par_iter()
        .filter(|&&(start, end)| {
            let line = &mmap[start..end];
            classify_line_fast(line, exclude)
        })
        .count();

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_jsonl_file(dir: &TempDir, filename: &str, content: &str) -> PathBuf {
        let file_path = dir.path().join(filename);
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_path
    }

    fn create_sample_user_message(uuid: &str, session_id: &str, content: &str) -> String {
        format!(
            r#"{{"uuid":"{uuid}","sessionId":"{session_id}","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{{"role":"user","content":"{content}"}}}}"#
        )
    }

    fn create_sample_user_message_at(
        uuid: &str,
        session_id: &str,
        timestamp: &str,
        content: &str,
    ) -> String {
        format!(
            r#"{{"uuid":"{uuid}","sessionId":"{session_id}","timestamp":"{timestamp}","type":"user","message":{{"role":"user","content":"{content}"}}}}"#
        )
    }

    fn create_sample_assistant_message(uuid: &str, session_id: &str, content: &str) -> String {
        format!(
            r#"{{"uuid":"{uuid}","sessionId":"{session_id}","timestamp":"2025-06-26T10:01:00Z","type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"{content}"}}],"id":"msg_123","model":"claude-opus-4-20250514","usage":{{"input_tokens":100,"output_tokens":50}}}}}}"#
        )
    }

    fn create_sample_summary_message(summary: &str) -> String {
        format!(r#"{{"type":"summary","summary":"{summary}","leafUuid":"leaf-123"}}"#)
    }

    #[tokio::test]
    async fn test_load_session_messages_basic() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!")
        );

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, "user");
        assert_eq!(messages[1].message_type, "assistant");
    }

    #[tokio::test]
    async fn test_message_surfaces_exclude_metadata_only_records_deterministically() {
        let temp_dir = TempDir::new().unwrap();
        let content = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            r#"{"type":"custom-title","sessionId":"session-1","customTitle":"Branch name"}"#,
            r#"{"type":"file-history-delta","messageId":"msg-1","snapshotMessageId":"snapshot-1","trackingPath":"/tmp/history","backup":{},"timestamp":"2025-06-26T10:00:30Z"}"#,
            r#"{"type":"permission-mode","sessionId":"session-1","permissionMode":"acceptEdits"}"#,
            r#"{"type":"relocated","sessionId":"session-1","relocatedCwd":"/tmp/project"}"#,
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!")
        );
        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);
        let path = file_path.to_string_lossy().to_string();

        let first = load_session_messages(path.clone()).await.unwrap();
        let second = load_session_messages(path.clone()).await.unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        assert_eq!(
            first
                .iter()
                .map(|message| message.message_type.as_str())
                .collect::<Vec<_>>(),
            ["user", "assistant"]
        );

        let page = load_session_messages_paginated(path.clone(), 0, 10, None)
            .await
            .unwrap();
        assert_eq!(page.total_count, 2);
        assert_eq!(page.messages.len(), 2);
        assert_eq!(
            get_session_message_count(path.clone(), None).await.unwrap(),
            2
        );
        assert_eq!(
            get_session_message_offset(path, "uuid-1".to_string(), None).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn test_metadata_only_records_never_normalize_into_messages() {
        for line in [
            r#"{"type":"custom-title","sessionId":"session-1","customTitle":"Branch name"}"#,
            r#"{"type":"file-history-delta","messageId":"msg-1","snapshotMessageId":"snapshot-1","trackingPath":"/tmp/history","backup":{},"timestamp":"2025-06-26T10:00:30Z"}"#,
            r#"{"type":"permission-mode","sessionId":"session-1","permissionMode":"acceptEdits"}"#,
            r#"{"type":"relocated","sessionId":"session-1","relocatedCwd":"/tmp/project"}"#,
        ] {
            assert!(parse_line_to_message(0, line, false).is_none());
            let mut bytes = line.as_bytes().to_vec();
            assert!(parse_line_simd(0, &mut bytes, false).is_none());
        }
    }

    #[tokio::test]
    async fn test_load_session_messages_excludes_summary() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi!"),
            create_sample_summary_message("Test conversation summary")
        );

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        // Summary messages should be excluded
        assert_eq!(messages.len(), 2);

        // Verify no summary message is present
        let summary_msg = messages.iter().find(|m| m.message_type == "summary");
        assert!(summary_msg.is_none());
    }

    #[tokio::test]
    async fn test_load_session_messages_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = create_test_jsonl_file(&temp_dir, "empty.jsonl", "");

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_load_session_messages_with_empty_lines() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "\n{}\n\n{}\n\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi!")
        );

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_load_session_messages_file_not_found() {
        let result = load_session_messages("/nonexistent/path/file.jsonl".to_string()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to open session file"));
    }

    #[tokio::test]
    async fn test_load_session_messages_with_malformed_json() {
        let temp_dir = TempDir::new().unwrap();

        // First line is valid, second is malformed
        let content = format!(
            "{}\n{{invalid json}}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi!")
        );

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        // Should still succeed with valid messages
        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn test_load_session_messages_paginated_basic() {
        let temp_dir = TempDir::new().unwrap();

        // Create 5 messages
        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result =
            load_session_messages_paginated(file_path.to_string_lossy().to_string(), 0, 3, None)
                .await;

        assert!(result.is_ok());
        let page = result.unwrap();
        assert_eq!(page.total_count, 5);
        assert_eq!(page.messages.len(), 3);
        assert!(page.has_more);
    }

    #[tokio::test]
    async fn test_load_session_messages_paginated_offset() {
        let temp_dir = TempDir::new().unwrap();

        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        // Get second page
        let result =
            load_session_messages_paginated(file_path.to_string_lossy().to_string(), 3, 3, None)
                .await;

        assert!(result.is_ok());
        let page = result.unwrap();
        assert_eq!(page.total_count, 5);
        assert_eq!(page.messages.len(), 2); // Only 2 remaining
        assert!(!page.has_more);
    }

    #[tokio::test]
    async fn test_load_session_messages_paginated_exclude_sidechain() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"},"isSidechain":false}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"Sidechain"},"isSidechain":true}
{"uuid":"uuid-3","sessionId":"session-1","timestamp":"2025-06-26T10:02:00Z","type":"user","message":{"role":"user","content":"World"},"isSidechain":false}
"#;

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", content);

        // With exclude_sidechain = true
        let result = load_session_messages_paginated(
            file_path.to_string_lossy().to_string(),
            0,
            10,
            Some(true),
        )
        .await;

        assert!(result.is_ok());
        let page = result.unwrap();
        assert_eq!(page.total_count, 2); // Sidechain message excluded
    }

    #[tokio::test]
    async fn test_get_session_message_count() {
        let temp_dir = TempDir::new().unwrap();

        let mut content = String::new();
        for i in 1..=10 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }
        // Add a summary (should not be counted)
        content.push_str(&format!("{}\n", create_sample_summary_message("Summary")));

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);

        let result = get_session_message_count(file_path.to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10); // Summary not counted
    }

    #[tokio::test]
    async fn test_get_session_message_count_exclude_sidechain() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"},"isSidechain":false}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"Sidechain"},"isSidechain":true}
{"uuid":"uuid-3","sessionId":"session-1","timestamp":"2025-06-26T10:02:00Z","type":"user","message":{"role":"user","content":"World"}}
"#;

        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", content);

        // Without exclude
        let count_all = get_session_message_count(file_path.to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(count_all, 3);

        // With exclude
        let count_filtered =
            get_session_message_count(file_path.to_string_lossy().to_string(), Some(true))
                .await
                .unwrap();
        assert_eq!(count_filtered, 2);
    }

    #[tokio::test]
    async fn test_get_session_message_offset_basic() {
        let temp_dir = TempDir::new().unwrap();

        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }
        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", &content);
        let path = file_path.to_string_lossy().to_string();

        // Newest message → offset 0
        let newest = get_session_message_offset(path.clone(), "uuid-5".to_string(), None).unwrap();
        assert_eq!(newest, Some(0));

        // Oldest message → offset 4 (4 visible messages are newer)
        let oldest = get_session_message_offset(path.clone(), "uuid-1".to_string(), None).unwrap();
        assert_eq!(oldest, Some(4));

        // Loading offset=0, limit=offset+1 must include the target uuid.
        let needed = oldest.unwrap() + 1;
        let page = load_session_messages_paginated(path.clone(), 0, needed, None)
            .await
            .unwrap();
        assert!(page.messages.iter().any(|m| m.uuid == "uuid-1"));

        // Unknown uuid → None
        let missing = get_session_message_offset(path, "no-such-uuid".to_string(), None).unwrap();
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_get_session_message_offset_skips_invisible_lines() {
        let temp_dir = TempDir::new().unwrap();

        // sidechain + summary lines must not shift the offset when excluded
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"},"isSidechain":false}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"Sidechain"},"isSidechain":true}
{"type":"summary","summary":"A summary","leafUuid":"uuid-3"}
{"uuid":"uuid-4","sessionId":"session-1","timestamp":"2025-06-26T10:02:00Z","type":"user","message":{"role":"user","content":"World"},"isSidechain":false}
"#;
        let file_path = create_test_jsonl_file(&temp_dir, "test.jsonl", content);
        let path = file_path.to_string_lossy().to_string();

        // With sidechain excluded: visible = [uuid-1, uuid-4]
        let offset =
            get_session_message_offset(path.clone(), "uuid-1".to_string(), Some(true)).unwrap();
        assert_eq!(offset, Some(1));

        // Without exclusion: visible = [uuid-1, uuid-2, uuid-4]
        let offset_all =
            get_session_message_offset(path.clone(), "uuid-1".to_string(), None).unwrap();
        assert_eq!(offset_all, Some(2));

        // Sidechain uuid itself is findable when not excluded
        let sidechain = get_session_message_offset(path, "uuid-2".to_string(), None).unwrap();
        assert_eq!(sidechain, Some(1));
    }

    #[tokio::test]
    async fn test_load_project_sessions_basic() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello from test"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi!")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        let sessions = result.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].message_count, 2);
    }

    #[tokio::test]
    async fn test_load_project_sessions_prefers_jsonl_cwd_for_project_name() {
        let temp_dir = TempDir::new().unwrap();
        let project_dir = temp_dir.path().join("-home-cym-claude-prompt-design");
        std::fs::create_dir_all(&project_dir).unwrap();
        let actual_cwd = temp_dir.path().join("claude_prompt_design");
        std::fs::create_dir_all(&actual_cwd).unwrap();
        let actual_cwd = actual_cwd.to_string_lossy();

        let content = format!(
            r#"{{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","cwd":"{actual_cwd}","message":{{"role":"user","content":"Hello world"}}}}
"#
        );
        let file_path = project_dir.join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_project_sessions(project_dir.to_string_lossy().to_string(), None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].project_name, "claude_prompt_design");
    }

    #[tokio::test]
    async fn test_load_project_sessions_prefers_verified_folder_over_stale_cwd() {
        let temp_dir = TempDir::new().unwrap();
        // Folder name decodes to an existing directory (/usr/lib); the
        // `.claude/projects/` marker must be present for verified decoding.
        let project_dir = temp_dir
            .path()
            .join(".claude")
            .join("projects")
            .join("-usr-lib");
        std::fs::create_dir_all(&project_dir).unwrap();

        // Stale embedded cwd simulates a session moved into this folder by hand.
        let content = concat!(
            r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","#,
            r#""type":"user","cwd":"/some/stale/Dev","#,
            r#""message":{"role":"user","content":"Hello world"}}"#,
            "\n"
        );
        let file_path = project_dir.join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_project_sessions(project_dir.to_string_lossy().to_string(), None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        // Verified folder name wins over the stale cwd.
        assert_eq!(result[0].project_name, "lib");
    }

    #[tokio::test]
    async fn test_load_project_sessions_with_summary() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi!"),
            create_sample_summary_message("This is the session summary")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        let sessions = result.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(
            sessions[0].summary,
            Some("This is the session summary".to_string())
        );
    }

    #[tokio::test]
    async fn test_load_project_sessions_multiple_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create first session file
        let content1 = format!(
            "{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello")
        );
        let file_path1 = temp_dir.path().join("session1.jsonl");
        let mut file1 = File::create(&file_path1).unwrap();
        file1.write_all(content1.as_bytes()).unwrap();

        // Create second session file
        let content2 = format!(
            "{}\n{}\n",
            create_sample_user_message("uuid-2", "session-2", "World"),
            create_sample_assistant_message("uuid-3", "session-2", "!")
        );
        let file_path2 = temp_dir.path().join("session2.jsonl");
        let mut file2 = File::create(&file_path2).unwrap();
        file2.write_all(content2.as_bytes()).unwrap();

        let result =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        let sessions = result.unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn test_load_project_sessions_page_uses_cache_and_offsets() {
        let temp_dir = TempDir::new().unwrap();

        let sessions = [
            (
                "old.jsonl",
                "session-old",
                "2025-06-26T09:00:00Z",
                "Old session",
            ),
            (
                "new.jsonl",
                "session-new",
                "2025-06-26T11:00:00Z",
                "New session",
            ),
            (
                "mid.jsonl",
                "session-mid",
                "2025-06-26T10:00:00Z",
                "Middle session",
            ),
        ];

        for (filename, session_id, timestamp, content) in sessions {
            let file_path = temp_dir.path().join(filename);
            let mut file = File::create(&file_path).unwrap();
            let line = create_sample_user_message_at(
                &format!("uuid-{session_id}"),
                session_id,
                timestamp,
                content,
            );
            file.write_all(format!("{line}\n").as_bytes()).unwrap();
        }

        let project_path = temp_dir.path().to_string_lossy().to_string();
        let full = load_project_sessions(project_path.clone(), None)
            .await
            .unwrap();
        assert_eq!(full.len(), 3);

        let first_page = load_project_sessions_page(project_path.clone(), None, Some(0), Some(2))
            .await
            .unwrap();

        assert_eq!(first_page.total, 3);
        assert_eq!(first_page.sessions.len(), 2);
        assert_eq!(first_page.offset, 0);
        assert_eq!(first_page.next_offset, 2);
        assert!(first_page.has_more);
        assert_eq!(first_page.sessions[0].actual_session_id, "session-new");
        assert_eq!(first_page.sessions[1].actual_session_id, "session-mid");

        let second_page =
            load_project_sessions_page(project_path, None, Some(first_page.next_offset), Some(2))
                .await
                .unwrap();

        assert_eq!(second_page.total, 3);
        assert_eq!(second_page.sessions.len(), 1);
        assert_eq!(second_page.offset, 2);
        assert_eq!(second_page.next_offset, 3);
        assert!(!second_page.has_more);
        assert_eq!(second_page.sessions[0].actual_session_id, "session-old");
    }

    #[tokio::test]
    async fn test_load_project_sessions_page_rejects_invalid_project_path() {
        let empty = load_project_sessions_page("  ".to_string(), None, None, None).await;
        assert_eq!(empty.err().unwrap(), "project_path is required");

        let relative =
            load_project_sessions_page("relative/project".to_string(), None, None, None).await;
        assert_eq!(
            relative.err().unwrap(),
            "project_path must be an absolute path"
        );
    }

    #[tokio::test]
    async fn test_load_project_sessions_page_skips_invalid_candidates() {
        let temp_dir = TempDir::new().unwrap();

        let valid_1 = create_test_jsonl_file(
            &temp_dir,
            "valid-1.jsonl",
            &format!(
                "{}\n",
                create_sample_user_message("uuid-valid-1", "session-valid-1", "Hello")
            ),
        );
        let invalid = create_test_jsonl_file(&temp_dir, "invalid.jsonl", "{}\n");
        let valid_2 = create_test_jsonl_file(
            &temp_dir,
            "valid-2.jsonl",
            &format!(
                "{}\n",
                create_sample_user_message("uuid-valid-2", "session-valid-2", "World")
            ),
        );

        // Pin distinct mtimes so the recency-sorted candidate order is
        // deterministic and the invalid file falls inside the first page's
        // scan window (valid-1 newest, invalid middle, valid-2 oldest).
        let base = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_750_000_000);
        for (path, age_secs) in [(&valid_1, 0u64), (&invalid, 60), (&valid_2, 120)] {
            let file = File::options().write(true).open(path).unwrap();
            file.set_modified(base - std::time::Duration::from_secs(age_secs))
                .unwrap();
        }

        let project_path = temp_dir.path().to_string_lossy().to_string();
        let page = load_project_sessions_page(project_path.clone(), None, Some(0), Some(2))
            .await
            .unwrap();

        let session_ids: Vec<&str> = page
            .sessions
            .iter()
            .map(|session| session.actual_session_id.as_str())
            .collect();
        assert_eq!(page.total, 2);
        assert_eq!(page.sessions.len(), 2);
        assert_eq!(page.next_offset, 3);
        assert!(!page.has_more);
        assert!(session_ids.contains(&"session-valid-1"));
        assert!(session_ids.contains(&"session-valid-2"));

        let second_page =
            load_project_sessions_page(project_path, None, Some(page.next_offset), Some(2))
                .await
                .unwrap();
        assert_eq!(second_page.total, 2);
        assert!(second_page.sessions.is_empty());
        assert!(!second_page.has_more);
    }

    #[tokio::test]
    async fn test_load_project_sessions_page_total_stays_cumulative_without_disk_cache() {
        let temp_dir = TempDir::new().unwrap();

        let valid_1 = create_test_jsonl_file(
            &temp_dir,
            "valid-1.jsonl",
            &format!(
                "{}\n",
                create_sample_user_message("uuid-valid-1", "session-valid-1", "Hello")
            ),
        );
        let invalid = create_test_jsonl_file(&temp_dir, "invalid.jsonl", "{}\n");
        let valid_2 = create_test_jsonl_file(
            &temp_dir,
            "valid-2.jsonl",
            &format!(
                "{}\n",
                create_sample_user_message("uuid-valid-2", "session-valid-2", "World")
            ),
        );

        let base = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_750_000_000);
        for (path, age_secs) in [(&valid_1, 0u64), (&invalid, 60), (&valid_2, 120)] {
            let file = File::options().write(true).open(path).unwrap();
            file.set_modified(base - std::time::Duration::from_secs(age_secs))
                .unwrap();
        }

        let project_path = temp_dir.path().to_string_lossy().to_string();
        let first_page = load_project_sessions_page(project_path.clone(), None, Some(0), Some(2))
            .await
            .unwrap();
        assert_eq!(first_page.total, 2);
        assert_eq!(first_page.sessions.len(), 2);

        // Simulate the best-effort cache write being lost (e.g. full disk or
        // read-only project dir): later pages must still account for the
        // candidates already dropped by earlier pages via the in-memory
        // registry, so `total` stays consistent on every page.
        std::fs::remove_file(temp_dir.path().join(".session_cache.json")).unwrap();

        let second_page =
            load_project_sessions_page(project_path, None, Some(first_page.next_offset), Some(2))
                .await
                .unwrap();
        assert_eq!(second_page.total, 2);
        assert!(second_page.sessions.is_empty());
        assert!(!second_page.has_more);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_load_project_sessions_page_ignores_symlinked_jsonl_outside_project() {
        let temp_dir = TempDir::new().unwrap();
        let outside_dir = TempDir::new().unwrap();
        let outside_file = create_test_jsonl_file(
            &outside_dir,
            "outside.jsonl",
            &format!(
                "{}\n",
                create_sample_user_message("uuid-outside", "session-outside", "Outside")
            ),
        );
        let link_path = temp_dir.path().join("linked-outside.jsonl");
        std::os::unix::fs::symlink(&outside_file, link_path).unwrap();

        let page = load_project_sessions_page(
            temp_dir.path().to_string_lossy().to_string(),
            None,
            Some(0),
            Some(2),
        )
        .await
        .unwrap();

        assert_eq!(page.total, 0);
        assert!(page.sessions.is_empty());
        assert!(!page.has_more);
    }

    #[tokio::test]
    async fn test_load_project_sessions_exclude_sidechain() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"},"isSidechain":false}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"Sidechain"},"isSidechain":true}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        // Without exclude
        let result_all = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result_all[0].message_count, 2);

        // With exclude
        let result_filtered =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), Some(true))
                .await
                .unwrap();
        assert_eq!(result_filtered[0].message_count, 1);
    }

    #[tokio::test]
    async fn test_load_project_sessions_with_tool_use() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Read file"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tool_1","name":"Read","input":{}}]}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        let sessions = result.unwrap();
        assert!(sessions[0].has_tool_use);
    }

    #[tokio::test]
    async fn test_load_project_sessions_empty_directory() {
        let temp_dir = TempDir::new().unwrap();

        let result =
            load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_incremental_parsing_on_file_append() {
        use std::io::Write;

        let temp_dir = TempDir::new().unwrap();

        // Initial content with 2 messages
        let initial_content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"assistant","message":{"role":"assistant","content":"Hi there"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, initial_content).unwrap();

        // First load - creates cache
        let result1 = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].message_count, 2);

        // Append more messages to the file
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&file_path)
            .unwrap();
        writeln!(file, r#"{{"uuid":"uuid-3","sessionId":"session-1","timestamp":"2025-06-26T10:02:00Z","type":"user","message":{{"role":"user","content":"How are you?"}}}}"#).unwrap();
        writeln!(file, r#"{{"uuid":"uuid-4","sessionId":"session-1","timestamp":"2025-06-26T10:03:00Z","type":"assistant","message":{{"role":"assistant","content":"I'm doing great!"}}}}"#).unwrap();
        drop(file);

        // Second load - should use incremental parsing
        let result2 = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].message_count, 4); // 2 original + 2 appended
        assert_eq!(result2[0].last_message_time, "2025-06-26T10:03:00Z");
    }

    #[tokio::test]
    async fn test_message_with_missing_uuid_generates_new_one() {
        let temp_dir = TempDir::new().unwrap();

        // Message without uuid
        let content = r#"{"sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 1);
        // Should have a generated UUID
        assert!(!messages[0].uuid.is_empty());
        assert!(messages[0].uuid.contains("-line-"));
    }

    #[tokio::test]
    async fn test_message_with_missing_session_id() {
        let temp_dir = TempDir::new().unwrap();

        // Message without sessionId
        let content = r#"{"uuid":"uuid-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].session_id, "unknown-session");
    }

    #[tokio::test]
    async fn test_assistant_message_with_usage_stats() {
        let temp_dir = TempDir::new().unwrap();

        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello!"}],"id":"msg_123","model":"claude-opus-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":20,"cache_read_input_tokens":10}}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 1);

        let msg = &messages[0];
        assert_eq!(msg.role, Some("assistant".to_string()));
        assert_eq!(msg.message_id, Some("msg_123".to_string()));
        assert_eq!(msg.model, Some("claude-opus-4-20250514".to_string()));
        assert_eq!(msg.stop_reason, Some("end_turn".to_string()));

        let usage = msg.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.cache_creation_input_tokens, Some(20));
        assert_eq!(usage.cache_read_input_tokens, Some(10));
    }

    #[tokio::test]
    async fn test_assistant_effort_maps_to_inference_reasoning_effort() {
        let temp_dir = TempDir::new().unwrap();

        // Claude Code stamps a top-level `effort` beside `message` on assistant
        // records; a user record without one must not grow an inference object.
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2026-07-26T10:00:00Z","type":"user","message":{"role":"user","content":"hi"}}
{"uuid":"uuid-2","parentUuid":"uuid-1","sessionId":"session-1","timestamp":"2026-07-26T10:00:01Z","type":"assistant","effort":"high","message":{"role":"assistant","content":[{"type":"text","text":"Hello!"}],"id":"msg_1","model":"claude-fable-5"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let result = load_session_messages(file_path.to_string_lossy().to_string()).await;

        assert!(result.is_ok());
        let messages = result.unwrap();
        assert_eq!(messages.len(), 2);
        assert!(messages[0].inference.is_none());

        let inference = messages[1].inference.as_ref().expect("inference seeded");
        assert_eq!(inference.reasoning_effort.as_deref(), Some("high"));
        // The legacy top-level model stays where it was; hydration folds it in
        // at the provider-loading boundary, not here.
        assert_eq!(messages[1].model.as_deref(), Some("claude-fable-5"));
    }

    #[tokio::test]
    async fn test_session_summary_fallback_first_user_message() {
        let temp_dir = TempDir::new().unwrap();

        // Session with no summary but has user messages
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Hello, can you help me?"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Of course!"}]}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].summary,
            Some("Hello, can you help me?".to_string())
        );
    }

    #[tokio::test]
    async fn test_session_summary_fallback_first_assistant_text() {
        let temp_dir = TempDir::new().unwrap();

        // Session with no summary, no user messages, but has assistant text
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"assistant","message":{"role":"assistant","content":"This is a resume of a previous conversation about Rust programming"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].summary,
            Some("This is a resume of a previous conversation about Rust programming".to_string())
        );
    }

    #[tokio::test]
    async fn test_session_summary_fallback_last_user_message() {
        let temp_dir = TempDir::new().unwrap();

        // Session with command message (not genuine text), followed by real user message
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"<command-message>init is analyzing...</command-message>\n<command-name>/init</command-name>"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"Can you review this code?"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Should use last_user_content as fallback since first is a command
        assert_eq!(
            result[0].summary,
            Some("Can you review this code?".to_string())
        );
    }

    #[tokio::test]
    async fn test_local_command_plumbing_is_not_counted_as_conversation() {
        let temp_dir = TempDir::new().unwrap();
        let content = concat!(
            r#"{"uuid":"cmd","sessionId":"session-1","timestamp":"2026-07-15T22:19:17Z","type":"user","message":{"role":"user","content":"<command-name>/model</command-name>\n<command-message>model</command-message>\n<command-args>claude-fable-5[1m]</command-args>"}}"#,
            "\n",
            r#"{"uuid":"stdout","parentUuid":"cmd","sessionId":"session-1","timestamp":"2026-07-15T22:19:17Z","type":"user","message":{"role":"user","content":"<local-command-stdout>Set model to claude-fable-5</local-command-stdout>"}}"#,
            "\n",
            r#"{"uuid":"u1","parentUuid":"stdout","sessionId":"session-1","timestamp":"2026-07-15T22:20:00Z","type":"user","message":{"role":"user","content":"Continue with the task."}}"#,
            "\n",
            r#"{"uuid":"a1","parentUuid":"u1","sessionId":"session-1","timestamp":"2026-07-15T22:20:01Z","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Continuing."}]}}"#,
            "\n"
        );
        std::fs::write(temp_dir.path().join("test.jsonl"), content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].message_count, 2);
    }

    #[tokio::test]
    async fn test_session_summary_fallback_incremental_preserves_values() {
        let temp_dir = TempDir::new().unwrap();

        // Initial content with user message
        let initial_content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"user","message":{"role":"user","content":"Initial question here"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"assistant","message":{"role":"assistant","content":"Answer to the question"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, initial_content).unwrap();

        // First load - creates cache with fallback values
        let result1 = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result1.len(), 1);
        assert_eq!(
            result1[0].summary,
            Some("Initial question here".to_string())
        );

        // Append more messages (no summary or user messages in new content)
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&file_path)
            .unwrap();
        use std::io::Write;
        writeln!(
            file,
            r#"{{"uuid":"uuid-3","sessionId":"session-1","timestamp":"2025-06-26T10:02:00Z","type":"assistant","message":{{"role":"assistant","content":"More content"}}}}"#
        )
        .unwrap();
        drop(file);

        // Second load - should preserve the fallback value from cache
        let result2 = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].message_count, 3);
        assert_eq!(
            result2[0].summary,
            Some("Initial question here".to_string())
        );
    }

    #[tokio::test]
    async fn test_extract_assistant_text_with_string_content() {
        let temp_dir = TempDir::new().unwrap();

        // Assistant message with string content (not array)
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"assistant","message":{"role":"assistant","content":"This is a string content message that should be extracted"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Should extract string content, not just array content
        assert!(result[0].summary.is_some());
        assert!(result[0]
            .summary
            .as_ref()
            .unwrap()
            .contains("string content message"));
    }

    #[tokio::test]
    async fn test_extract_assistant_text_min_length() {
        let temp_dir = TempDir::new().unwrap();

        // Assistant message with very short text (< 10 chars, should be ignored)
        let content = r#"{"uuid":"uuid-1","sessionId":"session-1","timestamp":"2025-06-26T10:00:00Z","type":"assistant","message":{"role":"assistant","content":"Short"}}
{"uuid":"uuid-2","sessionId":"session-1","timestamp":"2025-06-26T10:01:00Z","type":"user","message":{"role":"user","content":"User fallback message"}}
"#;

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Should fall back to user message since assistant text is too short
        assert_eq!(result[0].summary, Some("User fallback message".to_string()));
    }

    fn create_sample_rename_message(name: &str) -> String {
        format!(
            r#"{{"type":"system","subtype":"local_command","content":"<local-command-stdout>Session renamed to: {name}</local-command-stdout>","timestamp":"2025-06-26T10:05:00Z","sessionId":"session-1"}}"#
        )
    }

    #[tokio::test]
    async fn test_should_extract_rename_from_system_message() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_rename_message("MyProject")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("MyProject".to_string()));
    }

    #[tokio::test]
    async fn test_should_use_last_rename_when_multiple() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_rename_message("Alpha"),
            create_sample_rename_message("Beta")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("Beta".to_string()));
    }

    #[tokio::test]
    async fn test_should_prioritize_rename_over_other_summaries() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_summary_message("Auto summary"),
            create_sample_rename_message("Custom Name")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Rename takes priority over summary message
        assert_eq!(result[0].summary, Some("Custom Name".to_string()));
    }

    #[tokio::test]
    async fn test_should_fallback_to_existing_summary() {
        let temp_dir = TempDir::new().unwrap();

        // No rename message — should use first user content as summary
        let content = format!(
            "{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello world"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("Hello world".to_string()));
    }

    #[tokio::test]
    async fn test_should_not_count_system_as_message() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_rename_message("MyProject")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // System message should not be counted
        assert_eq!(result[0].message_count, 2);
    }

    #[tokio::test]
    async fn test_should_ignore_empty_rename() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello world"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            r#"{"type":"system","subtype":"local_command","content":"<local-command-stdout>Session renamed to: </local-command-stdout>","timestamp":"2025-06-26T10:05:00Z","sessionId":"session-1"}"#
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Empty rename should be ignored, falls back to first user content
        assert_eq!(result[0].summary, Some("Hello world".to_string()));
    }

    #[test]
    fn test_extract_rename_from_content() {
        // Valid rename
        let content = serde_json::json!(
            "<local-command-stdout>Session renamed to: MyProject</local-command-stdout>"
        );
        assert_eq!(
            extract_rename_from_content(&content),
            Some("MyProject".to_string())
        );

        // Empty name
        let content =
            serde_json::json!("<local-command-stdout>Session renamed to: </local-command-stdout>");
        assert_eq!(extract_rename_from_content(&content), None);

        // Not a rename message
        let content =
            serde_json::json!("<local-command-stdout>Some other command</local-command-stdout>");
        assert_eq!(extract_rename_from_content(&content), None);

        // Non-string content
        let content = serde_json::json!(42);
        assert_eq!(extract_rename_from_content(&content), None);

        // Name with special characters
        let content = serde_json::json!(
            "<local-command-stdout>Session renamed to: My [Project] v2.0</local-command-stdout>"
        );
        assert_eq!(
            extract_rename_from_content(&content),
            Some("My [Project] v2.0".to_string())
        );
    }

    #[tokio::test]
    async fn test_phase2_rename_beyond_metadata_lines() {
        let temp_dir = TempDir::new().unwrap();

        // Build a fixture with > METADATA_PHASE_LINES (100) to force Phase 2 parsing
        let mut content = String::new();
        for i in 1..=60 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-u{i}"),
                    "session-1",
                    &format!("User message {i}")
                )
            ));
            content.push_str(&format!(
                "{}\n",
                create_sample_assistant_message(
                    &format!("uuid-a{i}"),
                    "session-1",
                    &format!("Assistant reply {i}")
                )
            ));
        }
        // Append rename message after line 120 (beyond METADATA_PHASE_LINES=100)
        content.push_str(&format!("{}\n", create_sample_rename_message("LateRename")));

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Rename in Phase 2 (beyond metadata lines) should still be detected
        assert_eq!(result[0].summary, Some("LateRename".to_string()));
        // System message should not be counted (60 user + 60 assistant = 120)
        assert_eq!(result[0].message_count, 120);
    }

    #[tokio::test]
    async fn test_incremental_append_then_rename() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");

        // Initial content — populate cache
        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-u{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }
        std::fs::write(&file_path, &content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("Message 1".to_string()));
        assert_eq!(result[0].message_count, 5);

        // Append a rename message — triggers incremental parsing
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&file_path)
            .unwrap();
        writeln!(file, "{}", create_sample_rename_message("AppendedRename")).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Incremental parse should pick up the rename
        assert_eq!(result[0].summary, Some("AppendedRename".to_string()));
        // System message not counted
        assert_eq!(result[0].message_count, 5);
    }

    fn create_sample_custom_title_message(name: &str) -> String {
        format!(r#"{{"type":"custom-title","customTitle":"{name}","sessionId":"session-1"}}"#)
    }

    fn create_sample_agent_name_message(name: &str) -> String {
        format!(r#"{{"type":"agent-name","agentName":"{name}","sessionId":"session-1"}}"#)
    }

    #[tokio::test]
    async fn test_should_extract_rename_from_branch_custom_title() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_custom_title_message("HC1-migration"),
            create_sample_agent_name_message("HC1-migration")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("HC1-migration".to_string()));
        assert!(result[0].is_renamed);
        // custom-title and agent-name lines should not be counted as messages
        assert_eq!(result[0].message_count, 2);
    }

    #[tokio::test]
    async fn test_should_use_last_naming_event_regardless_of_kind() {
        let temp_dir = TempDir::new().unwrap();

        // /branch (custom-title) happens, then a later /rename overrides it
        let content = format!(
            "{}\n{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            create_sample_custom_title_message("BranchName"),
            create_sample_rename_message("LaterRename")
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        // Whichever naming event is chronologically last in the file wins
        assert_eq!(result[0].summary, Some("LaterRename".to_string()));
    }

    #[tokio::test]
    async fn test_should_ignore_empty_custom_title() {
        let temp_dir = TempDir::new().unwrap();

        let content = format!(
            "{}\n{}\n{}\n",
            create_sample_user_message("uuid-1", "session-1", "Hello world"),
            create_sample_assistant_message("uuid-2", "session-1", "Hi there!"),
            r#"{"type":"custom-title","customTitle":"","sessionId":"session-1"}"#
        );

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("Hello world".to_string()));
    }

    #[test]
    fn test_try_extract_custom_title() {
        assert_eq!(
            try_extract_custom_title("custom-title", Some("MyTitle")),
            Some("MyTitle".to_string())
        );
        // Wrong message type
        assert_eq!(
            try_extract_custom_title("agent-name", Some("MyTitle")),
            None
        );
        // Missing value
        assert_eq!(try_extract_custom_title("custom-title", None), None);
        // Empty/whitespace value
        assert_eq!(try_extract_custom_title("custom-title", Some("   ")), None);
    }

    #[tokio::test]
    async fn test_phase2_custom_title_beyond_metadata_lines() {
        let temp_dir = TempDir::new().unwrap();

        // Build a fixture with > METADATA_PHASE_LINES (100) to force Phase 2 parsing
        let mut content = String::new();
        for i in 1..=60 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-u{i}"),
                    "session-1",
                    &format!("User message {i}")
                )
            ));
            content.push_str(&format!(
                "{}\n",
                create_sample_assistant_message(
                    &format!("uuid-a{i}"),
                    "session-1",
                    &format!("Assistant reply {i}")
                )
            ));
        }
        // Append custom-title after line 120 (beyond METADATA_PHASE_LINES=100)
        content.push_str(&format!(
            "{}\n",
            create_sample_custom_title_message("LateBranchTitle")
        ));

        let file_path = temp_dir.path().join("test.jsonl");
        std::fs::write(&file_path, content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("LateBranchTitle".to_string()));
        assert_eq!(result[0].message_count, 120);
    }

    #[tokio::test]
    async fn test_incremental_append_then_custom_title() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");

        let mut content = String::new();
        for i in 1..=5 {
            content.push_str(&format!(
                "{}\n",
                create_sample_user_message(
                    &format!("uuid-u{i}"),
                    "session-1",
                    &format!("Message {i}")
                )
            ));
        }
        std::fs::write(&file_path, &content).unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("Message 1".to_string()));
        assert_eq!(result[0].message_count, 5);

        // Append a /branch custom-title event — triggers incremental parsing
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&file_path)
            .unwrap();
        writeln!(
            file,
            "{}",
            create_sample_custom_title_message("AppendedBranchTitle")
        )
        .unwrap();

        let result = load_project_sessions(temp_dir.path().to_string_lossy().to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].summary, Some("AppendedBranchTitle".to_string()));
        assert_eq!(result[0].message_count, 5);
    }

    #[test]
    fn workflow_run_id_for_detects_workflow_layout_only() {
        assert_eq!(
            workflow_run_id_for(Path::new(
                "/p/abc/subagents/workflows/wf_1a198a78-3be/agent-x.jsonl"
            )),
            Some("wf_1a198a78-3be".to_string())
        );
        // Flat subagents are not workflow-scoped.
        assert_eq!(
            workflow_run_id_for(Path::new("/p/abc/subagents/agent-x.jsonl")),
            None
        );
    }

    #[test]
    fn read_subagent_tool_use_id_reads_meta_json() {
        let dir = TempDir::new().unwrap();
        let meta = dir.path().join("agent-x.meta.json");
        std::fs::write(
            &meta,
            r#"{"agentType":"task","description":"d","toolUseId":"toolu_123"}"#,
        )
        .unwrap();
        assert_eq!(
            read_subagent_tool_use_id(&meta),
            Some("toolu_123".to_string())
        );
    }

    #[test]
    fn read_subagent_tool_use_id_none_when_missing_invalid_or_empty() {
        let dir = TempDir::new().unwrap();
        // Missing file.
        assert_eq!(
            read_subagent_tool_use_id(&dir.path().join("nope.meta.json")),
            None
        );
        // Present but no toolUseId (older meta or different shape).
        let meta = dir.path().join("agent-y.meta.json");
        std::fs::write(&meta, r#"{"agentType":"task"}"#).unwrap();
        assert_eq!(read_subagent_tool_use_id(&meta), None);
        // Empty toolUseId is treated as absent.
        let meta_empty = dir.path().join("agent-z.meta.json");
        std::fs::write(&meta_empty, r#"{"toolUseId":""}"#).unwrap();
        assert_eq!(read_subagent_tool_use_id(&meta_empty), None);
        // Invalid JSON is non-fatal.
        let meta_bad = dir.path().join("agent-bad.meta.json");
        std::fs::write(&meta_bad, "not json").unwrap();
        assert_eq!(read_subagent_tool_use_id(&meta_bad), None);
    }

    #[test]
    fn metadata_phase_counts_only_conversational_allowlist() {
        let temp_dir = TempDir::new().unwrap();
        // Keep metadata discovery incomplete until the final user record so every
        // preceding line exercises the full SessionMetadataEntry phase.
        let lines = [
            create_sample_assistant_message("a1", "session-1", "hello"),
            r#"{"type":"ai-title","aiTitle":"Generated","sessionId":"session-1","timestamp":"2026-01-01T00:00:01Z"}"#.to_string(),
            r#"{"type":"mode","mode":"plan","sessionId":"session-1","timestamp":"2026-01-01T00:00:02Z"}"#.to_string(),
            r#"{"type":"permission-mode","mode":"acceptEdits","sessionId":"session-1","timestamp":"2026-01-01T00:00:03Z"}"#.to_string(),
            r#"{"type":"future-metadata","sessionId":"session-1","timestamp":"2026-01-01T00:00:04Z"}"#.to_string(),
            r#"{"type":"user","uuid":"cmd","sessionId":"session-1","timestamp":"2026-01-01T00:00:04Z","message":{"role":"user","content":"<command-message>model</command-message>\n<command-name>/model</command-name>\n<command-args>default</command-args>"}}"#.to_string(),
            r#"{"type":"user","uuid":"stdout","sessionId":"session-1","timestamp":"2026-01-01T00:00:04Z","message":{"role":"user","content":"<local-command-stdout>Set model to default</local-command-stdout>"}}"#.to_string(),
            r#"{"type":"user","uuid":"init","sessionId":"session-1","timestamp":"2026-01-01T00:00:04Z","message":{"role":"user","content":"<command-message>init</command-message>\n<command-name>/init</command-name>"}}"#.to_string(),
            r#"{"type":"attachment","uuid":"q1","sessionId":"session-1","timestamp":"2026-01-01T00:00:05Z","attachment":{"type":"queued_command","prompt":[{"type":"text","text":"queued prompt"}]}}"#.to_string(),
            r#"{"type":"user","uuid":"task1","sessionId":"session-1","timestamp":"2026-01-01T00:00:05Z","origin":{"kind":"task-notification"},"message":{"role":"user","content":"<task-notification>completed</task-notification>"}}"#.to_string(),
            r#"{"type":"attachment","uuid":"task2","sessionId":"session-1","timestamp":"2026-01-01T00:00:05Z","attachment":{"type":"queued_command","commandMode":"task-notification","prompt":[{"type":"text","text":"synthetic completion"}]}}"#.to_string(),
            r#"{"type":"attachment","uuid":"r1","sessionId":"session-1","timestamp":"2026-01-01T00:00:06Z","attachment":{"type":"todo_reminder"}}"#.to_string(),
            create_sample_user_message("u1", "session-1", "authored prompt"),
        ];
        let path = create_test_jsonl_file(&temp_dir, "metadata-phase.jsonl", &lines.join("\n"));

        let result = extract_session_metadata_from_file(&path).unwrap();
        assert_eq!(result.session.message_count, 4);
        assert_eq!(result.session.summary.as_deref(), Some("authored prompt"));
    }

    #[test]
    fn fast_count_phase_uses_the_same_conversational_allowlist() {
        let temp_dir = TempDir::new().unwrap();
        // The first complete user record ends metadata discovery. Everything
        // after it therefore runs through QuickLineClassifier's fast phase.
        let lines = [
            create_sample_user_message("u1", "session-1", "start"),
            create_sample_assistant_message("a1", "session-1", "reply"),
            r#"{"type":"attachment","uuid":"q1","sessionId":"session-1","timestamp":"2026-01-01T00:00:05Z","attachment":{"type":"queued_command","prompt":[{"type":"text","text":"queued prompt"}]}}"#.to_string(),
            r#"{"type":"ai-title","aiTitle":"Generated","sessionId":"session-1","timestamp":"2026-01-01T00:00:06Z"}"#.to_string(),
            r#"{"type":"mode","mode":"plan","sessionId":"session-1","timestamp":"2026-01-01T00:00:07Z"}"#.to_string(),
            r#"{"type":"permission-mode","mode":"acceptEdits","sessionId":"session-1","timestamp":"2026-01-01T00:00:08Z"}"#.to_string(),
            r#"{"type":"future-metadata","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z"}"#.to_string(),
            r#"{"type":"user","uuid":"cmd","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z","message":{"role":"user","content":"<command-name>/model</command-name>\n<command-message>model</command-message>\n<command-args>default</command-args>"}}"#.to_string(),
            r#"{"type":"user","uuid":"stdout","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z","message":{"role":"user","content":"<local-command-stdout>Set model to default</local-command-stdout>"}}"#.to_string(),
            r#"{"type":"user","uuid":"init","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z","message":{"role":"user","content":"<command-message>init</command-message>\n<command-name>/init</command-name>"}}"#.to_string(),
            r#"{"type":"user","uuid":"task1","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z","origin":{"kind":"task-notification"},"message":{"role":"user","content":"<task-notification>completed</task-notification>"}}"#.to_string(),
            r#"{"type":"attachment","uuid":"task2","sessionId":"session-1","timestamp":"2026-01-01T00:00:09Z","attachment":{"type":"queued_command","commandMode":"task-notification","prompt":[{"type":"text","text":"synthetic completion"}]}}"#.to_string(),
            r#"{"type":"attachment","uuid":"r1","sessionId":"session-1","timestamp":"2026-01-01T00:00:10Z","attachment":{"type":"todo_reminder"}}"#.to_string(),
            r#"{"type":"user","sessionId":"session-1","timestamp":"2026-01-01T00:00:11Z","isMeta":true,"message":{"role":"user","content":"internal"}}"#.to_string(),
            r#"{"type":"user","message":{"role":"user","content":"invalid without identity or time"}}"#.to_string(),
        ];
        let path = create_test_jsonl_file(&temp_dir, "fast-phase.jsonl", &lines.join("\n"));

        let result = extract_session_metadata_from_file(&path).unwrap();
        assert_eq!(result.session.message_count, 4);
    }

    #[cfg(unix)]
    #[test]
    fn read_subagent_tool_use_id_rejects_symlink() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.meta.json");
        std::fs::write(&real, r#"{"toolUseId":"toolu_real"}"#).unwrap();
        let link = dir.path().join("agent-link.meta.json");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(read_subagent_tool_use_id(&link), None);
    }

    #[test]
    fn queued_command_attachment_normalizes_to_user_message() {
        // A message sent while the agent is still generating is persisted as a
        // `queued_command` attachment, not a `user` record. It must surface as a
        // user message carrying the prompt text, with uuid/parentUuid preserved
        // so the causal chain (and any reply) still links to it.
        let line = r#"{"type":"attachment","uuid":"q1","parentUuid":"p1","sessionId":"s1","timestamp":"2026-06-23T19:27:37.044Z","attachment":{"type":"queued_command","commandMode":"prompt","prompt":[{"type":"text","text":"do the thing"}]}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg =
            parse_line_simd(0, &mut bytes, false).expect("queued_command should yield a message");
        assert_eq!(msg.message_type, "user");
        assert_eq!(msg.role.as_deref(), Some("user"));
        assert_eq!(msg.uuid, "q1");
        assert_eq!(msg.parent_uuid.as_deref(), Some("p1"));
        // Provenance marker so consumers can present it distinctly.
        assert_eq!(msg.subtype.as_deref(), Some("queued_command"));
        let text = msg
            .content
            .as_ref()
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str());
        assert_eq!(text, Some("do the thing"));
    }

    #[test]
    fn compact_summary_user_record_gets_provenance_subtype() {
        // The summary Claude Code injects when it compacts a session is a
        // `type: "user"` record flagged `isCompactSummary`. It must keep its type
        // and content but gain a `compact_summary` subtype so consumers can tell it
        // apart from an authored user turn (rather than relying on a text heuristic).
        let line = r#"{"type":"user","uuid":"c1","parentUuid":"p0","sessionId":"s1","timestamp":"2026-06-24T10:00:00Z","isCompactSummary":true,"message":{"role":"user","content":"This session is being continued from a previous conversation that ran out of context."}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("compact summary should parse");
        assert_eq!(msg.message_type, "user");
        assert_eq!(msg.role.as_deref(), Some("user"));
        assert_eq!(msg.uuid, "c1");
        assert_eq!(msg.subtype.as_deref(), Some("compact_summary"));

        // A normal user record (no flag) keeps no subtype.
        let plain = r#"{"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-06-24T10:01:00Z","message":{"role":"user","content":"hello"}}"#;
        let mut pbytes = plain.as_bytes().to_vec();
        let pmsg = parse_line_simd(0, &mut pbytes, false).expect("plain user should parse");
        assert_eq!(pmsg.subtype, None);
    }

    #[test]
    fn task_notification_user_record_gets_provenance_subtype() {
        let line = r#"{"type":"user","uuid":"task1","parentUuid":"a1","sessionId":"s1","timestamp":"2026-07-23T23:07:25Z","origin":{"kind":"task-notification"},"promptSource":"sdk","userType":"external","message":{"role":"user","content":"<task-notification>\n<summary>Background command completed</summary>\n</task-notification>"}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("task notification should parse");
        assert_eq!(msg.message_type, "user");
        assert_eq!(msg.role.as_deref(), Some("user"));
        assert_eq!(msg.subtype.as_deref(), Some("task_notification"));
        let fallback =
            parse_line_to_message(0, line, false).expect("fallback parser should classify it");
        assert_eq!(fallback.subtype.as_deref(), Some("task_notification"));

        // Classification comes from provider metadata, not from XML-like text
        // that an authored prompt could legitimately quote.
        let quoted = r#"{"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-07-23T23:08:00Z","message":{"role":"user","content":"Please explain this literal <task-notification> tag."}}"#;
        let mut quoted_bytes = quoted.as_bytes().to_vec();
        let quoted_msg =
            parse_line_simd(0, &mut quoted_bytes, false).expect("quoted text should parse");
        assert_eq!(quoted_msg.subtype, None);
    }

    #[test]
    fn local_command_user_records_get_provenance_subtype() {
        let cases = [
            r#"{"type":"user","uuid":"cmd","sessionId":"s1","timestamp":"2026-07-15T22:19:17Z","message":{"role":"user","content":"<command-name>/model</command-name>\n<command-message>model</command-message>\n<command-args>claude-fable-5[1m]</command-args>"}}"#,
            r#"{"type":"user","uuid":"stdout","sessionId":"s1","timestamp":"2026-07-15T22:19:17Z","message":{"role":"user","content":"<local-command-stdout>Set model to claude-fable-5</local-command-stdout>"}}"#,
        ];
        for line in cases {
            let mut bytes = line.as_bytes().to_vec();
            let msg = parse_line_simd(0, &mut bytes, false).expect("local command should parse");
            assert_eq!(msg.subtype.as_deref(), Some("local_command"));
        }

        let quoted = r#"{"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-07-15T22:20:00Z","message":{"role":"user","content":"The literal <command-name>/model</command-name> tag is relevant."}}"#;
        let mut bytes = quoted.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("quoted tag should parse");
        assert_eq!(msg.subtype, None);

        let init = r#"{"type":"user","uuid":"init","sessionId":"s1","timestamp":"2026-07-21T09:40:02Z","message":{"role":"user","content":"<command-message>init</command-message>\n<command-name>/init</command-name>"}}"#;
        let mut bytes = init.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("prompt command should parse");
        assert_eq!(msg.subtype.as_deref(), Some("prompt_command"));
    }

    #[test]
    fn non_queued_attachment_is_not_reclassified() {
        // Other attachment subtypes are UI/plumbing, not authored content, and
        // must NOT become user messages.
        assert!(
            queued_command_prompt(Some(&serde_json::json!({"type":"todo_reminder"}))).is_none()
        );
        assert!(queued_command_prompt(None).is_none());
        assert!(queued_command_prompt(Some(
            &serde_json::json!({"type":"queued_command","prompt":[{"type":"text","text":"x"}]})
        ))
        .is_some());
        assert!(queued_command_prompt(Some(&serde_json::json!({
            "type": "queued_command",
            "commandMode": "task-notification",
            "prompt": [{"type":"text","text":"<task-notification>synthetic</task-notification>"}]
        })))
        .is_none());

        let line = r#"{"type":"attachment","uuid":"a1","sessionId":"s1","timestamp":"2026-06-23T19:27:37.044Z","attachment":{"type":"todo_reminder"}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("record still parses");
        assert_eq!(msg.message_type, "attachment");
        assert!(msg.role.is_none());
    }

    #[test]
    fn prompt_file_attachment_keeps_only_its_basename() {
        let line = r#"{"type":"attachment","uuid":"a1","parentUuid":"u1","sessionId":"s1","timestamp":"2026-07-21T12:00:00Z","attachment":{"type":"file","filename":"C:\\private\\project\\report.md","displayPath":"docs\\report.md","content":{"type":"text","text":"private contents"}}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("file attachment should parse");

        assert_eq!(msg.message_type, "attachment");
        assert_eq!(msg.parent_uuid.as_deref(), Some("u1"));
        assert_eq!(
            msg.data,
            Some(serde_json::json!({"promptAttachments": [{"name": "report.md"}]}))
        );
        let serialized = serde_json::to_string(&msg).expect("serialize normalized message");
        assert!(!serialized.contains("private contents"));
        assert!(!serialized.contains("private\\\\project"));
    }

    #[test]
    fn selected_editor_file_is_not_a_prompt_attachment() {
        let line = r#"{"type":"attachment","uuid":"a1","sessionId":"s1","timestamp":"2026-07-21T12:00:00Z","attachment":{"type":"edited_text_file","filename":"C:\\project\\selected.rs","snippet":"selected context"}}"#;
        let mut bytes = line.as_bytes().to_vec();
        let msg = parse_line_simd(0, &mut bytes, false).expect("editor attachment should parse");

        assert_eq!(msg.message_type, "attachment");
        assert!(msg.data.is_none());
    }
}
