use super::{ProviderInfo, SessionSnapshotLoad};
use crate::commands::multi_provider::finalize_loaded_messages;
use crate::models::{
    ClaudeMessage, ClaudeProject, ClaudeSession, InferenceMetadata, InferenceUsage, TokenUsage,
};
use crate::utils::{
    build_provider_message, estimate_message_count_from_size, find_line_ranges,
    search_json_value_case_insensitive,
};
use base64::prelude::{Engine as _, BASE64_URL_SAFE_NO_PAD};
use chrono::{DateTime, Utc};
use memchr::{memchr_iter, memmem};
use memmap2::Mmap;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::commands::session::NativeRenameResult;

const STATE_DB_FILENAME: &str = "state_5.sqlite";
const SESSION_INDEX_FILENAME: &str = "session_index.jsonl";
const EXTERNAL_AGENT_IMPORTS_FILENAME: &str = "external_agent_session_imports.json";
const STEER_SUBTYPE: &str = "steer";
const SNAPSHOT_CURSOR_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CodexParserState {
    session_id: String,
    meta_seen: bool,
    current_inference: InferenceMetadata,
    prev_input_tokens: u32,
    prev_output_tokens: u32,
    prev_cached_tokens: u32,
    prev_cache_write_tokens: u32,
    prev_reasoning_tokens: u32,
    msg_counter: u64,
}

impl CodexParserState {
    fn initial(path: &Path) -> Self {
        Self {
            session_id: session_id_from_rollout_filename(path).unwrap_or_default(),
            meta_seen: false,
            current_inference: InferenceMetadata::default(),
            prev_input_tokens: 0,
            prev_output_tokens: 0,
            prev_cached_tokens: 0,
            prev_cache_write_tokens: 0,
            prev_reasoning_tokens: 0,
            msg_counter: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CodexParserCheckpoint {
    byte_offset: u64,
    replace_from: usize,
    state: CodexParserState,
}

#[derive(Debug, Serialize, Deserialize)]
struct CodexSnapshotCursor {
    version: u32,
    provider: String,
    canonical_path: String,
    accepted_len: u64,
    accepted_digest: String,
    checkpoint: CodexParserCheckpoint,
}

struct CodexParseOutcome {
    messages: Vec<ClaudeMessage>,
    checkpoint: CodexParserCheckpoint,
    accepted_len: usize,
}

#[derive(Debug, Clone)]
struct NativeTitle {
    title: String,
    is_renamed: bool,
}

#[derive(Debug)]
struct SqliteTitle {
    title: String,
    preview: String,
}

#[derive(Debug, Deserialize)]
struct SessionIndexEntry {
    id: String,
    thread_name: String,
}

#[derive(Debug)]
struct IndexedName {
    latest: String,
    changed: bool,
}

#[derive(Debug, Deserialize)]
struct ExternalAgentImportLedger {
    #[serde(default)]
    records: Vec<ExternalAgentImportRecord>,
}

#[derive(Debug, Deserialize)]
struct ExternalAgentImportRecord {
    source_path: String,
    imported_thread_id: String,
}

fn mark_pending_steer(
    messages: &mut [ClaudeMessage],
    pending_user_message_index: &mut Option<usize>,
) {
    if let Some(index) = pending_user_message_index.take() {
        messages[index].subtype = Some(STEER_SUBTYPE.to_string());
    }
}

/// Detect Codex CLI installation
pub fn detect() -> Option<ProviderInfo> {
    let base_path = get_base_path()?;
    let sessions_path = Path::new(&base_path).join("sessions");
    let archived_sessions_path = Path::new(&base_path).join("archived_sessions");

    Some(ProviderInfo {
        id: "codex".to_string(),
        display_name: "Codex CLI".to_string(),
        base_path: base_path.clone(),
        is_available: (sessions_path.exists() && sessions_path.is_dir())
            || (archived_sessions_path.exists() && archived_sessions_path.is_dir()),
    })
}

/// Get the Codex base path
pub fn get_base_path() -> Option<String> {
    // Check $CODEX_HOME first
    if let Ok(codex_home) = std::env::var("CODEX_HOME") {
        let path = PathBuf::from(&codex_home);
        if path.exists() {
            return Some(codex_home);
        }
    }

    // Default: ~/.codex
    let home = dirs::home_dir()?;
    let codex_path = home.join(".codex");
    if codex_path.exists() {
        Some(codex_path.to_string_lossy().to_string())
    } else {
        None
    }
}

fn get_sessions_dir() -> Result<PathBuf, String> {
    let base_path = get_base_path().ok_or_else(|| "Codex not found".to_string())?;
    Ok(Path::new(&base_path).join("sessions"))
}

fn get_archived_sessions_dir() -> Result<PathBuf, String> {
    let base_path = get_base_path().ok_or_else(|| "Codex not found".to_string())?;
    Ok(Path::new(&base_path).join("archived_sessions"))
}

/// Whether a discovered rollout belongs to Codex's archived-session root.
///
/// Keep this classification at the provider/storage boundary: callers should
/// not infer lifecycle state from a coincidental `archived_sessions` path
/// component. Both paths come from the same configured `CODEX_HOME`, so a
/// component-aware prefix check preserves the exact scan-root provenance.
pub(crate) fn is_archived_session_path(path: &Path) -> bool {
    get_archived_sessions_dir().is_ok_and(|archived_dir| path.starts_with(archived_dir))
}

/// Imported Codex thread ids and their source providers, read from Codex's own
/// external-agent import ledger. The ledger is authoritative for import state;
/// rollout fields such as `source:"vscode"` and `history_mode:"legacy"` are also
/// used by native sessions and therefore cannot classify an import.
///
/// Best-effort by design: an absent or malformed ledger simply yields no import
/// stamps, matching the other independently-changing metadata stores.
pub(crate) fn external_agent_imports() -> HashMap<String, Option<String>> {
    let Some(base_path) = get_base_path() else {
        return HashMap::new();
    };
    let Ok(bytes) = fs::read(Path::new(&base_path).join(EXTERNAL_AGENT_IMPORTS_FILENAME)) else {
        return HashMap::new();
    };
    let Ok(ledger) = serde_json::from_slice::<ExternalAgentImportLedger>(&bytes) else {
        return HashMap::new();
    };

    ledger
        .records
        .into_iter()
        .filter(|record| !record.imported_thread_id.trim().is_empty())
        .map(|record| {
            let provider = imported_provider_from_path(&record.source_path);
            (record.imported_thread_id, provider)
        })
        .collect()
}

/// The current ledger records the original file path rather than a provider id.
/// Recognize the storage roots the provider registry already supports, without
/// exposing the source path itself through the headless listing.
fn imported_provider_from_path(source_path: &str) -> Option<String> {
    let normalized = source_path.replace('\\', "/").to_ascii_lowercase();
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if parts.contains(&".claude") {
        Some("claude".to_string())
    } else if parts.contains(&".copilot")
        || parts.iter().any(|part| part.contains("github.copilot"))
    {
        Some("copilot".to_string())
    } else if parts.contains(&"opencode") {
        Some("opencode".to_string())
    } else {
        None
    }
}

fn get_existing_session_dirs() -> Result<Vec<PathBuf>, String> {
    let sessions_dir = get_sessions_dir()?;
    let archived_sessions_dir = get_archived_sessions_dir()?;

    Ok([sessions_dir, archived_sessions_dir]
        .into_iter()
        .filter(|path| path.exists() && path.is_dir())
        .collect())
}

// Codex generates these filenames itself, always lowercase — a
// case-insensitive comparison would accept files Codex never writes.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub(crate) fn is_rollout_jsonl(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.starts_with("rollout-")
                && (name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"))
        })
}

/// Discovery filter for session walkers: accepts every rollout
/// [`is_rollout_jsonl`] does, but skips a compressed `.jsonl.zst` whose plain
/// `.jsonl` twin exists — Codex materializes the plain file for appends, so
/// the plain one is the current version and listing both would duplicate the
/// session.
pub(crate) fn is_discoverable_rollout(path: &Path) -> bool {
    if !is_rollout_jsonl(path) {
        return false;
    }
    let is_compressed = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl.zst"));
    if is_compressed {
        // "rollout-….jsonl.zst" → "rollout-….jsonl"
        let plain = path.with_extension("");
        if plain.exists() {
            return false;
        }
    }
    true
}

/// Rollout file contents as a linear byte buffer: an mmap for plain `.jsonl`,
/// a decompressed buffer for `.jsonl.zst` (Codex compresses old rollouts).
enum RolloutBytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for RolloutBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            RolloutBytes::Mapped(mmap) => mmap,
            RolloutBytes::Owned(bytes) => bytes,
        }
    }
}

#[allow(unsafe_code)] // Required for mmap performance optimization
fn read_rollout_bytes(path: &Path) -> Result<RolloutBytes, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let is_compressed = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl.zst"));
    if is_compressed {
        return zstd::decode_all(std::io::BufReader::new(file))
            .map(RolloutBytes::Owned)
            .map_err(|e| format!("Failed to decompress rollout: {e}"));
    }
    // SAFETY: File is read-only and we only read from the mapping
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
    Ok(RolloutBytes::Mapped(mmap))
}

/// Return true when `session_path` is a Codex rollout JSONL inside the active
/// or archived session roots.
pub fn is_session_path(session_path: &str) -> bool {
    let path = Path::new(session_path);
    validate_session_path(path, session_path)
        .map(|canonical_path| is_rollout_jsonl(&canonical_path))
        .unwrap_or(false)
}

fn validate_session_path(session_path: &Path, raw_session_path: &str) -> Result<PathBuf, String> {
    let canonical_session = session_path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve session path: {e}"))?;

    let mut canonical_session_dirs = Vec::new();
    for dir in [get_sessions_dir()?, get_archived_sessions_dir()?] {
        if !dir.exists() || !dir.is_dir() {
            continue;
        }
        canonical_session_dirs.push(
            dir.canonicalize()
                .map_err(|e| format!("Failed to resolve Codex session directory: {e}"))?,
        );
    }

    if canonical_session_dirs.is_empty() {
        return Err("No Codex session directories found".to_string());
    }

    let is_allowed = canonical_session_dirs
        .iter()
        .any(|allowed_dir| canonical_session.starts_with(allowed_dir));

    if !is_allowed {
        return Err(format!(
            "Session path is outside Codex session directories: {raw_session_path}"
        ));
    }

    Ok(canonical_session)
}

/// Session metadata extracted from rollout files. `pub(crate)` so providers
/// that share the Codex rollout format (e.g. Open Interpreter) can reuse the
/// extractors below.
pub(crate) struct SessionInfo {
    pub(crate) session_id: String,
    pub(crate) cwd: Option<String>,
    /// Provider-qualified source from the rollout's first `session_meta`.
    /// User-facing surfaces are `codex-cli` / `codex-vscode`; structured
    /// subagent provenance is normalized separately as `codex-subagent`.
    pub(crate) entrypoint: Option<String>,
    /// Authoritative parent id from the rollout's first `session_meta`.
    pub(crate) forked_from_id: Option<String>,
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    pub(crate) message_count: usize,
    pub(crate) first_message_time: String,
    pub(crate) last_message_time: String,
    pub(crate) last_modified: String,
    pub(crate) file_path: String,
    pub(crate) has_tool_use: bool,
    pub(crate) summary: Option<String>,
}

/// Map Codex's rollout `source` to the shared provider entrypoint namespace.
/// User-facing clients use strings; spawned agents use a structured
/// `{ "subagent": { "thread_spawn": ... } }` source. Unknown structured
/// variants remain unset rather than inventing a classification.
fn codex_entrypoint(source: Option<&Value>) -> Option<String> {
    match source? {
        Value::String(source) => {
            let source = source.trim();
            if source.is_empty() {
                None
            } else {
                Some(match source {
                    "cli" => "codex-cli".to_string(),
                    "vscode" => "codex-vscode".to_string(),
                    other => format!("codex-{other}"),
                })
            }
        }
        Value::Object(source)
            if source
                .get("subagent")
                .and_then(Value::as_object)
                .and_then(|subagent| subagent.get("thread_spawn"))
                .is_some_and(Value::is_object) =>
        {
            Some("codex-subagent".to_string())
        }
        _ => None,
    }
}

/// Lightweight metadata used by project-level scans.
pub(crate) struct ProjectScanInfo {
    pub(crate) cwd: Option<String>,
    pub(crate) message_count: usize,
    pub(crate) last_modified: String,
}

/// Scan Codex projects from a specific base path.
pub fn scan_projects_from_path(base_path: &str) -> Result<Vec<ClaudeProject>, String> {
    crate::utils::require_absolute_path(base_path, "Codex base path")?;
    let base = Path::new(base_path);

    let sessions_dir = base.join("sessions");
    let archived_sessions_dir = base.join("archived_sessions");

    let session_dirs: Vec<PathBuf> = [sessions_dir, archived_sessions_dir]
        .into_iter()
        .filter(|path| {
            std::fs::symlink_metadata(path)
                .map(|m| m.file_type().is_dir())
                .unwrap_or(false)
        })
        .collect();

    if session_dirs.is_empty() {
        return Ok(vec![]);
    }

    // Group sessions by cwd
    let mut project_map: HashMap<String, Vec<ProjectScanInfo>> = HashMap::new();

    for session_dir in session_dirs {
        for entry in WalkDir::new(session_dir)
            .min_depth(1)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| is_discoverable_rollout(e.path()))
        {
            let rollout_path = entry.path();

            if let Ok(info) = extract_project_scan_info(rollout_path) {
                let cwd = info.cwd.clone().unwrap_or_else(|| "unknown".to_string());
                project_map.entry(cwd).or_default().push(info);
            }
        }
    }

    let mut projects: Vec<ClaudeProject> = project_map
        .into_iter()
        .map(|(cwd, sessions)| {
            let name = Path::new(&cwd)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| cwd.clone());

            let session_count = sessions.len();
            let message_count: usize = sessions.iter().map(|s| s.message_count).sum();
            let last_modified = sessions
                .iter()
                .map(|s| s.last_modified.as_str())
                .max()
                .unwrap_or("")
                .to_string();

            ClaudeProject {
                name,
                path: format!("codex://{cwd}"),
                actual_path: cwd,
                session_count,
                message_count,
                last_modified,
                git_info: None,
                provider: Some("codex".to_string()),
                storage_type: None,
                custom_directory_label: None,
            }
        })
        .collect();

    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// Scan Codex projects from the default location.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let base = get_base_path().ok_or("Codex base path not found")?;
    scan_projects_from_path(&base)
}

/// Load sessions for a Codex project (filtered by cwd)
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let session_dirs = get_existing_session_dirs()?;
    let title_index = get_base_path()
        .map(|base_path| load_native_title_index(&base_path))
        .unwrap_or_default();

    if session_dirs.is_empty() {
        return Ok(vec![]);
    }

    // Extract cwd from virtual path "codex://{cwd}"
    let target_cwd = project_path
        .strip_prefix("codex://")
        .unwrap_or(project_path);

    let mut sessions = Vec::new();

    for session_dir in session_dirs {
        for entry in WalkDir::new(session_dir)
            .min_depth(1)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| is_discoverable_rollout(e.path()))
        {
            let rollout_path = entry.path();

            match extract_session_cwd(rollout_path) {
                Ok(Some(session_cwd)) if session_cwd != target_cwd => continue,
                Ok(_) | Err(_) => {}
            }

            if let Ok(info) = extract_session_info(rollout_path) {
                let native_title = title_index.get(&info.session_id);
                let session_cwd = info.cwd.as_deref().unwrap_or("unknown");
                if session_cwd != target_cwd {
                    continue;
                }

                sessions.push(ClaudeSession {
                    session_id: info.file_path.clone(),
                    actual_session_id: info.session_id,
                    file_path: info.file_path,
                    project_name: Path::new(target_cwd)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    message_count: info.message_count,
                    first_message_time: info.first_message_time,
                    last_message_time: info.last_message_time,
                    last_modified: info.last_modified,
                    has_tool_use: info.has_tool_use,
                    has_errors: false,
                    summary: native_title
                        .map(|native| native.title.clone())
                        .or(info.summary),
                    is_renamed: native_title.is_some_and(|native| native.is_renamed),
                    provider: Some("codex".to_string()),
                    storage_type: None,
                    entrypoint: info.entrypoint,
                    forked_from_id: info.forked_from_id,
                });
            }
        }
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Load all messages from a Codex rollout file
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let path = Path::new(session_path);
    if !path.exists() {
        return Err(format!("Session file not found: {session_path}"));
    }
    let canonical_path = validate_session_path(path, session_path)?;
    parse_rollout_file(&canonical_path)
}

/// Parse an already-validated Codex rollout JSONL file into messages. Pure of
/// base-path/scheme concerns so providers sharing the identical rollout format
/// (e.g. Open Interpreter) can validate against their own root, call this, and
/// re-tag the provider on the result.
#[allow(unsafe_code)] // Required for mmap performance optimization
pub(crate) fn parse_rollout_file(canonical_path: &Path) -> Result<Vec<ClaudeMessage>, String> {
    let mmap = read_rollout_bytes(canonical_path)?;
    let ranges = find_line_ranges(&mmap);
    let state = CodexParserState::initial(canonical_path);
    let checkpoint = CodexParserCheckpoint {
        byte_offset: 0,
        replace_from: 0,
        state: state.clone(),
    };
    parse_rollout_slice(&mmap, &ranges, state, checkpoint, false)
        .map(|outcome| outcome.messages)
        .map_err(|()| "Codex rollout unexpectedly referenced an earlier prefix".to_string())
}

fn parse_rollout_slice(
    bytes: &[u8],
    ranges: &[(usize, usize)],
    mut state: CodexParserState,
    mut checkpoint: CodexParserCheckpoint,
    resumed: bool,
) -> Result<CodexParseOutcome, ()> {
    let mut messages = Vec::new();
    let slice_base_replace_from = checkpoint.replace_from;
    let mut accepted_len = usize::try_from(checkpoint.byte_offset).map_err(|_| ())?;
    let mut active_turn_id: Option<String> = None;
    let mut active_turn_message_start = 0usize;
    let mut authored_user_messages_in_turn = 0usize;
    let mut pending_user_message_index: Option<usize> = None;
    let mut pending_compacted_notification = false;

    for (range_index, &(start, end)) in ranges.iter().enumerate() {
        if start < usize::try_from(checkpoint.byte_offset).map_err(|_| ())? {
            continue;
        }
        let line = &bytes[start..end];
        let mut buf = line.to_vec();
        let val: Value = match simd_json::from_slice(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        accepted_len = if bytes.get(end) == Some(&b'\n') {
            end + 1
        } else {
            end
        };
        let line_timestamp = val
            .get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let line_type = val.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let event_type = if line_type == "event_msg" {
            val.get("payload")
                .and_then(|payload| payload.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("")
        } else {
            ""
        };

        // Current Codex rollouts persist one logical compaction twice: the
        // authoritative `compacted` record carries replacement history, then
        // `context_compacted` announces that same event after bookkeeping.
        // Retain a standalone notification, but suppress the companion while
        // the exact provider sequence remains intact.
        if pending_compacted_notification
            && !matches!(line_type, "world_state" | "turn_context" | "compacted")
            && !(line_type == "event_msg"
                && matches!(event_type, "token_count" | "context_compacted"))
        {
            pending_compacted_notification = false;
        }

        match line_type {
            // First session_meta only — later ones are history replayed by
            // `codex fork` and must not re-tag messages with the source's id.
            "session_meta" if !state.meta_seen => {
                state.meta_seen = true;
                if let Some(payload) = val.get("payload") {
                    state.session_id = payload
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                }
            }
            "turn_context" => {
                if let Some(payload) = val.get("payload") {
                    if let Some(m) = payload.get("model").and_then(|v| v.as_str()) {
                        state.current_inference.model = Some(m.to_string());
                    }
                    if let Some(effort) = payload.get("effort").and_then(Value::as_str) {
                        state.current_inference.reasoning_effort = Some(effort.to_string());
                    }
                    // `thread_settings_applied.reasoning_summary` is the applied
                    // setting. Older rollouts may expose only this fallback.
                    if state.current_inference.reasoning_summary.is_none() {
                        if let Some(summary) = payload.get("summary").and_then(Value::as_str) {
                            state.current_inference.reasoning_summary = Some(summary.to_string());
                        }
                    }
                }
            }
            "response_item" => {
                if let Some(payload) = val.get("payload") {
                    let is_user_message = payload.get("type").and_then(Value::as_str)
                        == Some("message")
                        && payload.get("role").and_then(Value::as_str) == Some("user");
                    if let Some(msg) = convert_codex_item(
                        payload,
                        &state.session_id,
                        state.current_inference.model.as_ref(),
                        &line_timestamp,
                        &mut state.msg_counter,
                    ) {
                        let mut msg = msg;
                        if msg.message_type == "assistant" {
                            msg.inference = (!state.current_inference.is_empty())
                                .then(|| state.current_inference.clone());
                        }
                        if try_merge_tool_result_into_previous(&mut messages, &msg) {
                            continue;
                        }
                        if resumed && extract_tool_result_block(&msg).is_some() {
                            // A fresh complete parse may merge this result into a
                            // tool call before the retained prefix. Never guess.
                            return Err(());
                        }
                        messages.push(msg);
                        if is_user_message && active_turn_id.is_some() {
                            // The matching `event_msg.user_message` immediately follows the
                            // persisted response item. Remember its normalized row so that the
                            // event stream can classify authored input without mistaking
                            // injected user-role context for a steer.
                            pending_user_message_index = Some(messages.len() - 1);
                        }
                    }
                }
            }
            "event_msg" => {
                if let Some(payload) = val.get("payload") {
                    if event_type == "context_compacted" && pending_compacted_notification {
                        pending_compacted_notification = false;
                        continue;
                    }

                    // Skip events that duplicate response_item messages.
                    // Codex logs user/assistant text in both response_item (type=message)
                    // and event_msg (type=user_message / agent_message). Keep the
                    // response_item version for content, but use the authored-user event to
                    // distinguish same-turn steering from injected user-role context.
                    if event_type == "user_message" {
                        if active_turn_id.is_some() {
                            if authored_user_messages_in_turn > 0 {
                                mark_pending_steer(&mut messages, &mut pending_user_message_index);
                            } else {
                                pending_user_message_index = None;
                            }
                            authored_user_messages_in_turn += 1;
                        }
                        continue;
                    }
                    if event_type == "agent_message" {
                        continue;
                    }

                    if event_type == "thread_settings_applied" {
                        if let Some(settings) = payload.get("thread_settings") {
                            state.current_inference = codex_inference_settings(settings);
                        }
                    }

                    if event_type == "task_started" {
                        active_turn_id = payload
                            .get("turn_id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        active_turn_message_start = messages.len();
                        state.current_inference.context_window = payload
                            .get("model_context_window")
                            .and_then(Value::as_u64)
                            .and_then(|value| u32::try_from(value).ok());
                        if let Some(mode) = payload
                            .get("collaboration_mode_kind")
                            .and_then(Value::as_str)
                        {
                            state.current_inference.interaction_mode = Some(mode.to_string());
                        }
                        authored_user_messages_in_turn = 0;
                        pending_user_message_index = None;
                    }

                    if event_type == "token_count" {
                        let usage_totals = extract_token_totals(payload);
                        let (usage, cumulative) = match usage_totals {
                            Some(usage) => (usage, true),
                            None => match extract_last_token_usage(payload) {
                                Some(usage) => (usage, false),
                                None => continue,
                            },
                        };

                        let delta = if cumulative {
                            let delta_optional = |current: Option<u32>, previous: &mut u32| {
                                current.map(|value| {
                                    let delta = value.saturating_sub(*previous);
                                    *previous = value;
                                    delta
                                })
                            };
                            let delta = CodexTokenUsage {
                                input: usage.input.saturating_sub(state.prev_input_tokens),
                                output: usage.output.saturating_sub(state.prev_output_tokens),
                                cached: delta_optional(usage.cached, &mut state.prev_cached_tokens),
                                cache_write: delta_optional(
                                    usage.cache_write,
                                    &mut state.prev_cache_write_tokens,
                                ),
                                reasoning: delta_optional(
                                    usage.reasoning,
                                    &mut state.prev_reasoning_tokens,
                                ),
                            };
                            state.prev_input_tokens = usage.input;
                            state.prev_output_tokens = usage.output;
                            delta
                        } else {
                            usage
                        };

                        // Apply to the last assistant message without usage.
                        if resumed && active_turn_id.is_none() {
                            return Err(());
                        }
                        let Some(last_msg) = messages[active_turn_message_start..]
                            .iter_mut()
                            .rev()
                            .find(|message| {
                                message.message_type == "assistant" && message.usage.is_none()
                            })
                        else {
                            continue;
                        };

                        // Separate non-cached input from cached input for correct billing.
                        // OpenAI's input_tokens includes cached_input_tokens as a subset,
                        // but they are billed at different rates (cached gets 90% discount).
                        let non_cached_input =
                            delta.input.saturating_sub(delta.cached.unwrap_or(0));
                        last_msg.usage = Some(TokenUsage {
                            input_tokens: Some(non_cached_input),
                            output_tokens: Some(delta.output),
                            cache_creation_input_tokens: delta.cache_write,
                            cache_read_input_tokens: delta.cached,
                            service_tier: state.current_inference.service_tier.clone(),
                        });
                        let inference = last_msg
                            .inference
                            .get_or_insert_with(|| state.current_inference.clone());
                        inference.usage = Some(InferenceUsage {
                            input_tokens: Some(delta.input),
                            output_tokens: Some(delta.output),
                            cached_input_tokens: delta.cached,
                            cache_write_input_tokens: delta.cache_write,
                            reasoning_output_tokens: delta.reasoning,
                        });
                    } else if let Some(mut msg) = convert_codex_event(
                        payload,
                        &state.session_id,
                        &line_timestamp,
                        &mut state.msg_counter,
                    ) {
                        if msg.message_type == "assistant" {
                            if msg.model.is_none() {
                                msg.model.clone_from(&state.current_inference.model);
                            }
                            msg.inference = (!state.current_inference.is_empty())
                                .then(|| state.current_inference.clone());
                        }
                        messages.push(msg);
                    }

                    if event_type == "task_complete"
                        && payload.get("turn_id").and_then(Value::as_str)
                            == active_turn_id.as_deref()
                    {
                        if let Some(last_msg) = messages[active_turn_message_start..]
                            .iter_mut()
                            .rev()
                            .find(|message| message.message_type == "assistant")
                        {
                            let inference = last_msg
                                .inference
                                .get_or_insert_with(|| state.current_inference.clone());
                            inference.duration_ms =
                                payload.get("duration_ms").and_then(Value::as_u64);
                            inference.time_to_first_token_ms = payload
                                .get("time_to_first_token_ms")
                                .and_then(Value::as_u64);
                        }
                        active_turn_id = None;
                        authored_user_messages_in_turn = 0;
                        pending_user_message_index = None;
                        let next_offset = ranges
                            .get(range_index + 1)
                            .map_or(bytes.len(), |&(next_start, _)| next_start);
                        checkpoint = CodexParserCheckpoint {
                            byte_offset: u64::try_from(next_offset).map_err(|_| ())?,
                            replace_from: slice_base_replace_from + messages.len(),
                            state: state.clone(),
                        };
                    }
                }
            }
            "compacted" => {
                if let Some(payload) = val.get("payload") {
                    let msg = convert_codex_compacted(
                        payload,
                        &state.session_id,
                        &line_timestamp,
                        &mut state.msg_counter,
                    );
                    messages.push(msg);
                    pending_compacted_notification = true;
                }
            }
            _ => {}
        }
    }

    Ok(CodexParseOutcome {
        messages,
        checkpoint,
        accepted_len,
    })
}

fn digest_bytes(bytes: &[u8]) -> String {
    BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(bytes))
}

fn encode_snapshot_cursor(cursor: &CodexSnapshotCursor) -> Result<String, String> {
    serde_json::to_vec(cursor)
        .map(|bytes| BASE64_URL_SAFE_NO_PAD.encode(bytes))
        .map_err(|error| format!("Failed to encode Codex snapshot cursor: {error}"))
}

fn decode_snapshot_cursor(encoded: &str) -> Result<CodexSnapshotCursor, String> {
    const MAX_CURSOR_BYTES: usize = 64 * 1024;
    if encoded.len() > MAX_CURSOR_BYTES {
        return Err("Codex snapshot cursor is too large".to_string());
    }
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|error| format!("Invalid Codex snapshot cursor encoding: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid Codex snapshot cursor payload: {error}"))
}

/** Authoritative normalized replacement boundary carried by a provider cursor. */
pub(crate) fn snapshot_cursor_replace_from(encoded: &str) -> Result<usize, String> {
    Ok(decode_snapshot_cursor(encoded)?.checkpoint.replace_from)
}

fn cursor_for(
    canonical_path: &Path,
    bytes: &[u8],
    checkpoint: CodexParserCheckpoint,
) -> Result<String, String> {
    encode_snapshot_cursor(&CodexSnapshotCursor {
        version: SNAPSHOT_CURSOR_VERSION,
        provider: "codex".to_string(),
        canonical_path: canonical_path.to_string_lossy().into_owned(),
        accepted_len: u64::try_from(bytes.len())
            .map_err(|_| "Codex rollout is too large to cursor".to_string())?,
        accepted_digest: digest_bytes(bytes),
        checkpoint,
    })
}

fn is_plain_rollout(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
}

#[allow(unsafe_code)] // Required for mmap performance optimization
fn map_plain_rollout(path: &Path) -> Result<(Mmap, std::fs::Metadata), String> {
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

fn complete_snapshot_from_path(
    canonical_path: &Path,
    reason: impl Into<String>,
) -> Result<SessionSnapshotLoad, String> {
    let reason = reason.into();
    if !is_plain_rollout(canonical_path) {
        return Ok(SessionSnapshotLoad::Full {
            reason,
            messages: finalize_loaded_messages(parse_rollout_file(canonical_path)?),
            cursor: None,
            cursor_replace_from: None,
        });
    }

    let (mmap, before) = map_plain_rollout(canonical_path)?;
    let ranges = find_line_ranges(&mmap);
    let state = CodexParserState::initial(canonical_path);
    let checkpoint = CodexParserCheckpoint {
        byte_offset: 0,
        replace_from: 0,
        state: state.clone(),
    };
    let outcome = parse_rollout_slice(&mmap, &ranges, state, checkpoint, false)
        .map_err(|()| "Codex complete parse unexpectedly crossed its prefix".to_string())?;
    let after = fs::metadata(canonical_path).map_err(|error| error.to_string())?;
    let original_len = outcome.messages.len();
    let messages = finalize_loaded_messages(outcome.messages);
    let cursor =
        if messages.len() == original_len && source_stayed_stable(&before, &after, mmap.len()) {
            Some(cursor_for(
                canonical_path,
                &mmap[..outcome.accepted_len],
                outcome.checkpoint.clone(),
            )?)
        } else {
            None
        };

    Ok(SessionSnapshotLoad::Full {
        reason,
        messages,
        cursor_replace_from: cursor
            .as_deref()
            .map(snapshot_cursor_replace_from)
            .transpose()?,
        cursor,
    })
}

fn cursor_checkpoint_is_valid(cursor: &CodexSnapshotCursor, bytes: &[u8]) -> bool {
    let Ok(offset) = usize::try_from(cursor.checkpoint.byte_offset) else {
        return false;
    };
    let Ok(accepted_len) = usize::try_from(cursor.accepted_len) else {
        return false;
    };
    if offset > accepted_len || accepted_len > bytes.len() {
        return false;
    }
    offset == 0
        || offset == accepted_len
        || bytes
            .get(offset.wrapping_sub(1))
            .is_some_and(|byte| *byte == b'\n')
}

/// Load a cursor-aware normalized Codex snapshot.
///
/// A valid cursor proves the complete previously accepted byte prefix before
/// the parser resumes from its provider-owned completed-turn checkpoint. Any
/// uncertainty returns the ordinary complete result instead.
pub(crate) fn load_session_snapshot(
    session_path: &str,
    encoded_cursor: Option<&str>,
) -> Result<SessionSnapshotLoad, String> {
    let path = Path::new(session_path);
    if !path.exists() {
        return Err(format!("Session file not found: {session_path}"));
    }
    let canonical_path = validate_session_path(path, session_path)?;

    let Some(encoded_cursor) = encoded_cursor else {
        return complete_snapshot_from_path(&canonical_path, "initial");
    };
    if !is_plain_rollout(&canonical_path) {
        return complete_snapshot_from_path(&canonical_path, "unsupported-source");
    }

    let cursor = match decode_snapshot_cursor(encoded_cursor) {
        Ok(cursor) => cursor,
        Err(_) => return complete_snapshot_from_path(&canonical_path, "invalid-cursor"),
    };
    if cursor.version != SNAPSHOT_CURSOR_VERSION
        || cursor.provider != "codex"
        || cursor.canonical_path != canonical_path.to_string_lossy()
    {
        return complete_snapshot_from_path(&canonical_path, "incompatible-cursor");
    }

    let (mmap, before) = map_plain_rollout(&canonical_path)?;
    let Ok(accepted_len) = usize::try_from(cursor.accepted_len) else {
        return complete_snapshot_from_path(&canonical_path, "invalid-cursor");
    };
    if accepted_len > mmap.len() {
        return complete_snapshot_from_path(&canonical_path, "source-shrank");
    }
    if !cursor_checkpoint_is_valid(&cursor, &mmap) {
        return complete_snapshot_from_path(&canonical_path, "invalid-checkpoint");
    }

    let mut hasher = Sha256::new();
    hasher.update(&mmap[..accepted_len]);
    let accepted_digest = BASE64_URL_SAFE_NO_PAD.encode(hasher.clone().finalize());
    if accepted_digest != cursor.accepted_digest {
        return complete_snapshot_from_path(&canonical_path, "prefix-mismatch");
    }

    if accepted_len == mmap.len() {
        let after = fs::metadata(&canonical_path).map_err(|error| error.to_string())?;
        if source_stayed_stable(&before, &after, mmap.len()) {
            return Ok(SessionSnapshotLoad::Unchanged {
                cursor: encoded_cursor.to_string(),
            });
        }
        return complete_snapshot_from_path(&canonical_path, "source-changed-during-read");
    }

    let replace_from = cursor.checkpoint.replace_from;
    let ranges = find_line_ranges(&mmap);
    let outcome = match parse_rollout_slice(
        &mmap,
        &ranges,
        cursor.checkpoint.state.clone(),
        cursor.checkpoint.clone(),
        true,
    ) {
        Ok(outcome) => outcome,
        Err(()) => {
            return complete_snapshot_from_path(&canonical_path, "unsafe-backward-reference");
        }
    };
    let after = fs::metadata(&canonical_path).map_err(|error| error.to_string())?;
    if !source_stayed_stable(&before, &after, mmap.len()) {
        return complete_snapshot_from_path(&canonical_path, "source-changed-during-read");
    }

    if outcome.accepted_len == accepted_len {
        return Ok(SessionSnapshotLoad::Unchanged {
            cursor: encoded_cursor.to_string(),
        });
    }

    hasher.update(&mmap[accepted_len..outcome.accepted_len]);
    let full_digest = BASE64_URL_SAFE_NO_PAD.encode(hasher.finalize());
    let cursor_replace_from = outcome.checkpoint.replace_from;
    let original_len = outcome.messages.len();
    let messages = finalize_loaded_messages(outcome.messages);
    if messages.len() != original_len {
        return complete_snapshot_from_path(&canonical_path, "post-normalization-count-changed");
    }
    let next_cursor = encode_snapshot_cursor(&CodexSnapshotCursor {
        version: SNAPSHOT_CURSOR_VERSION,
        provider: "codex".to_string(),
        canonical_path: canonical_path.to_string_lossy().into_owned(),
        accepted_len: u64::try_from(outcome.accepted_len)
            .map_err(|_| "Codex rollout is too large to cursor".to_string())?,
        accepted_digest: full_digest,
        checkpoint: outcome.checkpoint,
    })?;

    Ok(SessionSnapshotLoad::Replace {
        replace_from,
        messages,
        cursor_replace_from,
        cursor: next_cursor,
    })
}

/// Search Codex sessions for a query string
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let session_dirs = get_existing_session_dirs()?;

    if session_dirs.is_empty() {
        return Ok(vec![]);
    }

    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for session_dir in session_dirs {
        for entry in WalkDir::new(session_dir)
            .min_depth(1)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter(|e| is_discoverable_rollout(e.path()))
        {
            let rollout_path = entry.path();

            if let Ok(messages) = load_messages(&rollout_path.to_string_lossy()) {
                for msg in messages {
                    if results.len() >= limit {
                        return Ok(results);
                    }

                    if let Some(content) = &msg.content {
                        if search_json_value_case_insensitive(content, &query_lower) {
                            results.push(msg);
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Rename a Codex CLI session by updating its native thread title in
/// `state_5.sqlite`. Codex stores the authoritative, resume-picker-visible
/// name in `threads.title`; the rollout JSONL remains the immutable transcript.
pub fn rename_session_title(
    session_path: &str,
    new_title: &str,
) -> Result<NativeRenameResult, String> {
    let base_path = get_base_path().ok_or_else(|| "Codex not found".to_string())?;
    rename_session_title_from_path(&base_path, session_path, new_title)
}

fn rename_session_title_from_path(
    base_path: &str,
    session_path: &str,
    new_title: &str,
) -> Result<NativeRenameResult, String> {
    let canonical_path = validate_session_path(Path::new(session_path), session_path)?;
    if !is_rollout_jsonl(&canonical_path) {
        return Err(format!("Invalid Codex rollout path: {session_path}"));
    }

    let info = extract_session_info(&canonical_path)?;
    if info.session_id.is_empty() {
        return Err("Codex rollout is missing session metadata id".to_string());
    }

    let conn = open_state_db_read_write(base_path)?;
    let (previous_title, first_user_message): (String, String) = conn
        .query_row(
            "SELECT title, first_user_message FROM threads WHERE id = ?1",
            rusqlite::params![&info.session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| {
            if matches!(e, rusqlite::Error::QueryReturnedNoRows) {
                format!(
                    "Codex thread not found in state database: {}",
                    info.session_id
                )
            } else {
                format!("Failed to read Codex thread metadata: {e}")
            }
        })?;

    let normalized_title = new_title.trim();
    if normalized_title.chars().any(|ch| ch == '\n' || ch == '\r') {
        return Err("Invalid title: Title cannot contain newline characters".to_string());
    }

    let reset_title = if first_user_message.trim().is_empty() {
        info.summary.clone().unwrap_or_default()
    } else {
        first_user_message
    };
    let next_title = if normalized_title.is_empty() {
        reset_title
    } else {
        normalized_title.to_string()
    };

    let affected_rows = conn
        .execute(
            "UPDATE threads SET title = ?1 WHERE id = ?2",
            rusqlite::params![&next_title, &info.session_id],
        )
        .map_err(|e| format!("Failed to rename Codex session: {e}"))?;

    if affected_rows == 0 {
        return Err(format!(
            "Codex thread not found in state database: {}",
            info.session_id
        ));
    }

    Ok(NativeRenameResult {
        success: true,
        previous_title,
        new_title: next_title,
        file_path: session_path.to_string(),
    })
}

/// Best-effort removal of a Codex session's `threads` row from `state_5.sqlite`
/// when the session is deleted, so a native-rename title (see
/// `rename_session_title`) does not linger as an orphaned row after the rollout
/// transcript is gone. Must be called BEFORE the rollout file is trashed — the
/// session id is read from the rollout itself.
///
/// Returns `Ok(())` when there is nothing to clean up (no state database, or no
/// matching row); only a genuine DB/IO failure is an `Err`.
pub fn delete_session_title(session_path: &str) -> Result<(), String> {
    let base_path = get_base_path().ok_or_else(|| "Codex not found".to_string())?;
    let canonical_path = validate_session_path(Path::new(session_path), session_path)?;
    if !is_rollout_jsonl(&canonical_path) {
        return Err(format!("Invalid Codex rollout path: {session_path}"));
    }

    // No state database means there is no native title to clean up.
    if !state_db_path(&base_path).is_file() {
        return Ok(());
    }

    let info = extract_session_info(&canonical_path)?;
    if info.session_id.is_empty() {
        return Ok(());
    }

    let conn = open_state_db_read_write(&base_path)?;
    conn.execute(
        "DELETE FROM threads WHERE id = ?1",
        rusqlite::params![&info.session_id],
    )
    .map_err(|e| format!("Failed to delete Codex thread row: {e}"))?;

    Ok(())
}

// ============================================================================
// Internal helpers
// ============================================================================

const JSON_TYPE_KEY: &[u8] = b"\"type\"";
fn skip_json_ws(bytes: &[u8], mut index: usize) -> usize {
    while bytes
        .get(index)
        .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
    {
        index += 1;
    }
    index
}

fn has_json_string_field_value(line: &[u8], key: &[u8], value: &[u8]) -> bool {
    let mut search_offset = 0;
    while search_offset < line.len() {
        let Some(relative_pos) = memmem::find(&line[search_offset..], key) else {
            return false;
        };

        let mut index = search_offset + relative_pos + key.len();
        index = skip_json_ws(line, index);
        if line.get(index) != Some(&b':') {
            search_offset = index.min(line.len());
            continue;
        }

        index = skip_json_ws(line, index + 1);
        if line.get(index) != Some(&b'"') {
            search_offset = index.min(line.len());
            continue;
        }

        let value_start = index + 1;
        let value_end = value_start + value.len();
        if line.get(value_start..value_end) == Some(value) && line.get(value_end) == Some(&b'"') {
            return true;
        }

        search_offset = value_end.min(line.len());
    }

    false
}

fn for_each_jsonl_line(data: &[u8], mut visit: impl FnMut(&[u8]) -> bool) {
    let mut start = 0;
    for end in memchr_iter(b'\n', data) {
        let line_end = if end > start && data[end - 1] == b'\r' {
            end - 1
        } else {
            end
        };
        if !visit(&data[start..line_end]) {
            return;
        }
        start = end + 1;
    }

    if start < data.len() {
        let line = &data[start..];
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        visit(line);
    }
}

fn parse_session_meta_cwd(line: &[u8]) -> Option<String> {
    if !has_json_string_field_value(line, JSON_TYPE_KEY, b"session_meta") {
        return None;
    }

    let mut buf = line.to_vec();
    let val: Value = simd_json::from_slice(&mut buf).ok()?;
    if val.get("type").and_then(|t| t.as_str()) != Some("session_meta") {
        return None;
    }

    val.get("payload")?
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// cwd from a `turn_context` line, the fallback identity source for
/// rollouts that carry no `session_meta` at all (issue #451 follow-up).
fn parse_turn_context_cwd(line: &[u8]) -> Option<String> {
    if !has_json_string_field_value(line, JSON_TYPE_KEY, b"turn_context") {
        return None;
    }

    let mut buf = line.to_vec();
    let val: Value = simd_json::from_slice(&mut buf).ok()?;
    if val.get("type").and_then(|t| t.as_str()) != Some("turn_context") {
        return None;
    }

    val.get("payload")?
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Session id derived from the rollout filename
/// (`rollout-<timestamp>-<uuid>.jsonl` → `<uuid>`); `None` when the stem
/// doesn't end in a UUID.
pub(crate) fn session_id_from_rollout_filename(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    // For "rollout-….jsonl.zst", file_stem still ends with ".jsonl".
    let stem = stem.strip_suffix(".jsonl").unwrap_or(stem);
    if stem.len() < 36 {
        return None;
    }
    let tail = &stem[stem.len() - 36..];
    let is_uuid = tail.bytes().enumerate().all(|(i, b)| match i {
        8 | 13 | 18 | 23 => b == b'-',
        _ => b.is_ascii_hexdigit(),
    });
    is_uuid.then(|| tail.to_string())
}

fn file_modified_rfc3339(path: &Path) -> String {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let dt: DateTime<Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_else(|| Utc::now().to_rfc3339())
}

fn estimate_rollout_message_count(path: &Path) -> usize {
    fs::metadata(path)
        .map(|metadata| estimate_message_count_from_size(metadata.len()))
        .unwrap_or(0)
}

#[allow(unsafe_code)] // Required for mmap performance optimization
pub(crate) fn extract_session_cwd(rollout_path: &Path) -> Result<Option<String>, String> {
    let mmap = read_rollout_bytes(rollout_path)?;

    let mut cwd = None;
    let mut turn_context_cwd = None;
    for_each_jsonl_line(&mmap, |line| {
        if let Some(found) = parse_session_meta_cwd(line) {
            cwd = Some(found);
            return false;
        }
        // Fallback for rollouts without any session_meta: the LAST
        // turn_context's cwd is where the session actually runs (a fork
        // replays the source's turn contexts first) — issue #451 follow-up.
        if let Some(found) = parse_turn_context_cwd(line) {
            turn_context_cwd = Some(found);
        }
        true
    });

    Ok(cwd.or(turn_context_cwd))
}

pub(crate) fn extract_project_scan_info(rollout_path: &Path) -> Result<ProjectScanInfo, String> {
    Ok(ProjectScanInfo {
        cwd: extract_session_cwd(rollout_path)?,
        // Project list scans stay lightweight; session-level message counts
        // are still computed exactly when the project is opened.
        message_count: estimate_rollout_message_count(rollout_path),
        last_modified: file_modified_rfc3339(rollout_path),
    })
}

fn state_db_path(base_path: &str) -> PathBuf {
    Path::new(base_path).join(STATE_DB_FILENAME)
}

fn open_state_db(base_path: &str) -> Option<Connection> {
    let db_path = state_db_path(base_path);
    if !db_path.is_file() {
        return None;
    }

    Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()
}

fn open_state_db_read_write(base_path: &str) -> Result<Connection, String> {
    let db_path = state_db_path(base_path);
    if !db_path.is_file() {
        return Err(format!(
            "Codex state database not found: {}",
            db_path.display()
        ));
    }

    Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("Failed to open Codex state database: {e}"))
}

fn load_native_title_index(base_path: &str) -> HashMap<String, NativeTitle> {
    let mut sqlite_titles = open_state_db(base_path)
        .and_then(|conn| load_sqlite_titles(&conn))
        .unwrap_or_default();
    let mut indexed_names = load_session_index_names(base_path);
    let mut titles = HashMap::with_capacity(sqlite_titles.len() + indexed_names.len());

    for (id, sqlite) in sqlite_titles.drain() {
        let stored_title = sqlite.title.trim();
        let preview = sqlite.preview.trim();
        let indexed_name = indexed_names.remove(&id);
        // The Codex extension sends both its initial generated title and manual
        // edits through `thread/name/set`. A single index name identical to the
        // SQLite title is therefore generated-title metadata, not rename
        // provenance. A non-reset name is explicit only when it differs from the
        // SQLite baseline or the append-only name history has changed.
        let is_renamed = indexed_name.as_ref().is_some_and(|indexed| {
            let name = indexed.latest.trim();
            name != preview && (name != stored_title || indexed.changed)
        });
        // Match Codex's LocalThreadStore precedence: a title distinct from the
        // preview is authoritative SQLite metadata; otherwise fall back to the
        // append-only compatibility index (latest entry wins).
        let resolved = if !stored_title.is_empty() && stored_title != preview {
            stored_title.to_string()
        } else if let Some(indexed) = indexed_name {
            indexed.latest
        } else if !preview.is_empty() {
            preview.to_string()
        } else if !stored_title.is_empty() {
            stored_title.to_string()
        } else {
            continue;
        };
        titles.insert(
            id,
            NativeTitle {
                title: resolved,
                is_renamed,
            },
        );
    }

    // A missing/unreadable SQLite row must not hide an explicitly indexed name.
    // Without a preview there is no reliable reset comparison, so a non-empty
    // index-only name is conservatively marked as renamed.
    for (id, indexed) in indexed_names {
        titles.insert(
            id,
            NativeTitle {
                title: indexed.latest,
                is_renamed: true,
            },
        );
    }

    titles
}

fn load_sqlite_titles(conn: &Connection) -> Option<HashMap<String, SqliteTitle>> {
    // `preview` is Codex's current user-facing original title. Fall back to the
    // older `first_user_message` column so pre-preview databases remain readable.
    query_sqlite_titles(conn, "preview").or_else(|| query_sqlite_titles(conn, "first_user_message"))
}

fn query_sqlite_titles(
    conn: &Connection,
    preview_column: &str,
) -> Option<HashMap<String, SqliteTitle>> {
    let sql = format!("SELECT id, title, {preview_column} FROM threads");
    let mut stmt = conn.prepare(&sql).ok()?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .ok()?;
    Some(
        rows.filter_map(std::result::Result::ok)
            .map(|(id, title, preview)| (id, SqliteTitle { title, preview }))
            .collect(),
    )
}

fn load_session_index_names(base_path: &str) -> HashMap<String, IndexedName> {
    let path = Path::new(base_path).join(SESSION_INDEX_FILENAME);
    let Ok(body) = fs::read_to_string(path) else {
        return HashMap::new();
    };
    let mut names = HashMap::<String, IndexedName>::new();
    for line in body.lines() {
        let Ok(entry) = serde_json::from_str::<SessionIndexEntry>(line.trim()) else {
            continue;
        };
        let name = entry.thread_name.trim();
        if entry.id.trim().is_empty() || name.is_empty() {
            continue;
        }
        match names.get_mut(&entry.id) {
            Some(indexed) => {
                indexed.changed |= indexed.latest != name;
                indexed.latest = name.to_string();
            }
            None => {
                names.insert(
                    entry.id,
                    IndexedName {
                        latest: name.to_string(),
                        changed: false,
                    },
                );
            }
        }
    }
    names
}

#[allow(unsafe_code)] // Required for mmap performance optimization
pub(crate) fn extract_session_info(rollout_path: &Path) -> Result<SessionInfo, String> {
    let mmap = read_rollout_bytes(rollout_path)?;
    let ranges = find_line_ranges(&mmap);

    let mut session_id = String::new();
    let mut meta_seen = false;
    let mut cwd = None;
    let mut source = None;
    let mut forked_from_id = None;
    let mut turn_context_cwd = None;
    let mut model = None;
    let mut message_count = 0usize;
    let mut first_time = String::new();
    let mut last_time = String::new();
    let mut has_tool_use = false;
    let mut summary = None;

    for &(start, end) in &ranges {
        let line = &mmap[start..end];
        let mut buf = line.to_vec();
        let val: Value = match simd_json::from_slice(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let line_type = val.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match line_type {
            // Only the FIRST session_meta identifies the file. `codex fork`
            // replays the source rollout verbatim into the new file, so a
            // forked rollout contains the source's session_meta as history
            // after its own — taking the last one misfiles the session under
            // the source cwd (issue #451).
            "session_meta" if !meta_seen => {
                meta_seen = true;
                if let Some(payload) = val.get("payload") {
                    session_id = payload
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    cwd = payload
                        .get("cwd")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    source = payload.get("source").cloned();
                    forked_from_id = payload
                        .get("forked_from_id")
                        .and_then(|v| v.as_str())
                        .map(str::trim)
                        .filter(|id| !id.is_empty())
                        .map(String::from);
                }
            }
            "turn_context" => {
                if let Some(payload) = val.get("payload") {
                    if model.is_none() {
                        model = payload
                            .get("model")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                    }
                    // Last turn_context wins — the fallback cwd for
                    // rollouts without any session_meta (issue #451).
                    if let Some(tc_cwd) = payload.get("cwd").and_then(|v| v.as_str()) {
                        turn_context_cwd = Some(tc_cwd.to_string());
                    }
                }
            }
            "response_item" => {
                if let Some(payload) = val.get("payload") {
                    let item_type = payload.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if item_type == "message" {
                        message_count += 1;

                        let ts = payload
                            .get("created_at")
                            .or_else(|| val.get("timestamp"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if first_time.is_empty() && !ts.is_empty() {
                            first_time.clone_from(&ts);
                        }
                        if !ts.is_empty() {
                            last_time.clone_from(&ts);
                        }

                        // Extract first user message as summary, skipping
                        // auto-injected wrapper blocks (e.g. <environment_context>)
                        // that codex CLI / Codex Desktop prepend to every session —
                        // they are system context, not a real user prompt.
                        if summary.is_none() {
                            if let Some(role) = payload.get("role").and_then(|r| r.as_str()) {
                                if role == "user" {
                                    if let Some(text) = extract_text_from_content(payload) {
                                        if !is_codex_auto_injected_user_text(&text) {
                                            summary = Some(text);
                                        }
                                    }
                                }
                            }
                        }
                    } else if item_type == "local_shell_call"
                        || item_type == "function_call"
                        || item_type == "custom_tool_call"
                        || item_type == "web_search_call"
                    {
                        has_tool_use = true;
                        message_count += 1;
                    } else if item_type == "function_call_output"
                        || item_type == "custom_tool_call_output"
                    {
                        message_count += 1;
                    }
                }
            }
            _ => {}
        }
    }

    let last_modified = if last_time.is_empty() {
        file_modified_rfc3339(rollout_path)
    } else {
        last_time.clone()
    };

    // Meta-less rollout fallbacks (issue #451 follow-up): session id from
    // the filename, cwd from the last turn_context.
    if session_id.is_empty() {
        if let Some(id) = session_id_from_rollout_filename(rollout_path) {
            session_id = id;
        }
    }
    if cwd.is_none() {
        cwd = turn_context_cwd;
    }

    Ok(SessionInfo {
        session_id,
        cwd,
        entrypoint: codex_entrypoint(source.as_ref()),
        forked_from_id,
        model,
        message_count,
        first_message_time: first_time,
        last_message_time: last_time,
        last_modified,
        file_path: rollout_path.to_string_lossy().to_string(),
        has_tool_use,
        summary,
    })
}

fn extract_text_from_content(item: &Value) -> Option<String> {
    let content = item.get("content")?.as_array()?;
    for c in content {
        let ctype = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if ctype == "input_text" || ctype == "output_text" || ctype == "text" {
            if let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                let truncated = match text.char_indices().nth(200) {
                    Some((idx, _)) => format!("{}...", &text[..idx]),
                    None => text.to_string(),
                };
                return Some(truncated);
            }
        }
    }
    None
}

/// Returns true when `text` is an auto-injected wrapper block prepended by
/// codex CLI / Codex Desktop to every session (currently
/// `<environment_context>...</environment_context>`). These look like user
/// messages structurally but contain no real prompt, so they should be
/// skipped when picking a session summary preview.
fn is_codex_auto_injected_user_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("<environment_context>")
}

fn absolute_prompt_path_basename(raw_path: &str) -> Option<String> {
    let normalized = raw_path.replace('\\', "/");
    let absolute = normalized.starts_with('/')
        || normalized
            .as_bytes()
            .get(1)
            .is_some_and(|separator| *separator == b':')
            && normalized
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic)
            && normalized
                .as_bytes()
                .get(2)
                .is_some_and(|slash| *slash == b'/');
    if !absolute {
        return None;
    }
    normalized
        .rsplit('/')
        .find(|part| !part.is_empty())
        .map(str::to_string)
}

/// One entry from Codex's generated `# Files mentioned by the user` wrapper.
/// Requiring an absolute path keeps ordinary user-authored Markdown from
/// becoming attachment metadata; ordinary files add a basename/heading check
/// after pasted-note entries have been classified.
fn prompt_wrapper_entry(line: &str) -> Option<(&str, &str)> {
    let (label, raw_path) = line.strip_prefix("## ")?.rsplit_once(": ")?;
    let label = label.trim();
    let raw_path = raw_path.trim();
    if label.is_empty() || absolute_prompt_path_basename(raw_path).is_none() {
        return None;
    }
    Some((label, raw_path))
}

/// Whether a Codex wrapper entry is the provider's externalized pasted-text
/// carrier. This classification remains intentionally narrower than an
/// ordinary attached file: both the owned attachment directory and generated
/// filename must be present.
fn pasted_prompt_note(raw_path: &str) -> Option<Value> {
    let normalized = raw_path.replace('\\', "/");
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 3 || !parts[parts.len() - 3].eq_ignore_ascii_case("attachments") {
        return None;
    }
    let attachment_id = parts[parts.len() - 2];
    let name = parts[parts.len() - 1];
    let uuid_like = attachment_id.len() == 36
        && attachment_id.bytes().enumerate().all(|(i, b)| {
            matches!(i, 8 | 13 | 18 | 23) && b == b'-'
                || !matches!(i, 8 | 13 | 18 | 23) && b.is_ascii_hexdigit()
        });
    if !uuid_like
        || !name.to_ascii_lowercase().starts_with("pasted-text")
        || !name.to_ascii_lowercase().ends_with(".txt")
    {
        return None;
    }
    Some(serde_json::json!({
        "kind": "note",
        "name": name
    }))
}

/// Provider-neutral metadata for Codex's generated prompt-file carrier. Codex
/// persists both ordinary attached files and externalized pasted text inside an
/// exact Markdown wrapper rather than structured attachment blocks. Once the
/// complete wrapper is verified, retain only the authored request body and
/// surface safe structural artifacts.
struct CodexPromptArtifactCarrier {
    data: Value,
    input_text_index: usize,
    request_body: String,
}

fn codex_prompt_artifact_carrier(content: Option<&Value>) -> Option<CodexPromptArtifactCarrier> {
    let content = content?.as_array()?;
    let (input_text_index, text) = content.iter().enumerate().find_map(|(index, item)| {
        (item.get("type").and_then(Value::as_str) == Some("input_text"))
            .then(|| item.get("text")?.as_str().map(|text| (index, text)))
            .flatten()
    })?;

    let trimmed = text.trim_start();
    let leading_bytes = text.len() - trimmed.len();
    let mut lines = trimmed.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end() != "# Files mentioned by the user:" {
        return None;
    }

    let mut entries = Vec::new();
    let mut saw_instruction = false;
    let mut request_body = None;
    let mut cursor = leading_bytes + first.len();
    for raw_line in lines {
        let next_cursor = cursor + raw_line.len();
        let line = raw_line.trim_end();
        if line.is_empty() {
            cursor = next_cursor;
            continue;
        }
        if line == "The attached pasted text file(s) contain the user's request. Read and act on that content." {
            saw_instruction = true;
            cursor = next_cursor;
            continue;
        }
        if line == "## My request for Codex:" {
            request_body = Some(
                text[next_cursor..]
                    .trim_start_matches(['\r', '\n'])
                    .to_string(),
            );
            break;
        }
        if saw_instruction {
            return None;
        }
        entries.push(prompt_wrapper_entry(line)?);
        cursor = next_cursor;
    }
    if entries.is_empty() {
        return None;
    }
    let request_body = request_body?;

    let artifacts = if saw_instruction {
        entries
            .into_iter()
            .map(|(label, raw_path)| {
                let mut artifact = pasted_prompt_note(raw_path)?;
                artifact
                    .as_object_mut()?
                    .insert("label".to_string(), Value::String(label.to_string()));
                Some(artifact)
            })
            .collect::<Option<Vec<_>>>()?
    } else {
        entries
            .into_iter()
            .map(|(label, raw_path)| {
                let name = absolute_prompt_path_basename(raw_path)?;
                (label == name).then(|| {
                    serde_json::json!({
                        "kind": "file",
                        "name": name
                    })
                })
            })
            .collect::<Option<Vec<_>>>()?
    };
    Some(CodexPromptArtifactCarrier {
        data: serde_json::json!({ "promptArtifacts": artifacts }),
        input_text_index,
        request_body,
    })
}

fn convert_codex_item(
    item: &Value,
    session_id: &str,
    model: Option<&String>,
    line_timestamp: &str,
    counter: &mut u64,
) -> Option<ClaudeMessage> {
    let item_type = item.get("type").and_then(|t| t.as_str())?;
    *counter += 1;

    let uuid = item
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| format!("codex-{counter}"));

    let timestamp = item
        .get("created_at")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(line_timestamp)
        .to_string();

    match item_type {
        "message" => {
            let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let artifact_carrier = if role == "user" {
                codex_prompt_artifact_carrier(item.get("content"))
            } else {
                None
            };
            let content =
                convert_codex_content_array(item.get("content"), artifact_carrier.as_ref());

            let mut message = build_codex_message(
                uuid,
                session_id,
                timestamp,
                if role == "user" { "user" } else { "assistant" },
                Some(role),
                content,
                if role == "assistant" {
                    model.cloned()
                } else {
                    None
                },
            );
            if let Some(carrier) = artifact_carrier {
                message.data = Some(carrier.data);
            }
            Some(message)
        }
        "local_shell_call" => {
            let command = item
                .get("action")
                .and_then(|a| a.get("command"))
                .cloned()
                .unwrap_or(Value::Null);

            let command_str = if let Some(arr) = command.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                command.as_str().unwrap_or("").to_string()
            };

            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let content = serde_json::json!([{
                "type": "tool_use",
                "id": call_id,
                "name": "Bash",
                "input": { "command": command_str }
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "assistant",
                Some("assistant"),
                Some(content),
                model.cloned(),
            ))
        }
        "function_call" => {
            let raw_name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let name = map_codex_tool_name(raw_name);
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = item.get("arguments");
            let mut input = parse_tool_arguments(arguments);
            normalize_tool_input(name, &mut input);

            let content = serde_json::json!([{
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "assistant",
                Some("assistant"),
                Some(content),
                model.cloned(),
            ))
        }
        "function_call_output" => {
            let output = item.get("output").cloned().unwrap_or(Value::Null);
            let output = normalize_tool_output(output);
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let content = serde_json::json!([{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "user",
                Some("user"),
                Some(content),
                None,
            ))
        }
        "custom_tool_call" => {
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("custom_tool");
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| uuid.clone());
            let mut input = item.get("input").cloned().unwrap_or(Value::Null);
            normalize_custom_tool_input(name, &mut input);

            let content = serde_json::json!([{
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "assistant",
                Some("assistant"),
                Some(content),
                model.cloned(),
            ))
        }
        "custom_tool_call_output" => {
            let output = item.get("output").cloned().unwrap_or(Value::Null);
            let output = normalize_tool_output(output);
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let content = serde_json::json!([{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "user",
                Some("user"),
                Some(content),
                None,
            ))
        }
        "web_search_call" => {
            let search_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| uuid.clone());
            let action = item
                .get("action")
                .cloned()
                .unwrap_or_else(|| Value::Object(serde_json::Map::default()));
            let input = normalize_web_search_input(action);

            let content = serde_json::json!([{
                "type": "tool_use",
                "id": search_id,
                "name": "WebSearch",
                "input": input
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "assistant",
                Some("assistant"),
                Some(content),
                model.cloned(),
            ))
        }
        "reasoning" => {
            let thinking_text = item
                .get("summary")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            if thinking_text.is_empty() {
                return None;
            }

            let content = serde_json::json!([{
                "type": "thinking",
                "thinking": thinking_text
            }]);

            Some(build_codex_message(
                uuid,
                session_id,
                timestamp,
                "assistant",
                Some("assistant"),
                Some(content),
                model.cloned(),
            ))
        }
        _ => None,
    }
}

fn convert_codex_event(
    payload: &Value,
    session_id: &str,
    line_timestamp: &str,
    counter: &mut u64,
) -> Option<ClaudeMessage> {
    let event_type = payload.get("type").and_then(|t| t.as_str())?;

    match event_type {
        "task_started" => {
            *counter += 1;
            let mut msg = build_codex_message(
                format!("codex-event-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "progress",
                None,
                None,
                None,
            );
            msg.data = Some(serde_json::json!({
                "type": "waiting_for_task",
                "status": "started",
                "taskId": payload.get("turn_id").and_then(Value::as_str).unwrap_or_default(),
                "message": "Task started"
            }));
            msg.tool_use_id = payload
                .get("turn_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(msg)
        }
        "task_complete" => {
            *counter += 1;
            let mut msg = build_codex_message(
                format!("codex-event-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "progress",
                None,
                None,
                None,
            );
            msg.data = Some(serde_json::json!({
                "type": "waiting_for_task",
                "status": "completed",
                "taskId": payload.get("turn_id").and_then(Value::as_str).unwrap_or_default(),
                "message": "Task completed"
            }));
            msg.tool_use_id = payload
                .get("turn_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(msg)
        }
        "context_compacted" => {
            *counter += 1;
            let mut msg = build_codex_message(
                format!("codex-event-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "system",
                None,
                Some(serde_json::json!("Context compacted")),
                None,
            );
            msg.subtype = Some("microcompact_boundary".to_string());
            msg.level = Some("info".to_string());
            msg.microcompact_metadata = Some(serde_json::json!({
                "trigger": "context_compacted"
            }));
            Some(msg)
        }
        "thread_rolled_back" => {
            // Codex emits this durable branch boundary when it rolls back one or more
            // completed turns (including the send-then-edit flow). Keep the count as
            // structured data so downstream consumers can mark the superseded turns
            // without guessing from timestamps, text similarity, or task ids.
            let num_turns = payload
                .get("num_turns")
                .and_then(Value::as_u64)
                .filter(|count| *count > 0)?;
            *counter += 1;
            let mut msg = build_codex_message(
                format!("codex-rollback-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "system",
                None,
                None,
                None,
            );
            msg.subtype = Some("thread_rolled_back".to_string());
            msg.level = Some("info".to_string());
            msg.data = Some(serde_json::json!({ "numTurns": num_turns }));
            Some(msg)
        }
        "agent_reasoning" => {
            let text = payload.get("text").and_then(Value::as_str)?.trim();
            if text.is_empty() {
                return None;
            }
            *counter += 1;
            let content = serde_json::json!([{
                "type": "thinking",
                "thinking": text
            }]);
            Some(build_codex_message(
                format!("codex-event-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "assistant",
                Some("assistant"),
                Some(content),
                None,
            ))
        }
        "turn_aborted" => {
            *counter += 1;
            let reason = payload
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let turn_id = payload.get("turn_id").and_then(Value::as_str).unwrap_or("");
            let content = serde_json::json!([{
                "type": "text",
                "text": format!("[Turn Aborted] reason: {reason}, turn: {turn_id}")
            }]);
            let mut msg = build_codex_message(
                format!("codex-abort-{counter}"),
                session_id,
                line_timestamp.to_string(),
                "system",
                None,
                Some(content),
                None,
            );
            msg.subtype = Some("turn_aborted".to_string());
            msg.level = Some("warning".to_string());
            Some(msg)
        }
        // Unsupported/duplicated Codex events are intentionally ignored.
        _ => None,
    }
}

fn convert_codex_compacted(
    payload: &Value,
    session_id: &str,
    line_timestamp: &str,
    counter: &mut u64,
) -> ClaudeMessage {
    *counter += 1;
    let replacement_history_count = payload
        .get("replacement_history")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    let mut msg = build_codex_message(
        format!("codex-compacted-{counter}"),
        session_id,
        line_timestamp.to_string(),
        "system",
        None,
        Some(serde_json::json!("Conversation compacted")),
        None,
    );
    msg.subtype = Some("compact_boundary".to_string());
    msg.level = Some("info".to_string());
    msg.compact_metadata = Some(serde_json::json!({
        "trigger": "compacted",
        "replacementHistoryCount": replacement_history_count
    }));
    msg
}

fn codex_inference_settings(settings: &Value) -> InferenceMetadata {
    let string = |name: &str| {
        settings
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let interaction_mode = settings
        .get("collaboration_mode")
        .and_then(|value| {
            value
                .as_str()
                .or_else(|| value.get("mode").and_then(Value::as_str))
        })
        .map(str::to_string);
    InferenceMetadata {
        model: string("model"),
        model_provider: string("model_provider_id"),
        reasoning_effort: string("reasoning_effort"),
        reasoning_summary: string("reasoning_summary"),
        service_tier: string("service_tier"),
        interaction_mode,
        personality: string("personality"),
        ..InferenceMetadata::default()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct CodexTokenUsage {
    input: u32,
    output: u32,
    cached: Option<u32>,
    cache_write: Option<u32>,
    reasoning: Option<u32>,
}

fn codex_token_usage(value: &Value) -> Option<CodexTokenUsage> {
    let number = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|number| u32::try_from(number).ok())
    };
    Some(CodexTokenUsage {
        input: u32::try_from(value.get("input_tokens")?.as_u64()?).ok()?,
        output: u32::try_from(value.get("output_tokens")?.as_u64()?).ok()?,
        cached: number("cached_input_tokens"),
        cache_write: number("cache_write_input_tokens"),
        reasoning: number("reasoning_output_tokens"),
    })
}

fn extract_token_totals(payload: &Value) -> Option<CodexTokenUsage> {
    // Recent Codex logs store usage in payload.info.total_token_usage.
    codex_token_usage(payload.get("info")?.get("total_token_usage")?)
}

fn extract_last_token_usage(payload: &Value) -> Option<CodexTokenUsage> {
    // Fallback for older/newer variants that only include last token usage.
    codex_token_usage(payload.get("info")?.get("last_token_usage")?)
}

fn map_codex_tool_name(name: &str) -> &str {
    match name {
        "exec_command" | "shell" | "write_stdin" => "Bash",
        _ => name,
    }
}

fn parse_tool_arguments(arguments: Option<&Value>) -> Value {
    match arguments {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::Object(serde_json::Map::default()))
        }
        Some(v) if v.is_object() || v.is_array() => v.clone(),
        _ => Value::Object(serde_json::Map::default()),
    }
}

fn normalize_tool_input(tool_name: &str, input: &mut Value) {
    if tool_name != "Bash" {
        return;
    }

    let Some(obj) = input.as_object_mut() else {
        return;
    };

    // Codex exec_command uses "cmd"; UI Bash renderer expects "command".
    if !obj.contains_key("command") {
        if let Some(cmd) = obj.get("cmd").cloned() {
            match cmd {
                Value::String(_) => {
                    obj.insert("command".to_string(), cmd);
                }
                Value::Array(arr) => {
                    let joined = arr
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ");
                    obj.insert("command".to_string(), Value::String(joined));
                }
                _ => {}
            }
        }
    }

    if let Some(Value::Array(arr)) = obj.get("command").cloned() {
        let joined = arr
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        obj.insert("command".to_string(), Value::String(joined));
    }
}

fn normalize_custom_tool_input(tool_name: &str, input: &mut Value) {
    if input.is_object() {
        return;
    }

    if tool_name == "apply_patch" {
        let patch = input.as_str().unwrap_or("").to_string();
        *input = serde_json::json!({ "patch": patch });
        return;
    }

    *input = serde_json::json!({ "input": input.clone() });
}

fn normalize_web_search_input(action: Value) -> Value {
    let Some(action_obj) = action.as_object() else {
        return Value::Object(serde_json::Map::default());
    };

    let mut input = serde_json::Map::default();
    if let Some(query) = action_obj.get("query").and_then(Value::as_str) {
        input.insert("query".to_string(), Value::String(query.to_string()));
    } else if let Some(url) = action_obj.get("url").and_then(Value::as_str) {
        input.insert("query".to_string(), Value::String(url.to_string()));
    } else if let Some(pattern) = action_obj.get("pattern").and_then(Value::as_str) {
        input.insert("query".to_string(), Value::String(pattern.to_string()));
    }
    if let Some(queries) = action_obj.get("queries").cloned() {
        input.insert("queries".to_string(), queries);
    }
    if let Some(action_type) = action_obj.get("type").and_then(Value::as_str) {
        input.insert(
            "action_type".to_string(),
            Value::String(action_type.to_string()),
        );
    }

    Value::Object(input)
}

fn normalize_tool_output(output: Value) -> Value {
    let Value::String(raw) = output else {
        return output;
    };

    // exec_command tool output can be a JSON string: {"output":"...", ...}
    if let Ok(parsed) = serde_json::from_str::<Value>(&raw) {
        if let Some(inner_output) = parsed.get("output") {
            return inner_output.clone();
        }
    }

    // Codex function wrapper output usually embeds "Output:\n{actual stdout}".
    if let Some((_, out)) = raw.split_once("\nOutput:\n") {
        return Value::String(out.to_string());
    }

    Value::String(raw)
}

fn try_merge_tool_result_into_previous(
    messages: &mut [ClaudeMessage],
    msg: &ClaudeMessage,
) -> bool {
    if msg.message_type != "user" {
        return false;
    }

    let Some((tool_use_id, mut tool_result_block)) = extract_tool_result_block(msg) else {
        return false;
    };

    for prev in messages.iter_mut().rev() {
        if prev.message_type != "assistant" {
            continue;
        }
        if let Some(tool_name) = matching_tool_use_name(prev, &tool_use_id) {
            if tool_name == "apply_patch" && failed_apply_patch_result(&tool_result_block) {
                if let Some(result) = tool_result_block.as_object_mut() {
                    result.insert("is_error".to_string(), Value::Bool(true));
                }
            }
            append_content_block(prev, tool_result_block);
            return true;
        }
    }

    false
}

fn extract_tool_result_block(msg: &ClaudeMessage) -> Option<(String, Value)> {
    let arr = msg.content.as_ref()?.as_array()?;
    let first = arr.first()?;
    if first.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    let tool_use_id = first
        .get("tool_use_id")
        .and_then(Value::as_str)?
        .to_string();
    Some((tool_use_id, first.clone()))
}

fn matching_tool_use_name<'a>(msg: &'a ClaudeMessage, tool_use_id: &str) -> Option<&'a str> {
    let arr = msg.content.as_ref().and_then(Value::as_array)?;
    arr.iter()
        .find(|item| {
            item.get("type").and_then(Value::as_str) == Some("tool_use")
                && item.get("id").and_then(Value::as_str) == Some(tool_use_id)
        })
        .and_then(|item| item.get("name"))
        .and_then(Value::as_str)
}

fn failed_apply_patch_result(block: &Value) -> bool {
    let Some(content) = block.get("content").and_then(Value::as_str) else {
        return false;
    };
    let content = content.trim_start();
    content.starts_with("apply_patch verification failed:") || content.starts_with("Invalid patch")
}

fn append_content_block(msg: &mut ClaudeMessage, block: Value) {
    match &mut msg.content {
        Some(Value::Array(arr)) => arr.push(block),
        _ => msg.content = Some(Value::Array(vec![block])),
    }
}

fn extract_first_tool_use(content: Option<&Value>) -> Option<Value> {
    let arr = content?.as_array()?;
    arr.iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("tool_use"))
        .cloned()
}

fn convert_codex_content_array(
    content: Option<&Value>,
    artifact_carrier: Option<&CodexPromptArtifactCarrier>,
) -> Option<Value> {
    let arr = content?.as_array()?;

    let items: Vec<Value> = arr
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let ctype = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match ctype {
                "input_text" | "output_text" | "text" => {
                    let replacement = artifact_carrier
                        .filter(|carrier| carrier.input_text_index == index)
                        .map(|carrier| carrier.request_body.as_str());
                    let text = replacement
                        .unwrap_or_else(|| item.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                    if replacement.is_some() && text.is_empty() {
                        return None;
                    }
                    Some(serde_json::json!({
                        "type": "text",
                        "text": text
                    }))
                }
                "input_image" => {
                    let image_url = item.get("image_url").and_then(Value::as_str).unwrap_or("");
                    if image_url.is_empty() {
                        return None;
                    }
                    Some(serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url
                        }
                    }))
                }
                "refusal" => {
                    let refusal = item
                        .get("refusal")
                        .and_then(|t| t.as_str())
                        .unwrap_or("Refused");
                    Some(serde_json::json!({
                        "type": "text",
                        "text": format!("[Refusal] {refusal}")
                    }))
                }
                _ => None,
            }
        })
        .collect();

    if items.is_empty() {
        None
    } else {
        Some(Value::Array(items))
    }
}

fn build_codex_message(
    uuid: String,
    session_id: &str,
    timestamp: String,
    message_type: &str,
    role: Option<&str>,
    content: Option<Value>,
    model: Option<String>,
) -> ClaudeMessage {
    let tool_use = if message_type == "assistant" {
        extract_first_tool_use(content.as_ref())
    } else {
        None
    };

    let mut msg = build_provider_message(
        "codex",
        uuid,
        session_id,
        timestamp,
        message_type,
        role,
        content,
        model,
    );
    msg.tool_use = tool_use;
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serial_test::serial;
    use std::ffi::OsString;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let original = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.original.as_ref() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn imported_provider_classifies_known_storage_roots() {
        assert_eq!(
            imported_provider_from_path(r"C:\Users\example\.claude\projects\x\session.jsonl"),
            Some("claude".to_string())
        );
        assert_eq!(
            imported_provider_from_path("/home/example/.copilot/session-state/session"),
            Some("copilot".to_string())
        );
        assert_eq!(
            imported_provider_from_path("/home/example/.local/share/opencode/storage/session"),
            Some("opencode".to_string())
        );
        assert_eq!(
            imported_provider_from_path("/home/example/other-provider/session.jsonl"),
            None
        );
    }

    fn write_codex_rollout(
        sessions_dir: &Path,
        filename: &str,
        session_id: &str,
        cwd: &str,
        first_prompt: &str,
    ) -> PathBuf {
        let rollout_path = sessions_dir.join(filename);
        let lines = [
            json!({
                "type": "session_meta",
                "payload": { "id": session_id, "cwd": cwd }
            }),
            json!({
                "timestamp": "2026-02-21T10:00:00Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "created_at": "2026-02-21T10:00:00Z",
                    "content": [{ "type": "input_text", "text": first_prompt }]
                }
            }),
        ];
        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n"))
            .expect("rollout fixture should be written");
        rollout_path
    }

    fn append_rollout_lines(path: &Path, lines: &[Value]) {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("rollout should open for append");
        for line in lines {
            writeln!(file, "{line}").expect("rollout line should append");
        }
        file.sync_all().expect("appended rollout should flush");
    }

    fn snapshot_fixture_prefix(session_id: &str) -> Vec<Value> {
        vec![
            json!({
                "timestamp": "2026-07-29T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": session_id, "cwd": "C:/repo", "source": "cli" }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:01Z",
                "type": "turn_context",
                "payload": { "model": "gpt-test", "effort": "high" }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "first prompt" }]
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:03Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "first prompt" }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_started",
                    "turn_id": "turn-1",
                    "model_context_window": 128000,
                    "collaboration_mode_kind": "default"
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:05Z",
                "type": "response_item",
                "payload": {
                    "id": "assistant-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "first answer" }]
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:06Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 20,
                            "output_tokens": 5,
                            "cached_input_tokens": 3,
                            "reasoning_output_tokens": 2
                        }
                    }
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:00:07Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn-1",
                    "duration_ms": 100,
                    "time_to_first_token_ms": 10
                }
            }),
        ]
    }

    fn write_snapshot_fixture(sessions_dir: &Path, session_id: &str) -> PathBuf {
        let path = sessions_dir.join(format!("rollout-2026-07-29T10-00-00-{session_id}.jsonl"));
        let lines = snapshot_fixture_prefix(session_id);
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, format!("{body}\n")).expect("snapshot fixture should be written");
        path
    }

    fn message_values(messages: &[ClaudeMessage]) -> Value {
        serde_json::to_value(messages).expect("messages should serialize")
    }

    fn assert_snapshot_matches_fresh(messages: &[ClaudeMessage], path: &Path) {
        let fresh = parse_rollout_file(path).expect("fresh complete parse should succeed");
        assert_eq!(message_values(messages), message_values(&fresh));

        let finalized_snapshot =
            crate::commands::multi_provider::finalize_loaded_messages(messages.to_vec());
        let finalized_fresh = crate::commands::multi_provider::finalize_loaded_messages(fresh);
        assert_eq!(
            message_values(&finalized_snapshot),
            message_values(&finalized_fresh)
        );
    }

    #[test]
    #[serial]
    fn snapshot_replacements_equal_fresh_complete_parses() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-equality");
        let path_text = path.to_string_lossy();

        let (mut cached, initial_cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    reason,
                    messages,
                    cursor: Some(cursor),
                    ..
                } => {
                    assert_eq!(reason, "initial");
                    (messages, cursor)
                }
                _ => panic!("initial Codex snapshot should be complete and cursor-bearing"),
            };
        let initial_len = cached.len();
        assert_snapshot_matches_fresh(&cached, &path);

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-07-29T10:01:00Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": "second prompt" }]
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01Z",
                    "type": "event_msg",
                    "payload": { "type": "user_message", "message": "second prompt" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:02Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": "turn-2",
                        "model_context_window": 128000,
                        "collaboration_mode_kind": "default"
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:03Z",
                    "type": "response_item",
                    "payload": {
                        "id": "tool-2",
                        "type": "function_call",
                        "call_id": "call-2",
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"echo hi\"}"
                    }
                }),
            ],
        );

        let active_cursor =
            match load_session_snapshot(&path_text, Some(&initial_cursor)).expect("active delta") {
                SessionSnapshotLoad::Replace {
                    replace_from,
                    messages,
                    cursor,
                    ..
                } => {
                    assert_eq!(replace_from, initial_len);
                    cached.truncate(replace_from);
                    cached.extend(messages);
                    cursor
                }
                _ => panic!("an appended active turn should return a replacement suffix"),
            };
        assert_snapshot_matches_fresh(&cached, &path);

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-07-29T10:01:04Z",
                    "type": "response_item",
                    "payload": {
                        "id": "result-2",
                        "type": "function_call_output",
                        "call_id": "call-2",
                        "output": "hi"
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:05Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 50,
                                "output_tokens": 15,
                                "cached_input_tokens": 8,
                                "reasoning_output_tokens": 4
                            }
                        }
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:06Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "turn_id": "turn-2",
                        "duration_ms": 200,
                        "time_to_first_token_ms": 20
                    }
                }),
            ],
        );

        let completed_cursor = match load_session_snapshot(&path_text, Some(&active_cursor))
            .expect("completion delta")
        {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor,
                ..
            } => {
                assert_eq!(replace_from, initial_len);
                cached.truncate(replace_from);
                cached.extend(messages);
                cursor
            }
            _ => panic!("an appended completion should replace the active turn"),
        };
        assert_snapshot_matches_fresh(&cached, &path);

        match load_session_snapshot(&path_text, Some(&completed_cursor))
            .expect("unchanged snapshot")
        {
            SessionSnapshotLoad::Unchanged { cursor } => {
                assert_eq!(cursor, completed_cursor);
            }
            _ => panic!("an identical source should return unchanged"),
        }
    }

    #[test]
    #[serial]
    fn snapshot_falls_back_when_the_verified_prefix_changes_or_shrinks() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-prefix");
        let path_text = path.to_string_lossy();
        let cursor = match load_session_snapshot(&path_text, None).expect("initial snapshot") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("initial snapshot should carry a cursor"),
        };

        let original = fs::read_to_string(&path).expect("fixture should be readable");
        let changed = original.replace("first prompt", "FIRST prompt");
        assert_eq!(changed.len(), original.len());
        fs::write(&path, changed).expect("same-length rewrite should succeed");
        match load_session_snapshot(&path_text, Some(&cursor)).expect("rewrite fallback") {
            SessionSnapshotLoad::Full {
                reason, messages, ..
            } => {
                assert_eq!(reason, "prefix-mismatch");
                assert_eq!(
                    message_values(&messages),
                    message_values(&parse_rollout_file(&path).unwrap())
                );
            }
            _ => panic!("a rewritten prefix must force a complete snapshot"),
        }

        fs::write(
            &path,
            format!("{}\n", snapshot_fixture_prefix("snapshot-prefix")[0]),
        )
        .expect("truncation should succeed");
        match load_session_snapshot(&path_text, Some(&cursor)).expect("shrink fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "source-shrank");
            }
            _ => panic!("a shrunk source must force a complete snapshot"),
        }
    }

    #[test]
    #[serial]
    fn snapshot_falls_back_for_a_tool_result_targeting_the_retained_prefix() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-backward");
        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-07-29T10:00:05.1Z",
                    "type": "response_item",
                    "payload": {
                        "id": "late-tool",
                        "type": "function_call",
                        "call_id": "late-call",
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"echo late\"}"
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:08Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": "turn-late"
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:09Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "turn_id": "turn-late"
                    }
                }),
            ],
        );
        let path_text = path.to_string_lossy();
        let cursor = match load_session_snapshot(&path_text, None).expect("initial snapshot") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("initial snapshot should carry a cursor"),
        };

        append_rollout_lines(
            &path,
            &[json!({
                "timestamp": "2026-07-29T10:00:10Z",
                "type": "response_item",
                "payload": {
                    "id": "late-result",
                    "type": "function_call_output",
                    "call_id": "late-call",
                    "output": "late"
                }
            })],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("safe fallback") {
            SessionSnapshotLoad::Full {
                reason, messages, ..
            } => {
                assert_eq!(reason, "unsafe-backward-reference");
                assert_eq!(
                    message_values(&messages),
                    message_values(&parse_rollout_file(&path).unwrap())
                );
            }
            _ => panic!("a backward tool result must force a complete snapshot"),
        }
    }

    #[test]
    #[serial]
    fn snapshot_replays_compaction_fork_metadata_and_steers_exactly() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-structural");
        let path_text = path.to_string_lossy();

        let (mut cached, cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("initial snapshot should carry a cursor"),
            };

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-07-29T10:01:00Z",
                    "type": "session_meta",
                    "payload": { "id": "replayed-source-id", "cwd": "C:/old-repo" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01Z",
                    "type": "compacted",
                    "payload": { "replacement_history": [{ "type": "message" }] }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01.005Z",
                    "type": "world_state",
                    "payload": {}
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01.010Z",
                    "type": "turn_context",
                    "payload": {}
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01.015Z",
                    "type": "event_msg",
                    "payload": { "type": "token_count" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:01.020Z",
                    "type": "event_msg",
                    "payload": { "type": "context_compacted" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:02Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-2" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:03Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": "first mid-turn input" }]
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:04Z",
                    "type": "event_msg",
                    "payload": { "type": "user_message", "message": "first mid-turn input" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:05Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": "second mid-turn input" }]
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:06Z",
                    "type": "event_msg",
                    "payload": { "type": "user_message", "message": "second mid-turn input" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:07Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-2" }
                }),
            ],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("structural delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("append-only structural records should produce a replacement"),
        }
        assert_snapshot_matches_fresh(&cached, &path);
        assert!(
            cached
                .iter()
                .all(|message| message.session_id == "snapshot-structural"),
            "replayed fork metadata must not replace the destination session id"
        );
        assert_eq!(
            cached
                .iter()
                .filter(|message| message.subtype.as_deref() == Some("compact_boundary"))
                .count(),
            1
        );
        assert!(!cached
            .iter()
            .any(|message| message.subtype.as_deref() == Some("microcompact_boundary")));
        assert!(cached
            .iter()
            .any(|message| message.subtype.as_deref() == Some(STEER_SUBTYPE)));
    }

    #[test]
    #[serial]
    fn snapshot_correlates_a_compaction_split_across_refreshes() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-split-compaction");
        let path_text = path.to_string_lossy();

        let (mut cached, cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("initial snapshot should carry a cursor"),
            };

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-07-29T10:01:00Z",
                    "type": "compacted",
                    "payload": { "replacement_history": [{ "type": "message" }] }
                }),
                json!({
                    "timestamp": "2026-07-29T10:01:00.005Z",
                    "type": "world_state",
                    "payload": {}
                }),
            ],
        );

        let cursor = match load_session_snapshot(&path_text, Some(&cursor))
            .expect("authoritative compaction delta")
        {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
                cursor
            }
            _ => panic!("an appended compaction should produce a replacement"),
        };

        append_rollout_lines(
            &path,
            &[json!({
                "timestamp": "2026-07-29T10:01:00.020Z",
                "type": "event_msg",
                "payload": { "type": "context_compacted" }
            })],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("companion delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("the appended companion should produce a replacement"),
        }
        assert_snapshot_matches_fresh(&cached, &path);
        assert_eq!(
            cached
                .iter()
                .filter(|message| message.subtype.as_deref() == Some("compact_boundary"))
                .count(),
            1
        );
        assert!(!cached
            .iter()
            .any(|message| message.subtype.as_deref() == Some("microcompact_boundary")));
    }

    #[test]
    #[serial]
    fn snapshot_rejects_invalid_and_incompatible_cursors() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-cursor");
        let path_text = path.to_string_lossy();

        match load_session_snapshot(&path_text, Some("not-base64")).expect("invalid fallback") {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "invalid-cursor");
            }
            _ => panic!("an invalid cursor must force a complete snapshot"),
        }

        let encoded = match load_session_snapshot(&path_text, None).expect("initial snapshot") {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("initial snapshot should carry a cursor"),
        };
        let mut cursor = decode_snapshot_cursor(&encoded).expect("cursor should decode");
        cursor.version += 1;
        let incompatible = encode_snapshot_cursor(&cursor).expect("cursor should encode");
        match load_session_snapshot(&path_text, Some(&incompatible)).expect("incompatible fallback")
        {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "incompatible-cursor");
            }
            _ => panic!("an incompatible cursor must force a complete snapshot"),
        }

        let archived_dir = codex_home.join("archived_sessions");
        fs::create_dir_all(&archived_dir).expect("archive directory should be created");
        let archived_path =
            archived_dir.join(path.file_name().expect("fixture should have a name"));
        fs::rename(&path, &archived_path).expect("fixture should move to the archive");
        match load_session_snapshot(&archived_path.to_string_lossy(), Some(&encoded))
            .expect("archive transition fallback")
        {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "incompatible-cursor");
            }
            _ => panic!("moving a rollout must invalidate its source-bound cursor"),
        }
    }

    #[test]
    #[serial]
    fn snapshot_excludes_an_incomplete_trailing_record_until_it_is_completed() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-partial-line");
        let path_text = path.to_string_lossy();

        let partial = r#"{"timestamp":"2026-07-29T10:01:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"second"#;
        {
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("rollout should open for append");
            write!(file, "{partial}").expect("partial record should append");
            file.sync_all().expect("partial rollout should flush");
        }

        let (initial_messages, cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("a stable plain rollout should carry a cursor"),
            };
        assert_snapshot_matches_fresh(&initial_messages, &path);

        match load_session_snapshot(&path_text, Some(&cursor)).expect("unchanged partial tail") {
            SessionSnapshotLoad::Unchanged {
                cursor: unchanged_cursor,
            } => assert_eq!(unchanged_cursor, cursor),
            _ => panic!("an unchanged incomplete tail must not advance the cursor"),
        }

        {
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("rollout should open for completion");
            writeln!(file, " prompt\"}}]}}}}").expect("record completion should append");
            file.sync_all().expect("completed rollout should flush");
        }

        let mut reconstructed = initial_messages;
        match load_session_snapshot(&path_text, Some(&cursor)).expect("completed tail delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                reconstructed.truncate(replace_from);
                reconstructed.extend(messages);
            }
            _ => panic!("completing the trailing record should produce a replacement"),
        }
        assert_snapshot_matches_fresh(&reconstructed, &path);
    }

    #[test]
    #[serial]
    fn snapshot_returns_a_cursorless_full_result_for_compressed_rollouts() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let plain = write_snapshot_fixture(&sessions_dir, "snapshot-compressed");
        let body = fs::read(&plain).expect("plain rollout should be readable");
        let compressed =
            sessions_dir.join("rollout-2026-07-29T10-00-00-snapshot-compressed.jsonl.zst");
        fs::write(
            &compressed,
            zstd::encode_all(&body[..], 3).expect("fixture should compress"),
        )
        .expect("compressed rollout should be written");
        fs::remove_file(&plain).expect("plain rollout should be removed");

        match load_session_snapshot(&compressed.to_string_lossy(), Some("ignored"))
            .expect("compressed fallback")
        {
            SessionSnapshotLoad::Full {
                reason,
                messages,
                cursor,
                ..
            } => {
                assert_eq!(reason, "unsupported-source");
                assert!(cursor.is_none());
                assert_snapshot_matches_fresh(&messages, &compressed);
            }
            _ => panic!("compressed rollouts must use a cursorless complete snapshot"),
        }
    }

    fn create_codex_state_db(codex_home: &Path, rows: &[(&str, &str, &str)]) {
        let conn = Connection::open(codex_home.join(STATE_DB_FILENAME))
            .expect("codex state db should be created");
        conn.execute(
            "CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                preview TEXT NOT NULL,
                first_user_message TEXT NOT NULL
            )",
            [],
        )
        .expect("threads table should be created");

        for (id, title, preview) in rows {
            conn.execute(
                "INSERT INTO threads (id, title, preview, first_user_message)
                 VALUES (?1, ?2, ?3, ?3)",
                rusqlite::params![id, title, preview],
            )
            .expect("thread row should be inserted");
        }
    }

    fn write_session_index(codex_home: &Path, entries: &[Value]) {
        let body = entries
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(codex_home.join(SESSION_INDEX_FILENAME), format!("{body}\n"))
            .expect("session index should be written");
    }

    #[test]
    fn map_exec_command_to_bash() {
        assert_eq!(map_codex_tool_name("exec_command"), "Bash");
        assert_eq!(map_codex_tool_name("shell"), "Bash");
        assert_eq!(map_codex_tool_name("write_stdin"), "Bash");
        assert_eq!(map_codex_tool_name("batch_execute"), "batch_execute");
    }

    #[test]
    fn normalize_bash_input_maps_cmd_to_command() {
        let mut input = json!({ "cmd": "pwd && ls -la" });
        normalize_tool_input("Bash", &mut input);
        assert_eq!(
            input.get("command").and_then(Value::as_str),
            Some("pwd && ls -la")
        );
    }

    #[test]
    fn normalize_bash_input_maps_command_array_to_string() {
        let mut input = json!({ "command": ["bash", "-lc", "pwd"] });
        normalize_tool_input("Bash", &mut input);
        assert_eq!(
            input.get("command").and_then(Value::as_str),
            Some("bash -lc pwd")
        );
    }

    #[test]
    fn normalize_tool_output_extracts_wrapped_output() {
        let wrapped = "Chunk ID: abc\nWall time: 0.01 seconds\nOutput:\nhello\nworld";
        let out = normalize_tool_output(Value::String(wrapped.to_string()));
        assert_eq!(out.as_str(), Some("hello\nworld"));
    }

    #[test]
    fn normalize_tool_output_extracts_json_output_field() {
        let out = normalize_tool_output(Value::String(
            r#"{"output":"done","metadata":{"exit_code":0}}"#.to_string(),
        ));
        assert_eq!(out.as_str(), Some("done"));
    }

    #[test]
    fn parse_nested_token_count_totals() {
        let payload = json!({
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 120,
                    "output_tokens": 30
                }
            }
        });
        assert_eq!(
            extract_token_totals(&payload),
            Some(CodexTokenUsage {
                input: 120,
                output: 30,
                ..CodexTokenUsage::default()
            })
        );
    }

    #[test]
    fn normalizes_authoritative_turn_inference_metadata() {
        let tmp = TempDir::new().expect("temp dir");
        let rollout_path = tmp.path().join("rollout-2026-07-22-test-session.jsonl");
        let lines = [
            json!({"timestamp":"2026-07-22T10:00:00Z","type":"session_meta","payload":{"id":"test-session"}}),
            json!({"timestamp":"2026-07-22T10:00:01Z","type":"event_msg","payload":{"type":"thread_settings_applied","thread_settings":{
                "model":"gpt-test","model_provider_id":"openai","service_tier":"default",
                "reasoning_effort":"high","reasoning_summary":"none","personality":"pragmatic",
                "collaboration_mode":{"mode":"default"}
            }}}),
            json!({"timestamp":"2026-07-22T10:00:02Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-1","model_context_window":258400,"collaboration_mode_kind":"default"}}),
            json!({"timestamp":"2026-07-22T10:00:03Z","type":"turn_context","payload":{"turn_id":"turn-1","model":"gpt-test","effort":"high","summary":"auto"}}),
            json!({"timestamp":"2026-07-22T10:00:04Z","type":"response_item","payload":{"id":"a1","type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}),
            json!({"timestamp":"2026-07-22T10:00:05Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{
                "input_tokens":100,"cached_input_tokens":40,"cache_write_input_tokens":3,
                "output_tokens":20,"reasoning_output_tokens":7
            }}}}),
            json!({"timestamp":"2026-07-22T10:00:06Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1","duration_ms":1500,"time_to_first_token_ms":250}}),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("write rollout");

        let messages = parse_rollout_file(&rollout_path).expect("parse rollout");
        let assistant = messages
            .iter()
            .find(|message| message.message_type == "assistant")
            .expect("assistant");
        let inference = assistant.inference.as_ref().expect("inference metadata");
        assert_eq!(inference.model.as_deref(), Some("gpt-test"));
        assert_eq!(inference.model_provider.as_deref(), Some("openai"));
        assert_eq!(inference.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(inference.reasoning_summary.as_deref(), Some("none"));
        assert_eq!(inference.service_tier.as_deref(), Some("default"));
        assert_eq!(inference.context_window, Some(258_400));
        assert_eq!(inference.interaction_mode.as_deref(), Some("default"));
        assert_eq!(inference.personality.as_deref(), Some("pragmatic"));
        assert_eq!(inference.duration_ms, Some(1500));
        assert_eq!(inference.time_to_first_token_ms, Some(250));
        assert_eq!(
            inference
                .usage
                .as_ref()
                .and_then(|usage| usage.input_tokens),
            Some(100)
        );
        assert_eq!(
            inference
                .usage
                .as_ref()
                .and_then(|usage| usage.reasoning_output_tokens),
            Some(7)
        );
    }

    #[test]
    fn normalize_custom_tool_input_wraps_apply_patch_text() {
        let mut input = Value::String("*** Begin Patch".to_string());
        normalize_custom_tool_input("apply_patch", &mut input);
        assert_eq!(
            input.get("patch").and_then(Value::as_str),
            Some("*** Begin Patch")
        );
    }

    #[test]
    fn normalize_web_search_input_extracts_query_and_type() {
        let input = normalize_web_search_input(json!({
            "type": "search",
            "query": "codex parser",
            "queries": ["codex parser", "codex rollout"]
        }));
        assert_eq!(
            input.get("query").and_then(Value::as_str),
            Some("codex parser")
        );
        assert_eq!(
            input.get("action_type").and_then(Value::as_str),
            Some("search")
        );
        assert!(input.get("queries").is_some());
    }

    #[test]
    fn convert_content_array_maps_input_image_to_image() {
        let converted = convert_codex_content_array(
            Some(&json!([
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc"
                }
            ])),
            None,
        )
        .expect("content should be converted");

        let arr = converted
            .as_array()
            .expect("converted content should be an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].get("type").and_then(Value::as_str), Some("image"));
        assert_eq!(
            arr[0]
                .get("source")
                .and_then(|v| v.get("url"))
                .and_then(Value::as_str),
            Some("data:image/png;base64,abc")
        );
    }

    #[test]
    fn pasted_prompt_note_emits_safe_artifact_metadata() {
        let content = json!([{
            "type": "input_text",
            "text": "\n# Files mentioned by the user:\n\n## Build output: C:\\Users\\person\\.codex/attachments/d5aa3020-8ebb-49f9-b5fd-c1cfa3374e3a/pasted-text.txt\n\nThe attached pasted text file(s) contain the user's request. Read and act on that content.\n\n## My request for Codex:\n\n"
        }]);

        let carrier = codex_prompt_artifact_carrier(Some(&content))
            .expect("complete pasted-note wrapper should be detected");
        assert_eq!(
            carrier.data,
            json!({
                "promptArtifacts": [{
                    "kind": "note",
                    "name": "pasted-text.txt",
                    "label": "Build output"
                }]
            })
        );
        assert_eq!(carrier.request_body, "");
        let serialized = serde_json::to_string(&carrier.data).unwrap();
        assert!(!serialized.contains("Users"));
        assert!(!serialized.contains("d5aa3020"));

        let mut counter = 0;
        let message = convert_codex_item(
            &json!({ "type": "message", "role": "user", "content": content }),
            "session-1",
            None,
            "2026-07-21T12:00:00Z",
            &mut counter,
        )
        .expect("user message should be converted");
        assert_eq!(
            message.data,
            Some(json!({
                "promptArtifacts": [{
                    "kind": "note",
                    "name": "pasted-text.txt",
                    "label": "Build output"
                }]
            }))
        );
        assert!(message.content.is_none());
    }

    #[test]
    fn ordinary_prompt_files_emit_safe_artifact_metadata() {
        let content = json!([{
            "type": "input_text",
            "text": "\n# Files mentioned by the user:\n\n## Claude-Rust vs Go compilation speed.md: e:\\Programas\\Artificial Intelligence (Ai)\\Claude\\My Commit Message Generator\\Claude-Rust vs Go compilation speed.md\n\n## schema.sql: /home/person/project/schema.sql\n\n## My request for Codex:\nUse these files.\n"
        }]);

        let carrier = codex_prompt_artifact_carrier(Some(&content))
            .expect("complete file wrapper should be detected");
        assert_eq!(
            carrier.data,
            json!({
                "promptArtifacts": [
                    {
                        "kind": "file",
                        "name": "Claude-Rust vs Go compilation speed.md"
                    },
                    {
                        "kind": "file",
                        "name": "schema.sql"
                    }
                ]
            })
        );
        assert_eq!(carrier.request_body, "Use these files.\n");
        let serialized = serde_json::to_string(&carrier.data).unwrap();
        assert!(!serialized.contains("Programas"));
        assert!(!serialized.contains("/home/person"));

        let mut counter = 0;
        let message = convert_codex_item(
            &json!({ "type": "message", "role": "user", "content": content }),
            "session-1",
            None,
            "2026-07-22T21:11:44Z",
            &mut counter,
        )
        .expect("user message should be converted");
        assert_eq!(
            message.data,
            Some(json!({
                "promptArtifacts": [
                    {
                        "kind": "file",
                        "name": "Claude-Rust vs Go compilation speed.md"
                    },
                    {
                        "kind": "file",
                        "name": "schema.sql"
                    }
                ]
            }))
        );
        assert_eq!(
            message.content,
            Some(json!([{
                "type": "text",
                "text": "Use these files.\n"
            }]))
        );
    }

    #[test]
    fn prompt_artifacts_require_the_complete_owned_wrapper() {
        let mentioned = json!([{
            "type": "input_text",
            "text": "The log mentions C:\\tmp\\attachments\\d5aa3020-8ebb-49f9-b5fd-c1cfa3374e3a\\pasted-text.txt"
        }]);
        let outside = json!([{
            "type": "input_text",
            "text": "# Files mentioned by the user:\n\n## Build output: C:\\tmp\\pasted-text.txt\n\nThe attached pasted text file(s) contain the user's request. Read and act on that content.\n\n## My request for Codex:\n"
        }]);
        let relative = json!([{
            "type": "input_text",
            "text": "# Files mentioned by the user:\n\n## report.md: docs/report.md\n\n## My request for Codex:\n"
        }]);
        let mismatched = json!([{
            "type": "input_text",
            "text": "# Files mentioned by the user:\n\n## report.md: C:\\project\\other.md\n\n## My request for Codex:\n"
        }]);
        let incomplete = json!([{
            "type": "input_text",
            "text": "# Files mentioned by the user:\n\n## report.md: C:\\project\\report.md\n"
        }]);

        assert!(codex_prompt_artifact_carrier(Some(&mentioned)).is_none());
        assert!(codex_prompt_artifact_carrier(Some(&outside)).is_none());
        assert!(codex_prompt_artifact_carrier(Some(&relative)).is_none());
        assert!(codex_prompt_artifact_carrier(Some(&mismatched)).is_none());
        assert!(codex_prompt_artifact_carrier(Some(&incomplete)).is_none());

        let original_text = mismatched[0]["text"]
            .as_str()
            .expect("test input should contain text");
        let mut counter = 0;
        let message = convert_codex_item(
            &json!({ "type": "message", "role": "user", "content": mismatched }),
            "session-1",
            None,
            "2026-07-22T21:11:44Z",
            &mut counter,
        )
        .expect("unrecognized user prose should still be converted");
        assert_eq!(message.data, None);
        assert_eq!(
            message.content,
            Some(json!([{
                "type": "text",
                "text": original_text
            }]))
        );
    }

    #[test]
    fn convert_custom_tool_call_to_tool_use() {
        let mut counter = 0u64;
        let msg = convert_codex_item(
            &json!({
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "call_patch_1",
                "input": "*** Begin Patch"
            }),
            "session-1",
            None,
            "2026-02-19T12:00:00Z",
            &mut counter,
        )
        .expect("custom_tool_call should be converted");

        assert_eq!(msg.message_type, "assistant");
        let arr = msg
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("content should be an array");
        assert_eq!(arr[0].get("type").and_then(Value::as_str), Some("tool_use"));
        assert_eq!(
            arr[0].get("name").and_then(Value::as_str),
            Some("apply_patch")
        );
        assert_eq!(
            arr[0]
                .get("input")
                .and_then(|v| v.get("patch"))
                .and_then(Value::as_str),
            Some("*** Begin Patch")
        );
    }

    #[test]
    fn convert_parallel_agent_function_calls_preserves_protocol_fields() {
        let fixtures = [
            (
                json!({
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "call_spawn_1",
                    "arguments": "{\"message\":\"Check the API\"}"
                }),
                "spawn_agent",
            ),
            (
                json!({
                    "type": "function_call",
                    "name": "wait_agent",
                    "call_id": "call_wait_1",
                    "arguments": "{\"targets\":[\"agent-1\",\"agent-2\"]}"
                }),
                "wait_agent",
            ),
        ];
        let mut counter = 0u64;

        for (item, expected_name) in fixtures {
            let msg = convert_codex_item(
                &item,
                "session-1",
                None,
                "2026-07-07T00:00:00Z",
                &mut counter,
            )
            .expect("collaboration function call should be converted");
            let block = msg
                .content
                .as_ref()
                .and_then(Value::as_array)
                .and_then(|blocks| blocks.first())
                .expect("tool use block should exist");

            assert_eq!(block["type"], "tool_use");
            assert_eq!(block["name"], expected_name);
            assert!(block["input"].is_object());
        }
    }

    #[test]
    fn convert_custom_tool_call_output_to_tool_result() {
        let mut counter = 0u64;
        let msg = convert_codex_item(
            &json!({
                "type": "custom_tool_call_output",
                "call_id": "call_patch_1",
                "output": "{\"output\":\"Success. Updated files\",\"metadata\":{\"exit_code\":0}}"
            }),
            "session-1",
            None,
            "2026-02-19T12:00:01Z",
            &mut counter,
        )
        .expect("custom_tool_call_output should be converted");

        assert_eq!(msg.message_type, "user");
        let arr = msg
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("content should be an array");
        assert_eq!(
            arr[0].get("type").and_then(Value::as_str),
            Some("tool_result")
        );
        assert_eq!(
            arr[0].get("tool_use_id").and_then(Value::as_str),
            Some("call_patch_1")
        );
        assert_eq!(
            arr[0].get("content").and_then(Value::as_str),
            Some("Success. Updated files")
        );
    }

    #[test]
    fn convert_web_search_call_to_web_search_tool_use() {
        let mut counter = 0u64;
        let msg = convert_codex_item(
            &json!({
                "type": "web_search_call",
                "action": {
                    "type": "open_page",
                    "url": "https://example.com"
                }
            }),
            "session-1",
            None,
            "2026-02-19T12:00:02Z",
            &mut counter,
        )
        .expect("web_search_call should be converted");

        assert_eq!(msg.message_type, "assistant");
        let arr = msg
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("content should be an array");
        assert_eq!(arr[0].get("type").and_then(Value::as_str), Some("tool_use"));
        assert_eq!(
            arr[0].get("name").and_then(Value::as_str),
            Some("WebSearch")
        );
        assert_eq!(
            arr[0]
                .get("input")
                .and_then(|v| v.get("query"))
                .and_then(Value::as_str),
            Some("https://example.com")
        );
    }

    #[test]
    fn merge_tool_result_into_previous_tool_use_message() {
        let mut messages = vec![build_codex_message(
            "assistant-1".to_string(),
            "session-1",
            "2026-02-19T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call_abc",
                "name": "Bash",
                "input": { "command": "pwd" }
            }])),
            None,
        )];

        let result_msg = build_codex_message(
            "user-1".to_string(),
            "session-1",
            "2026-02-19T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call_abc",
                "content": "ok"
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let merged_arr = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("assistant message content should be an array");
        assert_eq!(merged_arr.len(), 2);
        assert_eq!(
            merged_arr[1].get("type").and_then(Value::as_str),
            Some("tool_result")
        );
    }

    #[test]
    fn merge_marks_failed_apply_patch_result_as_error() {
        let mut messages = vec![build_codex_message(
            "assistant-1".to_string(),
            "session-1",
            "2026-07-17T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call_patch",
                "name": "apply_patch",
                "input": { "patch": "*** Begin Patch\n*** End Patch" }
            }])),
            None,
        )];
        let result_msg = build_codex_message(
            "user-1".to_string(),
            "session-1",
            "2026-07-17T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call_patch",
                "content": "apply_patch verification failed: expected context was not found"
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let result = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.get(1))
            .expect("merged tool result should exist");
        assert_eq!(result.get("is_error").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn merge_keeps_successful_apply_patch_result_non_error() {
        let mut messages = vec![build_codex_message(
            "assistant-1".to_string(),
            "session-1",
            "2026-07-17T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call_patch",
                "name": "apply_patch",
                "input": { "patch": "*** Begin Patch\n*** End Patch" }
            }])),
            None,
        )];
        let result_msg = build_codex_message(
            "user-1".to_string(),
            "session-1",
            "2026-07-17T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call_patch",
                "content": "Success. Updated the following files:\nM src/lib.rs"
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let result = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.get(1))
            .expect("merged tool result should exist");
        assert!(result.get("is_error").is_none());
    }

    #[test]
    fn merge_does_not_reclassify_another_tool_result() {
        let mut messages = vec![build_codex_message(
            "assistant-1".to_string(),
            "session-1",
            "2026-07-17T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell_command",
                "input": { "command": "echo test" }
            }])),
            None,
        )];
        let result_msg = build_codex_message(
            "user-1".to_string(),
            "session-1",
            "2026-07-17T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call_shell",
                "content": "apply_patch verification failed: quoted diagnostic"
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let result = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|blocks| blocks.get(1))
            .expect("merged tool result should exist");
        assert!(result.get("is_error").is_none());
    }

    #[test]
    fn failed_apply_patch_result_recognizes_invalid_patch_output() {
        assert!(failed_apply_patch_result(&json!({
            "type": "tool_result",
            "content": "Invalid patch text"
        })));
    }

    #[test]
    fn build_codex_message_sets_tool_use_from_content() {
        let msg = build_codex_message(
            "assistant-1".to_string(),
            "session-1",
            "2026-02-19T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call_1",
                "name": "Bash",
                "input": {"command": "pwd"}
            }])),
            None,
        );

        assert!(msg.tool_use.is_some());
        assert_eq!(
            msg.tool_use
                .as_ref()
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str),
            Some("Bash")
        );
    }

    #[test]
    fn convert_task_started_event_to_progress_message() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "task_started",
                "turn_id": "turn_1"
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        )
        .expect("task_started should be converted");

        assert_eq!(msg.message_type, "progress");
        assert_eq!(
            msg.data
                .as_ref()
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("started")
        );
    }

    #[test]
    fn convert_context_compacted_event_to_system_message() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "context_compacted"
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        )
        .expect("context_compacted should be converted");

        assert_eq!(msg.message_type, "system");
        assert_eq!(msg.subtype.as_deref(), Some("microcompact_boundary"));
    }

    #[test]
    fn convert_agent_reasoning_event_to_thinking_message() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "agent_reasoning",
                "text": "**Inspecting parsers**"
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        )
        .expect("agent_reasoning should be converted");

        assert_eq!(msg.message_type, "assistant");
        let arr = msg
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("content should be an array");
        assert_eq!(arr[0].get("type").and_then(Value::as_str), Some("thinking"));
        assert_eq!(
            arr[0].get("thinking").and_then(Value::as_str),
            Some("**Inspecting parsers**")
        );
    }

    #[test]
    fn convert_agent_reasoning_event_skips_empty_text() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "agent_reasoning",
                "text": "   "
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        );

        assert!(msg.is_none());
        assert_eq!(counter, 0);
    }

    #[test]
    fn convert_agent_message_event_not_handled() {
        // agent_message events are skipped in load_messages() to avoid
        // duplicating response_item messages. convert_codex_event should
        // return None for them.
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "agent_message",
                "message": "Working on requested changes"
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        );
        assert!(msg.is_none());
    }

    #[test]
    fn convert_user_message_event_not_handled() {
        // user_message events are skipped in load_messages() to avoid
        // duplicating response_item messages. convert_codex_event should
        // return None for them.
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "user_message",
                "message": "Please patch this file"
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        );
        assert!(msg.is_none());
    }

    #[test]
    fn convert_thread_rolled_back_to_system_boundary() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "thread_rolled_back",
                "num_turns": 2
            }),
            "session-1",
            "2026-07-16T03:06:56Z",
            &mut counter,
        )
        .expect("rollback event should be retained");

        assert_eq!(msg.message_type, "system");
        assert_eq!(msg.subtype.as_deref(), Some("thread_rolled_back"));
        assert_eq!(
            msg.data
                .as_ref()
                .and_then(|value| value.get("numTurns"))
                .and_then(Value::as_u64),
            Some(2)
        );
    }

    #[test]
    fn convert_compacted_line_to_system_message() {
        let mut counter = 0u64;
        let msg = convert_codex_compacted(
            &json!({
                "message": "",
                "replacement_history": [{"type":"message"}]
            }),
            "session-1",
            "2026-02-19T12:00:00Z",
            &mut counter,
        );

        assert_eq!(msg.message_type, "system");
        assert_eq!(msg.subtype.as_deref(), Some("compact_boundary"));
        assert_eq!(
            msg.compact_metadata
                .as_ref()
                .and_then(|v| v.get("replacementHistoryCount"))
                .and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    #[serial]
    fn load_messages_parses_codex_rollout_end_to_end() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-2026-02-19.jsonl");

        let lines = vec![
            json!({
                "timestamp": "2026-02-19T12:00:00Z",
                "type": "session_meta",
                "payload": { "id": "sess-1" }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:01Z",
                "type": "turn_context",
                "payload": { "model": "gpt-5-codex" }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "item-1",
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": "{\"cmd\":\"pwd\"}"
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:03Z",
                "type": "response_item",
                "payload": {
                    "id": "item-2",
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"output\":\"/tmp\",\"metadata\":{\"exit_code\":0}}"
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:04Z",
                "type": "response_item",
                "payload": {
                    "id": "item-3",
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "done" }]
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:05Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "output_tokens": 20
                        }
                    }
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:06Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_started",
                    "turn_id": "turn_1"
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:07Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn_1"
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:08.000Z",
                "type": "compacted",
                "payload": {
                    "replacement_history": [{ "type": "message" }, { "type": "summary" }]
                }
            }),
            json!({
                "timestamp": "2026-02-19T12:00:08.020Z",
                "type": "event_msg",
                "payload": {
                    "type": "context_compacted"
                }
            }),
        ];

        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n")).expect("fixture should be written");

        let messages = load_messages(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("rollout should be parsed");

        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].message_type, "assistant");
        assert_eq!(messages[1].message_type, "assistant");
        assert_eq!(messages[2].message_type, "progress");
        assert_eq!(messages[3].message_type, "progress");
        assert_eq!(messages[4].message_type, "system");

        let first_blocks = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("first message content should be an array");
        assert_eq!(first_blocks.len(), 2);
        assert_eq!(
            first_blocks[0].get("type").and_then(Value::as_str),
            Some("tool_use")
        );
        assert_eq!(
            first_blocks[1].get("type").and_then(Value::as_str),
            Some("tool_result")
        );
        assert_eq!(
            first_blocks[1].get("content").and_then(Value::as_str),
            Some("/tmp")
        );

        assert_eq!(
            messages[0]
                .tool_use
                .as_ref()
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str),
            Some("Bash")
        );
        assert_eq!(messages[0].model.as_deref(), Some("gpt-5-codex"));
        assert_eq!(messages[1].model.as_deref(), Some("gpt-5-codex"));

        assert_eq!(
            messages[1].usage.as_ref().and_then(|u| u.input_tokens),
            Some(100)
        );
        assert_eq!(
            messages[1].usage.as_ref().and_then(|u| u.output_tokens),
            Some(20)
        );

        assert_eq!(
            messages[2]
                .data
                .as_ref()
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("started")
        );
        assert_eq!(
            messages[3]
                .data
                .as_ref()
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("completed")
        );
        assert_eq!(messages[4].subtype.as_deref(), Some("compact_boundary"));
        assert_eq!(
            messages[4]
                .compact_metadata
                .as_ref()
                .and_then(|v| v.get("replacementHistoryCount"))
                .and_then(Value::as_u64),
            Some(2)
        );

        assert!(messages
            .iter()
            .all(|m| m.provider.as_deref() == Some("codex")));
        assert!(messages.iter().all(|m| m.session_id == "sess-1"));
    }

    #[test]
    #[serial]
    fn load_messages_skips_duplicate_event_msg_for_user_and_agent() {
        // Codex logs user/assistant text in both response_item (type=message)
        // and event_msg (type=user_message / agent_message). Only the
        // response_item version should be kept.
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-dedup-test.jsonl");

        let lines = [
            json!({
                "timestamp": "2026-03-01T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "sess-dedup" }
            }),
            // User message via response_item (canonical)
            json!({
                "timestamp": "2026-03-01T10:00:01Z",
                "type": "response_item",
                "payload": {
                    "id": "item-u1",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hello" }]
                }
            }),
            // Duplicate user message via event_msg (should be skipped)
            json!({
                "timestamp": "2026-03-01T10:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "hello"
                }
            }),
            // Assistant message via response_item (canonical)
            json!({
                "timestamp": "2026-03-01T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "item-a1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "hi there" }]
                }
            }),
            // Duplicate assistant message via event_msg (should be skipped)
            json!({
                "timestamp": "2026-03-01T10:00:02Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "hi there"
                }
            }),
            // Non-duplicate event (token_count) should still be processed
            json!({
                "timestamp": "2026-03-01T10:00:03Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 50,
                            "output_tokens": 10
                        }
                    }
                }
            }),
        ];

        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n")).expect("fixture should be written");

        let messages = load_messages(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("rollout should be parsed");

        // Only 2 messages: 1 user + 1 assistant (no duplicates from event_msg)
        // Before this fix, there were 4 messages (each duplicated by event_msg).
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, "user");
        assert_eq!(messages[1].message_type, "assistant");

        // Verify content is correct
        let user_text = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|arr| arr[0].get("text"))
            .and_then(Value::as_str);
        assert_eq!(user_text, Some("hello"));

        let assistant_text = messages[1]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|arr| arr[0].get("text"))
            .and_then(Value::as_str);
        assert_eq!(assistant_text, Some("hi there"));

        // token_count event should still be applied to assistant message
        assert_eq!(
            messages[1].usage.as_ref().and_then(|u| u.input_tokens),
            Some(50)
        );
    }

    #[test]
    #[serial]
    fn load_messages_dedup_multi_turn_conversation() {
        // Simulates a realistic multi-turn Codex conversation where each
        // user/assistant message appears as both response_item and event_msg.
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-multiturn.jsonl");

        let lines = [
            json!({
                "timestamp": "2026-03-01T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "sess-multi" }
            }),
            // Turn 1: user
            json!({
                "timestamp": "2026-03-01T10:00:01Z",
                "type": "response_item",
                "payload": {
                    "id": "u1", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "first question" }]
                }
            }),
            json!({
                "timestamp": "2026-03-01T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "first question" }
            }),
            // Turn 1: assistant
            json!({
                "timestamp": "2026-03-01T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "a1", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "first answer" }]
                }
            }),
            json!({
                "timestamp": "2026-03-01T10:00:02Z",
                "type": "event_msg",
                "payload": { "type": "agent_message", "message": "first answer" }
            }),
            // Turn 2: user
            json!({
                "timestamp": "2026-03-01T10:00:03Z",
                "type": "response_item",
                "payload": {
                    "id": "u2", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "follow-up" }]
                }
            }),
            json!({
                "timestamp": "2026-03-01T10:00:03Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "follow-up" }
            }),
            // Turn 2: assistant
            json!({
                "timestamp": "2026-03-01T10:00:04Z",
                "type": "response_item",
                "payload": {
                    "id": "a2", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "second answer" }]
                }
            }),
            json!({
                "timestamp": "2026-03-01T10:00:04Z",
                "type": "event_msg",
                "payload": { "type": "agent_message", "message": "second answer" }
            }),
            // Turn 3: user (final, no assistant reply yet)
            json!({
                "timestamp": "2026-03-01T10:00:05Z",
                "type": "response_item",
                "payload": {
                    "id": "u3", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "one more thing" }]
                }
            }),
            json!({
                "timestamp": "2026-03-01T10:00:05Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "one more thing" }
            }),
        ];

        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n")).expect("fixture should be written");

        let messages = load_messages(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("rollout should be parsed");

        // 5 messages: user, assistant, user, assistant, user (no duplicates)
        // Without the fix this would be 10 messages.
        assert_eq!(messages.len(), 5);

        let expected = [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "follow-up"),
            ("assistant", "second answer"),
            ("user", "one more thing"),
        ];
        for (i, (msg_type, text)) in expected.iter().enumerate() {
            assert_eq!(messages[i].message_type, *msg_type, "message {i} type");
            let actual_text = messages[i]
                .content
                .as_ref()
                .and_then(Value::as_array)
                .and_then(|arr| arr[0].get("text"))
                .and_then(Value::as_str);
            assert_eq!(actual_text, Some(*text), "message {i} content");
        }
    }

    #[test]
    #[serial]
    fn load_messages_marks_only_same_turn_steers() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-steer.jsonl");

        let lines = [
            json!({
                "timestamp": "2026-07-14T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "sess-steer" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-1" }
            }),
            // User-role context is model input, not an authored prompt, because it has
            // no matching user_message event. It must never consume the turn's first
            // authored-message slot or receive steer provenance.
            json!({
                "timestamp": "2026-07-14T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "context-1", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>...</environment_context>" }]
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:03Z",
                "type": "response_item",
                "payload": {
                    "id": "u1", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "start the work" }]
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:03Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "start the work" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:04Z",
                "type": "response_item",
                "payload": {
                    "id": "a1", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "working" }]
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:05Z",
                "type": "response_item",
                "payload": {
                    "id": "u2", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "focus on tests first" }]
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:05Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "focus on tests first" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:06Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-1" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:07Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-2" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:08Z",
                "type": "response_item",
                "payload": {
                    "id": "u3", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "ordinary follow-up" }]
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:08Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "ordinary follow-up" }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:09Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-2" }
            }),
        ];

        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n"))
            .expect("rollout fixture should be written");

        let messages = load_messages(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("rollout should be parsed");
        let users = messages
            .iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 4);
        assert_eq!(users[0].subtype, None, "injected context is not a steer");
        assert_eq!(users[1].subtype, None, "the turn's first prompt is normal");
        assert_eq!(users[2].subtype.as_deref(), Some(STEER_SUBTYPE));
        assert_eq!(
            users[3].subtype, None,
            "the next task's first prompt resets steer detection"
        );
    }

    #[test]
    #[serial]
    fn load_sessions_includes_archived_sessions() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        let archived_dir = codex_home.join("archived_sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        fs::create_dir_all(&archived_dir).expect("archived dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        let active_rollout = sessions_dir.join("rollout-active.jsonl");
        let archived_rollout = archived_dir.join("rollout-archived.jsonl");
        let active_lines = [
            json!({
                "type": "session_meta",
                "payload": { "id": "active-session", "cwd": project_cwd }
            }),
            json!({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "created_at": "2026-02-21T10:00:00Z",
                    "content": [{ "type": "input_text", "text": "active" }]
                }
            }),
        ];
        let archived_lines = [
            json!({
                "type": "session_meta",
                "payload": { "id": "archived-session", "cwd": project_cwd }
            }),
            json!({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "created_at": "2026-02-21T11:00:00Z",
                    "content": [{ "type": "input_text", "text": "archived" }]
                }
            }),
        ];
        let active_content = active_lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let archived_content = archived_lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&active_rollout, format!("{active_content}\n"))
            .expect("active fixture should be written");
        fs::write(&archived_rollout, format!("{archived_content}\n"))
            .expect("archived fixture should be written");

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(sessions.len(), 2);
        let active = sessions
            .iter()
            .find(|s| s.actual_session_id == "active-session")
            .expect("active session should be listed");
        let archived = sessions
            .iter()
            .find(|s| s.actual_session_id == "archived-session")
            .expect("archived session should be listed");
        assert!(!is_archived_session_path(Path::new(&active.file_path)));
        assert!(is_archived_session_path(Path::new(&archived.file_path)));
    }

    #[test]
    #[serial]
    fn missing_cwd_sessions_load_from_unknown_project() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let rollout_path = sessions_dir.join("rollout-no-cwd.jsonl");
        let lines = [
            json!({
                "type": "session_meta",
                "payload": { "id": "no-cwd-session" }
            }),
            json!({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "created_at": "2026-02-21T10:00:00Z",
                    "content": [{ "type": "input_text", "text": "missing cwd" }]
                }
            }),
        ];
        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n")).expect("fixture should be written");

        let projects = scan_projects_from_path(
            codex_home
                .to_str()
                .expect("codex home path should be valid UTF-8"),
        )
        .expect("projects should be scanned");
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].path, "codex://unknown");

        let sessions = load_sessions("codex://unknown", false).expect("sessions should be loaded");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].actual_session_id, "no-cwd-session");
    }

    #[test]
    #[serial]
    fn load_sessions_does_not_mark_generated_sqlite_title_as_renamed() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        write_codex_rollout(
            &sessions_dir,
            "rollout-native-title.jsonl",
            "native-title-session",
            project_cwd,
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "native-title-session",
                "Generated Codex title",
                "Original first prompt",
            )],
        );
        write_session_index(
            &codex_home,
            &[json!({
                "id": "native-title-session",
                "thread_name": "Generated Codex title"
            })],
        );

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(sessions.len(), 1);
        assert_eq!(
            sessions[0].summary.as_deref(),
            Some("Generated Codex title")
        );
        assert!(!sessions[0].is_renamed);
    }

    #[test]
    #[serial]
    fn native_title_uses_latest_session_index_name_when_sqlite_title_is_preview() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        write_codex_rollout(
            &sessions_dir,
            "rollout-index-title.jsonl",
            "index-title-session",
            project_cwd,
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "index-title-session",
                "Original first prompt",
                "Original first prompt",
            )],
        );
        write_session_index(
            &codex_home,
            &[
                json!({"id":"index-title-session","thread_name":"Older title","updated_at":"2026-02-21T10:00:00Z"}),
                json!({"malformed":"ignored"}),
                json!({"id":"index-title-session","thread_name":"Persisted title","updated_at":"2026-02-21T11:00:00Z"}),
            ],
        );

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(sessions[0].summary.as_deref(), Some("Persisted title"));
        assert!(sessions[0].is_renamed);
    }

    #[test]
    #[serial]
    fn native_title_treats_latest_preview_name_as_reset() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        write_codex_rollout(
            &sessions_dir,
            "rollout-reset-title.jsonl",
            "reset-title-session",
            project_cwd,
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "reset-title-session",
                "Original first prompt",
                "Original first prompt",
            )],
        );
        write_session_index(
            &codex_home,
            &[
                json!({"id":"reset-title-session","thread_name":"Temporary title","updated_at":"2026-02-21T10:00:00Z"}),
                json!({"id":"reset-title-session","thread_name":"Original first prompt","updated_at":"2026-02-21T11:00:00Z"}),
            ],
        );

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(
            sessions[0].summary.as_deref(),
            Some("Original first prompt")
        );
        assert!(!sessions[0].is_renamed);
    }

    #[test]
    #[serial]
    fn native_title_prefers_distinct_sqlite_title_over_legacy_index() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        write_codex_rollout(
            &sessions_dir,
            "rollout-sqlite-title.jsonl",
            "sqlite-title-session",
            project_cwd,
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "sqlite-title-session",
                "Current SQLite title",
                "Original first prompt",
            )],
        );
        write_session_index(
            &codex_home,
            &[
                json!({"id":"sqlite-title-session","thread_name":"Legacy index title","updated_at":"2026-02-21T10:00:00Z"}),
            ],
        );

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(sessions[0].summary.as_deref(), Some("Current SQLite title"));
        assert!(sessions[0].is_renamed);
    }

    #[test]
    #[serial]
    fn native_title_compares_with_preview_not_injected_first_user_message() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let project_cwd = "/Users/jack/client/claude-code-history-viewer";
        write_codex_rollout(
            &sessions_dir,
            "rollout-preview-title.jsonl",
            "preview-title-session",
            project_cwd,
            "Injected wrapper",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "preview-title-session",
                "Actual first prompt",
                "Actual first prompt",
            )],
        );
        let conn = Connection::open(codex_home.join(STATE_DB_FILENAME))
            .expect("codex state db should be readable");
        conn.execute(
            "UPDATE threads SET first_user_message = ?1 WHERE id = ?2",
            rusqlite::params!["Injected wrapper", "preview-title-session"],
        )
        .expect("first user message should be updated");

        let sessions = load_sessions(&format!("codex://{project_cwd}"), false)
            .expect("sessions should be loaded");

        assert_eq!(sessions[0].summary.as_deref(), Some("Actual first prompt"));
        assert!(!sessions[0].is_renamed);
    }

    #[test]
    fn native_title_uses_session_index_without_sqlite() {
        let tmp = TempDir::new().expect("temp dir should be created");
        write_session_index(
            tmp.path(),
            &[json!({
                "id": "index-only-session",
                "thread_name": "Persisted index-only title"
            })],
        );

        let titles = load_native_title_index(tmp.path().to_str().unwrap());
        let title = titles
            .get("index-only-session")
            .expect("index-only title should be loaded");

        assert_eq!(title.title, "Persisted index-only title");
        assert!(title.is_renamed);
    }

    #[test]
    fn native_title_marks_changed_index_history_when_latest_matches_sqlite() {
        let tmp = TempDir::new().expect("temp dir should be created");
        create_codex_state_db(
            tmp.path(),
            &[(
                "renamed-after-generation",
                "Manual title",
                "Original first prompt",
            )],
        );
        write_session_index(
            tmp.path(),
            &[
                json!({"id":"renamed-after-generation","thread_name":"Generated title"}),
                json!({"id":"renamed-after-generation","thread_name":"Manual title"}),
            ],
        );

        let titles = load_native_title_index(tmp.path().to_str().unwrap());
        let title = titles
            .get("renamed-after-generation")
            .expect("changed title history should be loaded");

        assert_eq!(title.title, "Manual title");
        assert!(title.is_renamed);
    }

    #[test]
    fn native_title_supports_legacy_database_without_preview() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let conn = Connection::open(tmp.path().join(STATE_DB_FILENAME))
            .expect("legacy Codex state db should be created");
        conn.execute(
            "CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                first_user_message TEXT NOT NULL
            )",
            [],
        )
        .expect("legacy threads table should be created");
        conn.execute(
            "INSERT INTO threads (id, title, first_user_message) VALUES (?1, ?2, ?3)",
            rusqlite::params![
                "legacy-session",
                "Legacy native title",
                "Original first prompt"
            ],
        )
        .expect("legacy thread row should be inserted");
        drop(conn);

        let titles = load_native_title_index(tmp.path().to_str().unwrap());
        let title = titles
            .get("legacy-session")
            .expect("legacy title should be loaded");

        assert_eq!(title.title, "Legacy native title");
        assert!(!title.is_renamed);
    }

    #[test]
    #[serial]
    fn rename_session_title_updates_codex_state_db_and_resets_to_first_prompt() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let rollout_path = write_codex_rollout(
            &sessions_dir,
            "rollout-rename-title.jsonl",
            "rename-title-session",
            "/Users/jack/client/claude-code-history-viewer",
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[(
                "rename-title-session",
                "Original first prompt",
                "Original first prompt",
            )],
        );

        let result = rename_session_title(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
            "  Better Codex title  ",
        )
        .expect("rename should update state db");

        assert_eq!(result.previous_title, "Original first prompt");
        assert_eq!(result.new_title, "Better Codex title");

        let conn = Connection::open(codex_home.join(STATE_DB_FILENAME))
            .expect("codex state db should be readable");
        let title: String = conn
            .query_row(
                "SELECT title FROM threads WHERE id = ?1",
                rusqlite::params!["rename-title-session"],
                |row| row.get(0),
            )
            .expect("renamed title should be readable");
        assert_eq!(title, "Better Codex title");

        let reset = rename_session_title(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
            "",
        )
        .expect("reset should update state db");
        assert_eq!(reset.previous_title, "Better Codex title");
        assert_eq!(reset.new_title, "Original first prompt");
    }

    #[test]
    #[serial]
    fn delete_session_title_removes_only_the_matching_thread_row() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let rollout_path = write_codex_rollout(
            &sessions_dir,
            "rollout-delete-cleanup.jsonl",
            "delete-cleanup-session",
            "/Users/jack/client/claude-code-history-viewer",
            "Original first prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[
                (
                    "delete-cleanup-session",
                    "Pinned title",
                    "Original first prompt",
                ),
                ("unrelated-session", "Keep me", "other prompt"),
            ],
        );

        delete_session_title(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("delete should clean the thread row");

        let conn = Connection::open(codex_home.join(STATE_DB_FILENAME))
            .expect("codex state db should be readable");
        let removed: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM threads WHERE id = ?1",
                rusqlite::params!["delete-cleanup-session"],
                |row| row.get(0),
            )
            .expect("count query should run");
        assert_eq!(removed, 0, "deleted session's thread row should be gone");

        let kept: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM threads WHERE id = ?1",
                rusqlite::params!["unrelated-session"],
                |row| row.get(0),
            )
            .expect("count query should run");
        assert_eq!(kept, 1, "unrelated thread rows must be untouched");
    }

    #[test]
    #[serial]
    fn delete_session_title_is_noop_without_state_db() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("02")
            .join("21");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);

        let rollout_path = write_codex_rollout(
            &sessions_dir,
            "rollout-no-state-db.jsonl",
            "no-db-session",
            "/tmp/project",
            "hello",
        );
        // No state_5.sqlite exists — cleanup must be a no-op, not an error.
        assert!(delete_session_title(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .is_ok());
    }

    #[test]
    #[serial]
    fn load_messages_accepts_archived_session_path() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let archived_dir = codex_home.join("archived_sessions");
        fs::create_dir_all(&archived_dir).expect("archived dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = archived_dir.join("rollout-archived-only.jsonl");
        let lines = [
            json!({
                "type": "session_meta",
                "payload": { "id": "archived-session", "cwd": "/tmp/project" }
            }),
            json!({
                "type": "response_item",
                "payload": {
                    "id": "item-1",
                    "type": "message",
                    "role": "assistant",
                    "created_at": "2026-02-21T10:00:00Z",
                    "content": [{ "type": "output_text", "text": "ok" }]
                }
            }),
        ];
        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n")).expect("fixture should be written");

        let messages = load_messages(
            rollout_path
                .to_str()
                .expect("rollout path should be valid UTF-8"),
        )
        .expect("archived rollout should be parsed");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].session_id, "archived-session");
    }

    /// Helper: write `lines` as one JSON-per-line into a fresh rollout file
    /// and run `extract_session_info` against it. Returns the resulting
    /// `SessionInfo`. Used by the env-context-skip tests below.
    fn run_extract_session_info_on_lines(lines: Vec<Value>) -> SessionInfo {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("rollout-2026-05-13.jsonl");
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");
        extract_session_info(&rollout_path).expect("extract_session_info should succeed")
    }

    fn run_extract_project_scan_info_on_lines(lines: Vec<Value>) -> ProjectScanInfo {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("rollout-2026-05-13.jsonl");
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");
        extract_project_scan_info(&rollout_path).expect("extract_project_scan_info should succeed")
    }

    fn session_meta_line() -> Value {
        json!({
            "timestamp": "2026-05-13T08:00:00Z",
            "type": "session_meta",
            "payload": { "id": "sess-env-ctx", "cwd": "/tmp/proj" }
        })
    }

    fn user_message_line(timestamp: &str, text: &str) -> Value {
        json!({
            "timestamp": timestamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            }
        })
    }

    const ENV_CONTEXT_BLOCK: &str = "<environment_context>\n  <cwd>/tmp/proj</cwd>\n  <shell>powershell</shell>\n  <current_date>2026-05-13</current_date>\n  <timezone>Asia/Shanghai</timezone>\n</environment_context>";

    #[test]
    fn project_scan_info_uses_lightweight_metadata() {
        let info = run_extract_project_scan_info_on_lines(vec![
            session_meta_line(),
            json!({
                "timestamp": "2026-05-13T08:00:01Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "duplicate event" }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hello" }]
                }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": "{}"
                }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:04Z",
                "type": "response_item",
                "payload": { "type": "reasoning", "summary": [] }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "done"
                }
            }),
        ]);

        assert_eq!(info.cwd.as_deref(), Some("/tmp/proj"));
        assert!(info.message_count > 0);
        assert!(!info.last_modified.is_empty());
    }

    #[test]
    fn json_field_matcher_accepts_whitespace_around_colon() {
        let line = br#"{ "type" : "response_item", "payload": { "type" : "message" } }"#;

        assert!(has_json_string_field_value(
            line,
            JSON_TYPE_KEY,
            b"response_item"
        ));
        assert!(has_json_string_field_value(line, JSON_TYPE_KEY, b"message"));
    }

    #[test]
    /// First user message is an auto-injected `<environment_context>` block;
    /// second user message is a real prompt — the summary should be the
    /// real prompt, not the env-context block.
    fn extract_session_info_skips_environment_context_wrapper() {
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line("2026-05-13T08:00:01Z", ENV_CONTEXT_BLOCK),
            user_message_line(
                "2026-05-13T08:00:02Z",
                "Please review my PR for the Antigravity provider.",
            ),
        ]);

        assert_eq!(
            info.summary.as_deref(),
            Some("Please review my PR for the Antigravity provider.")
        );
        // message_count counts *every* response_item type=message,
        // including the skipped wrapper, so the count surfaces real
        // activity volume.
        assert_eq!(info.message_count, 2);
    }

    #[test]
    /// First user message is a real prompt — extractor must not regress
    /// pre-existing behaviour for sessions without an env-context wrapper.
    fn extract_session_info_uses_first_real_user_prompt() {
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line("2026-05-13T08:00:01Z", "fix the WSL crash"),
            user_message_line("2026-05-13T08:00:02Z", "second message"),
        ]);

        assert_eq!(info.summary.as_deref(), Some("fix the WSL crash"));
        assert_eq!(info.message_count, 2);
    }

    #[test]
    fn extract_session_info_stamps_provider_qualified_source() {
        let info = run_extract_session_info_on_lines(vec![json!({
            "timestamp": "2026-05-13T08:00:00Z",
            "type": "session_meta",
            "payload": { "id": "sess-source", "cwd": "/tmp/proj", "source": "vscode" }
        })]);
        assert_eq!(info.entrypoint.as_deref(), Some("codex-vscode"));

        assert_eq!(
            codex_entrypoint(Some(&json!("cli"))).as_deref(),
            Some("codex-cli")
        );
        assert_eq!(
            codex_entrypoint(Some(&json!("future-surface"))).as_deref(),
            Some("codex-future-surface")
        );
        assert_eq!(codex_entrypoint(Some(&json!(" "))), None);
    }

    #[test]
    fn extract_session_info_stamps_structured_subagent_source() {
        let info = run_extract_session_info_on_lines(vec![json!({
            "timestamp": "2026-05-13T08:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "sess-subagent",
                "cwd": "/tmp/proj",
                "forked_from_id": "sess-parent",
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": "sess-parent",
                            "depth": 1,
                            "agent_path": "/root/research",
                            "agent_nickname": "Parfit"
                        }
                    }
                }
            }
        })]);

        assert_eq!(info.entrypoint.as_deref(), Some("codex-subagent"));
        assert_eq!(info.forked_from_id.as_deref(), Some("sess-parent"));
        assert_eq!(codex_entrypoint(Some(&json!({ "subagent": {} }))), None);
        assert_eq!(codex_entrypoint(Some(&json!({ "future": {} }))), None);
    }

    #[test]
    /// Session contains only auto-injected wrapper messages and no real
    /// prompt — summary stays None, matching legacy empty-session behaviour.
    fn extract_session_info_env_context_only_yields_no_summary() {
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line("2026-05-13T08:00:01Z", ENV_CONTEXT_BLOCK),
        ]);

        assert!(
            info.summary.is_none(),
            "env-context-only sessions should not produce a misleading summary; got {:?}",
            info.summary
        );
        // The wrapper still counts as a message — only the summary is gated.
        assert_eq!(info.message_count, 1);
    }

    fn session_meta_line_with(timestamp: &str, id: &str, cwd: &str) -> Value {
        json!({
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": { "id": id, "cwd": cwd }
        })
    }

    fn forked_session_meta_line_with(
        timestamp: &str,
        id: &str,
        cwd: &str,
        forked_from_id: Value,
    ) -> Value {
        json!({
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": {
                "id": id,
                "cwd": cwd,
                "forked_from_id": forked_from_id
            }
        })
    }

    #[test]
    /// `codex fork` creates the new rollout with its own `session_meta` first,
    /// then replays the source rollout verbatim — including the source's
    /// `session_meta` line. The first meta is the file's identity; later metas
    /// are replayed history and must not override it (issue #451: forked
    /// sessions vanished because the session filter used the last meta's cwd
    /// while project scanning used the first).
    fn extract_session_info_keeps_first_session_meta_on_forked_rollout() {
        let info = run_extract_session_info_on_lines(vec![
            forked_session_meta_line_with(
                "2026-05-13T08:00:00Z",
                "sess-fork-new",
                "/tmp/proj-b",
                json!("sess-orig"),
            ),
            forked_session_meta_line_with(
                "2026-05-12T08:00:00Z",
                "sess-orig",
                "/tmp/proj-a",
                json!("older-origin"),
            ),
            user_message_line("2026-05-13T08:00:01Z", "continue from the forked session"),
        ]);

        assert_eq!(info.session_id, "sess-fork-new");
        assert_eq!(info.cwd.as_deref(), Some("/tmp/proj-b"));
        assert_eq!(info.forked_from_id.as_deref(), Some("sess-orig"));
    }

    #[test]
    fn extract_session_info_ignores_invalid_fork_provenance() {
        for forked_from_id in [Value::Null, json!(""), json!("  "), json!(42), json!({})] {
            let info = run_extract_session_info_on_lines(vec![
                forked_session_meta_line_with(
                    "2026-05-13T08:00:00Z",
                    "sess-native",
                    "/tmp/proj",
                    forked_from_id,
                ),
                forked_session_meta_line_with(
                    "2026-05-12T08:00:00Z",
                    "replayed-session",
                    "/tmp/older-proj",
                    json!("replayed-parent"),
                ),
                user_message_line("2026-05-13T08:00:01Z", "native session"),
            ]);

            assert_eq!(info.forked_from_id, None);
        }
    }

    #[test]
    /// Messages replayed after the source's `session_meta` line in a forked
    /// rollout must carry the forked file's own session id, not the source's.
    fn parse_rollout_file_keeps_first_session_meta_id_on_forked_rollout() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("rollout-2026-05-13.jsonl");
        let lines = [
            session_meta_line_with("2026-05-13T08:00:00Z", "sess-fork-new", "/tmp/proj-b"),
            session_meta_line_with("2026-05-12T08:00:00Z", "sess-orig", "/tmp/proj-a"),
            user_message_line("2026-05-13T08:00:01Z", "continue from the forked session"),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");

        let messages =
            parse_rollout_file(&rollout_path).expect("parse_rollout_file should succeed");

        assert!(!messages.is_empty());
        assert!(
            messages.iter().all(|m| m.session_id == "sess-fork-new"),
            "all messages should carry the forked file's own session id; got {:?}",
            messages
                .iter()
                .map(|m| m.session_id.clone())
                .collect::<Vec<_>>()
        );
    }

    fn turn_context_line(timestamp: &str, cwd: &str) -> Value {
        json!({
            "timestamp": timestamp,
            "type": "turn_context",
            "payload": { "turn_id": "turn-1", "cwd": cwd, "model": "gpt-5" }
        })
    }

    fn write_rollout_lines(dir: &Path, file_name: &str, lines: &[Value]) -> std::path::PathBuf {
        let rollout_path = dir.join(file_name);
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");
        rollout_path
    }

    #[test]
    /// Newer Codex builds can leave rollouts with no `session_meta` line at
    /// all (issue #451 follow-up). Identity must then come from fallbacks:
    /// cwd from the LAST `turn_context` (a fork replays the source's turn
    /// contexts first, so the last one is where the session actually runs)
    /// and the session id from the rollout filename.
    fn extract_session_info_falls_back_when_rollout_has_no_session_meta() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = write_rollout_lines(
            tmp.path(),
            "rollout-2026-07-09T10-00-00-019cf000-aaaa-7000-8000-f986e7b4c56a.jsonl",
            &[
                turn_context_line("2026-07-09T10:00:00Z", "/tmp/proj-a"),
                user_message_line("2026-07-09T10:00:01Z", "replayed from the source session"),
                turn_context_line("2026-07-09T10:00:02Z", "/tmp/proj-b"),
                user_message_line("2026-07-09T10:00:03Z", "continue in the fork's folder"),
            ],
        );

        let info = extract_session_info(&rollout_path).expect("extract_session_info");
        assert_eq!(info.cwd.as_deref(), Some("/tmp/proj-b"));
        assert_eq!(info.session_id, "019cf000-aaaa-7000-8000-f986e7b4c56a");

        let cwd = extract_session_cwd(&rollout_path).expect("extract_session_cwd");
        assert_eq!(cwd.as_deref(), Some("/tmp/proj-b"));
    }

    #[test]
    /// `session_meta`, when present, still wins over any `turn_context` fallback.
    fn extract_session_info_prefers_session_meta_over_turn_context() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = write_rollout_lines(
            tmp.path(),
            "rollout-2026-07-09T10-00-00-019cf000-bbbb-7000-8000-f986e7b4c56a.jsonl",
            &[
                session_meta_line_with("2026-07-09T10:00:00Z", "sess-meta", "/tmp/proj-meta"),
                turn_context_line("2026-07-09T10:00:01Z", "/tmp/proj-turn"),
                user_message_line("2026-07-09T10:00:02Z", "hello"),
            ],
        );

        let info = extract_session_info(&rollout_path).expect("extract_session_info");
        assert_eq!(info.cwd.as_deref(), Some("/tmp/proj-meta"));
        assert_eq!(info.session_id, "sess-meta");
        assert_eq!(
            extract_session_cwd(&rollout_path).unwrap().as_deref(),
            Some("/tmp/proj-meta")
        );
    }

    #[test]
    /// Messages in a meta-less rollout carry the filename-derived session id.
    fn parse_rollout_file_uses_filename_session_id_without_meta() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = write_rollout_lines(
            tmp.path(),
            "rollout-2026-07-09T10-00-00-019cf000-cccc-7000-8000-f986e7b4c56a.jsonl",
            &[
                turn_context_line("2026-07-09T10:00:00Z", "/tmp/proj-b"),
                user_message_line("2026-07-09T10:00:01Z", "no meta anywhere"),
            ],
        );

        let messages = parse_rollout_file(&rollout_path).expect("parse_rollout_file");
        assert!(!messages.is_empty());
        assert!(messages
            .iter()
            .all(|m| m.session_id == "019cf000-cccc-7000-8000-f986e7b4c56a"));
    }

    #[test]
    /// Codex compresses old rollouts to `.jsonl.zst`; they must stay
    /// discoverable and parseable, and a compressed file whose plain twin
    /// exists must be skipped (the plain one is the materialized, current
    /// version).
    fn compressed_rollouts_are_discovered_and_parsed() {
        let tmp = TempDir::new().expect("temp dir should be created");

        let lines = [
            session_meta_line_with("2026-07-09T10:00:00Z", "sess-zst", "/tmp/proj-z"),
            user_message_line("2026-07-09T10:00:01Z", "hello from a compressed rollout"),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let compressed = zstd::encode_all(body.as_bytes(), 3).expect("zstd encode");
        let zst_path = tmp
            .path()
            .join("rollout-2026-07-09T10-00-00-019cf000-dddd-7000-8000-f986e7b4c56a.jsonl.zst");
        fs::write(&zst_path, compressed).expect("write zst fixture");

        assert!(is_rollout_jsonl(&zst_path));
        assert!(is_discoverable_rollout(&zst_path));

        let info = extract_session_info(&zst_path).expect("extract_session_info on zst");
        assert_eq!(info.session_id, "sess-zst");
        assert_eq!(info.cwd.as_deref(), Some("/tmp/proj-z"));

        let messages = parse_rollout_file(&zst_path).expect("parse zst rollout");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].session_id, "sess-zst");

        // Once the plain twin exists, the compressed copy is no longer listed.
        let plain_path = zst_path.with_extension("");
        fs::write(&plain_path, format!("{body}\n")).expect("write plain twin");
        assert!(!is_discoverable_rollout(&zst_path));
        assert!(is_discoverable_rollout(&plain_path));
    }

    #[test]
    /// Filename-derived session ids also work for compressed rollouts, whose
    /// `file_stem` still carries a ".jsonl" tail.
    fn session_id_from_rollout_filename_handles_zst() {
        assert_eq!(
            session_id_from_rollout_filename(Path::new(
                "rollout-2026-07-09T10-00-00-019cf000-eeee-7000-8000-f986e7b4c56a.jsonl.zst"
            )),
            Some("019cf000-eeee-7000-8000-f986e7b4c56a".to_string())
        );
        assert_eq!(
            session_id_from_rollout_filename(Path::new("rollout-short.jsonl")),
            None
        );
    }
}
