//! Headless normalized session dump (`--dump-session`).
//!
//! Routes a session id (or provider-scoped session path) through the
//! multi-provider registry and prints the normalized `Vec<ClaudeMessage>` as
//! JSON to stdout — the same shape the GUI consumes. Unlike `--export`
//! (Claude-only, raw JSONL `Value`s), this works for ANY provider and emits the
//! unified model while preserving message content verbatim.
//!
//! It reuses the existing multi-provider commands (`scan_all_projects`,
//! `load_provider_sessions`, `load_provider_messages`); those take plain
//! `String`s and no Tauri state. They are `async` only as Tauri-command
//! scaffolding — their bodies do synchronous std work and merely `.await` each
//! other, never a reactor — so a trivial inline `block_on` drives them with no
//! Tokio dependency and no GUI/webview.
//!
//! Two companion commands share this file: `--list-sessions` (which also stamps
//! each session with its decoded project directory, so callers can match the cwd
//! without reproducing Claude's storage-path encoding) and `--capabilities` (a
//! tiny version/feature probe so callers can fail fast on an incompatible build).

use crate::cli_args::extract_flag_value;
use crate::commands::multi_provider::{
    load_provider_messages, load_provider_sessions, scan_all_projects,
};
use crate::models::ClaudeSession;
use base64::prelude::{Engine as _, BASE64_STANDARD};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Contract version of the headless JSON surface. Bump on a breaking change to
/// the commands or their output shapes so callers can assert compatibility.
const HEADLESS_API_VERSION: u32 = 1;

/// Minimal executor for the sync-bodied futures in the load path. They resolve
/// on the first poll; the `Pending` arm is defensive and never expected here.
/// Uses the safe no-op waker (the crate denies `unsafe`).
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Waker};

    let mut cx = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Serialize `value` as pretty JSON. With `--output <file>` it is written there
/// (so the caller reads clean JSON regardless of anything else on stdout, e.g.
/// upstream debug logs); otherwise it goes to stdout. Returns the exit code.
fn emit_json<T: serde::Serialize>(args: &[String], value: &T) -> i32 {
    let json = match serde_json::to_string_pretty(value) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to serialize JSON: {e}");
            return 1;
        }
    };
    match extract_flag_value(args, "--output") {
        Some(path) => match std::fs::write(&path, json.as_bytes()) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Failed to write {path}: {e}");
                1
            }
        },
        None => {
            println!("{json}");
            0
        }
    }
}

/// Version/feature probe emitted by `--capabilities`. Lets a caller (e.g. ccmsg)
/// confirm the binary speaks the headless protocol it needs, and which version,
/// rather than inferring it from a failed call.
#[derive(serde::Serialize)]
struct Capabilities {
    api_version: u32,
    /// The base app version (crate version), for diagnostics.
    version: &'static str,
    /// Headless commands this build supports (flag names without the `--`).
    commands: Vec<&'static str>,
}

/// Handle the `--capabilities` CLI flag. Prints the headless protocol version and
/// supported commands as JSON. Returns the process exit code.
pub fn run_capabilities(args: &[String]) -> i32 {
    let caps = Capabilities {
        api_version: HEADLESS_API_VERSION,
        version: env!("CARGO_PKG_VERSION"),
        commands: vec!["dump-session", "list-sessions", "capabilities"],
    };
    emit_json(args, &caps)
}

const USAGE: &str = "Usage: --dump-session <session-id|session-path> [--provider <name>] [--format json] [--output <file>]\n\n\
Resolve a session through the multi-provider registry and print its normalized\n\
messages (Vec<ClaudeMessage>) as JSON. <session-id> is matched against the\n\
provider's sessions (full id, or an unambiguous id prefix); an absolute path or\n\
a provider-scoped path (e.g. cursor://<id>) is used directly. --provider\n\
defaults to 'claude'. With --output the JSON is written to <file> (so the\n\
caller is unaffected by anything else on stdout); otherwise it goes to stdout.";

/// A value that already names a concrete session location, so no id resolution
/// is needed: an absolute filesystem path, or a provider scheme like
/// `cursor://composerId`.
fn looks_like_session_path(value: &str) -> bool {
    value.contains("://") || std::path::Path::new(value).is_absolute()
}

/// Resolve a session id to the provider-scoped `session_path` expected by
/// `load_provider_messages`. A direct path passes through unchanged.
async fn resolve_session_path(provider: &str, id: &str) -> Result<String, String> {
    if looks_like_session_path(id) {
        return Ok(id.to_string());
    }

    // Narrow the scan to just the requested provider.
    let projects =
        scan_all_projects(None, Some(vec![provider.to_string()]), None, None, None).await?;

    // Layered matching, most-specific first. The file-stem layer disambiguates a
    // Claude main session (file `<id>.jsonl`) from its sidechains, which share
    // the same `actual_session_id`. Non-Claude providers fall through to id
    // matching, where the session path isn't a `<id>.jsonl` file.
    let mut by_stem: Vec<String> = Vec::new();
    let mut by_id: Vec<String> = Vec::new();
    let mut by_prefix: Vec<String> = Vec::new();
    for project in projects {
        let sessions =
            load_provider_sessions(provider.to_string(), project.path.clone(), Some(false)).await?;
        for s in sessions {
            let stem = std::path::Path::new(&s.file_path)
                .file_stem()
                .and_then(|os| os.to_str());
            if stem == Some(id) || s.session_id == id {
                by_stem.push(s.session_id);
            } else if s.actual_session_id == id {
                by_id.push(s.session_id);
            } else if s.actual_session_id.starts_with(id) || stem.is_some_and(|st| st.starts_with(id)) {
                by_prefix.push(s.session_id);
            }
        }
    }

    let mut matches = if !by_stem.is_empty() {
        by_stem
    } else if !by_id.is_empty() {
        by_id
    } else {
        by_prefix
    };
    match matches.len() {
        0 => Err(format!("No {provider} session found matching '{id}'")),
        1 => Ok(matches.remove(0)),
        n => Err(format!(
            "'{id}' is ambiguous — {n} {provider} sessions match; use the full id or a session path"
        )),
    }
}

/// Handle the `--dump-session` CLI flag. Returns the process exit code.
pub fn run_dump_session(args: &[String]) -> i32 {
    let Some(id) = extract_flag_value(args, "--dump-session") else {
        eprintln!("{USAGE}");
        return 2;
    };
    let provider = extract_flag_value(args, "--provider").unwrap_or_else(|| "claude".to_string());
    let format = extract_flag_value(args, "--format").unwrap_or_else(|| "json".to_string());
    if format != "json" {
        eprintln!("Unsupported --format '{format}' (only 'json' is supported)");
        return 2;
    }

    let result: Result<Vec<crate::models::ClaudeMessage>, String> = block_on(async {
        let session_path = resolve_session_path(&provider, &id).await?;
        load_provider_messages(provider.clone(), session_path).await
    });

    let messages = match result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    emit_json(args, &messages)
}

const LIST_USAGE: &str = "Usage: --list-sessions [--provider <name>] [--project <path>] [--format json] [--output <file>]\n\n\
List a provider's sessions as JSON (Vec<ClaudeSession> with summary/title,\n\
timestamps, message_count, ids). With --project <path> only that project's\n\
sessions are listed (path is the provider storage dir, e.g. a folder under\n\
~/.claude/projects); otherwise sessions across all of the provider's projects\n\
are returned. Each session is also stamped with `project_path` (the decoded\n\
project directory) when known. --provider defaults to 'claude'.";

/// The Claude Code VS Code extension "deletes" a session by adding its id to a
/// `hiddenSessionIds` array in the editor's global-state DB — a soft hide that
/// leaves the `.jsonl` on disk untouched, so a filesystem scan still lists it.
/// These helpers read that list so `--list-sessions` can stamp `is_hidden`,
/// letting a caller mark such sessions instead of showing them as active.
///
/// The store is `<user-data>/globalStorage/state.vscdb` (an SQLite DB with one
/// `ItemTable(key, value)`); the extension's blob lives under key
/// `Anthropic.claude-code`. The list is global (keyed by session id, not path),
/// so we union it across every installed editor flavor. Best-effort throughout:
/// a missing / locked / renamed store contributes nothing rather than failing
/// the listing — so detection degrades to "show everything", never an error.
const CLAUDE_VSCODE_STATE_KEY: &str = "Anthropic.claude-code";

/// `state.vscdb` paths for every installed editor flavor that can host the
/// Claude Code extension (stock VS Code, its insiders/OSS builds, and the forks).
fn claude_global_state_dbs() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    const FLAVORS: [&str; 6] = [
        "Code",
        "Code - Insiders",
        "VSCodium",
        "VSCodium - Insiders",
        "Cursor",
        "Windsurf",
    ];
    #[cfg(target_os = "windows")]
    let base = home.join("AppData/Roaming");
    #[cfg(target_os = "macos")]
    let base = home.join("Library/Application Support");
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let base = home.join(".config");

    FLAVORS
        .iter()
        .map(|flavor| {
            base.join(flavor)
                .join("User")
                .join("globalStorage")
                .join("state.vscdb")
        })
        .filter(|db| db.is_file())
        .collect()
}

/// Read the `hiddenSessionIds` array from one `state.vscdb`. Returns `None` on
/// any failure (locked/absent DB, missing key, non-JSON value) — best-effort.
fn read_hidden_ids(db: &Path) -> Option<Vec<String>> {
    // Read-only, like the other providers' `state.vscdb` reads; a WAL DB open by
    // a running editor still permits concurrent readers.
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
    let value: String = conn
        .query_row(
            "SELECT value FROM ItemTable WHERE key = ?1",
            [CLAUDE_VSCODE_STATE_KEY],
            |row| row.get(0),
        )
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&value).ok()?;
    let ids = parsed.get("hiddenSessionIds")?.as_array()?;
    Some(
        ids.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
    )
}

/// The union of Claude VS Code hidden (soft-deleted) session ids across every
/// installed editor flavor. Empty when nothing is installed or readable.
fn claude_hidden_session_ids() -> HashSet<String> {
    let mut hidden = HashSet::new();
    for db in claude_global_state_dbs() {
        if let Some(ids) = read_hidden_ids(&db) {
            hidden.extend(ids);
        }
    }
    hidden
}

// ── Copilot (VS Code) session lifecycle: `orphan` and `archived` ─────────────
//
// VS Code Copilot Chat writes each session to
// `workspaceStorage/<hash>/chatSessions/<id>.jsonl` and tracks two independent
// per-workspace states in that workspace's own `state.vscdb`:
//   • `chat.ChatSessionStore.index` — the recent-session list, hard-capped at 50
//     (VS Code's `trimEntries`). When a workspace exceeds 50, the oldest by
//     recency are dropped *from the index only* (the file stays). So a listed
//     session whose id is absent from this index has aged out of the visible
//     list — we mark it **orphan**.
//   • `agentSessions.state.cache` — an *uncapped* array of `{resource, archived}`
//     entries. `archived:true` on a `vscode-chat-session://local/<base64-id>`
//     resource is an explicit, reversible hide (the file is kept) — we mark it
//     **archived**.
// A user *delete* removes the file outright, so it never reaches this listing and
// needs no marker. `archived` takes precedence over `orphan` (an archived session
// is deliberately hidden, not merely aged out), so a session is at most one of the
// two. Both are read live from the workspace `state.vscdb`; best-effort — an
// unreadable store just yields neither mark. VS Code surface only.
const COPILOT_CHAT_INDEX_KEY: &str = "chat.ChatSessionStore.index";
const COPILOT_AGENT_STATE_KEY: &str = "agentSessions.state.cache";
const COPILOT_VSCODE_ENTRYPOINT: &str = "copilot-vscode";
const COPILOT_LOCAL_RESOURCE_PREFIX: &str = "vscode-chat-session://local/";

#[derive(Default)]
struct WorkspaceChatState {
    /// Session ids in `chat.ChatSessionStore.index` (VS Code's recent list).
    /// `None` when the index couldn't be read — so orphan is never asserted on a
    /// guess (a missing index means "unknown", not "everything orphaned").
    index_ids: Option<HashSet<String>>,
    /// Session ids marked `archived:true` in `agentSessions.state.cache`.
    archived_ids: HashSet<String>,
}

/// The workspace `state.vscdb` for a Copilot VS Code session, derived from the
/// session's own absolute file path `…/workspaceStorage/<hash>/chatSessions/
/// <id>.jsonl` — so it works regardless of where the editor's user-data dir lives
/// (default or a custom/portable location). `None` if the shape doesn't match or
/// the db is absent.
fn copilot_workspace_state_db(file_path: &str) -> Option<PathBuf> {
    let chat_sessions = Path::new(file_path).parent()?;
    if chat_sessions.file_name()?.to_str()? != "chatSessions" {
        return None;
    }
    let db = chat_sessions.parent()?.join("state.vscdb");
    db.is_file().then_some(db)
}

/// Decode a `vscode-chat-session://local/<base64>` resource into its session id
/// (the base64 payload is the plain UUID string). `None` for non-local resources
/// (e.g. `openai-codex://…` agent sessions) or malformed input.
fn decode_local_chat_resource(resource: &str) -> Option<String> {
    let b64 = resource.strip_prefix(COPILOT_LOCAL_RESOURCE_PREFIX)?;
    String::from_utf8(BASE64_STANDARD.decode(b64).ok()?).ok()
}

/// Read a Copilot VS Code workspace's chat state (recent-list ids + archived ids)
/// from its `state.vscdb`. Best-effort per key — a missing/locked db or key just
/// yields the empty/`None` default.
fn read_workspace_chat_state(db: &Path) -> WorkspaceChatState {
    let mut state = WorkspaceChatState::default();
    let Ok(conn) = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY) else {
        return state;
    };
    let read_value = |key: &str| -> Option<String> {
        conn.query_row(
            "SELECT value FROM ItemTable WHERE key = ?1",
            [key],
            |row| row.get(0),
        )
        .ok()
    };
    if let Some(value) = read_value(COPILOT_CHAT_INDEX_KEY) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&value) {
            if let Some(entries) = parsed.get("entries").and_then(|e| e.as_object()) {
                state.index_ids = Some(entries.keys().cloned().collect());
            }
        }
    }
    if let Some(value) = read_value(COPILOT_AGENT_STATE_KEY) {
        if let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(&value) {
            for entry in entries {
                let archived = entry.get("archived").and_then(|a| a.as_bool()) == Some(true);
                let resource = entry.get("resource").and_then(|r| r.as_str());
                if let (true, Some(id)) = (archived, resource.and_then(decode_local_chat_resource)) {
                    state.archived_ids.insert(id);
                }
            }
        }
    }
    state
}

/// Classifies Copilot VS Code sessions as `orphan` / `archived`, caching each
/// workspace's `state.vscdb` read so a workspace with many sessions is read once.
struct CopilotClassifier {
    enabled: bool,
    cache: RefCell<HashMap<PathBuf, WorkspaceChatState>>,
}

impl CopilotClassifier {
    fn new(provider: &str) -> Self {
        Self {
            // Only relevant when listing Copilot; every other provider is inert.
            enabled: provider == "copilot",
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Returns `(is_orphan, is_archived)` for a session — `(false, false)` unless
    /// it's a Copilot VS Code session with a readable workspace state.
    fn classify(&self, session: &ClaudeSession) -> (bool, bool) {
        if !self.enabled || session.entrypoint.as_deref() != Some(COPILOT_VSCODE_ENTRYPOINT) {
            return (false, false);
        }
        let Some(db) = copilot_workspace_state_db(&session.file_path) else {
            return (false, false);
        };
        let mut cache = self.cache.borrow_mut();
        let state = cache
            .entry(db.clone())
            .or_insert_with(|| read_workspace_chat_state(&db));
        let id = &session.actual_session_id;
        let archived = state.archived_ids.contains(id);
        // Orphan only when the index was read and this id is absent — and not when
        // already archived (archived is the more specific, deliberate state).
        let orphan = !archived
            && state
                .index_ids
                .as_ref()
                .is_some_and(|ids| !ids.contains(id));
        (orphan, archived)
    }
}

/// A `ClaudeSession` flattened with its decoded project directory. `project_path`
/// is the project's real filesystem path (its original working directory), so a
/// caller can match the cwd against it directly instead of reproducing Claude's
/// lossy storage-folder encoding. Omitted when not known (an explicit
/// `--project` storage path is not resolved back to its decoded form).
#[derive(serde::Serialize)]
struct SessionWithProjectPath {
    #[serde(flatten)]
    session: ClaudeSession,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_path: Option<String>,
    /// True when this session was soft-deleted (hidden) in the Claude Code VS
    /// Code extension. Claude-only; always `false` for other providers. Read
    /// live from the editor's global state — deliberately not cached, since the
    /// hidden list changes independently of the session file's (mtime, size).
    is_hidden: bool,
    /// Copilot VS Code only: the session's file is on disk but its id has aged
    /// out of the workspace's 50-entry `chat.ChatSessionStore.index` (VS Code no
    /// longer lists it). `false` for other providers/surfaces. See the classifier.
    is_orphan: bool,
    /// Copilot VS Code only: the session is explicitly archived (`archived:true`
    /// in the workspace's `agentSessions.state.cache`) — a deliberate, reversible
    /// hide that keeps the file. `false` for other providers/surfaces.
    is_archived: bool,
    /// Claude only: this local session was "teleported" — its conversation was
    /// relocated to a cloud (web) session and the local `.jsonl` emptied to a
    /// single `teleported-from` redirect stub. The normal metadata scan drops it
    /// (0 conversational messages), so it is re-surfaced here (see
    /// `scan_teleport_stubs`). `false` for a normal session and every other
    /// provider.
    is_teleported: bool,
    /// Claude only: the cloud session id a teleported stub points at (its
    /// `remoteSessionId`), so a caller can direct the user to the Web tab. `None`
    /// for a normal session, other providers, or a stub missing the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_session_id: Option<String>,
}

/// A teleport redirect stub: a Claude session whose local `.jsonl` was emptied to
/// a single `teleported-from` record when its conversation was relocated to a
/// cloud (web) session. The base's normal metadata scan drops such a file (it has
/// 0 conversational messages), so `--list-sessions` re-surfaces it here — stamped
/// `is_teleported` with the `remote_session_id` it points at — letting a caller
/// mark it (and point the user at the Web tab) instead of the session silently
/// vanishing. Claude-only: teleport is a Claude-cloud feature; no other provider
/// writes such a stub.
const TELEPORT_RECORD_TYPE: &str = "teleported-from";

/// A teleport stub is a single ~100-byte record. Cap candidate files well above
/// that but far below any real session, so a non-listed large/corrupt file is
/// never read (a real session is always already listed and skipped anyway).
const TELEPORT_STUB_MAX_BYTES: u64 = 1024;

#[derive(serde::Deserialize)]
struct TeleportStubRecord {
    #[serde(rename = "type")]
    record_type: String,
    #[serde(rename = "remoteSessionId")]
    remote_session_id: Option<String>,
}

/// If `path` is a teleport stub — its first non-empty line is a `teleported-from`
/// record — return that record's `remoteSessionId` (an inner `None` means the
/// field was absent). An outer `None` means "not a teleport stub". Best-effort:
/// any read/parse failure, or a file larger than a stub, is treated as "not a
/// stub" (a real session is far larger and never reaches here anyway).
fn read_teleport_stub(path: &Path) -> Option<Option<String>> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > TELEPORT_STUB_MAX_BYTES {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    let line = content.lines().find(|l| !l.trim().is_empty())?;
    let record: TeleportStubRecord = serde_json::from_str(line).ok()?;
    (record.record_type == TELEPORT_RECORD_TYPE).then_some(record.remote_session_id)
}

/// Build the listing entry for a teleported session. It has no local content, so
/// the fields are synthesized: id from the filename, timestamps from the file
/// mtime, and a placeholder summary (the original title lived in the now-relocated
/// conversation and is not recoverable from the stub).
fn synthesize_teleport_session(
    path: &Path,
    remote_session_id: Option<String>,
    project_path: Option<String>,
) -> SessionWithProjectPath {
    let file_path = path.to_string_lossy().to_string();
    let actual_session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-session")
        .to_string();
    let modified = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let dt: DateTime<Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_else(|| Utc::now().to_rfc3339());
    let project_name = project_path
        .as_deref()
        .and_then(|p| Path::new(p).file_name().and_then(|n| n.to_str()))
        .or_else(|| path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()))
        .unwrap_or("Unknown")
        .to_string();
    SessionWithProjectPath {
        session: ClaudeSession {
            session_id: file_path.clone(),
            actual_session_id,
            file_path,
            project_name,
            message_count: 0,
            first_message_time: modified.clone(),
            last_message_time: modified.clone(),
            last_modified: modified,
            has_tool_use: false,
            has_errors: false,
            // Neutral, direction-correct placeholder: the stub is a redirect
            // pointer to a cloud session (remote→local), not necessarily a session
            // that was "moved to the cloud". No real title survives in the stub.
            summary: Some("(teleported · cloud session)".to_string()),
            is_renamed: false,
            provider: None,
            storage_type: None,
            entrypoint: None,
        },
        project_path,
        is_hidden: false,
        is_orphan: false,
        is_archived: false,
        is_teleported: true,
        remote_session_id,
    }
}

/// Scan one project storage `dir` for teleport stubs not already in `listed` (the
/// file paths of the sessions the normal scan returned — every real session is
/// there, so only dropped/empty files are inspected, and only the tiny ones read).
/// Claude-only; the caller gates on the provider.
fn scan_teleport_stubs(
    dir: &Path,
    listed: &HashSet<String>,
    project_path: Option<String>,
) -> Vec<SessionWithProjectPath> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if listed.contains(&path.to_string_lossy().to_string()) {
            continue;
        }
        if let Some(remote_session_id) = read_teleport_stub(&path) {
            out.push(synthesize_teleport_session(
                &path,
                remote_session_id,
                project_path.clone(),
            ));
        }
    }
    out
}

/// List sessions for `provider`, optionally limited to one project storage path.
async fn list_sessions(
    provider: &str,
    project: Option<&str>,
) -> Result<Vec<SessionWithProjectPath>, String> {
    // The hidden list is a Claude VS Code concept; read it once per listing and
    // only for the Claude provider (empty set → every session is `is_hidden:false`).
    let hidden = if provider == "claude" {
        claude_hidden_session_ids()
    } else {
        HashSet::new()
    };
    let is_hidden = |session: &ClaudeSession| hidden.contains(&session.actual_session_id);
    // Copilot VS Code orphan/archived classification (inert for other providers),
    // caching each workspace's state.vscdb read across its sessions.
    let copilot = CopilotClassifier::new(provider);
    let wrap = |session: ClaudeSession, project_path: Option<String>| {
        let (is_orphan, is_archived) = copilot.classify(&session);
        SessionWithProjectPath {
            is_hidden: is_hidden(&session),
            is_orphan,
            is_archived,
            is_teleported: false,
            remote_session_id: None,
            session,
            project_path,
        }
    };

    if let Some(path) = project {
        // A non-existent project dir (e.g. no sessions for this cwd) is not an
        // error — it just yields an empty list.
        if !std::path::Path::new(path).exists() {
            return Ok(Vec::new());
        }
        let sessions =
            load_provider_sessions(provider.to_string(), path.to_string(), Some(true)).await?;
        let mut wrapped: Vec<SessionWithProjectPath> =
            sessions.into_iter().map(|session| wrap(session, None)).collect();
        // Teleport stubs (Claude only) are dropped by the metadata scan; re-surface
        // any in this storage dir that weren't already listed.
        if provider == "claude" {
            let listed = listed_paths(&wrapped);
            wrapped.extend(scan_teleport_stubs(std::path::Path::new(path), &listed, None));
        }
        return Ok(wrapped);
    }

    let projects =
        scan_all_projects(None, Some(vec![provider.to_string()]), None, None, None).await?;
    let mut all: Vec<SessionWithProjectPath> = Vec::new();
    for proj in projects {
        // Skip a project that fails to load rather than aborting the whole list.
        if let Ok(sessions) =
            load_provider_sessions(provider.to_string(), proj.path.clone(), Some(true)).await
        {
            let wrapped: Vec<SessionWithProjectPath> = sessions
                .into_iter()
                .map(|session| wrap(session, Some(proj.actual_path.clone())))
                .collect();
            // Re-surface this project's teleport stubs (Claude only), scoped to the
            // sessions the normal scan already returned for it.
            if provider == "claude" {
                let listed = listed_paths(&wrapped);
                all.extend(wrapped);
                all.extend(scan_teleport_stubs(
                    std::path::Path::new(&proj.path),
                    &listed,
                    Some(proj.actual_path.clone()),
                ));
            } else {
                all.extend(wrapped);
            }
        }
    }
    Ok(all)
}

/// The set of file paths in a wrapped-session list — the "already listed" guard
/// so `scan_teleport_stubs` never re-inspects a real session.
fn listed_paths(wrapped: &[SessionWithProjectPath]) -> HashSet<String> {
    wrapped
        .iter()
        .map(|w| w.session.file_path.clone())
        .collect()
}

/// Handle the `--list-sessions` CLI flag. Returns the process exit code.
pub fn run_list_sessions(args: &[String]) -> i32 {
    let format = extract_flag_value(args, "--format").unwrap_or_else(|| "json".to_string());
    if format != "json" {
        eprintln!("{LIST_USAGE}");
        eprintln!("Unsupported --format '{format}' (only 'json' is supported)");
        return 2;
    }
    let provider = extract_flag_value(args, "--provider").unwrap_or_else(|| "claude".to_string());
    let project = extract_flag_value(args, "--project");

    let result: Result<Vec<SessionWithProjectPath>, String> =
        block_on(async { list_sessions(&provider, project.as_deref()).await });

    match result {
        Ok(sessions) => emit_json(args, &sessions),
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use tempfile::TempDir;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn session(file_path: &Path, id: &str, entrypoint: Option<&str>) -> ClaudeSession {
        ClaudeSession {
            session_id: file_path.to_string_lossy().to_string(),
            actual_session_id: id.to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            project_name: "test-project".to_string(),
            message_count: 2,
            first_message_time: "2026-01-01T00:00:00Z".to_string(),
            last_message_time: "2026-01-01T00:01:00Z".to_string(),
            last_modified: "2026-01-01T00:01:00Z".to_string(),
            has_tool_use: false,
            has_errors: false,
            summary: Some("Test session".to_string()),
            is_renamed: false,
            provider: None,
            storage_type: None,
            entrypoint: entrypoint.map(str::to_string),
        }
    }

    fn create_item_table(db: &Path, values: &[(&str, Value)]) {
        let conn = Connection::open(db).unwrap();
        conn.execute(
            "CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();
        for (key, value) in values {
            conn.execute(
                "INSERT INTO ItemTable(key, value) VALUES (?1, ?2)",
                (key, value.to_string()),
            )
            .unwrap();
        }
    }

    #[test]
    fn capabilities_command_writes_the_headless_contract() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("capabilities.json");
        let argv = args(&[
            "viewer",
            "--capabilities",
            "--output",
            output.to_str().unwrap(),
        ]);

        assert_eq!(run_capabilities(&argv), 0);
        let value: Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(value["api_version"], HEADLESS_API_VERSION);
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(
            value["commands"],
            json!(["dump-session", "list-sessions", "capabilities"])
        );
    }

    #[test]
    fn dump_session_command_loads_an_absolute_claude_path() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("session.jsonl");
        let output = temp.path().join("dump.json");
        std::fs::write(
            &input,
            concat!(
                r#"{"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"hello"}}"#,
                "\n",
                r#"{"type":"assistant","uuid":"a1","parentUuid":"u1","sessionId":"s1","timestamp":"2026-01-01T00:01:00Z","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}"#,
                "\n"
            ),
        )
        .unwrap();
        let argv = args(&[
            "viewer",
            "--dump-session",
            input.to_str().unwrap(),
            "--provider",
            "claude",
            "--output",
            output.to_str().unwrap(),
        ]);

        assert_eq!(run_dump_session(&argv), 0);
        let messages: Vec<Value> = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], "user");
        assert_eq!(messages[0]["content"], "hello");
        assert_eq!(messages[1]["parentUuid"], "u1");
    }

    #[test]
    fn list_sessions_command_emits_normal_and_teleport_entries() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("sessions.json");
        std::fs::write(
            temp.path().join("normal.jsonl"),
            concat!(
                r#"{"type":"user","uuid":"u1","sessionId":"normal","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"hello"}}"#,
                "\n",
                r#"{"type":"assistant","uuid":"a1","sessionId":"normal","timestamp":"2026-01-01T00:01:00Z","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}"#,
                "\n"
            ),
        )
        .unwrap();
        std::fs::write(
            temp.path().join("teleported.jsonl"),
            r#"{"type":"teleported-from","remoteSessionId":"remote-123"}"#,
        )
        .unwrap();
        let argv = args(&[
            "viewer",
            "--list-sessions",
            "--provider",
            "claude",
            "--project",
            temp.path().to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ]);

        assert_eq!(run_list_sessions(&argv), 0);
        let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(rows.len(), 2);
        let normal = rows
            .iter()
            .find(|row| row["actual_session_id"] == "normal")
            .unwrap();
        assert_eq!(normal["message_count"], 2);
        assert_eq!(normal["is_teleported"], false);
        let teleport = rows
            .iter()
            .find(|row| row["actual_session_id"] == "teleported")
            .unwrap();
        assert_eq!(teleport["is_teleported"], true);
        assert_eq!(teleport["remote_session_id"], "remote-123");
        assert_eq!(teleport["message_count"], 0);
    }

    #[test]
    fn flattened_listing_serializes_decoded_project_path_and_flags() {
        let temp = TempDir::new().unwrap();
        let wrapped = SessionWithProjectPath {
            session: session(&temp.path().join("s.jsonl"), "s1", None),
            project_path: Some(r"C:\work\decoded-project".to_string()),
            is_hidden: true,
            is_orphan: false,
            is_archived: false,
            is_teleported: false,
            remote_session_id: None,
        };

        let value = serde_json::to_value(wrapped).unwrap();
        assert_eq!(value["project_path"], r"C:\work\decoded-project");
        assert_eq!(value["is_hidden"], true);
        assert!(
            value.get("session").is_none(),
            "ClaudeSession must stay flattened"
        );
    }

    #[test]
    fn hidden_ids_are_read_best_effort_from_vscode_state() {
        let temp = TempDir::new().unwrap();
        let db = temp.path().join("state.vscdb");
        create_item_table(
            &db,
            &[(
                CLAUDE_VSCODE_STATE_KEY,
                json!({"hiddenSessionIds":["hidden-1", 42, "hidden-2"]}),
            )],
        );

        assert_eq!(
            read_hidden_ids(&db),
            Some(vec!["hidden-1".to_string(), "hidden-2".to_string()])
        );
        assert_eq!(read_hidden_ids(&temp.path().join("missing.vscdb")), None);
    }

    #[test]
    fn copilot_classifier_distinguishes_recent_orphan_and_archived() {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspaceStorage").join("abc");
        let chats = workspace.join("chatSessions");
        std::fs::create_dir_all(&chats).unwrap();
        let db = workspace.join("state.vscdb");
        let archived_resource = format!(
            "{COPILOT_LOCAL_RESOURCE_PREFIX}{}",
            BASE64_STANDARD.encode("archived")
        );
        create_item_table(
            &db,
            &[
                (COPILOT_CHAT_INDEX_KEY, json!({"entries":{"recent":{}}})),
                (
                    COPILOT_AGENT_STATE_KEY,
                    json!([{"resource":archived_resource,"archived":true}]),
                ),
            ],
        );
        let classifier = CopilotClassifier::new("copilot");

        assert_eq!(
            classifier.classify(&session(
                &chats.join("recent.jsonl"),
                "recent",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, false)
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("orphan.jsonl"),
                "orphan",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (true, false)
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("archived.jsonl"),
                "archived",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, true),
            "archive must take precedence over absence from the recent index"
        );
        assert_eq!(
            CopilotClassifier::new("claude").classify(&session(
                &chats.join("orphan.jsonl"),
                "orphan",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, false)
        );
    }

    #[test]
    fn copilot_missing_recent_index_does_not_guess_orphan() {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspaceStorage").join("unknown");
        let chats = workspace.join("chatSessions");
        std::fs::create_dir_all(&chats).unwrap();
        create_item_table(
            &workspace.join("state.vscdb"),
            &[(COPILOT_AGENT_STATE_KEY, json!([]))],
        );

        let result = CopilotClassifier::new("copilot").classify(&session(
            &chats.join("unknown.jsonl"),
            "unknown",
            Some(COPILOT_VSCODE_ENTRYPOINT),
        ));
        assert_eq!(result, (false, false));
    }

    #[test]
    fn teleport_scan_skips_listed_non_stubs_and_oversized_files() {
        let temp = TempDir::new().unwrap();
        let listed_path = temp.path().join("listed.jsonl");
        let teleport_path = temp.path().join("redirect.jsonl");
        std::fs::write(
            &listed_path,
            r#"{"type":"teleported-from","remoteSessionId":"skip-me"}"#,
        )
        .unwrap();
        std::fs::write(
            &teleport_path,
            "\n{\"type\":\"teleported-from\",\"remoteSessionId\":\"remote-1\"}\n",
        )
        .unwrap();
        std::fs::write(temp.path().join("ordinary.jsonl"), r#"{"type":"user"}"#).unwrap();
        std::fs::write(
            temp.path().join("oversized.jsonl"),
            vec![b'x'; TELEPORT_STUB_MAX_BYTES as usize + 1],
        )
        .unwrap();
        let listed = HashSet::from([listed_path.to_string_lossy().to_string()]);

        let found = scan_teleport_stubs(temp.path(), &listed, Some(r"C:\work\project".to_string()));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].session.actual_session_id, "redirect");
        assert_eq!(found[0].remote_session_id.as_deref(), Some("remote-1"));
        assert_eq!(found[0].project_path.as_deref(), Some(r"C:\work\project"));
        assert!(found[0].is_teleported);
    }
}
