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
        return Ok(sessions
            .into_iter()
            .map(|session| wrap(session, None))
            .collect());
    }

    let projects =
        scan_all_projects(None, Some(vec![provider.to_string()]), None, None, None).await?;
    let mut all: Vec<SessionWithProjectPath> = Vec::new();
    for proj in projects {
        // Skip a project that fails to load rather than aborting the whole list.
        if let Ok(sessions) =
            load_provider_sessions(provider.to_string(), proj.path.clone(), Some(true)).await
        {
            all.extend(
                sessions
                    .into_iter()
                    .map(|session| wrap(session, Some(proj.actual_path.clone()))),
            );
        }
    }
    Ok(all)
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
