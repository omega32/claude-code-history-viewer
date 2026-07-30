//! Headless normalized session dumps (`--dump-session` and
//! `--dump-session-snapshot`).
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
//! The snapshot command is additive: providers without an incremental
//! implementation return a complete normalized array inside a `full` envelope.
//! Codex plain rollouts can instead prove the accepted byte prefix and return an
//! exact replacement suffix from a completed-turn checkpoint. The ordinary
//! `--dump-session` array contract remains unchanged.
//!
//! Other companion commands share this file: `--list-sessions` (which also
//! stamps each session with its decoded project directory, so callers can match
//! the cwd without reproducing Claude's storage-path encoding),
//! `--hide-session` (Claude's reversible VS Code deletion state),
//! `--archive-session` / `--unarchive-session` (Copilot VS Code's per-workspace
//! archive state), and `--capabilities` (a tiny version/feature probe so callers
//! can fail fast on an incompatible build).

use crate::cli_args::extract_flag_value;
use crate::commands::multi_provider::{
    finalize_loaded_messages, load_provider_messages, load_provider_sessions, scan_all_projects,
};
use crate::models::{ClaudeMessage, ClaudeSession};
use crate::providers::codex;
use base64::prelude::{Engine as _, BASE64_STANDARD};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior};
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
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWake;
    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
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
    if let Some(path) = extract_flag_value(args, "--output") {
        match std::fs::write(&path, json.as_bytes()) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Failed to write {path}: {e}");
                1
            }
        }
    } else {
        println!("{json}");
        0
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
        commands: vec![
            "dump-session",
            "dump-session-snapshot",
            "list-sessions",
            "hide-session",
            "archive-session",
            "unarchive-session",
            "capabilities",
        ],
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

const SNAPSHOT_USAGE: &str = "Usage: --dump-session-snapshot <session-id|session-path> [--provider <name>] [--cursor <opaque-token>] [--format json] [--output <file>]\n\n\
Return a normalized session envelope. A provider-owned cursor may produce an\n\
unchanged result or an exact normalized replacement suffix; unsupported or\n\
unverifiable transitions return the complete message array. The established\n\
--dump-session array contract remains unchanged.";

#[derive(serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum SessionSnapshotEnvelope {
    Full {
        reason: String,
        messages: Vec<ClaudeMessage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cursor: Option<String>,
        #[serde(rename = "cursorReplaceFrom", skip_serializing_if = "Option::is_none")]
        cursor_replace_from: Option<usize>,
    },
    Unchanged {
        cursor: String,
    },
    Replace {
        #[serde(rename = "replaceFrom")]
        replace_from: usize,
        messages: Vec<ClaudeMessage>,
        cursor: String,
        #[serde(rename = "cursorReplaceFrom")]
        cursor_replace_from: usize,
    },
}

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
            } else if s.actual_session_id.starts_with(id)
                || stem.is_some_and(|st| st.starts_with(id))
            {
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

/// Handle the cursor-aware normalized session refresh command.
pub fn run_dump_session_snapshot(args: &[String]) -> i32 {
    let Some(id) = extract_flag_value(args, "--dump-session-snapshot") else {
        eprintln!("{SNAPSHOT_USAGE}");
        return 2;
    };
    let provider = extract_flag_value(args, "--provider").unwrap_or_else(|| "claude".to_string());
    let format = extract_flag_value(args, "--format").unwrap_or_else(|| "json".to_string());
    if format != "json" {
        eprintln!("Unsupported --format '{format}' (only 'json' is supported)");
        return 2;
    }
    let cursor = extract_flag_value(args, "--cursor");

    let result: Result<SessionSnapshotEnvelope, String> = block_on(async {
        let session_path = resolve_session_path(&provider, &id).await?;
        if provider != "codex" {
            let messages = load_provider_messages(provider.clone(), session_path).await?;
            return Ok(SessionSnapshotEnvelope::Full {
                reason: "unsupported-provider".to_string(),
                messages,
                cursor: None,
                cursor_replace_from: None,
            });
        }

        match codex::load_session_snapshot(&session_path, cursor.as_deref())? {
            codex::SessionSnapshotLoad::Full {
                reason,
                messages,
                cursor,
            } => {
                let original_len = messages.len();
                let messages = finalize_loaded_messages(messages);
                let cursor = (messages.len() == original_len).then_some(cursor).flatten();
                let cursor_replace_from = cursor
                    .as_deref()
                    .map(codex::snapshot_cursor_replace_from)
                    .transpose()?;
                Ok(SessionSnapshotEnvelope::Full {
                    reason,
                    messages,
                    cursor,
                    cursor_replace_from,
                })
            }
            codex::SessionSnapshotLoad::Unchanged { cursor } => {
                Ok(SessionSnapshotEnvelope::Unchanged { cursor })
            }
            codex::SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor,
            } => {
                let original_len = messages.len();
                let messages = finalize_loaded_messages(messages);
                if messages.len() != original_len {
                    let messages = load_provider_messages(provider.clone(), session_path).await?;
                    return Ok(SessionSnapshotEnvelope::Full {
                        reason: "post-normalization-count-changed".to_string(),
                        messages,
                        cursor: None,
                        cursor_replace_from: None,
                    });
                }
                let cursor_replace_from = codex::snapshot_cursor_replace_from(&cursor)?;
                Ok(SessionSnapshotEnvelope::Replace {
                    replace_from,
                    messages,
                    cursor,
                    cursor_replace_from,
                })
            }
        }
    });

    match result {
        Ok(snapshot) => emit_json(args, &snapshot),
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
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
/// The store is `<user-data>/globalStorage/state.vscdb` (an `SQLite` DB with one
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

const HIDE_USAGE: &str =
    "Usage: --hide-session <session-id> [--provider claude] [--format json] [--output <file>]\n\n\
Apply Claude Code's reversible VS Code deletion state by appending <session-id>\n\
to hiddenSessionIds in each installed Claude extension state store. The session\n\
transcript remains on disk. Only the claude provider is supported.";

#[derive(Debug, serde::Serialize)]
struct HideSessionOutcome {
    session_id: String,
    hidden: bool,
    stores_updated: usize,
    stores_already_hidden: usize,
    stores_unavailable: usize,
}

/// Add one id to a single editor store. `Ok(Some(true))` means the row was
/// updated, `Ok(Some(false))` means it already contained the id, and `Ok(None)`
/// means that editor has no Claude extension state. Malformed or locked Claude
/// stores are errors to the caller, which can continue with the other flavors.
fn hide_session_in_db(db: &Path, session_id: &str) -> Result<Option<bool>, String> {
    let mut conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(2))
        .map_err(|e| e.to_string())?;
    // Reserve the writer before reading so another extension-state update cannot
    // land between our read and write and be lost by the JSON-object replacement.
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;
    let value: String = match tx.query_row(
        "SELECT value FROM ItemTable WHERE key = ?1",
        [CLAUDE_VSCODE_STATE_KEY],
        |row| row.get(0),
    ) {
        Ok(value) => value,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let mut parsed: serde_json::Value = serde_json::from_str(&value).map_err(|e| e.to_string())?;
    let state = parsed
        .as_object_mut()
        .ok_or_else(|| "Claude extension state is not a JSON object".to_string())?;
    let ids = state
        .entry("hiddenSessionIds")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| "hiddenSessionIds is not an array".to_string())?;
    if ids.iter().any(|v| v.as_str() == Some(session_id)) {
        return Ok(Some(false));
    }
    ids.push(serde_json::Value::String(session_id.to_string()));
    let updated = serde_json::to_string(&parsed).map_err(|e| e.to_string())?;
    tx.execute(
        "UPDATE ItemTable SET value = ?1 WHERE key = ?2",
        (&updated, CLAUDE_VSCODE_STATE_KEY),
    )
    .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(Some(true))
}

fn hide_claude_session(session_id: &str, dbs: Vec<PathBuf>) -> Result<HideSessionOutcome, String> {
    let mut outcome = HideSessionOutcome {
        session_id: session_id.to_string(),
        hidden: true,
        stores_updated: 0,
        stores_already_hidden: 0,
        stores_unavailable: 0,
    };
    for db in dbs {
        match hide_session_in_db(&db, session_id) {
            Ok(Some(true)) => outcome.stores_updated += 1,
            Ok(Some(false)) => outcome.stores_already_hidden += 1,
            Ok(None) => {}
            Err(_) => outcome.stores_unavailable += 1,
        }
    }
    if outcome.stores_updated + outcome.stores_already_hidden == 0 {
        return Err(
            "No writable Claude VS Code extension state store was found. Open Claude Code in VS Code, then try again."
                .to_string(),
        );
    }
    Ok(outcome)
}

/// Handle Claude's reversible session deletion. This intentionally updates the
/// same private VS Code global-state field as the extension; it never removes a
/// transcript file. Every readable editor flavor is updated so the union used
/// by `--list-sessions` cannot leave the row active in another installed editor.
pub fn run_hide_session(args: &[String]) -> i32 {
    let Some(session_id) = extract_flag_value(args, "--hide-session") else {
        eprintln!("{HIDE_USAGE}");
        return 2;
    };
    if session_id.trim().is_empty() {
        eprintln!("{HIDE_USAGE}");
        return 2;
    }
    let format = extract_flag_value(args, "--format").unwrap_or_else(|| "json".to_string());
    if format != "json" {
        eprintln!("Unsupported --format '{format}' (only 'json' is supported)");
        return 2;
    }
    let provider = extract_flag_value(args, "--provider").unwrap_or_else(|| "claude".to_string());
    if provider != "claude" {
        eprintln!("--hide-session supports only the claude provider (got '{provider}')");
        return 2;
    }
    match hide_claude_session(&session_id, claude_global_state_dbs()) {
        Ok(outcome) => emit_json(args, &outcome),
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

// ── Copilot (VS Code) session lifecycle: `orphan` and `archived` ─────────────
//
// VS Code Copilot Chat writes each session to either
// `workspaceStorage/<hash>/chatSessions/<id>.jsonl` or
// `globalStorage/emptyWindowChatSessions/<id>.jsonl`, and tracks two independent
// states in the owning workspace/global `state.vscdb`:
//   • `chat.ChatSessionStore.index` — the recent-session list, hard-capped at 50
//     (VS Code's `trimEntries`). When a workspace exceeds 50, the oldest by
//     recency are dropped *from the index only* (the file stays). So a listed
//     session whose id is absent from this index has aged out of the visible
//     list — we mark it **orphan**.
//   • `agentSessions.state.cache` — an *uncapped* array of session-state entries.
//     `archived:true` on a `vscode-chat-session://local/<base64-id>` resource is
//     an explicit, reversible hide (the file is kept), while `pinned:true` is the
//     independent pinned-list state — we expose both facts.
// A user *delete* removes the file outright, so it never reaches this listing and
// needs no marker. `archived` takes precedence over `orphan` (an archived session
// is deliberately hidden, not merely aged out), so a session is at most one of the
// two. Both are read live from the owning `state.vscdb`; best-effort — an
// unreadable store just yields neither mark. VS Code surface only.
const COPILOT_CHAT_INDEX_KEY: &str = "chat.ChatSessionStore.index";
const COPILOT_AGENT_STATE_KEY: &str = "agentSessions.state.cache";
const COPILOT_VSCODE_ENTRYPOINT: &str = "copilot-vscode";
const COPILOT_LOCAL_RESOURCE_PREFIX: &str = "vscode-chat-session://local/";

#[derive(Default)]
#[allow(clippy::struct_field_names)]
struct WorkspaceChatState {
    /// Session ids in `chat.ChatSessionStore.index` (VS Code's recent list).
    /// `None` when the index couldn't be read — so orphan is never asserted on a
    /// guess (a missing index means "unknown", not "everything orphaned").
    index_ids: Option<HashSet<String>>,
    /// Session ids marked `archived:true` in `agentSessions.state.cache`.
    archived_ids: HashSet<String>,
    /// Session ids marked `pinned:true` in `agentSessions.state.cache`.
    pinned_ids: HashSet<String>,
}

/// The owning `state.vscdb` for a Copilot VS Code session, derived from either
/// `…/workspaceStorage/<hash>/chatSessions/<id>.jsonl` or
/// `…/globalStorage/emptyWindowChatSessions/<id>.jsonl`. `None` if the shape
/// doesn't match or the database is absent.
fn copilot_workspace_state_db(file_path: &str) -> Option<PathBuf> {
    let sessions = Path::new(file_path).parent()?;
    let db = match sessions.file_name()?.to_str()? {
        "chatSessions" | "emptyWindowChatSessions" => sessions.parent()?.join("state.vscdb"),
        _ => return None,
    };
    db.is_file().then_some(db)
}

/// Decode a `vscode-chat-session://local/<base64>` resource into its session id
/// (the base64 payload is the plain UUID string). `None` for non-local resources
/// (e.g. `openai-codex://…` agent sessions) or malformed input.
fn decode_local_chat_resource(resource: &str) -> Option<String> {
    let b64 = resource.strip_prefix(COPILOT_LOCAL_RESOURCE_PREFIX)?;
    String::from_utf8(BASE64_STANDARD.decode(b64).ok()?).ok()
}

fn local_chat_resource(session_id: &str) -> String {
    format!(
        "{COPILOT_LOCAL_RESOURCE_PREFIX}{}",
        BASE64_STANDARD.encode(session_id)
    )
}

/// VS Code-family flavor that owns one workspace/global-state database. The
/// database path itself is authoritative even for custom user-data locations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditorFlavor {
    Code,
    CodeInsiders,
    Vscodium,
}

impl EditorFlavor {
    fn label(self) -> &'static str {
        match self {
            Self::Code => "Visual Studio Code",
            Self::CodeInsiders => "Visual Studio Code Insiders",
            Self::Vscodium => "VSCodium",
        }
    }

    #[cfg(target_os = "windows")]
    fn process_name(self) -> &'static str {
        match self {
            Self::Code => "Code.exe",
            Self::CodeInsiders => "Code - Insiders.exe",
            Self::Vscodium => "VSCodium.exe",
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn process_needles(self) -> &'static [&'static str] {
        match self {
            Self::Code => &[
                "/visual studio code.app/",
                "/usr/share/code/code",
                "/bin/code",
            ],
            Self::CodeInsiders => &[
                "/visual studio code - insiders.app/",
                "/usr/share/code-insiders/code-insiders",
                "/bin/code-insiders",
            ],
            Self::Vscodium => &["/vscodium.app/", "/usr/share/codium/codium", "/bin/codium"],
        }
    }
}

fn editor_flavor_for_user_data_root(user: &Path) -> Option<EditorFlavor> {
    match user.parent()?.file_name()?.to_str()? {
        "Code" => Some(EditorFlavor::Code),
        "Code - Insiders" => Some(EditorFlavor::CodeInsiders),
        "VSCodium" => Some(EditorFlavor::Vscodium),
        _ => None,
    }
}

fn known_editor_user_data_roots() -> Vec<(PathBuf, EditorFlavor)> {
    crate::providers::vscode::get_base_paths()
        .into_iter()
        .filter_map(|root| {
            let flavor = editor_flavor_for_user_data_root(&root)?;
            Some((root.canonicalize().ok()?, flavor))
        })
        .collect()
}

fn editor_flavor_for_state_db_in(
    db: &Path,
    known_roots: &[(PathBuf, EditorFlavor)],
) -> Option<EditorFlavor> {
    let owner = db.parent()?;
    let user = if owner.file_name()?.to_str()? == "globalStorage" {
        owner.parent()?
    } else {
        let workspace_storage = owner.parent()?;
        if workspace_storage.file_name()?.to_str()? != "workspaceStorage" {
            return None;
        }
        workspace_storage.parent()?
    };
    if user.file_name()?.to_str()? != "User" {
        return None;
    }
    if let Some(flavor) = editor_flavor_for_user_data_root(user) {
        return Some(flavor);
    }

    let canonical_user = user.canonicalize().ok()?;
    known_roots
        .iter()
        .find_map(|(root, flavor)| (root == &canonical_user).then_some(*flavor))
}

fn editor_flavor_for_state_db(db: &Path) -> Option<EditorFlavor> {
    editor_flavor_for_state_db_in(db, &known_editor_user_data_roots())
}

/// External edits to VS Code workspace state can be overwritten by the editor's
/// in-memory cache on its next save. Refuse unless the owning flavor is stopped;
/// an inability to inspect processes is also a refusal, never silent best effort.
fn ensure_editor_stopped(db: &Path) -> Result<(), String> {
    let flavor = editor_flavor_for_state_db(db).ok_or_else(|| {
        format!(
            "Cannot identify the VS Code flavor that owns {}",
            db.display()
        )
    })?;

    #[cfg(target_os = "windows")]
    let running = {
        let image_filter = format!("IMAGENAME eq {}", flavor.process_name());
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &image_filter, "/FO", "CSV", "/NH"])
            .output()
            .map_err(|e| format!("Failed to inspect running editor processes: {e}"))?;
        if !output.status.success() {
            return Err(
                "Failed to inspect running editor processes; archive state was not changed"
                    .to_string(),
            );
        }
        String::from_utf8_lossy(&output.stdout)
            .to_ascii_lowercase()
            .contains(&flavor.process_name().to_ascii_lowercase())
    };

    #[cfg(not(target_os = "windows"))]
    let running = {
        let output = std::process::Command::new("ps")
            .args(["-A", "-o", "command="])
            .output()
            .map_err(|e| format!("Failed to inspect running editor processes: {e}"))?;
        if !output.status.success() {
            return Err(
                "Failed to inspect running editor processes; archive state was not changed"
                    .to_string(),
            );
        }
        let processes = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
        flavor
            .process_needles()
            .iter()
            .any(|needle| processes.contains(needle))
    };

    if running {
        return Err(format!(
            "{} is running and may overwrite external session-state changes. Exit every {} window completely, then try again.",
            flavor.label(),
            flavor.label()
        ));
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct CopilotArchiveOutcome {
    session_id: String,
    archived: bool,
    changed: bool,
}

/// Update exactly one local Copilot session entry while preserving the other
/// entries and every field VS Code may have added. Archiving mirrors VS Code's
/// native behavior by marking the session read at the current timestamp.
fn set_copilot_archive_in_db(
    db: &Path,
    session_id: &str,
    archived: bool,
    now_ms: i64,
) -> Result<bool, String> {
    let mut conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(2))
        .map_err(|e| e.to_string())?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;
    let stored: Option<String> = tx
        .query_row(
            "SELECT value FROM ItemTable WHERE key = ?1",
            [COPILOT_AGENT_STATE_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let mut entries = match stored {
        Some(value) => serde_json::from_str::<Vec<serde_json::Value>>(&value)
            .map_err(|e| format!("Invalid {COPILOT_AGENT_STATE_KEY} value: {e}"))?,
        None => Vec::new(),
    };
    let resource = local_chat_resource(session_id);
    let mut changed = false;
    let mut found = false;
    for entry in &mut entries {
        let matches = entry
            .get("resource")
            .and_then(|value| value.as_str())
            .and_then(decode_local_chat_resource)
            .as_deref()
            == Some(session_id);
        if !matches {
            continue;
        }
        found = true;
        let state = entry
            .as_object_mut()
            .ok_or_else(|| "Copilot session state entry is not a JSON object".to_string())?;
        let current_archived = state
            .get("archived")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if current_archived != archived {
            state.insert("archived".to_string(), serde_json::Value::Bool(archived));
            changed = true;
        }
        if archived {
            let previous = state
                .get("read")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            if previous < now_ms {
                state.insert("read".to_string(), serde_json::Value::Number(now_ms.into()));
                changed = true;
            }
        }
        break;
    }
    if !found && archived {
        let mut state = serde_json::Map::new();
        state.insert("resource".to_string(), serde_json::Value::String(resource));
        state.insert("archived".to_string(), serde_json::Value::Bool(archived));
        if archived {
            state.insert("read".to_string(), serde_json::Value::Number(now_ms.into()));
        }
        entries.push(serde_json::Value::Object(state));
        changed = true;
    }
    if changed {
        let updated = serde_json::to_string(&entries).map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO ItemTable(key, value) VALUES (?1, ?2)\n\
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (COPILOT_AGENT_STATE_KEY, &updated),
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(changed)
}

const COPILOT_ARCHIVE_USAGE: &str =
    "Usage: --archive-session <session-path> | --unarchive-session <session-path> --provider copilot [--format json] [--output <file>]\n\n\
Update Copilot VS Code's reversible archive state. The selected session must be\n\
a local chatSessions or emptyWindowChatSessions JSONL file, and its editor must\n\
be fully stopped so it cannot overwrite the external state change.";

/// Handle Copilot VS Code archive/unarchive. The session path is deliberately
/// required: it identifies the exact workspace database without an ambiguous
/// cross-workspace id scan.
pub fn run_set_session_archived(args: &[String], archived: bool) -> i32 {
    let flag = if archived {
        "--archive-session"
    } else {
        "--unarchive-session"
    };
    let Some(session_path) = extract_flag_value(args, flag) else {
        eprintln!("{COPILOT_ARCHIVE_USAGE}");
        return 2;
    };
    let provider = extract_flag_value(args, "--provider").unwrap_or_else(|| "copilot".to_string());
    if provider != "copilot" {
        eprintln!("{flag} supports only the copilot provider (got '{provider}')");
        return 2;
    }
    let format = extract_flag_value(args, "--format").unwrap_or_else(|| "json".to_string());
    if format != "json" {
        eprintln!("Unsupported --format '{format}' (only 'json' is supported)");
        return 2;
    }
    let path = Path::new(&session_path);
    let Some(db) = copilot_workspace_state_db(&session_path) else {
        eprintln!("The selected session is not a managed Copilot VS Code chat file with an owning state.vscdb");
        return 1;
    };
    let Some(session_id) = path.file_stem().and_then(|stem| stem.to_str()) else {
        eprintln!("The selected Copilot session path has no valid session id");
        return 1;
    };
    if let Err(error) = ensure_editor_stopped(&db) {
        eprintln!("{error}");
        return 1;
    }
    let now_ms = Utc::now().timestamp_millis();
    match set_copilot_archive_in_db(&db, session_id, archived, now_ms) {
        Ok(changed) => emit_json(
            args,
            &CopilotArchiveOutcome {
                session_id: session_id.to_string(),
                archived,
                changed,
            },
        ),
        Err(error) => {
            eprintln!("Failed to update Copilot session archive state: {error}");
            1
        }
    }
}

/// Read a Copilot VS Code store's chat state (recent-list, archived, and pinned
/// ids) from its workspace/global `state.vscdb`. Best-effort per key — a
/// missing/locked db or key just yields the empty/`None` default.
fn read_workspace_chat_state(db: &Path) -> WorkspaceChatState {
    let mut state = WorkspaceChatState::default();
    let Ok(conn) = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY) else {
        return state;
    };
    let read_value = |key: &str| -> Option<String> {
        conn.query_row("SELECT value FROM ItemTable WHERE key = ?1", [key], |row| {
            row.get(0)
        })
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
                let archived =
                    entry.get("archived").and_then(serde_json::Value::as_bool) == Some(true);
                let pinned = entry.get("pinned").and_then(serde_json::Value::as_bool) == Some(true);
                let resource = entry.get("resource").and_then(|r| r.as_str());
                if let Some(id) = resource.and_then(decode_local_chat_resource) {
                    if archived {
                        state.archived_ids.insert(id.clone());
                    }
                    if pinned {
                        state.pinned_ids.insert(id);
                    }
                }
            }
        }
    }
    state
}

/// Classifies Copilot VS Code sessions as `orphan` / `archived` / `pinned`, caching each
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

    /// Returns `(is_orphan, is_archived, is_pinned)` for a session — all false
    /// unless it's a Copilot VS Code session with readable workspace state.
    fn classify(&self, session: &ClaudeSession) -> (bool, bool, bool) {
        if !self.enabled || session.entrypoint.as_deref() != Some(COPILOT_VSCODE_ENTRYPOINT) {
            return (false, false, false);
        }
        let Some(db) = copilot_workspace_state_db(&session.file_path) else {
            return (false, false, false);
        };
        let mut cache = self.cache.borrow_mut();
        let state = cache
            .entry(db.clone())
            .or_insert_with(|| read_workspace_chat_state(&db));
        let id = &session.actual_session_id;
        let archived = state.archived_ids.contains(id);
        let pinned = state.pinned_ids.contains(id);
        // Orphan only when the index was read and this id is absent — and not when
        // already archived (archived is the more specific, deliberate state).
        let orphan = !archived
            && state
                .index_ids
                .as_ref()
                .is_some_and(|ids| !ids.contains(id));
        (orphan, archived, pinned)
    }
}

/// A `ClaudeSession` flattened with its decoded project directory. `project_path`
/// is the project's real filesystem path (its original working directory), so a
/// caller can match the cwd against it directly instead of reproducing Claude's
/// lossy storage-folder encoding. Omitted when not known (an explicit
/// `--project` storage path is not resolved back to its decoded form).
#[derive(serde::Serialize)]
#[allow(clippy::struct_excessive_bools)]
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
    /// True when the provider reports an archived session. For Copilot VS Code,
    /// this comes from `archived:true` in `agentSessions.state.cache`; for Codex,
    /// it records that the rollout was discovered under `archived_sessions`.
    is_archived: bool,
    /// Copilot VS Code only: `pinned:true` in `agentSessions.state.cache`.
    /// Independent of archive/orphan state and `false` for every other surface.
    is_pinned: bool,
    /// Claude only: this local session was "teleported" — its conversation was
    /// relocated to a cloud (web) session and the local `.jsonl` emptied to a
    /// single `teleported-from` redirect stub. The normal metadata scan drops it
    /// (0 conversational messages), so it is re-surfaced here (see
    /// `scan_teleport_stubs`). `false` for a normal session and every other
    /// provider.
    is_teleported: bool,
    /// Codex only: this thread appears in Codex's authoritative external-agent
    /// import ledger.
    is_imported: bool,
    /// Codex only: source provider inferred from the ledger's private source
    /// path. The path itself is never emitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    imported_from: Option<String>,
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
#[allow(clippy::option_option)]
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
/// conversation and is not recoverable from the stub). Hidden state still comes
/// from Claude VS Code's authoritative `hiddenSessionIds`, keyed by the filename.
fn synthesize_teleport_session(
    path: &Path,
    remote_session_id: Option<String>,
    project_path: Option<String>,
    hidden: &HashSet<String>,
) -> SessionWithProjectPath {
    let file_path = path.to_string_lossy().to_string();
    let actual_session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-session")
        .to_string();
    let is_hidden = hidden.contains(&actual_session_id);
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
        .or_else(|| {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
        })
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
        is_hidden,
        is_orphan: false,
        is_archived: false,
        is_pinned: false,
        is_teleported: true,
        is_imported: false,
        imported_from: None,
        remote_session_id,
    }
}

/// Scan one project storage `dir` for teleport stubs not already in `listed` (the
/// file paths of the sessions the normal scan returned — every real session is
/// there, so only dropped/empty files are inspected, and only the tiny ones read).
/// Claude-only; the caller gates on the provider and supplies the same live
/// hidden-id snapshot used to wrap ordinary Claude sessions.
fn scan_teleport_stubs(
    dir: &Path,
    listed: &HashSet<String>,
    project_path: Option<String>,
    hidden: &HashSet<String>,
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
                hidden,
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
    // Copilot VS Code orphan/archive/pin classification (inert for other providers),
    // caching each workspace's state.vscdb read across its sessions. Codex's
    // archived state is cheaper and authoritative: it is the scan-root provenance.
    let copilot = CopilotClassifier::new(provider);
    // Codex's import ledger changes independently of rollout files, so read it
    // once per listing and join by the generated thread id.
    let codex_imports = if provider == "codex" {
        codex::external_agent_imports()
    } else {
        HashMap::new()
    };
    let wrap = |session: ClaudeSession, project_path: Option<String>| {
        let (is_orphan, copilot_archived, is_pinned) = copilot.classify(&session);
        let is_archived = copilot_archived
            || (provider == "codex"
                && codex::is_archived_session_path(Path::new(&session.file_path)));
        let is_imported = codex_imports.contains_key(&session.actual_session_id);
        let imported_from = codex_imports
            .get(&session.actual_session_id)
            .cloned()
            .flatten();
        SessionWithProjectPath {
            is_hidden: is_hidden(&session),
            is_orphan,
            is_archived,
            is_pinned,
            is_teleported: false,
            is_imported,
            imported_from,
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
        let mut wrapped: Vec<SessionWithProjectPath> = sessions
            .into_iter()
            .map(|session| wrap(session, None))
            .collect();
        // Teleport stubs (Claude only) are dropped by the metadata scan; re-surface
        // any in this storage dir that weren't already listed.
        if provider == "claude" {
            let listed = listed_paths(&wrapped);
            wrapped.extend(scan_teleport_stubs(
                std::path::Path::new(path),
                &listed,
                None,
                &hidden,
            ));
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
                    &hidden,
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
    use serial_test::serial;
    use std::ffi::OsString;
    use tempfile::TempDir;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &Path) -> Self {
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
            json!([
                "dump-session",
                "dump-session-snapshot",
                "list-sessions",
                "hide-session",
                "archive-session",
                "unarchive-session",
                "capabilities"
            ])
        );
    }

    #[test]
    fn dump_session_command_loads_an_absolute_claude_path() {
        let temp = TempDir::new().unwrap();
        let input = temp.path().join("session.jsonl");
        let output = temp.path().join("dump.json");
        let snapshot_output = temp.path().join("snapshot.json");
        std::fs::write(
            &input,
            concat!(
                r#"{"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"hello"}}"#,
                "\n",
                r#"{"type":"custom-title","sessionId":"s1","customTitle":"Branch name"}"#,
                "\n",
                r#"{"type":"file-history-delta","messageId":"m1","snapshotMessageId":"snapshot-1","trackingPath":"/tmp/history","backup":{},"timestamp":"2026-01-01T00:00:30Z"}"#,
                "\n",
                r#"{"type":"permission-mode","sessionId":"s1","permissionMode":"acceptEdits"}"#,
                "\n",
                r#"{"type":"relocated","sessionId":"s1","relocatedCwd":"/tmp/project"}"#,
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

        let snapshot_argv = args(&[
            "viewer",
            "--dump-session-snapshot",
            input.to_str().unwrap(),
            "--provider",
            "claude",
            "--output",
            snapshot_output.to_str().unwrap(),
        ]);
        assert_eq!(run_dump_session_snapshot(&snapshot_argv), 0);
        let snapshot: Value =
            serde_json::from_slice(&std::fs::read(snapshot_output).unwrap()).unwrap();
        assert_eq!(snapshot["kind"], "full");
        assert_eq!(snapshot["reason"], "unsupported-provider");
        assert_eq!(snapshot["messages"].as_array().unwrap().len(), 2);
        assert!(snapshot["cursor"].is_null());
    }

    #[test]
    #[serial]
    fn dump_session_snapshot_command_emits_codex_replacement_envelopes() {
        let temp = TempDir::new().unwrap();
        let codex_home = temp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        std::fs::create_dir_all(&sessions_dir).unwrap();
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let input = sessions_dir.join("rollout-snapshot-command.jsonl");
        let output = temp.path().join("snapshot.json");
        std::fs::write(
            &input,
            [
                json!({
                    "timestamp": "2026-07-29T10:00:00Z",
                    "type": "session_meta",
                    "payload": { "id": "snapshot-command", "cwd": "C:/repo" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:01Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": "first" }]
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:02Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-1" }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:03Z",
                    "type": "response_item",
                    "payload": {
                        "id": "assistant-1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "answer" }]
                    }
                }),
                json!({
                    "timestamp": "2026-07-29T10:00:04Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-1" }
                }),
            ]
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
                + "\n",
        )
        .unwrap();

        let initial_args = args(&[
            "viewer",
            "--dump-session-snapshot",
            input.to_str().unwrap(),
            "--provider",
            "codex",
            "--output",
            output.to_str().unwrap(),
        ]);
        assert_eq!(run_dump_session_snapshot(&initial_args), 0);
        let initial: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(initial["kind"], "full");
        assert_eq!(initial["reason"], "initial");
        let initial_count = initial["messages"].as_array().unwrap().len();
        assert_eq!(initial_count, 4);
        assert_eq!(initial["cursorReplaceFrom"], initial_count);
        let cursor = initial["cursor"].as_str().unwrap().to_string();

        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&input)
            .unwrap();
        for line in [
            json!({
                "timestamp": "2026-07-29T10:01:00Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "second" }]
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:01:01Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "second" }
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
                    "id": "assistant-2",
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "second answer" }]
                }
            }),
            json!({
                "timestamp": "2026-07-29T10:01:04Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-2" }
            }),
        ] {
            writeln!(file, "{line}").unwrap();
        }
        file.sync_all().unwrap();

        let refresh_args = args(&[
            "viewer",
            "--dump-session-snapshot",
            input.to_str().unwrap(),
            "--provider",
            "codex",
            "--cursor",
            &cursor,
            "--output",
            output.to_str().unwrap(),
        ]);
        assert_eq!(run_dump_session_snapshot(&refresh_args), 0);
        let refresh: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(refresh["kind"], "replace");
        assert_eq!(refresh["replaceFrom"], initial_count);
        let replacement_count = refresh["messages"].as_array().unwrap().len();
        assert_eq!(replacement_count, 4);
        assert_eq!(
            refresh["cursorReplaceFrom"],
            initial_count + replacement_count
        );
        assert!(refresh["cursor"].is_string());
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
    #[serial]
    fn list_sessions_marks_codex_archive_root() {
        let temp = TempDir::new().unwrap();
        let codex_home = temp.path().join("codex-home");
        let active_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("07")
            .join("16");
        let archived_dir = codex_home.join("archived_sessions");
        std::fs::create_dir_all(&active_dir).unwrap();
        std::fs::create_dir_all(&archived_dir).unwrap();
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let output = temp.path().join("sessions.json");
        let cwd = "/redacted/project";
        let write_rollout = |path: &Path, id: &str, prompt: &str| {
            let records = [
                json!({"type":"session_meta","payload":{"id":id,"cwd":cwd}}),
                json!({"type":"response_item","payload":{"type":"message","role":"user","created_at":"2026-07-16T10:00:00Z","content":[{"type":"input_text","text":prompt}]}}),
            ];
            let content = records
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(path, format!("{content}\n")).unwrap();
        };
        write_rollout(&active_dir.join("rollout-active.jsonl"), "active", "active");
        write_rollout(
            &archived_dir.join("rollout-archived.jsonl"),
            "archived",
            "archived",
        );
        let argv = args(&[
            "viewer",
            "--list-sessions",
            "--provider",
            "codex",
            "--output",
            output.to_str().unwrap(),
        ]);

        assert_eq!(run_list_sessions(&argv), 0);
        let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        let active = rows
            .iter()
            .find(|row| row["actual_session_id"] == "active")
            .unwrap();
        let archived = rows
            .iter()
            .find(|row| row["actual_session_id"] == "archived")
            .unwrap();
        assert_eq!(active["is_archived"], false);
        assert_eq!(archived["is_archived"], true);
    }

    #[test]
    #[serial]
    fn list_sessions_marks_codex_external_agent_imports() {
        let temp = TempDir::new().unwrap();
        let codex_home = temp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("07")
            .join("17");
        std::fs::create_dir_all(&sessions_dir).unwrap();
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let output = temp.path().join("sessions.json");
        let write_rollout = |id: &str| {
            let path = sessions_dir.join(format!("rollout-2026-07-17T00-00-00-{id}.jsonl"));
            let records = [
                json!({"type":"session_meta","payload":{"id":id,"cwd":"/redacted/project","source":"vscode"}}),
                json!({"type":"response_item","payload":{"type":"message","role":"user","created_at":"2026-07-17T00:00:00Z","content":[{"type":"input_text","text":"hello"}]}}),
            ];
            let content = records
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(path, format!("{content}\n")).unwrap();
        };
        write_rollout("imported-thread");
        write_rollout("native-thread");
        std::fs::write(
            codex_home.join("external_agent_session_imports.json"),
            serde_json::to_vec(&json!({
                "records": [{
                    "source_path": "/home/test/.claude/projects/work/source-session.jsonl",
                    "content_sha256": "abc",
                    "imported_thread_id": "imported-thread",
                    "imported_at": 1,
                    "source_modified_at": 1
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        let argv = args(&[
            "viewer",
            "--list-sessions",
            "--provider",
            "codex",
            "--output",
            output.to_str().unwrap(),
        ]);

        assert_eq!(run_list_sessions(&argv), 0);
        let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        let imported = rows
            .iter()
            .find(|row| row["actual_session_id"] == "imported-thread")
            .unwrap();
        let native = rows
            .iter()
            .find(|row| row["actual_session_id"] == "native-thread")
            .unwrap();
        assert_eq!(imported["is_imported"], true);
        assert_eq!(imported["imported_from"], "claude");
        assert_eq!(native["is_imported"], false);
        assert!(native.get("imported_from").is_none());
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
            is_pinned: false,
            is_teleported: false,
            is_imported: false,
            imported_from: None,
            remote_session_id: None,
        };

        let value = serde_json::to_value(wrapped).unwrap();
        assert_eq!(value["project_path"], r"C:\work\decoded-project");
        assert_eq!(value["is_hidden"], true);
        assert_eq!(value["is_pinned"], false);
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
    fn hide_session_updates_each_eligible_store_without_removing_other_state() {
        let temp = TempDir::new().unwrap();
        let first = temp.path().join("first.vscdb");
        let second = temp.path().join("second.vscdb");
        let without_claude = temp.path().join("without-claude.vscdb");
        create_item_table(
            &first,
            &[(
                CLAUDE_VSCODE_STATE_KEY,
                json!({"theme":"dark","hiddenSessionIds":["old"]}),
            )],
        );
        create_item_table(&second, &[(CLAUDE_VSCODE_STATE_KEY, json!({"other":true}))]);
        create_item_table(&without_claude, &[("other-extension", json!({"value":1}))]);

        let outcome = hide_claude_session(
            "new-session",
            vec![first.clone(), second.clone(), without_claude.clone()],
        )
        .unwrap();
        assert_eq!(outcome.stores_updated, 2);
        assert_eq!(outcome.stores_already_hidden, 0);
        assert_eq!(outcome.stores_unavailable, 0);
        assert_eq!(
            read_hidden_ids(&first),
            Some(vec!["old".into(), "new-session".into()])
        );
        assert_eq!(read_hidden_ids(&second), Some(vec!["new-session".into()]));

        let repeated = hide_claude_session(
            "new-session",
            vec![first.clone(), second.clone(), without_claude],
        )
        .unwrap();
        assert_eq!(repeated.stores_updated, 0);
        assert_eq!(repeated.stores_already_hidden, 2);
        let conn = Connection::open(first).unwrap();
        let value: String = conn
            .query_row(
                "SELECT value FROM ItemTable WHERE key = ?1",
                [CLAUDE_VSCODE_STATE_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&value).unwrap()["theme"],
            "dark"
        );
    }

    #[test]
    fn hide_session_fails_when_no_store_can_persist_the_state() {
        let temp = TempDir::new().unwrap();
        let unrelated = temp.path().join("state.vscdb");
        create_item_table(&unrelated, &[("other-extension", json!({"value":1}))]);

        let err = hide_claude_session("session", vec![unrelated]).unwrap_err();
        assert!(err.contains("No writable Claude VS Code extension state store"));
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
        let pinned_resource = local_chat_resource("pinned");
        let both_resource = local_chat_resource("both");
        create_item_table(
            &db,
            &[
                (
                    COPILOT_CHAT_INDEX_KEY,
                    json!({"entries":{"recent":{},"pinned":{},"both":{}}}),
                ),
                (
                    COPILOT_AGENT_STATE_KEY,
                    json!([
                        {"resource":archived_resource,"archived":true},
                        {"resource":pinned_resource,"pinned":true},
                        {"resource":both_resource,"archived":true,"pinned":true}
                    ]),
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
            (false, false, false)
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("orphan.jsonl"),
                "orphan",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (true, false, false)
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("archived.jsonl"),
                "archived",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, true, false),
            "archive must take precedence over absence from the recent index"
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("pinned.jsonl"),
                "pinned",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, false, true)
        );
        assert_eq!(
            classifier.classify(&session(
                &chats.join("both.jsonl"),
                "both",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, true, true),
            "pinning is independent of archive state"
        );
        assert_eq!(
            CopilotClassifier::new("claude").classify(&session(
                &chats.join("orphan.jsonl"),
                "orphan",
                Some(COPILOT_VSCODE_ENTRYPOINT)
            )),
            (false, false, false)
        );
    }

    #[test]
    fn copilot_archive_updates_only_the_target_and_preserves_state() {
        let temp = TempDir::new().unwrap();
        let db = temp.path().join("state.vscdb");
        let target_resource = local_chat_resource("target");
        let other_resource = local_chat_resource("other");
        create_item_table(
            &db,
            &[(
                COPILOT_AGENT_STATE_KEY,
                json!([
                    {
                        "resource": target_resource,
                        "archived": false,
                        "pinned": true,
                        "read": 100,
                        "future": {"kept": true}
                    },
                    {"resource": other_resource, "archived": true, "read": 200}
                ]),
            )],
        );

        assert!(set_copilot_archive_in_db(&db, "target", true, 1_000).unwrap());
        let conn = Connection::open(&db).unwrap();
        let value: String = conn
            .query_row(
                "SELECT value FROM ItemTable WHERE key = ?1",
                [COPILOT_AGENT_STATE_KEY],
                |row| row.get(0),
            )
            .unwrap();
        let entries: Vec<Value> = serde_json::from_str(&value).unwrap();
        assert_eq!(entries[0]["archived"], true);
        assert_eq!(entries[0]["read"], 1_000);
        assert_eq!(entries[0]["pinned"], true);
        assert_eq!(entries[0]["future"], json!({"kept":true}));
        assert_eq!(entries[1]["archived"], true);
        assert_eq!(entries[1]["read"], 200);
        drop(conn);

        assert!(set_copilot_archive_in_db(&db, "target", false, 2_000).unwrap());
        assert!(!set_copilot_archive_in_db(&db, "target", false, 3_000).unwrap());
        let state = read_workspace_chat_state(&db);
        assert!(!state.archived_ids.contains("target"));
        assert!(state.archived_ids.contains("other"));
    }

    #[test]
    fn copilot_archive_creates_missing_state_but_unarchive_is_a_noop() {
        let temp = TempDir::new().unwrap();
        let db = temp.path().join("state.vscdb");
        create_item_table(&db, &[("unrelated", json!({"kept":true}))]);

        assert!(!set_copilot_archive_in_db(&db, "missing", false, 100).unwrap());
        assert!(set_copilot_archive_in_db(&db, "missing", true, 200).unwrap());
        let state = read_workspace_chat_state(&db);
        assert!(state.archived_ids.contains("missing"));
    }

    #[test]
    fn copilot_archive_identifies_the_owning_editor_from_the_state_path() {
        let root = Path::new("root");
        assert_eq!(
            editor_flavor_for_state_db(&root.join("Code/User/workspaceStorage/abc/state.vscdb")),
            Some(EditorFlavor::Code)
        );
        assert_eq!(
            editor_flavor_for_state_db(
                &root.join("Code - Insiders/User/workspaceStorage/abc/state.vscdb")
            ),
            Some(EditorFlavor::CodeInsiders)
        );
        assert_eq!(
            editor_flavor_for_state_db(
                &root.join("VSCodium/User/workspaceStorage/abc/state.vscdb")
            ),
            Some(EditorFlavor::Vscodium)
        );
        assert_eq!(
            editor_flavor_for_state_db(&root.join("Code/User/globalStorage/state.vscdb")),
            Some(EditorFlavor::Code)
        );
        assert_eq!(
            editor_flavor_for_state_db(&root.join("unknown/state.vscdb")),
            None
        );
    }

    #[test]
    fn copilot_archive_identifies_a_redirected_editor_user_data_root() {
        let temp = TempDir::new().unwrap();
        let physical_user = temp.path().join("Visual Studio Code").join("User");
        let db = physical_user.join("globalStorage").join("state.vscdb");
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        std::fs::write(&db, "").unwrap();
        let known_roots = vec![(physical_user.canonicalize().unwrap(), EditorFlavor::Code)];

        assert_eq!(
            editor_flavor_for_state_db_in(&db, &known_roots),
            Some(EditorFlavor::Code)
        );
    }

    #[test]
    fn copilot_empty_window_session_resolves_global_state_database() {
        let temp = TempDir::new().unwrap();
        let global = temp.path().join("User").join("globalStorage");
        let chats = global.join("emptyWindowChatSessions");
        std::fs::create_dir_all(&chats).unwrap();
        let db = global.join("state.vscdb");
        create_item_table(&db, &[]);
        let session = chats.join("session.jsonl");
        std::fs::write(&session, "{}").unwrap();

        assert_eq!(
            copilot_workspace_state_db(&session.to_string_lossy()),
            Some(db)
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
        assert_eq!(result, (false, false, false));
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

        let found = scan_teleport_stubs(
            temp.path(),
            &listed,
            Some(r"C:\work\project".to_string()),
            &HashSet::new(),
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].session.actual_session_id, "redirect");
        assert_eq!(found[0].remote_session_id.as_deref(), Some("remote-1"));
        assert_eq!(found[0].project_path.as_deref(), Some(r"C:\work\project"));
        assert!(found[0].is_teleported);
        assert!(!found[0].is_hidden);

        let hidden = HashSet::from(["redirect".to_string()]);
        let found = scan_teleport_stubs(
            temp.path(),
            &listed,
            Some(r"C:\work\project".to_string()),
            &hidden,
        );
        assert_eq!(found.len(), 1);
        assert!(found[0].is_hidden);
    }
}
