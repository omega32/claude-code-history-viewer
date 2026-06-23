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
}

/// List sessions for `provider`, optionally limited to one project storage path.
async fn list_sessions(
    provider: &str,
    project: Option<&str>,
) -> Result<Vec<SessionWithProjectPath>, String> {
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
            .map(|session| SessionWithProjectPath {
                session,
                project_path: None,
            })
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
            all.extend(sessions.into_iter().map(|session| SessionWithProjectPath {
                session,
                project_path: Some(proj.actual_path.clone()),
            }));
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
