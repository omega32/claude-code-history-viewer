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

use crate::cli_args::extract_flag_value;
use crate::commands::multi_provider::{
    load_provider_messages, load_provider_sessions, scan_all_projects,
};

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

    let json = match serde_json::to_string_pretty(&messages) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to serialize JSON: {e}");
            return 1;
        }
    };

    // `--output` writes to a file so the caller reads clean JSON regardless of
    // anything else this process prints to stdout (e.g. upstream debug logs).
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
