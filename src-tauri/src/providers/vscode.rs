//! VS Code Copilot Chat history provider.
//!
//! VS Code stores Copilot Chat conversations either per workspace under
//! `<UserData>/workspaceStorage/<hash>/chatSessions/<sessionUuid>.jsonl`, or
//! without a workspace under
//! `<UserData>/globalStorage/emptyWindowChatSessions/<sessionUuid>.jsonl`.
//! Each `.jsonl` file is *not* a stream of messages — it's an append-only
//! patch log on top of an initial snapshot:
//!
//! * line 1, `kind: 0`: full session snapshot
//!   (`requests[]`, `sessionId`, `creationDate`, `inputState`, …)
//! * subsequent `kind: 1`: set value at `k: ["a", "b", 2, …]` to `v`
//! * subsequent `kind: 2`: append every item of `v` (an array) to the
//!   array at path `k`
//!
//! We replay the log into an in-memory `serde_json::Value` to recover the
//! final session state, then iterate `requests[]` to emit user/assistant
//! `ClaudeMessage`s. The workspace ↔ folder mapping comes from
//! `workspace.json`'s `folder` URI (same convention Cursor uses), so
//! sessions are grouped per real project directory.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession, TokenUsage};
use crate::providers::ProviderInfo;
use crate::utils::{
    build_provider_message, is_symlink, ms_to_iso, prompt_attachment_name, prompt_attachments_data,
    search_json_value_case_insensitive,
};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Public provider id stamped on every project/session/message — unified
/// with the Copilot CLI/Desktop providers under "copilot". Per-session
/// disambiguation lives in `entrypoint = "copilot-vscode"`.
const PROVIDER_ID: &str = "copilot";
const ENTRYPOINT: &str = "copilot-vscode";
const EMPTY_WINDOW_DIR: &str = "emptyWindowChatSessions";
const EMPTY_WINDOW_PROJECT_SCHEME: &str = "vscode-empty-window://";

#[derive(Debug, Clone)]
struct UserDataRoot {
    path: PathBuf,
    label: &'static str,
}

/// Detect a VS Code (stable) installation that has Copilot Chat data.
pub fn detect() -> Option<ProviderInfo> {
    let roots = get_user_data_roots();
    let base = roots.first()?.path.clone();
    let is_available = roots.iter().any(|root| {
        root.path.join("workspaceStorage").is_dir() || empty_window_chat_dir(&root.path).is_dir()
    });
    Some(ProviderInfo {
        id: PROVIDER_ID.to_string(),
        display_name: "VS Code".to_string(),
        base_path: base.to_string_lossy().to_string(),
        is_available,
    })
}

/// First available `<UserData>` for VS Code-family builds, per OS.
pub fn get_base_path() -> Option<PathBuf> {
    get_user_data_roots()
        .into_iter()
        .next()
        .map(|root| root.path)
}

pub fn get_base_paths() -> Vec<PathBuf> {
    get_user_data_roots()
        .into_iter()
        .map(|root| root.path)
        .collect()
}

fn get_user_data_roots() -> Vec<UserDataRoot> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };

    #[cfg(target_os = "macos")]
    let candidates = [
        ("Code", "VS Code"),
        ("Code - Insiders", "VS Code Insiders"),
        ("VSCodium", "VSCodium"),
    ]
    .into_iter()
    .map(|(dir, label)| UserDataRoot {
        path: home
            .join("Library/Application Support")
            .join(dir)
            .join("User"),
        label,
    })
    .collect::<Vec<_>>();

    #[cfg(target_os = "linux")]
    let candidates = [
        ("Code", "VS Code"),
        ("Code - Insiders", "VS Code Insiders"),
        ("VSCodium", "VSCodium"),
    ]
    .into_iter()
    .map(|(dir, label)| UserDataRoot {
        path: home.join(".config").join(dir).join("User"),
        label,
    })
    .collect::<Vec<_>>();

    #[cfg(target_os = "windows")]
    let candidates = [
        ("Code", "VS Code"),
        ("Code - Insiders", "VS Code Insiders"),
        ("VSCodium", "VSCodium"),
    ]
    .into_iter()
    .map(|(dir, label)| UserDataRoot {
        path: home.join("AppData/Roaming").join(dir).join("User"),
        label,
    })
    .collect::<Vec<_>>();

    candidates
        .into_iter()
        .filter(|candidate| candidate.path.is_dir())
        .collect()
}

fn get_workspace_storage_roots() -> Result<Vec<PathBuf>, String> {
    let roots = get_base_paths()
        .into_iter()
        .map(|base| base.join("workspaceStorage"))
        .collect::<Vec<_>>();
    if roots.is_empty() {
        Err("VS Code user data directory not found".to_string())
    } else {
        Ok(roots)
    }
}

fn empty_window_chat_dir(user_data_path: &Path) -> PathBuf {
    user_data_path.join("globalStorage").join(EMPTY_WINDOW_DIR)
}

fn empty_window_project_path(user_data_path: &Path) -> String {
    format!(
        "{EMPTY_WINDOW_PROJECT_SCHEME}{}",
        user_data_path.to_string_lossy()
    )
}

fn empty_window_project_identity(custom_directory_label: Option<&str>) -> String {
    let id = match custom_directory_label {
        None | Some("VS Code") => "code",
        Some("VS Code Insiders") => "code-insiders",
        Some("VSCodium") => "vscodium",
        Some(label) => label,
    };
    format!("{EMPTY_WINDOW_PROJECT_SCHEME}{id}")
}

fn empty_window_project_name(
    user_data_path: &Path,
    custom_directory_label: Option<&str>,
) -> String {
    let flavor = custom_directory_label
        .map(ToString::to_string)
        .or_else(|| {
            user_data_path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                .map(|name| match name {
                    "Code - Insiders" => "VS Code Insiders".to_string(),
                    "VSCodium" => "VSCodium".to_string(),
                    _ => "VS Code".to_string(),
                })
        })
        .unwrap_or_else(|| "VS Code".to_string());
    format!("{flavor} — Empty Window")
}

fn user_data_roots_from_workspace_roots(workspace_storage_roots: &[PathBuf]) -> Vec<PathBuf> {
    workspace_storage_roots
        .iter()
        .filter_map(|root| root.parent().map(Path::to_path_buf))
        .collect()
}

fn is_wsl_unc_path(path: &Path) -> bool {
    let path = path.to_string_lossy();
    path.starts_with(r"\\wsl.localhost\")
        || path.starts_with(r"\\wsl$\")
        || path.starts_with(r"\\?\UNC\wsl.localhost\")
        || path.starts_with(r"\\?\UNC\wsl$\")
}

fn is_within_any_root(canonical: &Path, roots: &[PathBuf]) -> bool {
    for root in roots {
        let root = match root.canonicalize() {
            Ok(root) => root,
            Err(_) => continue,
        };
        if canonical.starts_with(&root) {
            return true;
        }
    }
    false
}

fn is_wsl_workspace_storage_path(path: &Path) -> bool {
    if !is_wsl_unc_path(path) {
        return false;
    }
    let path = path.to_string_lossy().replace('/', "\\");
    [
        r"\.vscode-server\data\User\workspaceStorage\",
        r"\.vscode-server-insiders\data\User\workspaceStorage\",
        r"\.vscodium-server\data\User\workspaceStorage\",
    ]
    .iter()
    .any(|segment| path.contains(segment))
}

fn is_wsl_empty_window_path(path: &Path) -> bool {
    if !is_wsl_unc_path(path) {
        return false;
    }
    let path = path.to_string_lossy().replace('/', "\\");
    [
        r"\.vscode-server\data\User\globalStorage\emptyWindowChatSessions\",
        r"\.vscode-server-insiders\data\User\globalStorage\emptyWindowChatSessions\",
        r"\.vscodium-server\data\User\globalStorage\emptyWindowChatSessions\",
    ]
    .iter()
    .any(|segment| path.contains(segment))
}

fn is_wsl_user_data_path(path: &Path) -> bool {
    if !is_wsl_unc_path(path) {
        return false;
    }
    let path = path.to_string_lossy().replace('/', "\\");
    [
        r"\.vscode-server\data\User",
        r"\.vscode-server-insiders\data\User",
        r"\.vscodium-server\data\User",
    ]
    .iter()
    .any(|suffix| path.ends_with(suffix))
}

fn validate_workspace_path_in(
    raw: &str,
    workspace_storage_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    let ws_path = raw.strip_prefix("vscode://").unwrap_or(raw);
    let path = PathBuf::from(ws_path);
    if !path.is_absolute() {
        return Err("VS Code workspace path must be absolute".to_string());
    }

    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve VS Code workspace path: {e}"))?;

    if !is_within_any_root(&canonical, workspace_storage_roots)
        && !is_wsl_workspace_storage_path(&canonical)
    {
        return Err("VS Code workspace path is outside workspaceStorage".to_string());
    }

    Ok(canonical)
}

fn validate_session_path_in(
    raw: &str,
    workspace_storage_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err("VS Code session path must be absolute".to_string());
    }
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
        != Some("jsonl")
    {
        return Err("VS Code session path must be a JSONL file".to_string());
    }
    let parent_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|n| n.to_str());
    if parent_name != Some("chatSessions") && parent_name != Some(EMPTY_WINDOW_DIR) {
        return Err(
            "VS Code session path must be inside chatSessions or emptyWindowChatSessions"
                .to_string(),
        );
    }

    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve VS Code session path: {e}"))?;

    let managed_workspace = parent_name == Some("chatSessions")
        && (is_within_any_root(&canonical, workspace_storage_roots)
            || is_wsl_workspace_storage_path(&canonical));
    let user_data_roots = user_data_roots_from_workspace_roots(workspace_storage_roots);
    let managed_empty_window = parent_name == Some(EMPTY_WINDOW_DIR)
        && (user_data_roots.iter().any(|root| {
            empty_window_chat_dir(root)
                .canonicalize()
                .is_ok_and(|dir| canonical.starts_with(dir))
        }) || is_wsl_empty_window_path(&canonical));
    if !managed_workspace && !managed_empty_window {
        return Err("VS Code session path is outside managed chat storage".to_string());
    }

    Ok(canonical)
}

fn validate_session_path(session_path: &str) -> Result<PathBuf, String> {
    let roots = get_workspace_storage_roots().unwrap_or_default();
    validate_session_path_in(session_path, &roots)
}

/// One workspace folder → one project.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let mut projects = Vec::new();
    for root in get_user_data_roots() {
        let label = (root.label != "VS Code").then_some(root.label);
        projects.extend(scan_projects_from_user_data_path(&root.path, label)?);
    }
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

pub fn scan_projects_from_user_data_path(
    user_data_path: &Path,
    custom_directory_label: Option<&str>,
) -> Result<Vec<ClaudeProject>, String> {
    let mut projects = scan_projects_in(
        &user_data_path.join("workspaceStorage"),
        custom_directory_label,
    )?;
    if let Some(project) = scan_empty_window_project(user_data_path, custom_directory_label)? {
        projects.push(project);
    }
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

fn scan_projects_in(
    ws_root: &Path,
    custom_directory_label: Option<&str>,
) -> Result<Vec<ClaudeProject>, String> {
    if !ws_root.is_dir() {
        return Ok(Vec::new());
    }

    let ws_paths: Vec<PathBuf> = fs::read_dir(ws_root)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|entry| entry.path())
        .collect();

    // Probing every chatSessions/*.jsonl per workspace is I/O-heavy, so the
    // per-workspace work runs on a bounded pool. Order-preserving, and the
    // sequential loop's error semantics are kept: an unreadable chatSessions
    // dir still fails the scan (first error in input order), while workspaces
    // without usable sessions are skipped.
    let results = crate::utils::par_map_bounded(ws_paths, |ws_path| {
        scan_workspace(&ws_path, custom_directory_label)
    });

    let mut projects = Vec::new();
    for result in results {
        if let Some(project) = result? {
            projects.push(project);
        }
    }

    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

/// One workspace dir → one project (`Ok(None)` = no usable chat sessions,
/// `Err` = the chatSessions dir exists but cannot be read).
fn scan_workspace(
    ws_path: &Path,
    custom_directory_label: Option<&str>,
) -> Result<Option<ClaudeProject>, String> {
    if is_symlink(ws_path) || !ws_path.is_dir() {
        return Ok(None);
    }

    let Some(folder) = read_workspace_folder(&ws_path.join("workspace.json")) else {
        return Ok(None);
    };

    let chat_dir = ws_path.join("chatSessions");
    if !chat_dir.is_dir() {
        return Ok(None);
    }

    let mut session_count = 0usize;
    let mut last_modified_ms: u64 = 0;
    let mut message_count = 0usize;

    for (_, info) in list_session_metadata(&chat_dir)? {
        // Empty chat panels (kind:0 with requests:[]) should not be
        // counted as sessions or contribute to the project's tally.
        if info.message_count == 0 {
            continue;
        }
        session_count += 1;
        message_count += info.message_count;
        if info.last_modified_ms > last_modified_ms {
            last_modified_ms = info.last_modified_ms;
        }
    }

    if session_count == 0 {
        return Ok(None);
    }

    let name = PathBuf::from(&folder)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| folder.clone());

    Ok(Some(ClaudeProject {
        name,
        path: format!("vscode://{}", ws_path.to_string_lossy()),
        actual_path: folder,
        session_count,
        message_count,
        last_modified: ms_to_iso(last_modified_ms),
        git_info: None,
        provider: Some(PROVIDER_ID.to_string()),
        storage_type: None,
        custom_directory_label: custom_directory_label.map(ToString::to_string),
    }))
}

fn scan_empty_window_project(
    user_data_path: &Path,
    custom_directory_label: Option<&str>,
) -> Result<Option<ClaudeProject>, String> {
    let chat_dir = empty_window_chat_dir(user_data_path);
    if !chat_dir.is_dir() {
        return Ok(None);
    }

    let mut session_count = 0usize;
    let mut message_count = 0usize;
    let mut last_modified_ms = 0u64;
    for (_, info) in list_session_metadata(&chat_dir)? {
        if info.message_count == 0 {
            continue;
        }
        session_count += 1;
        message_count += info.message_count;
        last_modified_ms = last_modified_ms.max(info.last_modified_ms);
    }
    if session_count == 0 {
        return Ok(None);
    }

    Ok(Some(ClaudeProject {
        name: empty_window_project_name(user_data_path, custom_directory_label),
        path: empty_window_project_path(user_data_path),
        actual_path: empty_window_project_identity(custom_directory_label),
        session_count,
        message_count,
        last_modified: ms_to_iso(last_modified_ms),
        git_info: None,
        provider: Some(PROVIDER_ID.to_string()),
        storage_type: None,
        custom_directory_label: custom_directory_label.map(ToString::to_string),
    }))
}

/// Sessions for a single workspace.
pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let roots = get_workspace_storage_roots().unwrap_or_default();
    load_sessions_in(project_path, &roots)
}

fn load_sessions_in(
    project_path: &str,
    workspace_storage_roots: &[PathBuf],
) -> Result<Vec<ClaudeSession>, String> {
    if let Some(raw_user_data_path) = project_path.strip_prefix(EMPTY_WINDOW_PROJECT_SCHEME) {
        let raw = PathBuf::from(raw_user_data_path);
        if !raw.is_absolute() {
            return Err("VS Code empty-window user-data path must be absolute".to_string());
        }
        let canonical = raw
            .canonicalize()
            .map_err(|e| format!("Failed to resolve VS Code user-data path: {e}"))?;
        let allowed = user_data_roots_from_workspace_roots(workspace_storage_roots);
        let managed = allowed.iter().any(|root| {
            root.canonicalize()
                .is_ok_and(|allowed| canonical == allowed)
        }) || is_wsl_user_data_path(&canonical);
        if !managed {
            return Err("VS Code empty-window path is outside managed user data".to_string());
        }
        return load_sessions_from_chat_dir(
            &empty_window_chat_dir(&canonical),
            &empty_window_project_name(&canonical, None),
        );
    }

    let ws_path_buf = validate_workspace_path_in(project_path, workspace_storage_roots)?;

    let chat_dir = ws_path_buf.join("chatSessions");
    if !chat_dir.is_dir() {
        return Ok(Vec::new());
    }

    let folder = read_workspace_folder(&ws_path_buf.join("workspace.json"));
    let project_name = folder
        .as_deref()
        .and_then(|f| {
            PathBuf::from(f)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "VS Code".to_string());

    load_sessions_from_chat_dir(&chat_dir, &project_name)
}

fn load_sessions_from_chat_dir(
    chat_dir: &Path,
    project_name: &str,
) -> Result<Vec<ClaudeSession>, String> {
    if !chat_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    for (session_path, info) in list_session_metadata(chat_dir)? {
        // Skip empty sessions (e.g., chat panels opened but never used).
        if info.message_count == 0 {
            continue;
        }

        sessions.push(ClaudeSession {
            session_id: session_path.to_string_lossy().to_string(),
            actual_session_id: info.session_id,
            file_path: session_path.to_string_lossy().to_string(),
            project_name: project_name.to_string(),
            message_count: info.message_count,
            first_message_time: ms_to_iso(info.first_message_ms),
            last_message_time: ms_to_iso(info.last_modified_ms),
            last_modified: ms_to_iso(info.last_modified_ms),
            has_tool_use: info.has_tool_use,
            has_errors: false,
            // A user rename wins over the first-request preview (same precedence
            // as the Claude/codex scanners give their native titles).
            is_renamed: info.custom_title.is_some(),
            summary: info.custom_title.or(info.summary),
            provider: Some(PROVIDER_ID.to_string()),
            storage_type: None,
            entrypoint: Some(ENTRYPOINT.to_string()),
        });
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Replay the patch log, then convert each request into messages.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let path = validate_session_path(session_path)?;
    load_messages_from_path(&path)
}

fn load_messages_from_path(path: &Path) -> Result<Vec<ClaudeMessage>, String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let state = replay_session(&raw)?;
    Ok(messages_from_state(&state))
}

/// Naive case-insensitive search across every chat session.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let mut results = Vec::new();
    let query_lower = query.to_lowercase();
    for root in get_user_data_roots() {
        search_workspace_storage(
            &root.path.join("workspaceStorage"),
            &query_lower,
            limit,
            &mut results,
        )?;
        search_chat_dir(
            &empty_window_chat_dir(&root.path),
            &query_lower,
            limit,
            &mut results,
        )?;
        if results.len() >= limit {
            break;
        }
    }
    Ok(results)
}

pub fn search_from_user_data_path(
    user_data_path: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<ClaudeMessage>, String> {
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    search_workspace_storage(
        &user_data_path.join("workspaceStorage"),
        &query_lower,
        limit,
        &mut results,
    )?;
    search_chat_dir(
        &empty_window_chat_dir(user_data_path),
        &query_lower,
        limit,
        &mut results,
    )?;
    Ok(results)
}

fn search_workspace_storage(
    ws_root: &Path,
    query_lower: &str,
    limit: usize,
    results: &mut Vec<ClaudeMessage>,
) -> Result<(), String> {
    if !ws_root.is_dir() {
        return Ok(());
    }

    for ws_entry in fs::read_dir(ws_root).map_err(|e| e.to_string())?.flatten() {
        let ws_path = ws_entry.path();
        if is_symlink(&ws_path) || !ws_path.is_dir() {
            continue;
        }
        let chat_dir = ws_path.join("chatSessions");
        if !chat_dir.is_dir() {
            continue;
        }

        search_chat_dir(&chat_dir, query_lower, limit, results)?;
        if results.len() >= limit {
            return Ok(());
        }
    }

    Ok(())
}

fn search_chat_dir(
    chat_dir: &Path,
    query_lower: &str,
    limit: usize,
    results: &mut Vec<ClaudeMessage>,
) -> Result<(), String> {
    if !chat_dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(chat_dir).map_err(|e| e.to_string())?.flatten() {
        let session_path = entry.path();
        if is_symlink(&session_path) || !session_path.is_file() {
            continue;
        }
        if session_path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
            != Some("jsonl")
        {
            continue;
        }

        if let Ok(messages) = load_messages_from_path(&session_path) {
            for msg in messages {
                if results.len() >= limit {
                    return Ok(());
                }
                if let Some(content) = &msg.content {
                    if search_json_value_case_insensitive(content, query_lower) {
                        results.push(msg);
                    }
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Patch log replay
// ============================================================================

/// Resolved final state of a chat session.
fn replay_session(raw: &str) -> Result<Value, String> {
    let mut lines = raw.split('\n').filter(|l| !l.trim().is_empty());

    let first = lines
        .next()
        .ok_or_else(|| "Empty VS Code session file".to_string())?;
    let header: Value =
        serde_json::from_str(first).map_err(|e| format!("Invalid VS Code session header: {e}"))?;
    if header.get("kind").and_then(Value::as_u64) != Some(0) {
        return Err("VS Code session file missing initial snapshot (kind=0)".to_string());
    }
    let mut state = header
        .get("v")
        .cloned()
        .ok_or_else(|| "VS Code session snapshot has no `v` field".to_string())?;

    for line in lines {
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            // Tolerate truncated/corrupt trailing lines, like Codex does.
            Err(_) => continue,
        };

        let kind = entry.get("kind").and_then(Value::as_u64).unwrap_or(0);
        let path = entry.get("k").and_then(Value::as_array).cloned();
        let value = entry.get("v").cloned();
        let (path, value) = match (path, value) {
            (Some(p), Some(v)) => (p, v),
            _ => continue,
        };

        match kind {
            1 => {
                let _ = set_at_path(&mut state, &path, value);
            }
            2 => {
                if let Some(items) = value.as_array() {
                    let _ = append_at_path(&mut state, &path, items);
                }
            }
            _ => {}
        }
    }

    Ok(state)
}

/// Upper bound on array indices materialised during patch-log replay. VS Code
/// writes small sequential indices; a wildly out-of-range index indicates a
/// corrupt/truncated session file and must not drive an unbounded `push` loop.
const MAX_REPLAY_ARRAY_INDEX: usize = 1_000_000;

/// Walk to the parent of `path`, then assign `path.last()` to `value`.
fn set_at_path(state: &mut Value, path: &[Value], value: Value) -> Result<(), ()> {
    if path.is_empty() {
        *state = value;
        return Ok(());
    }
    let (last, parents) = path.split_last().expect("path non-empty here");
    let parent = traverse_mut(state, parents)?;
    match (parent, last) {
        (Value::Object(map), Value::String(key)) => {
            map.insert(key.clone(), value);
            Ok(())
        }
        (Value::Array(arr), Value::Number(n)) => {
            let idx = n.as_u64().ok_or(())? as usize;
            if idx > MAX_REPLAY_ARRAY_INDEX {
                return Err(());
            }
            while arr.len() <= idx {
                arr.push(Value::Null);
            }
            arr[idx] = value;
            Ok(())
        }
        _ => Err(()),
    }
}

/// Append every item to the array at `path` (creating arrays/maps as needed).
fn append_at_path(state: &mut Value, path: &[Value], items: &[Value]) -> Result<(), ()> {
    let target = traverse_mut(state, path)?;
    if let Value::Null = target {
        *target = Value::Array(Vec::new());
    }
    let arr = target.as_array_mut().ok_or(())?;
    arr.extend(items.iter().cloned());
    Ok(())
}

/// Walk `path` mutably, materialising missing intermediates.
fn traverse_mut<'a>(mut state: &'a mut Value, path: &[Value]) -> Result<&'a mut Value, ()> {
    for seg in path {
        state = match (state, seg) {
            (Value::Object(map), Value::String(key)) => map
                .entry(key.clone())
                .or_insert(Value::Object(serde_json::Map::default())),
            (Value::Array(arr), Value::Number(n)) => {
                let idx = n.as_u64().ok_or(())? as usize;
                if idx > MAX_REPLAY_ARRAY_INDEX {
                    return Err(());
                }
                while arr.len() <= idx {
                    arr.push(Value::Null);
                }
                &mut arr[idx]
            }
            _ => return Err(()),
        };
    }
    Ok(state)
}

// ============================================================================
// State → ClaudeMessage[]
// ============================================================================

fn messages_from_state(state: &Value) -> Vec<ClaudeMessage> {
    let session_id = state
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let creation_ms = state
        .get("creationDate")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let base_timestamp = ms_to_iso(creation_ms);

    let requests = match state.get("requests").and_then(Value::as_array) {
        Some(r) => r,
        None => return Vec::new(),
    };

    let mut messages = Vec::with_capacity(requests.len() * 2);
    let mut counter: u64 = 0;

    for (idx, req) in requests.iter().enumerate() {
        let req_ts = req
            .get("timestamp")
            .and_then(Value::as_u64)
            .map(ms_to_iso)
            .unwrap_or_else(|| base_timestamp.clone());

        if let Some(text) = extract_user_text(req) {
            counter += 1;
            let uuid = req
                .get("requestId")
                .and_then(Value::as_str)
                .map(String::from)
                .unwrap_or_else(|| format!("vscode-req-{idx}-{counter}"));
            let content = serde_json::json!([{ "type": "text", "text": text }]);
            let mut message = build_provider_message(
                PROVIDER_ID,
                uuid,
                &session_id,
                req_ts.clone(),
                "user",
                Some("user"),
                Some(content),
                None,
            );
            message.data = prompt_attachments_data(prompt_attachment_names(req));
            messages.push(message);
        }

        if let Some(assistant) =
            build_assistant_message(req, idx, &session_id, &req_ts, &mut counter)
        {
            messages.push(assistant);
        }

        // Compaction summary: VS Code records a background conversation summary on
        // `requests[n].result.metadata.summaries[].text` (flattened to
        // `metadata.summary`) when it compresses context. Surface it as a synthetic
        // user record stamped `subtype: "compact_summary"` — mirroring how the Claude
        // parser marks Claude Code's compaction summary — so the consumer detects it
        // structurally (never by sniffing text) and labels it, instead of treating it
        // as an authored turn. Emitted after the request's own turn, at the point the
        // earlier context was compressed.
        if let Some(summary) = extract_compaction_summary(req) {
            counter += 1;
            let uuid = format!("vscode-compact-{idx}-{counter}");
            let content = serde_json::json!([{ "type": "text", "text": summary }]);
            let mut msg = build_provider_message(
                PROVIDER_ID,
                uuid,
                &session_id,
                req_ts.clone(),
                "user",
                Some("user"),
                Some(content),
                None,
            );
            msg.subtype = Some("compact_summary".to_string());
            messages.push(msg);
        }
    }

    messages
}

fn extract_user_text(req: &Value) -> Option<String> {
    let msg = req.get("message")?;
    if let Some(text) = msg.get("text").and_then(Value::as_str) {
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }
    // Fallback: stitch together text parts.
    let parts = msg.get("parts").and_then(Value::as_array)?;
    let joined = parts
        .iter()
        .filter_map(|p| {
            let kind = p.get("kind").and_then(Value::as_str).unwrap_or("");
            if kind == "text" {
                p.get("text").and_then(Value::as_str).map(str::to_string)
            } else {
                None
            }
        })
        .collect::<String>();
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Explicit files attached through Copilot Chat's prompt UI. Historical
/// requests persist them as `kind:file` variables whose id is a real `file:`
/// URI. IDE-selected/open context uses the distinct `vscode.implicit.*` ids and
/// therefore cannot pass this classifier.
fn prompt_attachment_names(req: &Value) -> Vec<String> {
    req.get("variableData")
        .and_then(|value| value.get("variables"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|variable| variable.get("kind").and_then(Value::as_str) == Some("file"))
        .filter(|variable| {
            variable
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("file:"))
        })
        .filter_map(|variable| variable.get("name").and_then(Value::as_str))
        .filter_map(prompt_attachment_name)
        .collect()
}

/// The conversation-compaction summary attached to a request's result, if any.
/// VS Code writes a background summary under `result.metadata.summaries[].text`
/// (also flattened to `result.metadata.summary`). Returns the summary text, or
/// `None` when the request carries no compaction. Prefers the structured
/// `summaries[]` (the source of truth; joins multiple non-empty entries), falling
/// back to the flat `summary` string.
fn extract_compaction_summary(req: &Value) -> Option<String> {
    let metadata = req.get("result")?.get("metadata")?;
    if let Some(arr) = metadata.get("summaries").and_then(Value::as_array) {
        let joined = arr
            .iter()
            .filter_map(|s| s.get("text").and_then(Value::as_str))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !joined.is_empty() {
            return Some(joined);
        }
    }
    metadata
        .get("summary")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn build_assistant_message(
    req: &Value,
    idx: usize,
    session_id: &str,
    timestamp: &str,
    counter: &mut u64,
) -> Option<ClaudeMessage> {
    let response = req.get("response").and_then(Value::as_array)?;
    let mut blocks: Vec<Value> = Vec::new();
    let mut tool_use_block: Option<Value> = None;
    // A `vscode_askQuestions` call stores its questions + the user's answers in a
    // separate `questionCarousel` response part; pre-scan them (keyed by the owning
    // tool-call id) so the invocation below can fold them into the tool's input.
    let carousels = question_carousels(response);
    // Track whether the previous visible block was plain prose or an inline
    // reference, so consecutive spans from the same response (split at every
    // inlineReference boundary) can be coalesced into one text block. Thinking,
    // tool calls, and progress steps break the prose run.
    let mut last_was_prose = false;

    for part in response {
        let kind = part.get("kind").and_then(Value::as_str);
        match kind {
            None => {
                // Plain markdown content: just a {value, …} object. VS Code wraps a
                // tool invocation in a fenced code block and persists the opening and
                // closing fences as their *own* standalone markdown parts (a bare ```
                // line each); landing adjacent (the tool part sits beside them) they
                // render as an empty code block downstream. They are UI scaffolding,
                // not model-authored prose (a real code block arrives as one part with
                // its code inside), so drop a fence-delimiter-only part.
                if let Some(text) = part.get("value").and_then(Value::as_str) {
                    if !text.is_empty() && !is_fence_delimiter_only(text) {
                        // Coalesce into the preceding text block when in a prose run —
                        // the surrounding text parts at an inlineReference boundary
                        // belong to the same sentence; the JOIN (\n\n) between them
                        // would add a spurious blank line.
                        if last_was_prose {
                            if let Some(last) = blocks.last_mut() {
                                if last.get("type").and_then(Value::as_str) == Some("text") {
                                    let old = last["text"].as_str().unwrap_or("").to_string();
                                    last["text"] = Value::String(old + text);
                                    continue;
                                }
                            }
                        }
                        blocks.push(serde_json::json!({ "type": "text", "text": text }));
                        last_was_prose = true;
                    }
                }
            }
            Some("thinking") => {
                last_was_prose = false;
                let text = part.get("value").and_then(Value::as_str).unwrap_or("");
                // Skip empty/encrypted-only thinking blobs; render visible text only.
                if !text.is_empty() {
                    blocks.push(serde_json::json!({
                        "type": "thinking",
                        "thinking": text,
                    }));
                }
            }
            Some("toolInvocationSerialized") => {
                last_was_prose = false;
                let tool_id = part
                    .get("toolId")
                    .and_then(Value::as_str)
                    .unwrap_or("tool")
                    .to_string();
                let call_id = part
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| {
                        *counter += 1;
                        format!("vscode-tool-{idx}-{counter}")
                    });
                // `invocationMessage` is a markdown object ({value, uris?}) for
                // most tools, but a bare string for some (e.g. copilot_applyPatch).
                let invocation_text = part
                    .get("invocationMessage")
                    .and_then(markdown_or_str)
                    .unwrap_or("");
                let past_text = part
                    .get("pastTenseMessage")
                    .and_then(markdown_or_str)
                    .unwrap_or("");
                let is_complete = part
                    .get("isComplete")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

                let mut input = serde_json::Map::new();
                if !invocation_text.is_empty() {
                    input.insert(
                        "message".to_string(),
                        Value::String(invocation_text.to_string()),
                    );
                }
                // Structured arguments, where the serialized part carries them
                // (machine-readable next to the prose `message`): file-target
                // tools resolve their target into `invocationMessage.uris`, the
                // terminal tool carries the exact command line.
                if let Some(path) = first_invocation_uri_path(part) {
                    input.insert("path".to_string(), Value::String(path));
                }
                if let Some(command) = terminal_command_line(part) {
                    input.insert("command".to_string(), Value::String(command));
                }
                // The to-do tool (`manage_todo_list`) carries its structured list
                // in `toolSpecificData.todoList` — surface it beside the prose
                // `message` so the consumer can render a real checklist instead of
                // the "Created N todos" one-liner (the questions tool, by contrast,
                // has no `toolSpecificData`; its data is a separate `questionCarousel`
                // response part — handled elsewhere).
                if let Some(todos) = todo_list_items(part) {
                    input.insert("todoList".to_string(), todos);
                }
                // A questions tool (`vscode_askQuestions`): fold the carousel's
                // questions into `input.questions` (the AskUserQuestion shape) and turn
                // the user's `selectedValue`s into the paired answer text — so the whole
                // Q&A normalizes to Claude's prompt+reply shape and the consumer
                // reconstructs it with no provider-specific code.
                let question_answer = carousels.get(&call_id).map(|carousel| {
                    let (mapped, answers) = map_question_carousel(carousel);
                    input.insert("questions".to_string(), Value::Array(mapped));
                    answers
                });
                let tool_use = serde_json::json!({
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool_id,
                    "input": Value::Object(input),
                });
                if tool_use_block.is_none() {
                    tool_use_block = Some(tool_use.clone());
                }
                blocks.push(tool_use);

                // The paired result: the user's answers for a questions tool (even when
                // the invocation isn't flagged complete), else the prose past-tense line.
                // Unanswered questions emit no result (the prompt still shows, answerless).
                if let Some(answers) = question_answer {
                    if !answers.is_empty() {
                        blocks.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": answers,
                        }));
                    }
                } else if is_complete && !past_text.is_empty() {
                    blocks.push(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": past_text,
                    }));
                }
            }
            Some("progressTaskSerialized") => {
                last_was_prose = false;
                if let Some(text) = part
                    .get("content")
                    .and_then(|c| c.get("value"))
                    .and_then(Value::as_str)
                {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({ "type": "text", "text": text }));
                    }
                }
            }
            Some("inlineReference") => {
                if let Some(ref_text) = inline_reference_text(part) {
                    // Coalesce into the preceding prose block — an inline reference
                    // is a mid-sentence span, not a paragraph boundary.
                    if last_was_prose {
                        if let Some(last) = blocks.last_mut() {
                            if last.get("type").and_then(Value::as_str) == Some("text") {
                                let old = last["text"].as_str().unwrap_or("").to_string();
                                last["text"] = Value::String(old + &ref_text);
                                continue;
                            }
                        }
                    }
                    blocks.push(serde_json::json!({ "type": "text", "text": ref_text }));
                    last_was_prose = true;
                }
            }
            // Unknown / non-renderable kinds (e.g. "mcpServersStarting") are
            // intentionally skipped.
            Some(_) => {}
        }
    }

    if blocks.is_empty() {
        return None;
    }

    *counter += 1;
    let uuid = req
        .get("responseId")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| format!("vscode-resp-{idx}-{counter}"));

    let model = req.get("modelId").and_then(Value::as_str).map(String::from);
    let usage = req
        .get("completionTokens")
        .and_then(Value::as_u64)
        .map(|out| TokenUsage {
            input_tokens: None,
            output_tokens: Some(out as u32),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            service_tier: None,
        });
    let duration_ms = req.get("elapsedMs").and_then(Value::as_u64);

    let mut msg = build_provider_message(
        PROVIDER_ID,
        uuid,
        session_id,
        timestamp.to_string(),
        "assistant",
        Some("assistant"),
        Some(Value::Array(blocks)),
        model,
    );
    msg.tool_use = tool_use_block;
    msg.usage = usage;
    msg.duration_ms = duration_ms;
    Some(msg)
}

// ============================================================================
// Helpers shared with cursor.rs (kept private to avoid a cross-cutting refactor)
// ============================================================================

fn read_workspace_folder(workspace_json_path: &Path) -> Option<String> {
    let data = fs::read_to_string(workspace_json_path).ok()?;
    let json: Value = serde_json::from_str(&data).ok()?;
    let folder = json.get("folder").and_then(Value::as_str)?;
    folder.strip_prefix("file://").map(|s| {
        let path = if s.len() > 2 && s.as_bytes()[2] == b':' {
            // Windows drive letter (file:///C:/…)
            &s[1..]
        } else {
            s
        };
        percent_decode(path)
    })
}

fn percent_decode(input: &str) -> String {
    let mut buf = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                buf.push(byte);
                i += 3;
                continue;
            }
        }
        buf.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(buf).unwrap_or_else(|_| input.to_string())
}

// Serializable so it can be cached on disk (see the metadata cache below).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SessionMetadata {
    session_id: String,
    message_count: usize,
    first_message_ms: u64,
    last_modified_ms: u64,
    has_tool_use: bool,
    summary: Option<String>,
    /// The user-assigned title (`customTitle` in the replayed state) — written
    /// by VS Code's own rename UI as a `{"kind":1,"k":["customTitle"],"v":…}`
    /// patch record. Present ⇒ the session was deliberately renamed.
    custom_title: Option<String>,
}

// ── Persistent metadata cache ────────────────────────────────────────────────
//
// `probe_session_metadata` replays a session's whole append-only patch log to
// derive its metadata — costly for large sessions, and repeated on *every*
// listing (and twice per `--list-sessions`, once for the project tally and once
// for the session list). Cache the derived metadata per `chatSessions/` dir,
// keyed by each file's (mtime, size): the logs are append-only, so a matching
// (mtime, size) means the metadata is unchanged and the replay can be skipped.
// This mirrors the Claude scanner's `.session_cache.json` (see
// `commands/session/load.rs`); the copilot CLI scanner has only an in-process
// cache, which is cold across the separate processes a headless caller spawns.

/// One cached session's derived metadata plus the (mtime, size) it was valid for.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CachedSessionMetadata {
    /// File modification time (Unix seconds).
    modified_time: u64,
    /// File size in bytes (catches sub-second appends a seconds-mtime misses).
    file_size: u64,
    metadata: SessionMetadata,
}

/// The on-disk cache file for one `chatSessions/` directory.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct SessionMetadataCache {
    /// Bumped on any format change to `SessionMetadata`/this struct so stale
    /// caches are dropped rather than misread.
    version: u32,
    /// session filename -> cached metadata.
    entries: std::collections::HashMap<String, CachedSessionMetadata>,
}

const METADATA_CACHE_VERSION: u32 = 1;

/// The cache file lives alongside the sessions it describes (mirroring Claude's
/// `.session_cache.json`); VS Code only reads `*.jsonl` here, so the dotfile is
/// inert to it.
fn metadata_cache_path(chat_dir: &Path) -> PathBuf {
    chat_dir.join(".session_cache.json")
}

/// A file's freshness key: (modified-time seconds, size bytes). `None` if it
/// can't be stat'd (the caller then probes without caching).
fn file_freshness(path: &Path) -> Option<(u64, u64)> {
    let meta = path.metadata().ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((mtime, meta.len()))
}

/// Load a `chatSessions/` cache from disk (empty on any error or a version bump).
fn load_metadata_cache(chat_dir: &Path) -> SessionMetadataCache {
    if let Ok(content) = fs::read_to_string(metadata_cache_path(chat_dir)) {
        if let Ok(cache) = serde_json::from_str::<SessionMetadataCache>(&content) {
            if cache.version == METADATA_CACHE_VERSION {
                return cache;
            }
        }
    }
    SessionMetadataCache::default()
}

/// Save a `chatSessions/` cache atomically (best effort; errors ignored — the
/// cache is only an accelerator). Skips the write when there is nothing to cache.
fn save_metadata_cache(chat_dir: &Path, cache: &SessionMetadataCache) {
    if cache.entries.is_empty() {
        return;
    }
    let path = metadata_cache_path(chat_dir);
    let Ok(content) = serde_json::to_string(cache) else {
        return;
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("json.{nonce}.tmp"));
    if fs::write(&tmp, content.as_bytes()).is_ok() {
        #[cfg(target_os = "windows")]
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        let _ = fs::rename(&tmp, &path);
    }
}

/// Probe one session's metadata, reusing the cache when the file is unchanged.
/// On a hit the entry is carried into `next`; on a miss the file is replayed and
/// the fresh result stored. A file that can't be stat'd or replayed is probed
/// directly / skipped and left uncached (so a transient failure is never sticky).
fn probe_cached(
    path: &Path,
    old: &SessionMetadataCache,
    next: &mut SessionMetadataCache,
) -> Option<SessionMetadata> {
    let Some(key) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return probe_session_metadata(path);
    };
    let Some((modified_time, file_size)) = file_freshness(path) else {
        return probe_session_metadata(path);
    };
    if let Some(hit) = old.entries.get(&key) {
        if hit.modified_time == modified_time && hit.file_size == file_size {
            next.entries.insert(key, hit.clone());
            return Some(hit.metadata.clone());
        }
    }
    let metadata = probe_session_metadata(path)?;
    next.entries.insert(
        key,
        CachedSessionMetadata {
            modified_time,
            file_size,
            metadata: metadata.clone(),
        },
    );
    Some(metadata)
}

/// Every readable session in a `chatSessions/` dir as `(path, metadata)`, backed
/// by the persistent cache so unchanged sessions aren't re-replayed. Only entries
/// still present are carried forward, so deleted sessions are evicted naturally.
/// Callers filter empties (`message_count == 0`) — those stay cached, so the
/// common empty chat panels aren't re-replayed either.
fn list_session_metadata(chat_dir: &Path) -> Result<Vec<(PathBuf, SessionMetadata)>, String> {
    let old = load_metadata_cache(chat_dir);
    let mut next = SessionMetadataCache {
        version: METADATA_CACHE_VERSION,
        entries: std::collections::HashMap::new(),
    };
    let mut out = Vec::new();
    for entry in fs::read_dir(chat_dir).map_err(|e| e.to_string())?.flatten() {
        let path = entry.path();
        if is_symlink(&path) || !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
            != Some("jsonl")
        {
            continue;
        }
        if let Some(meta) = probe_cached(&path, &old, &mut next) {
            out.push((path, meta));
        }
    }
    save_metadata_cache(chat_dir, &next);
    Ok(out)
}

/// Text of a VS Code message field that is either a markdown object
/// (`{value, uris?, …}`) or a bare string.
fn markdown_or_str(m: &Value) -> Option<&str> {
    m.as_str()
        .or_else(|| m.get("value").and_then(Value::as_str))
}

/// True when `text` (ignoring surrounding whitespace) is *only* a Markdown
/// code-fence delimiter: three backticks, optionally followed by a bare language
/// token (letters/digits, hyphen, or underscore) and nothing else. VS Code
/// Copilot Chat wraps a tool invocation in a fenced block
/// and persists the opening/closing fences as their own standalone markdown parts;
/// these carry no model-authored prose and would render as empty code blocks, so
/// they are dropped. A real code block arrives as a single part with its code
/// *inside* the fences (so the remainder contains a newline), and is never matched.
fn is_fence_delimiter_only(text: &str) -> bool {
    match text.trim().strip_prefix("```") {
        Some(rest) => rest
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_'),
        None => false,
    }
}

/// First resolved file path of a tool invocation: `invocationMessage.uris` maps
/// each markdown link target to URI components whose `path` is already decoded
/// (e.g. `/e:/Programas/…/script.ps1`). Single-target tools (readFile,
/// createFile, replaceString, …) carry exactly one.
fn first_invocation_uri_path(part: &Value) -> Option<String> {
    let uris = part.get("invocationMessage")?.get("uris")?.as_object()?;
    uris.values()
        .find_map(|u| u.get("path").and_then(Value::as_str))
        .map(String::from)
}

/// The terminal tool's exact command line
/// (`toolSpecificData.commandLine.original`, falling back to `forDisplay`).
fn terminal_command_line(part: &Value) -> Option<String> {
    let data = part.get("toolSpecificData")?;
    if data.get("kind").and_then(Value::as_str) != Some("terminal") {
        return None;
    }
    let command_line = data.get("commandLine")?;
    command_line
        .get("original")
        .and_then(Value::as_str)
        .or_else(|| command_line.get("forDisplay").and_then(Value::as_str))
        .map(String::from)
}

/// The structured to-do items of a `manage_todo_list` invocation, if present.
/// VS Code carries them in `toolSpecificData.todoList` — a `[{id,title,status}]`
/// array — beside the prose `message`. Returned verbatim (the consumer owns the
/// `{id,title,status}` → common-model mapping); `None` for any non-todo tool.
fn todo_list_items(part: &Value) -> Option<Value> {
    let data = part.get("toolSpecificData")?;
    if data.get("kind").and_then(Value::as_str) != Some("todoList") {
        return None;
    }
    data.get("todoList").filter(|v| v.is_array()).cloned()
}

/// Text representation of a VS Code `inlineReference` response part.
///
/// Copilot Chat emits references as standalone parts (typically with
/// `inlineReference: "file:///..."`) interleaved in assistant prose. Render them
/// as text so the prose remains readable in-order after normalization.
///
/// Preference order:
/// 1) explicit label/title/name fields when present
/// 2) basename derived from a URI/path payload
/// 3) the decoded URI/path itself
fn inline_reference_text(part: &Value) -> Option<String> {
    fn non_empty(v: Option<&str>) -> Option<String> {
        v.map(str::trim).filter(|s| !s.is_empty()).map(String::from)
    }

    fn field_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
        v.get(key).and_then(Value::as_str)
    }

    fn display_name(raw: &str) -> Option<String> {
        if raw.trim().is_empty() {
            return None;
        }

        let mut decoded = if let Some(rest) = raw.strip_prefix("file://") {
            percent_decode(rest)
        } else {
            raw.to_string()
        };

        // Normalize file:// URI-drive form (`/e:/...`) to native-drive form.
        if decoded.len() > 2 {
            let bytes = decoded.as_bytes();
            if bytes[0] == b'/' && bytes[2] == b':' {
                decoded = decoded[1..].to_string();
            }
        }

        let without_fragment = decoded.split('#').next().unwrap_or(decoded.as_str());
        let trimmed = without_fragment.trim_end_matches(['/', '\\']);
        let basename = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
        if basename.is_empty() {
            Some(decoded)
        } else {
            Some(basename.to_string())
        }
    }

    let reference = part.get("inlineReference").unwrap_or(part);

    // Prefer a human-authored label if the shape carries one.
    for key in ["label", "title", "name", "displayName"] {
        if let Some(text) = non_empty(field_str(reference, key).or_else(|| field_str(part, key))) {
            return Some(text);
        }
    }

    let raw = reference
        .as_str()
        .or_else(|| field_str(reference, "uri"))
        .or_else(|| field_str(reference, "path"))
        .or_else(|| field_str(reference, "target"))
        .or_else(|| field_str(reference, "value"))
        .or_else(|| field_str(part, "uri"))
        .or_else(|| field_str(part, "path"));

    raw.and_then(display_name)
}

/// Map each `questionCarousel` response part to its owning tool-call id (`resolveId`)
/// → the whole carousel. The carousel holds the prompt (`questions[]`) and the user's
/// answers (`data[<questionId>].selectedValue`). A carousel can be snapshotted more
/// than once as it's answered, so prefer a snapshot that carries `data`.
fn question_carousels(response: &[Value]) -> std::collections::HashMap<String, Value> {
    let mut map: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    for part in response {
        if part.get("kind").and_then(Value::as_str) != Some("questionCarousel") {
            continue;
        }
        let id = match part.get("resolveId").and_then(Value::as_str) {
            Some(id) => id.to_string(),
            None => continue,
        };
        let has_data = part.get("data").and_then(Value::as_object).is_some();
        if has_data || !map.contains_key(&id) {
            map.insert(id, part.clone());
        }
    }
    map
}

/// Map a `questionCarousel` to the `AskUserQuestion` `input.questions` shape
/// (`{question, header, options:[{label}], multiSelect}`) and collect the user's
/// answers as readable text. A question's answer is `carousel.data[q.id].selectedValue`;
/// each answered question adds one `header: value` line. Returns `(questions, answers)`.
fn map_question_carousel(carousel: &Value) -> (Vec<Value>, String) {
    let questions = carousel.get("questions").and_then(Value::as_array);
    let data = carousel.get("data");
    let mut mapped = Vec::new();
    let mut answers: Vec<String> = Vec::new();
    for q in questions.into_iter().flatten() {
        let message = q.get("message").and_then(Value::as_str).unwrap_or("");
        let title = q.get("title").and_then(Value::as_str).unwrap_or("");
        let multi = q.get("type").and_then(Value::as_str) == Some("multiSelect");
        let options: Vec<Value> = q
            .get("options")
            .and_then(Value::as_array)
            .map(|opts| {
                opts.iter()
                    .filter_map(|o| o.get("label").and_then(Value::as_str))
                    .map(|label| serde_json::json!({ "label": label }))
                    .collect()
            })
            .unwrap_or_default();
        mapped.push(serde_json::json!({
            "question": message,
            "header": title,
            "options": options,
            "multiSelect": multi,
        }));
        if let Some(ans) = q
            .get("id")
            .and_then(Value::as_str)
            .and_then(|qid| data.and_then(|d| d.get(qid)))
            .and_then(selected_answer)
        {
            answers.push(if title.is_empty() {
                ans
            } else {
                format!("{title}: {ans}")
            });
        }
    }
    (mapped, answers.join("\n"))
}

/// The chosen answer in a carousel `data` entry — `selectedValue` as text (a string
/// for `singleSelect`, comma-joined for a `multiSelect` array). `None` when unanswered.
fn selected_answer(entry: &Value) -> Option<String> {
    match entry.get("selectedValue") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let joined = arr
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            (!joined.is_empty()).then_some(joined)
        }
        _ => None,
    }
}

/// Cheap metadata probe — replays the patch log and walks the final state once.
fn probe_session_metadata(session_path: &Path) -> Option<SessionMetadata> {
    let raw = fs::read_to_string(session_path).ok()?;
    let state = replay_session(&raw).ok()?;

    let session_id = state
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let creation_ms = state
        .get("creationDate")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let last_message_ms = state
        .get("lastMessageDate")
        .and_then(Value::as_u64)
        .unwrap_or(creation_ms);

    let mut message_count = 0usize;
    let mut has_tool_use = false;
    let mut summary: Option<String> = None;

    // A user rename (VS Code's own UI, or an external writer appending the same
    // patch record). A `null` set (a reset) yields None via `as_str`.
    let custom_title = state
        .get("customTitle")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| truncate_preview(s, 200));

    if let Some(requests) = state.get("requests").and_then(Value::as_array) {
        for req in requests {
            if let Some(text) = extract_user_text(req) {
                message_count += 1;
                if summary.is_none() && !text.is_empty() {
                    summary = Some(truncate_preview(&text, 200));
                }
            }
            if let Some(response) = req.get("response").and_then(Value::as_array) {
                let any_visible = response.iter().any(|part| {
                    let kind = part.get("kind").and_then(Value::as_str);
                    match kind {
                        None => part
                            .get("value")
                            .and_then(Value::as_str)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false),
                        Some("thinking") => part
                            .get("value")
                            .and_then(Value::as_str)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false),
                        Some("toolInvocationSerialized") => {
                            has_tool_use = true;
                            true
                        }
                        Some("progressTaskSerialized") => part
                            .get("content")
                            .and_then(|c| c.get("value"))
                            .and_then(Value::as_str)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false),
                        Some("inlineReference") => inline_reference_text(part).is_some(),
                        _ => false,
                    }
                });
                if any_visible {
                    message_count += 1;
                }
            }
        }
    }

    Some(SessionMetadata {
        session_id,
        message_count,
        first_message_ms: creation_ms,
        last_modified_ms: last_message_ms.max(creation_ms),
        has_tool_use,
        summary,
        custom_title,
    })
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}...", &text[..idx]),
        None => text.to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn build_log(initial: Value, patches: &[Value]) -> String {
        let mut lines = vec![json!({"kind": 0, "v": initial}).to_string()];
        for p in patches {
            lines.push(p.to_string());
        }
        lines.join("\n")
    }

    #[test]
    fn replay_applies_set_patches() {
        let log = build_log(
            json!({"sessionId": "abc", "requests": [], "creationDate": 1000}),
            &[
                json!({"kind": 1, "k": ["customTitle"], "v": "Hello"}),
                json!({"kind": 1, "k": ["creationDate"], "v": 2000}),
            ],
        );
        let state = replay_session(&log).unwrap();
        assert_eq!(state["customTitle"], "Hello");
        assert_eq!(state["creationDate"], 2000);
    }

    #[test]
    fn replay_applies_array_appends() {
        let log = build_log(
            json!({"sessionId": "abc", "requests": []}),
            &[
                json!({
                    "kind": 2,
                    "k": ["requests"],
                    "v": [{
                        "message": {"text": "hi"},
                        "response": [{"value": "hello"}],
                        "requestId": "r1",
                        "modelId": "copilot/gpt-5",
                        "timestamp": 5000
                    }]
                }),
                json!({
                    "kind": 2,
                    "k": ["requests", 0, "response"],
                    "v": [{"kind": "thinking", "value": "thoughts"}]
                }),
                json!({
                    "kind": 1,
                    "k": ["requests", 0, "completionTokens"],
                    "v": 17
                }),
            ],
        );
        let state = replay_session(&log).unwrap();
        let req = &state["requests"][0];
        assert_eq!(req["message"]["text"], "hi");
        assert_eq!(req["response"].as_array().unwrap().len(), 2);
        assert_eq!(req["completionTokens"], 17);
    }

    #[test]
    fn replay_skips_corrupt_trailing_line() {
        let log = format!(
            "{}\n{}\n{}",
            json!({"kind": 0, "v": {"sessionId": "abc", "requests": [], "creationDate": 1}}),
            json!({"kind": 1, "k": ["customTitle"], "v": "Hello"}),
            "garbage line"
        );
        let state = replay_session(&log).unwrap();
        assert_eq!(state["customTitle"], "Hello");
    }

    #[test]
    fn messages_render_user_assistant_pair() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "responseId": "resp-1",
                "modelId": "copilot/auto",
                "completionTokens": 42,
                "elapsedMs": 1200,
                "timestamp": 1700000005000u64,
                "message": {"text": "What is foo?"},
                "response": [
                    {"value": "Foo is bar."},
                    {"kind": "thinking", "value": "reasoning…"},
                    {"kind": "toolInvocationSerialized",
                        "toolId": "copilot_readFile",
                        "toolCallId": "tc-1",
                        "isComplete": true,
                        "invocationMessage": {"value": "Reading foo.txt"},
                        "pastTenseMessage": {"value": "Read foo.txt"}
                    }
                ]
            }]
        });
        let msgs = messages_from_state(&state);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].message_type, "user");
        assert_eq!(msgs[0].provider.as_deref(), Some("copilot"));
        let user_blocks = msgs[0].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(user_blocks[0]["text"], "What is foo?");

        assert_eq!(msgs[1].message_type, "assistant");
        assert_eq!(msgs[1].model.as_deref(), Some("copilot/auto"));
        assert_eq!(
            msgs[1].usage.as_ref().and_then(|u| u.output_tokens),
            Some(42)
        );
        assert_eq!(msgs[1].duration_ms, Some(1200));
        let kinds: Vec<&str> = msgs[1]
            .content
            .as_ref()
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(kinds, vec!["text", "thinking", "tool_use", "tool_result"]);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(blocks[2]["id"], "tc-1");
        assert_eq!(blocks[3]["tool_use_id"], "tc-1");
        assert!(msgs[1].tool_use.is_some());
    }

    #[test]
    fn messages_keep_explicit_prompt_files_but_not_implicit_editor_context() {
        let state = json!({
            "sessionId": "sess-attachments",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "message": {"text": "Review these"},
                "variableData": {"variables": [
                    {"kind": "file", "id": "file:///repo/docs/report.md", "name": "report.md", "value": "private"},
                    {"kind": "file", "id": "vscode.implicit.selection", "name": "selected.rs", "value": "selected"},
                    {"kind": "file", "id": "vscode.implicit.file", "name": "open.rs", "value": "open"}
                ]}
            }]
        });

        let messages = messages_from_state(&state);
        assert_eq!(
            messages[0].data,
            Some(json!({"promptAttachments": [{"name": "report.md"}]}))
        );
        let serialized = serde_json::to_string(&messages[0]).unwrap();
        assert!(!serialized.contains("selected.rs"));
        assert!(!serialized.contains("open.rs"));
        assert!(!serialized.contains("private"));
    }

    #[test]
    fn is_fence_delimiter_only_matches_bare_fences_not_real_blocks() {
        // Bare fence parts (VS Code's tool-wrapper delimiters), any surrounding ws:
        assert!(is_fence_delimiter_only("\n```\n"));
        assert!(is_fence_delimiter_only("```"));
        assert!(is_fence_delimiter_only("```json"));
        assert!(is_fence_delimiter_only("  ```ts-x_1  "));
        // Real code blocks / prose are never matched (the code is inside the part):
        assert!(!is_fence_delimiter_only("```js\nconst x = 1;\n```"));
        assert!(!is_fence_delimiter_only("Editing now."));
        assert!(!is_fence_delimiter_only("``"));
        assert!(!is_fence_delimiter_only("```json extra"));
    }

    #[test]
    fn messages_drop_vscode_tool_wrapper_fence_parts() {
        // VS Code wraps a tool invocation in a code fence, storing the opening/closing
        // fences as their own standalone markdown parts; adjacent, they'd render as an
        // empty code block. The normalizer drops the fence-only parts (prose is kept).
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "responseId": "resp-1",
                "timestamp": 1700000005000u64,
                "message": {"text": "edit it"},
                "response": [
                    {"value": "Editing now."},
                    {"kind": "toolInvocationSerialized",
                        "toolId": "copilot_applyPatch",
                        "toolCallId": "tc-1",
                        "isComplete": true,
                        "invocationMessage": "Applying patch"
                    },
                    {"value": "\n```\n"},
                    {"value": "\n```\n"},
                    {"value": "Done."}
                ]
            }]
        });
        let msgs = messages_from_state(&state);
        let kinds: Vec<&str> = msgs[1]
            .content
            .as_ref()
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap_or(""))
            .collect();
        // The two fence-only parts are gone; only real prose + the tool call remain.
        assert_eq!(kinds, vec!["text", "tool_use", "text"]);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(blocks[0]["text"], "Editing now.");
        assert_eq!(blocks[2]["text"], "Done.");
    }

    #[test]
    fn messages_render_inline_references_as_text() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "responseId": "resp-1",
                "timestamp": 1700000005000u64,
                "message": {"text": "summarize"},
                "response": [
                    {"value": "Main reference: "},
                    {"kind": "inlineReference", "inlineReference": "file:///e%3A/proj/README.md"},
                    {"value": ", secondary: "},
                    {"kind": "inlineReference", "label": "Phase 1 plan", "inlineReference": "file:///e%3A/proj/implementation%20drafts/Phase%201%20-%20Implementation%20Plan.md"}
                ]
            }]
        });

        let msgs = messages_from_state(&state);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        let texts: Vec<&str> = blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect();
        // Consecutive plain-text and inlineReference spans from the same response are
        // coalesced into one text block; no blank lines are inserted between them.
        assert_eq!(
            texts,
            vec!["Main reference: README.md, secondary: Phase 1 plan"]
        );
    }

    #[test]
    fn messages_surface_compaction_summary_as_compact_summary_record() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "timestamp": 1700000005000u64,
                "message": {"text": "Analyze the script"},
                "response": [{"value": "Working on it."}],
                "result": {"metadata": {
                    "maxToolCallsExceeded": true,
                    "summaries": [{"text": "## Conversation Overview\n\n**Primary Objectives:** …"}]
                }}
            }]
        });
        let msgs = messages_from_state(&state);
        // user + assistant + the synthetic compaction record
        assert_eq!(msgs.len(), 3);
        let compact = &msgs[2];
        assert_eq!(compact.message_type, "user");
        assert_eq!(compact.role.as_deref(), Some("user"));
        assert_eq!(compact.subtype.as_deref(), Some("compact_summary"));
        let text = compact.content.as_ref().unwrap().as_array().unwrap()[0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("Conversation Overview"));
        // the ordinary turn is not flagged.
        assert!(msgs[0].subtype.is_none() && msgs[1].subtype.is_none());
    }

    #[test]
    fn no_compaction_record_when_metadata_lacks_summary() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "message": {"text": "hi"},
                "response": [{"value": "hello"}],
                "result": {"metadata": {"codeBlocks": []}}
            }]
        });
        let msgs = messages_from_state(&state);
        assert!(msgs.iter().all(|m| m.subtype.is_none()));
    }

    #[test]
    fn extract_compaction_summary_prefers_summaries_then_flat() {
        // structured summaries[] wins over the flat string.
        let req = json!({"result": {"metadata": {
            "summaries": [{"text": "structured"}],
            "summary": "flat"
        }}});
        assert_eq!(
            extract_compaction_summary(&req).as_deref(),
            Some("structured")
        );
        // falls back to the flat `summary`.
        let req = json!({"result": {"metadata": {"summary": "flat only"}}});
        assert_eq!(
            extract_compaction_summary(&req).as_deref(),
            Some("flat only")
        );
        // joins multiple non-empty summaries, skipping blanks.
        let req = json!({"result": {"metadata": {
            "summaries": [{"text": "a"}, {"text": ""}, {"text": "b"}]
        }}});
        assert_eq!(extract_compaction_summary(&req).as_deref(), Some("a\n\nb"));
        // none when absent.
        assert_eq!(
            extract_compaction_summary(&json!({"result": {"metadata": {}}})),
            None
        );
        assert_eq!(
            extract_compaction_summary(&json!({"message": {"text": "x"}})),
            None
        );
    }

    #[test]
    fn messages_pair_generated_tool_call_ids() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "responseId": "resp-1",
                "message": {"text": "Read the file"},
                "response": [{
                    "kind": "toolInvocationSerialized",
                    "toolId": "copilot_readFile",
                    "isComplete": true,
                    "invocationMessage": {"value": "Reading foo.txt"},
                    "pastTenseMessage": {"value": "Read foo.txt"}
                }]
            }]
        });

        let msgs = messages_from_state(&state);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[1]["type"], "tool_result");
        assert_eq!(blocks[0]["id"], blocks[1]["tool_use_id"]);
        assert_eq!(blocks[0]["id"], "vscode-tool-0-2");
    }

    #[test]
    fn tool_use_carries_structured_path_and_command_arguments() {
        let state = json!({
            "sessionId": "sess-1",
            "creationDate": 1700000000000u64,
            "requests": [{
                "requestId": "req-1",
                "responseId": "resp-1",
                "message": {"text": "do things"},
                "response": [
                    {
                        // File-target tool: the invocation markdown resolves the
                        // target into `uris` (decoded `path`, real shape).
                        "kind": "toolInvocationSerialized",
                        "toolId": "copilot_readFile",
                        "toolCallId": "tc-read",
                        "invocationMessage": {
                            "value": "Reading [](file:///e%3A/proj/a.ps1)",
                            "uris": {"file:///e%3A/proj/a.ps1#1-1": {"scheme": "file", "path": "/e:/proj/a.ps1", "fragment": "1-1"}}
                        }
                    },
                    {
                        // Terminal tool: the exact command line rides toolSpecificData.
                        "kind": "toolInvocationSerialized",
                        "toolId": "run_in_terminal",
                        "toolCallId": "tc-term",
                        "invocationMessage": {"value": "Running `dir`"},
                        "toolSpecificData": {"kind": "terminal", "commandLine": {"original": "dir /b", "forDisplay": "dir"}}
                    },
                    {
                        // Bare-string invocation message (copilot_applyPatch's shape):
                        // still captured as `message`; no structured args exist.
                        "kind": "toolInvocationSerialized",
                        "toolId": "copilot_applyPatch",
                        "toolCallId": "tc-patch",
                        "invocationMessage": "Apply Patch"
                    }
                ]
            }]
        });

        let msgs = messages_from_state(&state);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        assert_eq!(blocks[0]["input"]["path"], "/e:/proj/a.ps1");
        assert_eq!(
            blocks[0]["input"]["message"],
            "Reading [](file:///e%3A/proj/a.ps1)"
        );
        assert_eq!(blocks[1]["input"]["command"], "dir /b");
        assert!(blocks[1]["input"].get("path").is_none());
        assert_eq!(blocks[2]["input"]["message"], "Apply Patch");
        assert!(blocks[2]["input"].get("command").is_none());
    }

    #[test]
    fn tool_use_carries_todo_list_items() {
        let state = json!({
            "sessionId": "s",
            "creationDate": 1,
            "requests": [{
                "requestId": "r",
                "message": {"text": "plan it"},
                "response": [{
                    "kind": "toolInvocationSerialized",
                    "toolId": "manage_todo_list",
                    "toolCallId": "tc-1",
                    "isComplete": true,
                    "invocationMessage": {"value": "Created 2 todos"},
                    "toolSpecificData": {"kind": "todoList", "todoList": [
                        {"id": "1", "title": "Add flag", "status": "in-progress"},
                        {"id": "2", "title": "Write tests", "status": "not-started"}
                    ]}
                }]
            }]
        });
        let msgs = messages_from_state(&state);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        let tool = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
        assert_eq!(tool["name"], "manage_todo_list");
        // Structured items carried verbatim beside the prose message.
        assert_eq!(tool["input"]["message"], "Created 2 todos");
        let todos = tool["input"]["todoList"].as_array().unwrap();
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0]["title"], "Add flag");
        assert_eq!(todos[0]["status"], "in-progress");
        assert_eq!(todos[1]["id"], "2");
        // A non-todo tool gets no todoList key.
        let state2 = json!({
            "sessionId": "s", "creationDate": 1,
            "requests": [{
                "message": {"text": "x"},
                "response": [{
                    "kind": "toolInvocationSerialized",
                    "toolId": "copilot_readFile",
                    "invocationMessage": {"value": "Reading foo"}
                }]
            }]
        });
        let m2 = messages_from_state(&state2);
        let b2 = m2[1].content.as_ref().unwrap().as_array().unwrap();
        assert!(b2[0]["input"].get("todoList").is_none());
    }

    #[test]
    fn askquestions_folds_carousel_into_input_questions_and_answers() {
        let state = json!({
            "sessionId": "s",
            "creationDate": 1,
            "requests": [{
                "requestId": "r",
                "message": {"text": "ask me"},
                "response": [
                    {
                        "kind": "toolInvocationSerialized",
                        "toolId": "vscode_askQuestions",
                        "toolCallId": "call_x",
                        "isComplete": true,
                        "invocationMessage": {"value": "Asking 3 questions (A, B, C)"},
                        "pastTenseMessage": {"value": "Asked 3 questions (A, B, C)"}
                    },
                    {
                        "kind": "questionCarousel",
                        "resolveId": "call_x",
                        "questions": [
                            {"id": "call_x:0", "type": "singleSelect", "title": "A", "message": "Pick A?",
                             "options": [{"id":"y","label":"Yes","value":"Yes"},{"id":"n","label":"No","value":"No"}]},
                            {"id": "call_x:1", "type": "multiSelect", "title": "B", "message": "Pick B?",
                             "options": [{"id":"1","label":"One","value":"One"},{"id":"2","label":"Two","value":"Two"}]},
                            {"id": "call_x:2", "type": "singleSelect", "title": "C", "message": "Pick C?",
                             "options": [{"id":"a","label":"A","value":"A"}]}
                        ],
                        "data": {
                            "call_x:0": {"selectedValue": "Yes"},
                            "call_x:1": {"selectedValue": ["One", "Two"]},
                            "call_x:2": {"selectedValue": null}
                        }
                    }
                ]
            }]
        });
        let msgs = messages_from_state(&state);
        let blocks = msgs[1].content.as_ref().unwrap().as_array().unwrap();
        let tool = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
        assert_eq!(tool["name"], "vscode_askQuestions");
        // Questions folded into the AskUserQuestion shape (message→question, title→header).
        let qs = tool["input"]["questions"].as_array().unwrap();
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0]["question"], "Pick A?");
        assert_eq!(qs[0]["header"], "A");
        assert_eq!(qs[0]["multiSelect"], false);
        assert_eq!(qs[0]["options"][0]["label"], "Yes");
        assert_eq!(qs[1]["multiSelect"], true);
        // The paired result carries the user's answers (unanswered C omitted);
        // multiSelect joins its array; the prose past-tense line is NOT used.
        let result = blocks.iter().find(|b| b["type"] == "tool_result").unwrap();
        assert_eq!(result["tool_use_id"], "call_x");
        assert_eq!(result["content"], "A: Yes\nB: One, Two");
    }

    #[test]
    fn probe_counts_progress_task_responses_as_visible() {
        let tmp = tempfile::TempDir::new().unwrap();
        let session_path = tmp.path().join("progress.jsonl");
        fs::write(
            &session_path,
            json!({"kind": 0, "v": {
                "sessionId": "progress-1111-1111-1111-111111111111",
                "creationDate": 1779490058917u64,
                "requests": [{
                    "response": [{
                        "kind": "progressTaskSerialized",
                        "content": {"value": "Working..."}
                    }]
                }]
            }})
            .to_string(),
        )
        .unwrap();

        let metadata = probe_session_metadata(&session_path).unwrap();
        assert_eq!(metadata.message_count, 1);
    }

    #[test]
    fn probe_counts_inline_reference_responses_as_visible() {
        let tmp = tempfile::TempDir::new().unwrap();
        let session_path = tmp.path().join("inline-reference.jsonl");
        fs::write(
            &session_path,
            json!({"kind": 0, "v": {
                "sessionId": "inline-ref-1111-1111-1111-111111111111",
                "creationDate": 1779490058917u64,
                "requests": [{
                    "response": [{
                        "kind": "inlineReference",
                        "inlineReference": "file:///e%3A/proj/README.md"
                    }]
                }]
            }})
            .to_string(),
        )
        .unwrap();

        let metadata = probe_session_metadata(&session_path).unwrap();
        assert_eq!(metadata.message_count, 1);
    }

    #[test]
    fn probe_reads_custom_title_latest_set_wins_and_null_clears() {
        let tmp = tempfile::TempDir::new().unwrap();
        let session_path = tmp.path().join("renamed.jsonl");
        let header = json!({"kind": 0, "v": {
            "sessionId": "rename-1111-1111-1111-111111111111",
            "creationDate": 1779490058917u64,
            "requests": [{"message": {"text": "hello"}}]
        }});

        // A rename set-patch: custom_title populated, latest wins.
        fs::write(
            &session_path,
            format!(
                "{}\n{}\n{}",
                header,
                json!({"kind": 1, "k": ["customTitle"], "v": "First Name"}),
                json!({"kind": 1, "k": ["customTitle"], "v": "Final Name"}),
            ),
        )
        .unwrap();
        let metadata = probe_session_metadata(&session_path).unwrap();
        assert_eq!(metadata.custom_title.as_deref(), Some("Final Name"));

        // A trailing null set (a reset) clears it; blank/whitespace also clears.
        fs::write(
            &session_path,
            format!(
                "{}\n{}\n{}",
                header,
                json!({"kind": 1, "k": ["customTitle"], "v": "Named"}),
                json!({"kind": 1, "k": ["customTitle"], "v": null}),
            ),
        )
        .unwrap();
        let metadata = probe_session_metadata(&session_path).unwrap();
        assert_eq!(metadata.custom_title, None);

        // No rename records at all: None.
        fs::write(&session_path, header.to_string()).unwrap();
        let metadata = probe_session_metadata(&session_path).unwrap();
        assert_eq!(metadata.custom_title, None);
    }

    #[test]
    fn read_workspace_folder_decodes_uri() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ws_json = tmp.path().join("workspace.json");
        fs::write(&ws_json, r#"{"folder":"file:///Users/me/my%20project"}"#).unwrap();
        assert_eq!(
            read_workspace_folder(&ws_json).as_deref(),
            Some("/Users/me/my project")
        );
    }

    // ── Metadata cache (A4) ─────────────────────────────────────────────────

    /// Write one real session file and return (`chat_dir`, filename, path).
    fn seed_session(tmp: &tempfile::TempDir, name: &str) -> (PathBuf, String, PathBuf) {
        let chat = tmp.path().join("chatSessions");
        fs::create_dir_all(&chat).unwrap();
        let file = format!("{name}.jsonl");
        let path = chat.join(&file);
        fs::write(
            &path,
            json!({"kind": 0, "v": {
                "sessionId": "real",
                "creationDate": 1000u64,
                "requests": [{"message": {"text": "hi"}}]
            }})
            .to_string(),
        )
        .unwrap();
        (chat, file, path)
    }

    fn sentinel_meta() -> SessionMetadata {
        SessionMetadata {
            session_id: "CACHED".into(),
            message_count: 999,
            first_message_ms: 0,
            last_modified_ms: 0,
            has_tool_use: false,
            summary: None,
            custom_title: None,
        }
    }

    #[test]
    fn metadata_cache_hit_returns_cached_and_skips_the_replay() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (chat, file, path) = seed_session(&tmp, "aaaa");
        // Seed a cache entry whose metadata differs from a real probe, stamped
        // with the file's actual (mtime, size) so it's considered fresh.
        let (modified_time, file_size) = file_freshness(&path).unwrap();
        let mut cache = SessionMetadataCache {
            version: METADATA_CACHE_VERSION,
            entries: std::collections::HashMap::default(),
        };
        cache.entries.insert(
            file,
            CachedSessionMetadata {
                modified_time,
                file_size,
                metadata: sentinel_meta(),
            },
        );
        save_metadata_cache(&chat, &cache);

        let listed = list_session_metadata(&chat).unwrap();
        assert_eq!(listed.len(), 1);
        // The sentinel came back → the replay was skipped (a real probe → "real").
        assert_eq!(listed[0].1.session_id, "CACHED");
        assert_eq!(listed[0].1.message_count, 999);
    }

    #[test]
    fn metadata_cache_misses_on_size_change_and_reprobes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (chat, file, path) = seed_session(&tmp, "bbbb");
        let (modified_time, file_size) = file_freshness(&path).unwrap();
        let mut cache = SessionMetadataCache {
            version: METADATA_CACHE_VERSION,
            entries: std::collections::HashMap::default(),
        };
        // Wrong size → stale entry → real probe runs (append-only ⇒ size always moves).
        cache.entries.insert(
            file,
            CachedSessionMetadata {
                modified_time,
                file_size: file_size + 1,
                metadata: sentinel_meta(),
            },
        );
        save_metadata_cache(&chat, &cache);

        let listed = list_session_metadata(&chat).unwrap();
        assert_eq!(listed[0].1.session_id, "real"); // re-probed, not the sentinel
    }

    #[test]
    fn metadata_cache_ignores_a_stale_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (chat, file, path) = seed_session(&tmp, "cccc");
        let (modified_time, file_size) = file_freshness(&path).unwrap();
        // Right freshness but wrong version → the whole cache is dropped.
        let mut cache = SessionMetadataCache {
            version: METADATA_CACHE_VERSION + 1,
            entries: std::collections::HashMap::default(),
        };
        cache.entries.insert(
            file,
            CachedSessionMetadata {
                modified_time,
                file_size,
                metadata: sentinel_meta(),
            },
        );
        save_metadata_cache(&chat, &cache);

        let listed = list_session_metadata(&chat).unwrap();
        assert_eq!(listed[0].1.session_id, "real"); // stale-version cache ignored
                                                    // …and the freshly written cache is at the current version.
        assert_eq!(load_metadata_cache(&chat).version, METADATA_CACHE_VERSION);
    }

    #[test]
    fn metadata_cache_evicts_vanished_sessions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (chat, file, path) = seed_session(&tmp, "dddd");
        let (modified_time, file_size) = file_freshness(&path).unwrap();
        let mut cache = SessionMetadataCache {
            version: METADATA_CACHE_VERSION,
            entries: std::collections::HashMap::default(),
        };
        cache.entries.insert(
            file.clone(),
            CachedSessionMetadata {
                modified_time,
                file_size,
                metadata: sentinel_meta(),
            },
        );
        cache.entries.insert(
            "ghost.jsonl".into(),
            CachedSessionMetadata {
                modified_time,
                file_size,
                metadata: sentinel_meta(),
            },
        );
        save_metadata_cache(&chat, &cache);

        list_session_metadata(&chat).unwrap();
        let after = load_metadata_cache(&chat);
        assert!(after.entries.contains_key(&file)); // present file kept
        assert!(!after.entries.contains_key("ghost.jsonl")); // vanished file evicted
    }

    #[test]
    fn metadata_cache_round_trips_a_cold_probe() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (chat, file, _) = seed_session(&tmp, "eeee");
        // Cold: no cache file yet → probe runs, then the cache is written.
        assert!(!metadata_cache_path(&chat).exists());
        let listed = list_session_metadata(&chat).unwrap();
        assert_eq!(listed[0].1.session_id, "real");
        let cached = load_metadata_cache(&chat);
        assert_eq!(cached.version, METADATA_CACHE_VERSION);
        assert_eq!(
            cached.entries.get(&file).unwrap().metadata.session_id,
            "real"
        );
    }

    #[test]
    fn header_without_kind_zero_errors() {
        let log = json!({"kind": 1, "k": ["x"], "v": 1}).to_string();
        assert!(replay_session(&log).is_err());
    }

    #[test]
    fn load_sessions_skips_empty_chat_panels() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chat_dir = tmp.path().join("chatSessions");
        fs::create_dir_all(&chat_dir).unwrap();
        fs::write(
            tmp.path().join("workspace.json"),
            r#"{"folder":"file:///Users/me/repo"}"#,
        )
        .unwrap();

        // Empty panel: only kind:0 header with requests:[]
        fs::write(
            chat_dir.join("empty-1111-1111-1111-111111111111.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "empty-1111-1111-1111-111111111111",
                "creationDate": 1779490058917u64,
                "requests": []
            }})
            .to_string(),
        )
        .unwrap();

        // Used session with at least one user request.
        let header = json!({"kind": 0, "v": {
            "sessionId": "used-2222-2222-2222-222222222222",
            "creationDate": 1779490058917u64,
            "requests": [{
                "message": {"text": "hello"},
                "response": []
            }]
        }})
        .to_string();
        fs::write(
            chat_dir.join("used-2222-2222-2222-222222222222.jsonl"),
            header,
        )
        .unwrap();

        let roots = vec![tmp.path().to_path_buf()];
        let sessions = load_sessions_in(&tmp.path().to_string_lossy(), &roots).unwrap();
        let ids: Vec<&str> = sessions
            .iter()
            .map(|s| s.actual_session_id.as_str())
            .collect();
        assert!(
            ids.iter().any(|id| id.starts_with("used-")),
            "non-empty session must surface: {ids:?}",
        );
        assert!(
            !ids.iter().any(|id| id.starts_with("empty-")),
            "empty chat panel must be skipped: {ids:?}",
        );
    }

    #[test]
    fn scan_projects_excludes_workspaces_with_only_empty_panels() {
        let ws_root = tempfile::TempDir::new().unwrap();

        // Workspace 1: only empty chat panels.
        let ws1 = ws_root.path().join("hash-empty");
        let chat1 = ws1.join("chatSessions");
        fs::create_dir_all(&chat1).unwrap();
        fs::write(
            ws1.join("workspace.json"),
            r#"{"folder":"file:///Users/me/empty-repo"}"#,
        )
        .unwrap();
        fs::write(
            chat1.join("empty-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "empty-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "creationDate": 1779490058917u64,
                "requests": []
            }})
            .to_string(),
        )
        .unwrap();

        // Workspace 2: one empty panel + one used session.
        let ws2 = ws_root.path().join("hash-used");
        let chat2 = ws2.join("chatSessions");
        fs::create_dir_all(&chat2).unwrap();
        fs::write(
            ws2.join("workspace.json"),
            r#"{"folder":"file:///Users/me/used-repo"}"#,
        )
        .unwrap();
        fs::write(
            chat2.join("empty-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "empty-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "creationDate": 1779490058917u64,
                "requests": []
            }})
            .to_string(),
        )
        .unwrap();
        fs::write(
            chat2.join("used-cccc-cccc-cccc-cccccccccccc.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "used-cccc-cccc-cccc-cccccccccccc",
                "creationDate": 1779490058917u64,
                "requests": [{
                    "message": {"text": "hello"},
                    "response": []
                }]
            }})
            .to_string(),
        )
        .unwrap();

        let projects = scan_projects_in(ws_root.path(), None).unwrap();
        let names: Vec<&str> = projects.iter().map(|p| p.name.as_str()).collect();
        assert!(
            !names.contains(&"empty-repo"),
            "workspace with only empty panels must be skipped: {names:?}",
        );
        let used = projects
            .iter()
            .find(|p| p.name == "used-repo")
            .expect("used-repo project must be present");
        assert_eq!(
            used.session_count, 1,
            "session count must exclude the empty panel",
        );
    }

    #[test]
    fn scan_and_load_empty_window_sessions_as_a_synthetic_project() {
        let user_data = tempfile::TempDir::new().unwrap();
        let chat_dir = user_data
            .path()
            .join("globalStorage")
            .join("emptyWindowChatSessions");
        fs::create_dir_all(&chat_dir).unwrap();

        fs::write(
            chat_dir.join("empty-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "empty-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "creationDate": 1779490058917u64,
                "requests": []
            }})
            .to_string(),
        )
        .unwrap();
        fs::write(
            chat_dir.join("used-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"),
            json!({"kind": 0, "v": {
                "sessionId": "used-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "creationDate": 1779490058917u64,
                "requests": [{
                    "message": {"text": "hello from an empty window"},
                    "response": []
                }]
            }})
            .to_string(),
        )
        .unwrap();

        let projects = scan_projects_from_user_data_path(user_data.path(), None).unwrap();
        let project = projects
            .iter()
            .find(|project| project.name == "VS Code — Empty Window")
            .expect("a non-empty empty-window chat must create a synthetic project");
        assert_eq!(project.actual_path, "vscode-empty-window://code");
        assert_eq!(project.session_count, 1);
        assert_eq!(project.message_count, 1);

        let workspace_roots = vec![user_data.path().join("workspaceStorage")];
        let sessions = load_sessions_in(&project.path, &workspace_roots).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(
            sessions[0].actual_session_id,
            "used-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        );
        assert_eq!(sessions[0].project_name, "VS Code — Empty Window");
        assert_eq!(sessions[0].entrypoint.as_deref(), Some(ENTRYPOINT));

        let matches =
            search_from_user_data_path(user_data.path(), "hello from an empty window", 10).unwrap();
        assert!(
            !matches.is_empty(),
            "global search must include empty-window chat messages"
        );
    }

    #[test]
    fn path_validation_rejects_paths_outside_workspace_storage() {
        let ws_root = tempfile::TempDir::new().unwrap();
        let workspace = ws_root.path().join("hash-used");
        let chat_dir = workspace.join("chatSessions");
        fs::create_dir_all(&chat_dir).unwrap();
        let session_path = chat_dir.join("session-1111-1111-1111-111111111111.jsonl");
        fs::write(
            &session_path,
            json!({"kind": 0, "v": {"sessionId": "session-1111-1111-1111-111111111111", "requests": []}})
                .to_string(),
        )
        .unwrap();

        let outside = tempfile::TempDir::new().unwrap();
        let outside_workspace = outside.path().join("workspace");
        let outside_chat_dir = outside_workspace.join("chatSessions");
        fs::create_dir_all(&outside_chat_dir).unwrap();
        let outside_session = outside_chat_dir.join("outside-1111-1111-1111-111111111111.jsonl");
        fs::write(&outside_session, "{}").unwrap();

        let roots = vec![ws_root.path().to_path_buf()];
        assert!(validate_workspace_path_in(&workspace.to_string_lossy(), &roots).is_ok());
        assert!(validate_session_path_in(&session_path.to_string_lossy(), &roots).is_ok());
        assert!(validate_workspace_path_in(&outside_workspace.to_string_lossy(), &roots).is_err());
        assert!(validate_session_path_in(&outside_session.to_string_lossy(), &roots).is_err());
    }

    #[test]
    fn session_validation_accepts_only_managed_empty_window_storage() {
        let user_data = tempfile::TempDir::new().unwrap();
        let managed = user_data
            .path()
            .join("globalStorage")
            .join("emptyWindowChatSessions")
            .join("managed.jsonl");
        fs::create_dir_all(managed.parent().unwrap()).unwrap();
        fs::write(&managed, "{}").unwrap();

        let outside = tempfile::TempDir::new().unwrap();
        let unmanaged = outside
            .path()
            .join("globalStorage")
            .join("emptyWindowChatSessions")
            .join("unmanaged.jsonl");
        fs::create_dir_all(unmanaged.parent().unwrap()).unwrap();
        fs::write(&unmanaged, "{}").unwrap();

        let roots = vec![user_data.path().join("workspaceStorage")];
        assert!(validate_session_path_in(&managed.to_string_lossy(), &roots).is_ok());
        assert!(validate_session_path_in(&unmanaged.to_string_lossy(), &roots).is_err());
    }

    #[test]
    fn wsl_user_data_validation_requires_a_known_server_root() {
        assert!(is_wsl_user_data_path(Path::new(
            r"\\wsl.localhost\Ubuntu\home\me\.vscode-server\data\User"
        )));
        assert!(is_wsl_user_data_path(Path::new(
            r"\\wsl$\Ubuntu\home\me\.vscodium-server\data\User"
        )));
        assert!(!is_wsl_user_data_path(Path::new(
            r"\\wsl.localhost\Ubuntu\home\me\other\data\User"
        )));
        assert!(!is_wsl_user_data_path(Path::new(
            r"\\wsl.localhost\Ubuntu\home\me\.vscode-server\data\User\extra"
        )));
    }
}
