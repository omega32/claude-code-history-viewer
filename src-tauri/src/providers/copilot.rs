//! Aggregator for the unified GitHub Copilot provider.
//!
//! All three Copilot client surfaces — terminal CLI, Desktop app, and the
//! VS Code Copilot Chat extension — surface to the frontend as a single
//! provider with id `"copilot"`. Per-session disambiguation lives in the
//! `entrypoint` field (`copilot-cli` / `copilot-desktop` / `copilot-vscode`),
//! which the existing source-filter UI already understands.
//!
//! The aggregator calls into the three concrete scanners
//! (`copilot_cli`, `copilot_desktop`, `vscode`) and groups their results by
//! `actual_path` so a folder that has, say, both Copilot CLI sessions AND a
//! VS Code Copilot Chat history collapses into one project entry.
//!
//! Routing back to the right sub-scanner is done lazily: project paths
//! produced by the aggregator are minted with the synthetic
//! `copilot://<actual_path>` scheme, and `load_sessions` re-scans the three
//! sub-scanners and filters their projects by matching `actual_path`. This
//! costs us one extra scan on session-load, but avoids encoding multiple
//! storage hashes into the project URL.

use crate::commands::multi_provider::finalize_loaded_messages;
use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::{copilot_cli, vscode, ProviderInfo, SessionSnapshotLoad};
use crate::utils::parse_rfc3339_utc;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Public provider id stamped on every record.
pub const PROVIDER_ID: &str = "copilot";
const SNAPSHOT_CURSOR_VERSION: u32 = 1;

/// Synthetic URL scheme for merged Copilot projects.
const PROJECT_SCHEME: &str = "copilot://";
const VSCODE_EMPTY_WINDOW_PROJECT_SCHEME: &str = "vscode-empty-window://";

/// Which sub-provider a source path belongs to. Stored inside the merged
/// project URL so `load_sessions` can dispatch directly without rescanning
/// every sub-provider.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SourceKind {
    /// `~/.copilot/session-state` walked by [`copilot_cli`] in CLI mode.
    Cli,
    /// Same storage walked by [`copilot_cli`] in Desktop mode.
    Desktop,
    /// VS Code Copilot Chat workspace storage walked by [`vscode`].
    VsCode,
}

/// One sub-source contributing to a merged project. `path` is whatever the
/// underlying scanner uses to identify the project (CLI: bare filesystem path
/// of the workspace folder; VS Code: encoded `vscode://...` path the vscode
/// scanner expects on load).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SourceRef {
    kind: SourceKind,
    path: String,
}

/// Payload encoded into the merged project URL so we can recover the original
/// sub-source paths on `load_sessions` without re-scanning everything.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectRef {
    actual: String,
    sources: Vec<SourceRef>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CopilotSnapshotCursor {
    version: u32,
    provider: String,
    surface: String,
    canonical_path: String,
    replace_from: usize,
    prefix_digest: String,
    messages_digest: String,
}

fn digest_messages(messages: &[ClaudeMessage]) -> Result<String, String> {
    let bytes = serde_json::to_vec(messages)
        .map_err(|error| format!("Failed to serialize Copilot snapshot messages: {error}"))?;
    Ok(BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(bytes)))
}

fn encode_snapshot_cursor(cursor: &CopilotSnapshotCursor) -> Result<String, String> {
    serde_json::to_vec(cursor)
        .map(|bytes| BASE64_URL_SAFE_NO_PAD.encode(bytes))
        .map_err(|error| format!("Failed to encode Copilot snapshot cursor: {error}"))
}

fn decode_snapshot_cursor(encoded: &str) -> Result<CopilotSnapshotCursor, String> {
    const MAX_CURSOR_BYTES: usize = 64 * 1024;
    if encoded.len() > MAX_CURSOR_BYTES {
        return Err("Copilot snapshot cursor is too large".to_string());
    }
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|error| format!("Invalid Copilot snapshot cursor encoding: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid Copilot snapshot cursor payload: {error}"))
}

fn replacement_checkpoint(messages: &[ClaudeMessage]) -> usize {
    messages
        .iter()
        .rposition(|message| message.message_type == "user" && message.subtype.is_none())
        .unwrap_or(0)
}

fn cursor_for_messages(
    surface: &str,
    canonical_path: &str,
    messages: &[ClaudeMessage],
) -> Result<(String, usize, String), String> {
    let replace_from = replacement_checkpoint(messages);
    let messages_digest = digest_messages(messages)?;
    let prefix_digest = digest_messages(&messages[..replace_from])?;
    let cursor = encode_snapshot_cursor(&CopilotSnapshotCursor {
        version: SNAPSHOT_CURSOR_VERSION,
        provider: PROVIDER_ID.to_string(),
        surface: surface.to_string(),
        canonical_path: canonical_path.to_string(),
        replace_from,
        prefix_digest,
        messages_digest: messages_digest.clone(),
    })?;
    Ok((cursor, replace_from, messages_digest))
}

fn full_snapshot(
    reason: &str,
    surface: &str,
    canonical_path: &str,
    messages: Vec<ClaudeMessage>,
) -> Result<SessionSnapshotLoad, String> {
    let (cursor, cursor_replace_from, _) = cursor_for_messages(surface, canonical_path, &messages)?;
    Ok(SessionSnapshotLoad::Full {
        reason: reason.to_string(),
        messages,
        cursor: Some(cursor),
        cursor_replace_from: Some(cursor_replace_from),
    })
}

fn snapshot_from_messages(
    surface: &str,
    canonical_path: &str,
    messages: Vec<ClaudeMessage>,
    previous_cursor: Option<&str>,
) -> Result<SessionSnapshotLoad, String> {
    let Some(encoded_cursor) = previous_cursor else {
        return full_snapshot("initial", surface, canonical_path, messages);
    };

    let previous = match decode_snapshot_cursor(encoded_cursor) {
        Ok(cursor)
            if cursor.version == SNAPSHOT_CURSOR_VERSION
                && cursor.provider == PROVIDER_ID
                && cursor.surface == surface
                && cursor.canonical_path == canonical_path
                && cursor.replace_from <= messages.len() =>
        {
            cursor
        }
        _ => return full_snapshot("invalid-cursor", surface, canonical_path, messages),
    };

    let (cursor, cursor_replace_from, messages_digest) =
        cursor_for_messages(surface, canonical_path, &messages)?;
    if messages_digest == previous.messages_digest {
        return Ok(SessionSnapshotLoad::Unchanged { cursor });
    }

    if cursor_replace_from < previous.replace_from {
        return full_snapshot("checkpoint-regressed", surface, canonical_path, messages);
    }

    let current_prefix_digest = digest_messages(&messages[..previous.replace_from])?;
    if current_prefix_digest != previous.prefix_digest {
        return full_snapshot(
            "normalized-prefix-mismatch",
            surface,
            canonical_path,
            messages,
        );
    }

    let suffix = messages[previous.replace_from..].to_vec();
    Ok(SessionSnapshotLoad::Replace {
        replace_from: previous.replace_from,
        messages: suffix,
        cursor,
        cursor_replace_from,
    })
}

fn encode_project_ref(r: &ProjectRef) -> String {
    let json = serde_json::to_string(r).unwrap_or_default();
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json);
    format!("{PROJECT_SCHEME}{b64}")
}

fn decode_project_ref(project_path: &str) -> Option<ProjectRef> {
    let payload = project_path.strip_prefix(PROJECT_SCHEME)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Detect a Copilot installation. Reports available if any of the three
/// sub-providers has data on disk.
pub fn detect() -> Option<ProviderInfo> {
    let cli = copilot_cli::detect();
    let desktop = copilot_cli::detect_desktop();
    let vsc = vscode::detect();

    // Prefer the Copilot CLI/Desktop base path (`~/.copilot`) when available,
    // since that's where the bulk of session data lives. Fall back to the
    // VS Code user-data root.
    let base_path = cli
        .as_ref()
        .map(|i| i.base_path.clone())
        .or_else(|| desktop.as_ref().map(|i| i.base_path.clone()))
        .or_else(|| vsc.as_ref().map(|i| i.base_path.clone()))?;

    let is_available = cli.as_ref().is_some_and(|i| i.is_available)
        || desktop.as_ref().is_some_and(|i| i.is_available)
        || vsc.as_ref().is_some_and(|i| i.is_available);

    Some(ProviderInfo {
        id: PROVIDER_ID.to_string(),
        display_name: "Copilot".to_string(),
        base_path,
        is_available,
    })
}

/// Normalise an `actual_path` so equivalent CLI and VS Code references
/// collapse to the same key. VS Code records workspace folders as
/// `file:///path` URIs while the CLI uses bare filesystem paths; we drop
/// the `file://` prefix so they group together.
fn group_key(actual_path: &str) -> String {
    let path = actual_path.strip_prefix("file://").unwrap_or(actual_path);
    let all_separators = !path.is_empty() && path.chars().all(|value| matches!(value, '/' | '\\'));
    let drive_root = path.len() >= 3
        && path.as_bytes()[1] == b':'
        && path.as_bytes()[2..]
            .iter()
            .all(|value| matches!(value, b'/' | b'\\'));
    if all_separators || drive_root {
        path.to_string()
    } else {
        path.trim_end_matches(['/', '\\']).to_string()
    }
}

/// Tag each project with its sub-source kind, then group by canonical folder.
fn merge_projects(parts: Vec<(SourceKind, ClaudeProject)>) -> Vec<ClaudeProject> {
    let mut grouped: HashMap<String, Vec<(SourceKind, ClaudeProject)>> = HashMap::new();
    for (kind, project) in parts {
        let key = group_key(&project.actual_path);
        grouped.entry(key).or_default().push((kind, project));
    }

    let mut merged: Vec<ClaudeProject> = grouped
        .into_iter()
        .map(|(group_key, mut group)| {
            // Use the most-recently-modified project as the display template.
            group.sort_by(|a, b| b.1.last_modified.cmp(&a.1.last_modified));
            let template = group
                .first()
                .map(|(_, p)| p.clone())
                .expect("group is non-empty");
            let session_count = group.iter().map(|(_, p)| p.session_count).sum();
            let message_count = group.iter().map(|(_, p)| p.message_count).sum();
            let last_modified = group
                .iter()
                .map(|(_, p)| p.last_modified.as_str())
                .max()
                .unwrap_or("")
                .to_string();
            // Use the grouping identity itself so the merged path is stable
            // regardless of which storage surface was modified most recently.
            let actual_path = group_key;
            let name = if actual_path.starts_with(VSCODE_EMPTY_WINDOW_PROJECT_SCHEME) {
                template.name.clone()
            } else {
                Path::new(&actual_path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| template.name.clone())
            };

            let sources: Vec<SourceRef> = group
                .iter()
                .map(|(kind, p)| SourceRef {
                    kind: *kind,
                    path: p.path.clone(),
                })
                .collect();
            let path = encode_project_ref(&ProjectRef {
                actual: actual_path.clone(),
                sources,
            });

            ClaudeProject {
                name,
                path,
                actual_path,
                session_count,
                message_count,
                last_modified,
                git_info: None,
                provider: Some(PROVIDER_ID.to_string()),
                storage_type: None,
                custom_directory_label: template.custom_directory_label,
            }
        })
        .collect();

    merged.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    merged
}

pub(crate) struct CopilotSessionListing {
    pub(crate) session: ClaudeSession,
    pub(crate) project_path: String,
}

/// Load one exact live Copilot carrier without reading sibling carrier contents.
/// Concrete providers may enumerate parent directory names to prove spelling.
pub(crate) fn load_session_metadata_by_path(
    raw: &str,
) -> Result<Option<CopilotSessionListing>, String> {
    let path = Path::new(raw);
    if !path.is_absolute() {
        return Err("Copilot session path must be absolute".to_string());
    }
    let parent = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    let loaded = if matches!(parent, Some("chatSessions" | "emptyWindowChatSessions")) {
        vscode::load_session_metadata_by_path(raw)?
    } else if path.file_name().is_some_and(|name| name == "events.jsonl") {
        copilot_cli::load_session_metadata_by_path(raw)?
    } else {
        return Err("Copilot session path is not a recognized live carrier".to_string());
    };
    Ok(loaded.map(|(session, project_path)| CopilotSessionListing {
        session,
        project_path: group_key(&project_path),
    }))
}

fn tag<I: IntoIterator<Item = ClaudeProject>>(
    kind: SourceKind,
    iter: I,
) -> impl Iterator<Item = (SourceKind, ClaudeProject)> {
    iter.into_iter().map(move |p| (kind, p))
}

/// Scan all three Copilot sub-providers and return one merged project list.
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let mut all = Vec::new();
    if let Ok(p) = copilot_cli::scan_projects() {
        all.extend(tag(SourceKind::Cli, p));
    }
    if let Ok(p) = copilot_cli::scan_desktop_projects() {
        all.extend(tag(SourceKind::Desktop, p));
    }
    if let Ok(p) = vscode::scan_projects() {
        all.extend(tag(SourceKind::VsCode, p));
    }
    Ok(merge_projects(all))
}

/// WSL/custom-path variant. `copilot_base_path` is the `~/.copilot` directory
/// for the CLI+Desktop scan; `vscode_user_data_path` is the VS Code user-data
/// dir. Either may be `None` to skip that sub-scan.
pub fn scan_projects_from_paths(
    copilot_base_path: Option<&str>,
    vscode_user_data_path: Option<&Path>,
    custom_directory_label: Option<&str>,
) -> Result<Vec<ClaudeProject>, String> {
    let mut all = Vec::new();
    if let Some(base) = copilot_base_path {
        if let Ok(p) = copilot_cli::scan_projects_from_path(base, custom_directory_label) {
            all.extend(tag(SourceKind::Cli, p));
        }
        if let Ok(p) = copilot_cli::scan_desktop_projects_from_path(base, custom_directory_label) {
            all.extend(tag(SourceKind::Desktop, p));
        }
    }
    if let Some(base) = vscode_user_data_path {
        if let Ok(p) = vscode::scan_projects_from_user_data_path(base, custom_directory_label) {
            all.extend(tag(SourceKind::VsCode, p));
        }
    }
    Ok(merge_projects(all))
}

/// Load sessions for a merged project. Decodes the source list embedded in
/// the project URL and dispatches each sub-source directly. No rescan.
pub fn load_sessions(project_path: &str, exclude: bool) -> Result<Vec<ClaudeSession>, String> {
    let Some(project_ref) = decode_project_ref(project_path) else {
        // Older/malformed URL — degrade to a rescan-and-filter fallback so we
        // don't break in case stale URLs survive in any cache.
        return Ok(load_sessions_fallback(project_path, exclude));
    };

    let mut sessions = Vec::new();
    for src in project_ref.sources {
        let result = match src.kind {
            SourceKind::Cli | SourceKind::Desktop => copilot_cli::load_sessions(&src.path, exclude),
            SourceKind::VsCode => vscode::load_sessions(&src.path, exclude),
        };
        if let Ok(s) = result {
            sessions.extend(s);
        }
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Legacy fallback: if a caller hands us a project URL without an embedded
/// source list (unlikely after this refactor, but defensive), fall back to
/// the old scan-and-filter behaviour.
fn load_sessions_fallback(project_path: &str, exclude: bool) -> Vec<ClaudeSession> {
    type Loader = dyn Fn(&str, bool) -> Result<Vec<ClaudeSession>, String>;
    let raw = project_path
        .strip_prefix(PROJECT_SCHEME)
        .unwrap_or(project_path);
    let target_key = group_key(raw);
    let mut sessions = Vec::new();

    let collect = |scanned: Vec<ClaudeProject>, loader: &Loader, sink: &mut Vec<ClaudeSession>| {
        for p in scanned {
            if group_key(&p.actual_path) == target_key {
                if let Ok(s) = loader(&p.path, exclude) {
                    sink.extend(s);
                }
            }
        }
    };

    if let Ok(scanned) = copilot_cli::scan_projects() {
        collect(scanned, &copilot_cli::load_sessions, &mut sessions);
    }
    if let Ok(scanned) = copilot_cli::scan_desktop_projects() {
        collect(scanned, &copilot_cli::load_sessions, &mut sessions);
    }
    if let Ok(scanned) = vscode::scan_projects() {
        collect(scanned, &vscode::load_sessions, &mut sessions);
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    sessions
}

/// Heuristic: does this look like a VS Code chat session file path?
fn is_vscode_session_path(session_path: &str) -> bool {
    ((session_path.contains("/workspaceStorage/") || session_path.contains("\\workspaceStorage\\"))
        && (session_path.contains("/chatSessions/") || session_path.contains("\\chatSessions\\")))
        || session_path.contains("/globalStorage/emptyWindowChatSessions/")
        || session_path.contains("\\globalStorage\\emptyWindowChatSessions\\")
}

fn sort_and_truncate_results(results: &mut Vec<ClaudeMessage>, limit: usize) {
    results.sort_by(|a, b| {
        match (
            parse_rfc3339_utc(&a.timestamp),
            parse_rfc3339_utc(&b.timestamp),
        ) {
            (Some(a_ts), Some(b_ts)) => b_ts.cmp(&a_ts),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => b.timestamp.cmp(&a.timestamp),
        }
    });
    results.truncate(limit);
}

/// Load messages by sniffing the session file path and dispatching to the
/// correct sub-scanner. Both `copilot_cli` and `vscode` loaders already stamp
/// `provider: "copilot"` on each message because we updated their constants.
pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    if is_vscode_session_path(session_path) {
        vscode::load_messages(session_path)
    } else {
        copilot_cli::load_messages(session_path)
    }
}

/// Load a cursor-aware Copilot snapshot.
///
/// Copilot's two storage families have different mutation semantics: the CLI
/// can append tool completions that update an earlier assistant record, while
/// VS Code replays patches that may target any prior request. The first safe
/// delta implementation therefore replays the provider source authoritatively,
/// finalizes the complete normalized sequence, and emits a suffix only after
/// the previous cursor's retained normalized prefix hashes byte-for-byte.
/// Provider-native parser checkpoints can optimize that replay later without
/// changing this envelope or its fallback guarantees.
pub(crate) fn load_session_snapshot(
    session_path: &str,
    previous_cursor: Option<&str>,
) -> Result<SessionSnapshotLoad, String> {
    let surface = if is_vscode_session_path(session_path) {
        "copilot-vscode"
    } else {
        "copilot-cli"
    };
    let messages = finalize_loaded_messages(load_messages(session_path)?);
    let canonical = fs::canonicalize(session_path)
        .map_err(|error| format!("Failed to resolve Copilot session path: {error}"))?;
    let canonical_path = canonical.to_string_lossy();
    snapshot_from_messages(surface, &canonical_path, messages, previous_cursor)
}

/// Search across all three sub-providers and merge results, capping at `limit`.
pub fn search(query: &str, limit: usize) -> Result<Vec<ClaudeMessage>, String> {
    let mut out = Vec::new();
    if let Ok(r) = copilot_cli::search(query, limit) {
        out.extend(r);
    }
    if let Ok(r) = copilot_cli::search_desktop(query, limit) {
        out.extend(r);
    }
    if let Ok(r) = vscode::search(query, limit) {
        out.extend(r);
    }
    sort_and_truncate_results(&mut out, limit);
    Ok(out)
}

/// WSL/custom-path search variant.
pub fn search_from_paths(
    copilot_base_path: Option<&str>,
    vscode_user_data_path: Option<&Path>,
    query: &str,
    limit: usize,
) -> Result<Vec<ClaudeMessage>, String> {
    let mut out = Vec::new();
    if let Some(base) = copilot_base_path {
        if let Ok(r) = copilot_cli::search_from_path(base, query, limit) {
            out.extend(r);
        }
        if let Ok(r) = copilot_cli::search_desktop_from_path(base, query, limit) {
            out.extend(r);
        }
    }
    if let Some(base) = vscode_user_data_path {
        if let Ok(r) = vscode::search_from_user_data_path(base, query, limit) {
            out.extend(r);
        }
    }
    sort_and_truncate_results(&mut out, limit);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(actual_path: &str, path: &str, sessions: usize, messages: usize) -> ClaudeProject {
        ClaudeProject {
            name: Path::new(actual_path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| actual_path.to_string()),
            path: path.to_string(),
            actual_path: actual_path.to_string(),
            session_count: sessions,
            message_count: messages,
            last_modified: "2026-01-01T00:00:00Z".to_string(),
            git_info: None,
            provider: Some("copilot".to_string()),
            storage_type: None,
            custom_directory_label: None,
        }
    }

    fn message(uuid: &str, timestamp: &str) -> ClaudeMessage {
        ClaudeMessage {
            uuid: uuid.to_string(),
            parent_uuid: None,
            session_id: "session".to_string(),
            timestamp: timestamp.to_string(),
            message_type: "user".to_string(),
            content: None,
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
            provider: Some(PROVIDER_ID.to_string()),
        }
    }

    fn authored_message(uuid: &str, message_type: &str, subtype: Option<&str>) -> ClaudeMessage {
        let mut value = message(uuid, "2026-01-01T00:00:00Z");
        value.message_type = message_type.to_string();
        value.subtype = subtype.map(str::to_string);
        value.content = Some(serde_json::json!(uuid));
        value
    }

    #[test]
    fn group_key_strips_file_prefix_and_trailing_slash() {
        assert_eq!(group_key("/Users/me/repo"), "/Users/me/repo");
        assert_eq!(group_key("file:///Users/me/repo"), "/Users/me/repo");
        assert_eq!(group_key("file:///Users/me/repo/"), "/Users/me/repo");
        assert_eq!(group_key(r"C:\Users\me\repo\"), r"C:\Users\me\repo");
        assert_eq!(group_key("/"), "/");
        assert_eq!(group_key(r"C:\"), r"C:\");
        assert_eq!(group_key("C:/"), "C:/");
    }

    #[test]
    fn merge_collapses_cli_and_vscode_for_same_folder() {
        let cli = project("/Users/me/repo", "copilot-cli:///Users/me/repo", 2, 50);
        let vsc = project(
            "file:///Users/me/repo",
            "vscode:///Users/me/.vscode/workspaceStorage/abc",
            3,
            70,
        );
        let merged = merge_projects(vec![(SourceKind::Cli, cli), (SourceKind::VsCode, vsc)]);
        assert_eq!(merged.len(), 1);
        let p = &merged[0];
        assert_eq!(p.session_count, 5);
        assert_eq!(p.message_count, 120);
        assert_eq!(p.actual_path, "/Users/me/repo");
        assert!(p.path.starts_with("copilot://"));
        assert_eq!(p.provider.as_deref(), Some("copilot"));

        // Round-trip: decoded ref should preserve both source paths.
        let decoded = decode_project_ref(&p.path).expect("decodes");
        assert_eq!(decoded.actual, "/Users/me/repo");
        assert_eq!(decoded.sources.len(), 2);
        let cli_src = decoded
            .sources
            .iter()
            .find(|s| s.kind == SourceKind::Cli)
            .unwrap();
        let vsc_src = decoded
            .sources
            .iter()
            .find(|s| s.kind == SourceKind::VsCode)
            .unwrap();
        assert_eq!(cli_src.path, "copilot-cli:///Users/me/repo");
        assert_eq!(
            vsc_src.path,
            "vscode:///Users/me/.vscode/workspaceStorage/abc"
        );
    }

    #[test]
    fn merge_keeps_distinct_folders_separate() {
        let a = project("/repo/a", "copilot-cli:///repo/a", 1, 5);
        let b = project("/repo/b", "copilot-cli:///repo/b", 2, 10);
        let merged = merge_projects(vec![(SourceKind::Cli, a), (SourceKind::Cli, b)]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_preserves_empty_window_project_identity_and_name() {
        let mut empty = project(
            "vscode-empty-window://code",
            "vscode-empty-window:///Users/me/Library/Application Support/Code/User",
            1,
            2,
        );
        empty.name = "VS Code — Empty Window".to_string();

        let merged = merge_projects(vec![(SourceKind::VsCode, empty)]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "VS Code — Empty Window");
        assert_eq!(merged[0].actual_path, "vscode-empty-window://code");
    }

    #[test]
    fn project_ref_round_trips() {
        let r = ProjectRef {
            actual: "/Users/me/repo".to_string(),
            sources: vec![
                SourceRef {
                    kind: SourceKind::Cli,
                    path: "copilot-cli:///x".to_string(),
                },
                SourceRef {
                    kind: SourceKind::VsCode,
                    path: "vscode:///y".to_string(),
                },
            ],
        };
        let encoded = encode_project_ref(&r);
        assert!(encoded.starts_with("copilot://"));
        let decoded = decode_project_ref(&encoded).unwrap();
        assert_eq!(decoded.actual, r.actual);
        assert_eq!(decoded.sources.len(), 2);
    }

    #[test]
    fn decode_project_ref_returns_none_for_legacy_url() {
        // Old format without base64 payload should not falsely decode.
        assert!(decode_project_ref("copilot:///repo/a").is_none());
    }

    #[test]
    fn search_results_sort_before_truncate() {
        let mut results = vec![
            message("old", "2026-01-01T00:00:00Z"),
            message("invalid", "not-a-timestamp"),
            message("new", "2026-01-02T00:00:00Z"),
        ];

        sort_and_truncate_results(&mut results, 2);

        assert_eq!(
            results.iter().map(|m| m.uuid.as_str()).collect::<Vec<_>>(),
            vec!["new", "old"]
        );
    }

    #[test]
    fn is_vscode_session_path_detects_chatsessions_files() {
        assert!(is_vscode_session_path(
            "/Users/me/Library/Application Support/Code/User/workspaceStorage/abc/chatSessions/x.jsonl"
        ));
        assert!(is_vscode_session_path(
            r"C:\Users\me\AppData\Roaming\Code\User\workspaceStorage\abc\chatSessions\x.jsonl"
        ));
        assert!(is_vscode_session_path(
            "/Users/me/Library/Application Support/Code/User/globalStorage/emptyWindowChatSessions/x.jsonl"
        ));
        assert!(is_vscode_session_path(
            r"C:\Users\me\AppData\Roaming\Code\User\globalStorage\emptyWindowChatSessions\x.jsonl"
        ));
        assert!(!is_vscode_session_path(
            "/Users/me/.copilot/session-state/abc/events.jsonl"
        ));
    }

    #[test]
    fn snapshot_replaces_from_the_previous_authored_user_checkpoint() {
        let initial = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
            authored_message("u2", "user", None),
            authored_message("a2", "assistant", None),
        ];
        let cursor = match snapshot_from_messages(
            "copilot-vscode",
            "C:/sessions/example.jsonl",
            initial,
            None,
        )
        .expect("initial snapshot")
        {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                cursor_replace_from: Some(2),
                ..
            } => cursor,
            _ => panic!("initial snapshot should seed an authored-user checkpoint"),
        };

        let appended = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
            authored_message("u2", "user", None),
            authored_message("a2-updated", "assistant", None),
            authored_message("u3", "user", None),
            authored_message("a3", "assistant", None),
        ];
        match snapshot_from_messages(
            "copilot-vscode",
            "C:/sessions/example.jsonl",
            appended,
            Some(&cursor),
        )
        .expect("delta snapshot")
        {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor_replace_from,
                ..
            } => {
                assert_eq!(replace_from, 2);
                assert_eq!(cursor_replace_from, 4);
                assert_eq!(
                    messages
                        .iter()
                        .map(|message| message.uuid.as_str())
                        .collect::<Vec<_>>(),
                    vec!["u2", "a2-updated", "u3", "a3"]
                );
            }
            _ => panic!("append should replace the prior open suffix"),
        }
    }

    #[test]
    fn snapshot_falls_back_when_a_patch_changes_the_retained_prefix() {
        let initial = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
            authored_message("u2", "user", None),
            authored_message("a2", "assistant", None),
        ];
        let cursor = match snapshot_from_messages(
            "copilot-vscode",
            "C:/sessions/example.jsonl",
            initial,
            None,
        )
        .expect("initial snapshot")
        {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };

        let patched = vec![
            authored_message("u1-patched", "user", None),
            authored_message("a1", "assistant", None),
            authored_message("u2", "user", None),
            authored_message("a2", "assistant", None),
        ];
        match snapshot_from_messages(
            "copilot-vscode",
            "C:/sessions/example.jsonl",
            patched,
            Some(&cursor),
        )
        .expect("fallback snapshot")
        {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "normalized-prefix-mismatch");
            }
            _ => panic!("a retained-prefix mutation must fall back"),
        }
    }

    #[test]
    fn snapshot_reports_unchanged_for_an_identical_normalized_sequence() {
        let messages = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
        ];
        let cursor = match snapshot_from_messages(
            "copilot-cli",
            "/sessions/events.jsonl",
            messages.clone(),
            None,
        )
        .expect("initial snapshot")
        {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };

        assert!(matches!(
            snapshot_from_messages(
                "copilot-cli",
                "/sessions/events.jsonl",
                messages,
                Some(&cursor),
            )
            .expect("unchanged snapshot"),
            SessionSnapshotLoad::Unchanged { .. }
        ));
    }

    #[test]
    fn snapshot_does_not_use_compaction_summaries_as_checkpoints() {
        let messages = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
            authored_message("summary", "user", Some("compact_summary")),
            authored_message("a2", "assistant", None),
        ];
        match snapshot_from_messages(
            "copilot-vscode",
            "C:/sessions/example.jsonl",
            messages,
            None,
        )
        .expect("initial snapshot")
        {
            SessionSnapshotLoad::Full {
                cursor_replace_from: Some(0),
                ..
            } => {}
            _ => panic!("only ordinary authored-user messages may own checkpoints"),
        }
    }

    #[test]
    fn snapshot_rejects_a_cursor_from_another_copilot_surface() {
        let messages = vec![
            authored_message("u1", "user", None),
            authored_message("a1", "assistant", None),
        ];
        let cursor = match snapshot_from_messages(
            "copilot-cli",
            "/sessions/events.jsonl",
            messages.clone(),
            None,
        )
        .expect("initial snapshot")
        {
            SessionSnapshotLoad::Full {
                cursor: Some(cursor),
                ..
            } => cursor,
            _ => panic!("cursor expected"),
        };

        match snapshot_from_messages(
            "copilot-vscode",
            "/sessions/events.jsonl",
            messages,
            Some(&cursor),
        )
        .expect("invalid cursor fallback")
        {
            SessionSnapshotLoad::Full { reason, .. } => {
                assert_eq!(reason, "invalid-cursor");
            }
            _ => panic!("surface-mismatched cursors must fall back"),
        }
    }
}
