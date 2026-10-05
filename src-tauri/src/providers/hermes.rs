//! Read-only Hermes display history, independent of its inference backend.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession, InferenceMetadata};
use crate::providers::ProviderInfo;
use crate::utils::{build_provider_message, inspect_path_without_aliases, is_symlink_or_reparse};
use base64::Engine;
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

const PROVIDER: &str = "hermes";
const SCHEME: &str = "hermes://";
const SESSION_COLUMNS: &[&str] = &[
    "id",
    "source",
    "created_source",
    "model",
    "model_config",
    "started_at",
    "ended_at",
    "end_reason",
    "parent_session_id",
    "cwd",
    "title",
    "title_source",
    "last_activity_at",
    "archived",
    "auto_archived",
    "hidden",
    "pinned",
];
const MESSAGE_COLUMNS: &[&str] = &[
    "id",
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "tool_name",
    "timestamp",
    "active",
    "compacted",
    "display_order",
    "display_identity",
    "display_metadata",
    "display_kind",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "finish_reason",
    "_compressed_summary",
    "tool_call_uid",
    "tool_call_uids",
];

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackupSource {
    pub id: String,
    pub root: String,
    pub database_path: String,
    pub relative_path: String,
}

pub(crate) fn backup_sources() -> Result<Vec<BackupSource>, String> {
    match selected_home() {
        Some(home) => backup_sources_in_home(&home),
        None => Err("Cannot resolve the selected Hermes home".to_string()),
    }
}

fn backup_source(
    root: &Path,
    id: String,
    relative_path: String,
) -> Result<Option<BackupSource>, String> {
    let database = root.join("state.db");
    let Some(metadata) = inspect_path_without_aliases(&database)? else {
        return Ok(None);
    };
    if !metadata.is_file() {
        return Err(format!(
            "Hermes database must be a regular file: {}",
            database.display()
        ));
    }
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let database = database.canonicalize().map_err(|e| e.to_string())?;
    if database.parent() != Some(root.as_path()) {
        return Err("Hermes database escaped its selected root".to_string());
    }
    Ok(Some(BackupSource {
        id,
        root: root
            .to_str()
            .ok_or("Hermes root is not valid UTF-8")?
            .to_string(),
        database_path: database
            .to_str()
            .ok_or("Hermes database path is not valid UTF-8")?
            .to_string(),
        relative_path,
    }))
}

fn backup_sources_in_home(home: &Path) -> Result<Vec<BackupSource>, String> {
    let Some(metadata) = inspect_path_without_aliases(home)? else {
        return Ok(Vec::new());
    };
    if !metadata.is_dir() {
        return Err("Hermes home must be a regular directory".to_string());
    }
    let mut sources = Vec::new();
    if let Some(source) =
        backup_source(home, "default".to_string(), "default/state.db".to_string())?
    {
        sources.push(source);
    }
    let profiles = home.join("profiles");
    if let Some(metadata) = inspect_path_without_aliases(&profiles)? {
        if !metadata.is_dir() {
            return Err("Hermes profiles must be a regular directory".to_string());
        }
        let entries = std::fs::read_dir(&profiles)
            .map_err(|e| format!("Cannot enumerate Hermes profiles: {e}"))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Cannot inspect Hermes profile: {e}"))?;
            let metadata = std::fs::symlink_metadata(entry.path()).map_err(|e| e.to_string())?;
            if is_symlink_or_reparse(&metadata) {
                return Err("Hermes profile contains a symbolic link or reparse point".to_string());
            }
            if !metadata.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name
                .to_str()
                .ok_or("Hermes profile name is not valid UTF-8")?;
            let id = format!("profile-{:x}", Sha256::digest(name.as_bytes()));
            if let Some(source) = backup_source(
                &entry.path(),
                id,
                format!("profiles/{}/state.db", urlencoding::encode(name)),
            )? {
                sources.push(source);
            }
        }
    }
    sources.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let mut paths = HashSet::new();
    for source in &sources {
        let identity = if cfg!(windows) {
            source.database_path.replace('\\', "/").to_lowercase()
        } else {
            source.database_path.clone()
        };
        if !paths.insert(identity) {
            return Err("Hermes backup sources contain aliased databases".to_string());
        }
    }
    Ok(sources)
}

pub(crate) struct OfflineSession {
    pub session: ClaudeSession,
    pub project_path: String,
    pub hidden: bool,
    pub archived: bool,
    pub pinned: bool,
}

pub(crate) fn load_offline_sessions(path: &Path) -> Result<Vec<OfflineSession>, String> {
    let conn = open_offline_db(path)?;
    let native = rows(&conn, "sessions", SESSION_COLUMNS, None)?;
    let sessions = sessions_in_conn(&conn, path, None, false)?;
    sessions
        .into_iter()
        .map(|session| {
            let row = native
                .iter()
                .find(|row| text(row, "id") == session.actual_session_id)
                .ok_or("Hermes session metadata disappeared")?;
            let cwd = text(row, "cwd");
            Ok(OfflineSession {
                project_path: if cwd.is_empty() {
                    selector(path, "project", "")
                } else {
                    cwd.to_string()
                },
                hidden: flag(row, "hidden"),
                archived: flag(row, "archived") || flag(row, "auto_archived"),
                pinned: flag(row, "pinned"),
                session,
            })
        })
        .collect()
}

pub(crate) fn load_offline_messages(path: &Path, id: &str) -> Result<Vec<ClaudeMessage>, String> {
    if id.trim().is_empty() || id.contains('\0') {
        return Err("Hermes backup requires an explicit nonempty native session ID".to_string());
    }
    let conn = open_offline_db(path)?;
    let native = rows(&conn, "sessions", SESSION_COLUMNS, None)?;
    messages_in_conn(&conn, &native, id)
}

fn selected_home() -> Option<PathBuf> {
    std::env::var("HERMES_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|value| {
            expand_home(
                &value,
                |name| std::env::var(name).ok(),
                dirs::home_dir().as_deref(),
                &std::env::current_dir().unwrap_or_default(),
            )
        })
        .or_else(|| {
            let suffix = std::env::var("HERMES_DATA_DIR_SUFFIX").unwrap_or_default();
            if cfg!(windows) {
                dirs::data_local_dir().map(|p| p.join(format!("hermes{suffix}")))
            } else {
                dirs::home_dir().map(|p| p.join(format!(".hermes{suffix}")))
            }
        })
}

fn homes() -> Vec<PathBuf> {
    selected_home().map_or_else(Vec::new, |home| stores_in_home(&home))
}

fn expand_home(
    value: &str,
    lookup: impl Fn(&str) -> Option<String>,
    user_home: Option<&Path>,
    current_dir: &Path,
) -> PathBuf {
    let pattern = if cfg!(windows) {
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)|%([^%]+)%"
    } else {
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)"
    };
    let re = regex::Regex::new(pattern).expect("static Hermes path expansion pattern");
    let expanded = re.replace_all(value, |caps: &regex::Captures<'_>| {
        caps.iter()
            .skip(1)
            .flatten()
            .next()
            .and_then(|m| lookup(m.as_str()))
            .unwrap_or_else(|| caps[0].to_string())
    });
    let path = if expanded == "~" {
        user_home
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(expanded.as_ref()))
    } else if let Some(tail) = expanded
        .strip_prefix("~/")
        .or_else(|| expanded.strip_prefix("~\\"))
    {
        user_home
            .map(|home| home.join(tail))
            .unwrap_or_else(|| PathBuf::from(expanded.as_ref()))
    } else {
        PathBuf::from(expanded.as_ref())
    };
    if path.is_absolute() {
        path
    } else {
        current_dir.join(path)
    }
}

fn stores_in_home(home: &Path) -> Vec<PathBuf> {
    let mut stores = Vec::new();
    let db = home.join("state.db");
    if db.is_file() {
        stores.push(db);
    }
    if let Ok(entries) = std::fs::read_dir(home.join("profiles")) {
        for entry in entries.flatten() {
            let db = entry.path().join("state.db");
            if db.is_file() {
                stores.push(db);
            }
        }
    }
    stores.sort();
    stores.dedup();
    stores
}

pub fn detect() -> Option<ProviderInfo> {
    let stores = homes();
    let first = stores.first()?;
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Hermes".to_string(),
        base_path: first.parent()?.to_string_lossy().into_owned(),
        is_available: true,
    })
}

fn open_db(path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("Failed to open Hermes database: {e}"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| e.to_string())?;
    // One SQLite snapshot must cover metadata, ancestry and messages, including the live WAL.
    conn.execute_batch("PRAGMA query_only=ON; BEGIN DEFERRED")
        .map_err(|e| e.to_string())?;
    Ok(conn)
}

fn open_offline_db(path: &Path) -> Result<Connection, String> {
    let metadata =
        inspect_path_without_aliases(path)?.ok_or("Hermes backup database is missing")?;
    if !metadata.is_file() {
        return Err("Hermes backup database must be a regular file".to_string());
    }
    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        if inspect_path_without_aliases(Path::new(&sidecar))?.is_some() {
            return Err(
                "Hermes backup database must be standalone without SQLite sidecars".to_string(),
            );
        }
    }
    let uri = format!(
        "file:{}?immutable=1",
        urlencoding::encode(
            path.to_str()
                .ok_or("Hermes backup path is not valid UTF-8")?
        )
    );
    // Published backups are standalone snapshots: immutable prevents sidecar creation or live WAL consultation.
    let conn = Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|e| format!("Cannot open Hermes backup database: {e}"))?;
    conn.execute_batch("PRAGMA query_only=ON; BEGIN DEFERRED")
        .map_err(|e| e.to_string())?;
    let mut check = conn
        .prepare("PRAGMA quick_check")
        .map_err(|e| e.to_string())?;
    let results = check
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if results != ["ok"] {
        return Err("Hermes backup database failed SQLite quick_check".to_string());
    }
    drop(check);
    schema_columns(&conn, "sessions")?;
    schema_columns(&conn, "messages")?;
    Ok(conn)
}

fn schema_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, String> {
    let mut schema = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| e.to_string())?;
    let available: HashSet<String> = schema
        .query_map([], |r| r.get(1))
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let required: &[&str] = if table == "sessions" {
        &["id", "started_at"]
    } else {
        &["id", "session_id", "role", "content", "timestamp"]
    };
    if required.iter().any(|column| !available.contains(*column)) {
        return Err(format!("Unsupported Hermes {table} schema"));
    }
    Ok(available)
}

fn rows(
    conn: &Connection,
    table: &str,
    wanted: &[&str],
    session: Option<&str>,
) -> Result<Vec<Value>, String> {
    let available = schema_columns(conn, table)?;
    let columns: Vec<&str> = wanted
        .iter()
        .copied()
        .filter(|c| available.contains(*c))
        .collect();
    let query = format!(
        "SELECT {} FROM {table}{} ORDER BY id",
        columns.join(","),
        if session.is_some() {
            " WHERE session_id = ?1"
        } else {
            ""
        }
    );
    let mut stmt = conn.prepare(&query).map_err(|e| e.to_string())?;
    let mut cursor = if let Some(id) = session {
        stmt.query([id])
    } else {
        stmt.query([])
    }
    .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    while let Some(row) = cursor.next().map_err(|e| e.to_string())? {
        let mut object = Map::new();
        for (i, name) in columns.iter().enumerate() {
            let value = match row.get_ref(i).map_err(|e| e.to_string())? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(n) => json!(n),
                ValueRef::Real(n) => json!(n),
                ValueRef::Text(s) => Value::String(
                    std::str::from_utf8(s)
                        .map_err(|e| e.to_string())?
                        .to_string(),
                ),
                ValueRef::Blob(b) => {
                    Value::String(base64::engine::general_purpose::STANDARD.encode(b))
                }
            };
            object.insert((*name).to_string(), value);
        }
        result.push(Value::Object(object));
    }
    Ok(result)
}

fn text<'a>(row: &'a Value, key: &str) -> &'a str {
    row[key].as_str().unwrap_or("")
}
fn optional(row: &Value, key: &str) -> Option<String> {
    let s = text(row, key);
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}
fn parsed(row: &Value, key: &str) -> Value {
    serde_json::from_str(text(row, key)).unwrap_or(Value::Null)
}
fn flag(row: &Value, key: &str) -> bool {
    row[key].as_i64().is_some_and(|n| n != 0) || row[key] == true
}
fn timestamp(value: &Value) -> String {
    value
        .as_f64()
        .filter(|n| n.is_finite())
        .and_then(|n| chrono::DateTime::from_timestamp_micros((n * 1_000_000.0) as i64))
        .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Micros, true))
        .unwrap_or_default()
}
fn selector(path: &Path, kind: &str, value: &str) -> String {
    format!(
        "{SCHEME}{}?{kind}={}",
        urlencoding::encode(&path.to_string_lossy()),
        urlencoding::encode(value)
    )
}
fn parse_selector(value: &str, kind: &str) -> Result<(PathBuf, String), String> {
    let tail = value
        .strip_prefix(SCHEME)
        .ok_or("Invalid Hermes selector")?;
    let (path, selection) = tail.split_once('?').ok_or("Invalid Hermes selector")?;
    let id = selection
        .strip_prefix(&format!("{kind}="))
        .ok_or("Invalid Hermes selector kind")?;
    let path = urlencoding::decode(path).map_err(|e| e.to_string())?;
    let id = urlencoding::decode(id).map_err(|e| e.to_string())?;
    if path.contains('\0') || id.contains('\0') || !Path::new(path.as_ref()).is_absolute() {
        return Err("Invalid Hermes database path".to_string());
    }
    Ok((PathBuf::from(path.as_ref()), id.into_owned()))
}

fn explicit_child(session: &Value) -> bool {
    if text(session, "source") == "tool" {
        return true;
    }
    let config = parsed(session, "model_config");
    let parent = text(session, "parent_session_id");
    ["_branched_from", "_delegate_from", "_reset_from"]
        .iter()
        .any(|key| {
            config[*key]
                .as_str()
                .is_some_and(|id| parent.is_empty() || id == parent)
        })
}
fn lineage<'a>(sessions: &'a [Value], id: &str) -> Result<Vec<&'a Value>, String> {
    let mut current = sessions
        .iter()
        .find(|s| text(s, "id") == id)
        .ok_or("Hermes session not found")?;
    let mut chain = vec![current];
    let mut seen = HashSet::from([id.to_string()]);
    while !explicit_child(current) && !text(current, "parent_session_id").is_empty() {
        let parent_id = text(current, "parent_session_id");
        let parent = sessions
            .iter()
            .find(|s| text(s, "id") == parent_id)
            .ok_or("Hermes parent session is missing")?;
        if text(parent, "end_reason") != "compression" {
            break;
        }
        if !seen.insert(parent_id.to_string()) {
            return Err("Invalid Hermes compression lineage".to_string());
        }
        chain.push(parent);
        current = parent;
    }
    chain.reverse();
    Ok(chain)
}

fn identity(row: &Value) -> String {
    let calls = parsed(row, "tool_calls");
    if text(row, "role") == "assistant" {
        if let Some(calls) = calls.as_array().filter(|calls| !calls.is_empty()) {
            let ids: Vec<String> = calls.iter().map(call_id).collect();
            if ids.iter().all(|id| !id.is_empty()) {
                return json!([
                    "assistant",
                    null,
                    row["timestamp"],
                    row["tool_call_id"],
                    ids,
                    row["tool_name"]
                ])
                .to_string();
            }
        }
    }
    json!([
        row["role"],
        row["content"],
        row["timestamp"],
        row["tool_call_id"],
        row["tool_calls"],
        row["tool_name"]
    ])
    .to_string()
}

fn call_id(call: &Value) -> String {
    ["call_id", "id"]
        .iter()
        .find_map(|key| {
            let id = text(call, key).trim();
            if id.is_empty() {
                None
            } else {
                Some(id.split('|').next().unwrap_or(id).trim().to_string())
            }
        })
        .unwrap_or_default()
}

fn display_rows(
    conn: &Connection,
    sessions: &[Value],
    id: &str,
) -> Result<Vec<(Value, Value)>, String> {
    let chain = lineage(sessions, id)?;
    if chain.len() > 1 {
        let mut merged = Vec::new();
        for session in chain {
            for row in rows(conn, "messages", MESSAGE_COLUMNS, Some(text(session, "id")))? {
                if visible(&row) {
                    merged.push((session.clone(), row));
                }
            }
        }
        // Hermes orders a multi-segment display by global row id; segment-local display orders cannot be concatenated.
        merged.sort_by_key(|(_, row)| row["id"].as_i64().unwrap_or_default());
        let mut result: Vec<(Value, Value)> = Vec::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        for (session, row) in merged {
            let key = identity(&row);
            if let Some(&pos) = positions.get(&key) {
                if rank(&row) > rank(&result[pos].1) {
                    result[pos] = (session, row);
                }
            } else {
                positions.insert(key, result.len());
                result.push((session, row));
            }
        }
        return Ok(result);
    }
    let mut selected: Vec<(Value, Value)> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for session in chain {
        let all = rows(conn, "messages", MESSAGE_COLUMNS, Some(text(session, "id")))?;
        let visible: Vec<Value> = all.into_iter().filter(visible).collect();
        let indexed = visible
            .iter()
            .all(|r| r["display_order"].as_i64().is_some_and(|n| n > 0));
        let mut groups: BTreeMap<i64, Value> = BTreeMap::new();
        let mut legacy: Vec<Value> = Vec::new();
        let mut legacy_positions = HashMap::new();
        for row in visible {
            if indexed {
                let key = row["display_order"].as_i64().unwrap_or_default();
                if groups.get(&key).map_or(true, |old| rank(&row) > rank(old)) {
                    groups.insert(key, row);
                }
            } else {
                let key = identity(&row);
                if let Some(&pos) = legacy_positions.get(&key) {
                    if rank(&row) > rank(&legacy[pos]) {
                        legacy[pos] = row;
                    }
                } else {
                    legacy_positions.insert(key, legacy.len());
                    legacy.push(row);
                }
            }
        }
        let ordered = if indexed {
            groups.into_values().collect()
        } else {
            legacy
        };
        for row in ordered {
            let key = identity(&row);
            if let Some(&pos) = seen.get(&key) {
                if rank(&row) > rank(&selected[pos].1) {
                    selected[pos] = (session.clone(), row);
                }
            } else {
                seen.insert(key, selected.len());
                selected.push((session.clone(), row));
            }
        }
    }
    Ok(selected)
}
fn visible(row: &Value) -> bool {
    (row.get("active").is_none() || flag(row, "active") || flag(row, "compacted"))
        && !flag(&parsed(row, "display_metadata"), "model_only")
}
fn rank(row: &Value) -> (bool, i64) {
    (
        row.get("active").is_none() || flag(row, "active"),
        row["id"].as_i64().unwrap_or_default(),
    )
}

fn content(row: &Value) -> Result<Value, String> {
    let value = &row["content"];
    if let Some(encoded) = value.as_str().and_then(|s| s.strip_prefix("\0json:")) {
        return serde_json::from_str(encoded)
            .map_err(|e| format!("Invalid Hermes multimodal content: {e}"));
    }
    Ok(value.clone())
}
fn blocks(value: Value) -> Vec<Value> {
    match value {
        Value::Null => Vec::new(),
        Value::String(s) if s.is_empty() => Vec::new(),
        Value::String(s) => vec![json!({"type":"text","text":s})],
        Value::Array(parts) => parts
            .into_iter()
            .map(|p| match text(&p, "type") {
                "image_url" => {
                    let url = text(&p["image_url"], "url");
                    if let Some((mime, data)) = url.strip_prefix("data:").and_then(|s| s.split_once(";base64,")) {
                        json!({"type":"image","source":{"type":"base64","media_type":mime,"data":data}})
                    } else {
                        json!({"type":"image","source":{"type":"url","url":url}})
                    }
                }
                "input_text" | "output_text" => json!({"type":"text","text":p["text"]}),
                _ => p,
            })
            .collect(),
        other => vec![other],
    }
}

fn tool_variants(value: &str) -> HashSet<String> {
    let mut variants = HashSet::new();
    let value = value.trim();
    if !value.is_empty() {
        variants.insert(value.to_string());
        variants.extend(
            value
                .split('|')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from),
        );
    }
    variants
}

fn pair_tools(rows: &mut [(Value, Value)]) -> Result<(), String> {
    let mut aliases: HashMap<String, HashSet<String>> = HashMap::new();
    for (session, row) in rows {
        if text(row, "role") == "user" {
            aliases.clear();
        }
        if text(row, "role") == "assistant" && !text(row, "tool_calls").is_empty() {
            let mut calls: Value = serde_json::from_str(text(row, "tool_calls"))
                .map_err(|e| format!("Invalid Hermes tool calls: {e}"))?;
            let calls = calls
                .as_array_mut()
                .ok_or("Hermes tool calls are not an array")?;
            let mut named = HashSet::new();
            for call in calls.iter() {
                for key in ["call_id", "id", "response_item_id"] {
                    named.extend(tool_variants(text(call, key)));
                }
            }
            for name in named {
                aliases.remove(&name);
            }
            let uids = parsed(row, "tool_call_uids");
            let mut occurrences: HashMap<String, usize> = HashMap::new();
            for (i, call) in calls.iter_mut().enumerate() {
                let native_id = call_id(call);
                if native_id.is_empty() {
                    return Err("Hermes tool call has no id".to_string());
                }
                let nth = occurrences.entry(native_id.clone()).or_default();
                let uid = if uids[&native_id].is_array() {
                    uids[&native_id][*nth].as_str()
                } else {
                    uids[&native_id].as_str()
                };
                *nth += 1;
                let normalized = uid
                    .filter(|s| !s.is_empty())
                    .map(|uid| format!("hermes-tool:{uid}"))
                    .unwrap_or_else(|| {
                        format!("hermes-tool:{}:{}:{i}", text(session, "id"), row["id"])
                    });
                for key in ["call_id", "id", "response_item_id"] {
                    for variant in tool_variants(text(call, key)) {
                        aliases
                            .entry(variant)
                            .or_default()
                            .insert(normalized.clone());
                    }
                }
                call["native_id"] = json!(native_id);
                call["id"] = json!(normalized);
                call["call_id"] = call["id"].clone();
            }
            row["tool_calls"] = json!(serde_json::to_string(calls).map_err(|e| e.to_string())?);
        } else if text(row, "role") == "tool" {
            let native_id =
                optional(row, "tool_call_id").ok_or("Hermes tool result has no call id")?;
            let resolved = if let Some(uid) = optional(row, "tool_call_uid") {
                Some(format!("hermes-tool:{uid}"))
            } else {
                let matched: HashSet<String> = tool_variants(&native_id)
                    .iter()
                    .filter_map(|v| aliases.get(v))
                    .flat_map(|ids| ids.iter().cloned())
                    .collect();
                if matched.len() > 1 {
                    return Err("Ambiguous Hermes tool-result identity".to_string());
                }
                matched.into_iter().next()
            };
            if let Some(resolved) = resolved {
                row["native_tool_call_id"] = json!(native_id);
                row["tool_call_id"] = json!(resolved);
            }
        }
    }
    Ok(())
}
fn normalize(session: &Value, row: &Value, id: &str) -> Result<ClaudeMessage, String> {
    let role = text(row, "role");
    let config = parsed(session, "model_config");
    let mut value = content(row)?;
    if value.is_array() {
        value = Value::Array(blocks(value));
    }
    let message_type;
    if role == "tool" {
        let call_id = optional(row, "tool_call_id").ok_or("Hermes tool result has no call id")?;
        value = json!([{"type":"tool_result","tool_use_id":call_id,"content":value,"native_id":optional(row,"native_tool_call_id")}]);
        message_type = "user";
    } else if role == "assistant" {
        let mut parts = Vec::new();
        if let Some(reasoning) =
            optional(row, "reasoning_content").or_else(|| optional(row, "reasoning"))
        {
            parts.push(json!({"type":"thinking","thinking":reasoning}));
        } else {
            if let Some(details) = parsed(row, "reasoning_details").as_array() {
                for detail in details {
                    match text(detail, "type") {
                        "redacted_thinking" | "reasoning.encrypted" => {
                            parts.push(json!({"type":"redacted_thinking","data":""}));
                        }
                        "thinking" | "reasoning.text" | "reasoning.summary" => {
                            if let Some(s) = ["summary", "thinking", "content", "text"]
                                .iter()
                                .find_map(|key| detail[*key].as_str().filter(|s| !s.is_empty()))
                            {
                                parts.push(json!({"type":"thinking","thinking":s}));
                            }
                        }
                        _ => {}
                    }
                }
            }
            let items = parsed(row, "codex_reasoning_items");
            if let Some(items) = items.as_array().filter(|_| parts.is_empty()) {
                for item in items {
                    if let Some(summary) = item["summary"].as_array() {
                        for part in summary {
                            if let Some(s) = part["text"].as_str().filter(|s| !s.is_empty()) {
                                parts.push(json!({"type":"thinking","thinking":s}));
                            }
                        }
                    }
                }
                if parts.is_empty() && items.iter().any(|i| i["encrypted_content"].is_string()) {
                    parts.push(json!({"type":"redacted_thinking","data":""}));
                }
            }
        }
        parts.extend(blocks(value));
        if !text(row, "tool_calls").is_empty() {
            let calls: Value = serde_json::from_str(text(row, "tool_calls"))
                .map_err(|e| format!("Invalid Hermes tool calls: {e}"))?;
            for call in calls
                .as_array()
                .ok_or("Hermes tool calls are not an array")?
            {
                let call_id = call_id(call);
                if call_id.is_empty() {
                    return Err("Hermes tool call has no id".to_string());
                }
                let name = call["function"]["name"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("Hermes tool call has no name")?;
                let input = match &call["function"]["arguments"] {
                    Value::String(s) => {
                        serde_json::from_str(s).unwrap_or_else(|_| json!({"raw_arguments":s}))
                    }
                    Value::Object(_) => call["function"]["arguments"].clone(),
                    Value::Null => json!({}),
                    _ => return Err("Unsupported Hermes tool arguments".to_string()),
                };
                parts.push(json!({"type":"tool_use","id":call_id,"name":name,"input":input,"native_id":call["native_id"]}));
            }
        }
        // Scalar authored replies retain their exact native text; structured replies carry tools/reasoning.
        value = if parts.len() == 1 && text(&parts[0], "type") == "text" {
            parts[0]["text"].clone()
        } else {
            Value::Array(parts)
        };
        message_type = "assistant";
    } else if role == "user" || role == "system" {
        message_type = role;
    } else {
        return Err(format!("Unsupported Hermes message role: {role}"));
    }
    let model = if role == "assistant" {
        optional(session, "model")
    } else {
        None
    };
    let mut message = build_provider_message(
        PROVIDER,
        format!("hermes:{}:{}", text(session, "id"), row["id"]),
        id,
        timestamp(&row["timestamp"]),
        message_type,
        Some(if role == "tool" { "user" } else { role }),
        Some(value),
        model.clone(),
    );
    if role == "assistant" {
        message.inference = Some(InferenceMetadata {
            model,
            model_provider: optional(&config, "provider"),
            reasoning_effort: optional(&config["reasoning_config"], "effort"),
            service_tier: optional(&config, "service_tier"),
            stop_reason: optional(row, "finish_reason"),
            ..Default::default()
        });
    }
    message.stop_reason = optional(row, "finish_reason");
    if flag(row, "_compressed_summary") {
        message.subtype = Some("compact_summary".to_string());
    } else if role == "user" && text(row, "display_kind") == "steer" {
        message.subtype = Some("steer".to_string());
    } else if role == "user" && !text(row, "display_kind").is_empty() {
        message.message_type = "system".to_string();
        message.subtype = optional(row, "display_kind");
    }
    Ok(message)
}

fn messages_in_conn(
    conn: &Connection,
    sessions: &[Value],
    id: &str,
) -> Result<Vec<ClaudeMessage>, String> {
    let mut messages = Vec::new();
    let mut parent = None;
    let mut rows = display_rows(conn, sessions, id)?;
    pair_tools(&mut rows)?;
    for (session, row) in rows {
        let mut message = normalize(&session, &row, id)?;
        message.parent_uuid = parent;
        parent = Some(message.uuid.clone());
        messages.push(message);
    }
    Ok(messages)
}
fn read_messages(path: &Path, id: &str) -> Result<Vec<ClaudeMessage>, String> {
    let conn = open_db(path)?;
    let sessions = rows(&conn, "sessions", SESSION_COLUMNS, None)?;
    messages_in_conn(&conn, &sessions, id)
}
pub fn load_messages(value: &str) -> Result<Vec<ClaudeMessage>, String> {
    let (path, id) = parse_selector(value, "session")?;
    read_messages(&path, &id)
}

fn sessions_in_conn(
    conn: &Connection,
    path: &Path,
    cwd: Option<&str>,
    exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let native = rows(conn, "sessions", SESSION_COLUMNS, None)?;
    let mut sessions = Vec::new();
    for session in &native {
        if cwd.is_some_and(|cwd| cwd != text(session, "cwd"))
            || (exclude_sidechain && text(session, "source") == "tool")
        {
            continue;
        }
        let id = text(session, "id");
        let messages = messages_in_conn(conn, &native, id)?;
        if messages.is_empty() {
            continue;
        }
        let path_selector = selector(path, "session", id);
        let last = messages
            .last()
            .map(|m| m.timestamp.clone())
            .unwrap_or_default();
        let source = optional(session, "created_source").or_else(|| optional(session, "source"));
        sessions.push(ClaudeSession {
            session_id: path_selector.clone(),
            actual_session_id: id.to_string(),
            file_path: path_selector,
            project_name: project_name(text(session, "cwd")),
            message_count: messages.len(),
            first_message_time: messages[0].timestamp.clone(),
            last_message_time: last.clone(),
            last_modified: if session["last_activity_at"].is_number() {
                timestamp(&session["last_activity_at"])
            } else {
                last
            },
            has_tool_use: messages.iter().any(|m| {
                m.content
                    .as_ref()
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.iter().any(|b| text(b, "type") == "tool_use"))
            }),
            has_errors: false,
            summary: optional(session, "title"),
            title_history: Vec::new(),
            is_renamed: text(session, "title_source") == "user",
            provider: Some(PROVIDER.to_string()),
            storage_type: Some("sqlite".to_string()),
            entrypoint: source.map(|s| format!("hermes-{s}")),
            forked_from_id: if explicit_child(session) && text(session, "source") != "tool" {
                optional(session, "parent_session_id")
            } else {
                None
            },
            subagent_provenance: None,
        });
    }
    sessions.sort_by(|a, b| {
        b.last_modified
            .cmp(&a.last_modified)
            .then(a.session_id.cmp(&b.session_id))
    });
    Ok(sessions)
}
fn project_name(cwd: &str) -> String {
    if cwd.is_empty() {
        "Hermes".to_string()
    } else {
        Path::new(cwd)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(cwd)
            .to_string()
    }
}
pub fn load_sessions(value: &str, exclude_sidechain: bool) -> Result<Vec<ClaudeSession>, String> {
    let (path, cwd) = parse_selector(value, "project")?;
    sessions_in_conn(&open_db(&path)?, &path, Some(&cwd), exclude_sidechain)
}
fn scan_store(path: &Path) -> Result<Vec<ClaudeProject>, String> {
    let conn = open_db(path)?;
    let native = rows(&conn, "sessions", SESSION_COLUMNS, None)?;
    let sessions = sessions_in_conn(&conn, path, None, false)?;
    let mut grouped: BTreeMap<String, Vec<&ClaudeSession>> = BTreeMap::new();
    for session in &sessions {
        let cwd = native
            .iter()
            .find(|s| text(s, "id") == session.actual_session_id)
            .map(|s| text(s, "cwd"))
            .unwrap_or_default();
        grouped.entry(cwd.to_string()).or_default().push(session);
    }
    Ok(grouped
        .into_iter()
        .map(|(cwd, sessions)| {
            let project_path = selector(path, "project", &cwd);
            ClaudeProject {
                name: project_name(&cwd),
                path: project_path.clone(),
                actual_path: if cwd.is_empty() { project_path } else { cwd },
                session_count: sessions.len(),
                message_count: sessions.iter().map(|s| s.message_count).sum(),
                last_modified: sessions
                    .iter()
                    .map(|s| s.last_modified.as_str())
                    .max()
                    .unwrap_or_default()
                    .to_string(),
                git_info: None,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("sqlite".to_string()),
                custom_directory_label: None,
            }
        })
        .collect())
}
pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let mut projects = Vec::new();
    for path in homes() {
        projects.extend(scan_store(&path)?);
    }
    Ok(projects)
}
pub(crate) fn lifecycle(value: &str) -> Result<(bool, bool, bool), String> {
    let (path, id) = parse_selector(value, "session")?;
    let conn = open_db(&path)?;
    let sessions = rows(&conn, "sessions", SESSION_COLUMNS, None)?;
    let row = sessions
        .iter()
        .find(|r| text(r, "id") == id)
        .ok_or("Hermes session not found")?;
    Ok((
        flag(row, "hidden"),
        flag(row, "archived") || flag(row, "auto_archived"),
        flag(row, "pinned"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn fixture() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let db = Connection::open(dir.path().join("state.db")).unwrap();
        db.execute_batch("PRAGMA journal_mode=WAL;
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, model_config TEXT, started_at REAL, ended_at REAL, end_reason TEXT, parent_session_id TEXT, cwd TEXT, title TEXT, title_source TEXT, last_activity_at REAL);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_calls TEXT, tool_call_id TEXT, tool_name TEXT, timestamp REAL, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0, display_order INTEGER, display_metadata TEXT, reasoning TEXT, reasoning_content TEXT, finish_reason TEXT);
            INSERT INTO sessions VALUES ('s', 'desktop', 'gpt-test', '{\"provider\":\"openai-codex\"}', 100, NULL, NULL, NULL, NULL, 'Native title', 'llm', 102);
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('s', 'user', 'Hello', 100, 1);
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('s', 'assistant', 'Response', 102, 2);").unwrap();
        (dir, db)
    }

    #[test]
    fn backup_inventory_covers_default_and_profiles_without_opening_stores() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("state.db"), "not opened by discovery").unwrap();
        let profile = dir.path().join("profiles/work space");
        std::fs::create_dir_all(&profile).unwrap();
        std::fs::write(profile.join("state.db"), "not opened by discovery").unwrap();
        let stores = backup_sources_in_home(dir.path()).unwrap();
        assert_eq!(stores.len(), 2);
        assert_eq!(stores[0].id, "default");
        assert_eq!(stores[0].relative_path, "default/state.db");
        assert_eq!(
            stores[1].id,
            format!("profile-{:x}", Sha256::digest(b"work space"))
        );
        assert_eq!(stores[1].relative_path, "profiles/work%20space/state.db");
        assert_eq!(
            stores[1].root,
            profile.canonicalize().unwrap().to_string_lossy()
        );
        assert!(backup_sources_in_home(&dir.path().join("missing"))
            .unwrap()
            .is_empty());
        std::fs::remove_file(profile.join("state.db")).unwrap();
        std::fs::create_dir(profile.join("state.db")).unwrap();
        assert!(backup_sources_in_home(dir.path()).is_err());
    }

    #[test]
    fn offline_backup_reads_standalone_store_and_native_flags_without_sidecars() {
        let (dir, db) = fixture();
        db.execute_batch("ALTER TABLE sessions ADD hidden INTEGER; ALTER TABLE sessions ADD archived INTEGER; ALTER TABLE sessions ADD pinned INTEGER;
            UPDATE sessions SET hidden=1, archived=1, pinned=1;
            PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;").unwrap();
        drop(db);
        let path = dir.path().join("state.db");
        let before = std::fs::read(&path).unwrap();
        let listed = load_offline_sessions(&path).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].session.actual_session_id, "s");
        assert!(listed[0].project_path.starts_with("hermes://"));
        assert!(listed[0].hidden && listed[0].archived && listed[0].pinned);
        let messages = load_offline_messages(&path, "s").unwrap();
        assert_eq!(messages[0].content, Some(json!("Hello")));
        assert_eq!(
            messages[1]
                .inference
                .as_ref()
                .unwrap()
                .model_provider
                .as_deref(),
            Some("openai-codex")
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(!dir.path().join("state.db-shm").exists());
        assert!(load_offline_messages(&path, "").is_err());
        std::fs::write(dir.path().join("state.db-wal"), []).unwrap();
        assert!(load_offline_sessions(&path).is_err());
    }

    #[test]
    fn offline_backup_refuses_corrupt_or_unsupported_databases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.db");
        std::fs::write(&path, "not a SQLite database").unwrap();
        assert!(load_offline_sessions(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        let db = Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE sessions(id TEXT, started_at REAL); CREATE TABLE messages(id INTEGER, role TEXT, content TEXT, timestamp REAL);").unwrap();
        drop(db);
        assert!(load_offline_sessions(&path)
            .err()
            .unwrap()
            .contains("Unsupported Hermes messages schema"));
    }

    #[test]
    fn backup_inventory_rejects_symlink_or_reparse_paths() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("state.db"), "opaque bytes").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, home.join("profiles")).unwrap();
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(&real, home.join("profiles")).is_err() {
                return;
            }
        }
        assert!(backup_sources_in_home(&home)
            .unwrap_err()
            .contains("symbolic link or reparse point"));
    }

    #[test]
    fn reads_live_wal_without_writing_and_keeps_hermes_ownership() {
        let (dir, db) = fixture();
        let before = std::fs::read(dir.path().join("state.db")).unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "s").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, Some(serde_json::json!("Hello")));
        assert_eq!(messages[1].provider.as_deref(), Some("hermes"));
        assert_eq!(messages[1].model.as_deref(), Some("gpt-test"));
        assert_eq!(std::fs::read(dir.path().join("state.db")).unwrap(), before);
        assert!(db.is_autocommit());
    }

    #[test]
    fn display_history_keeps_compaction_but_excludes_rewinds_and_model_only_rows() {
        let (dir, db) = fixture();
        db.execute_batch("UPDATE messages SET active = 0, compacted = 1 WHERE id = 1;
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('s', 'user', 'Hello', 100, 1);
            INSERT INTO messages (session_id, role, content, timestamp, display_order, active) VALUES ('s', 'user', 'Rewound', 103, 3, 0);
            INSERT INTO messages (session_id, role, content, timestamp, display_order, display_metadata) VALUES ('s', 'user', 'Model only', 104, 4, '{\"model_only\":true}');").unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "s").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, Some(serde_json::json!("Hello")));
    }

    #[test]
    fn tools_keep_native_pairing_and_malformed_arguments_verbatim() {
        let (dir, db) = fixture();
        db.execute("UPDATE messages SET content = '', reasoning = 'Plan', tool_calls = ?1 WHERE id = 2",
            [r#"[{"id":"call-1","type":"function","function":{"name":"terminal","arguments":"{bad"}}]"#]).unwrap();
        db.execute_batch("INSERT INTO messages (session_id, role, content, timestamp, display_order, tool_call_id, tool_name) VALUES ('s', 'tool', 'Output', 103, 3, 'call-1', 'terminal');").unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "s").unwrap();
        let parts = messages[1].content.as_ref().unwrap();
        assert_eq!(parts[0]["thinking"], "Plan");
        assert_eq!(parts[1]["input"]["raw_arguments"], "{bad");
        assert_eq!(
            parts[1]["id"],
            messages[2].content.as_ref().unwrap()[0]["tool_use_id"]
        );
        assert_eq!(
            messages[2].parent_uuid.as_deref(),
            Some(messages[1].uuid.as_str())
        );
        assert_eq!(
            messages[1]
                .inference
                .as_ref()
                .unwrap()
                .model_provider
                .as_deref(),
            Some("openai-codex")
        );
        db.execute("UPDATE messages SET tool_calls = '{bad' WHERE id = 2", [])
            .unwrap();
        assert!(read_messages(&dir.path().join("state.db"), "s").is_err());
    }

    #[test]
    fn synthetic_projects_do_not_invent_cwd_and_selectors_round_trip() {
        let (dir, _db) = fixture();
        let path = dir.path().join("state.db");
        let projects = scan_store(&path).unwrap();
        assert_eq!(projects.len(), 1);
        assert!(projects[0].actual_path.starts_with(SCHEME));
        let sessions = load_sessions(&projects[0].path, false).unwrap();
        assert_eq!(sessions[0].summary.as_deref(), Some("Native title"));
        assert!(!sessions[0].is_renamed);
        assert_eq!(sessions[0].entrypoint.as_deref(), Some("hermes-desktop"));
        assert_eq!(load_messages(&sessions[0].file_path).unwrap().len(), 2);
        let special = "id?#% /Unicode-é";
        assert_eq!(
            parse_selector(&selector(&path, "session", special), "session").unwrap(),
            (path, special.to_string())
        );
        assert!(parse_selector("hermes://relative?session=s", "session").is_err());
        assert!(load_messages(&projects[0].path).is_err());
    }

    #[test]
    fn only_verified_compression_ancestors_are_loaded() {
        let (dir, db) = fixture();
        db.execute_batch("UPDATE sessions SET end_reason = 'compression' WHERE id = 's';
            INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES ('tip', 'cli', 105, 's');
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('tip', 'user', 'Continuation', 105, 1);").unwrap();
        let path = dir.path().join("state.db");
        assert_eq!(read_messages(&path, "tip").unwrap().len(), 3);
        db.execute(
            "UPDATE sessions SET model_config = ?1 WHERE id = 'tip'",
            [r#"{"_branched_from":"s"}"#],
        )
        .unwrap();
        assert_eq!(read_messages(&path, "tip").unwrap().len(), 1);
        db.execute(
            "UPDATE sessions SET model_config = ?1 WHERE id = 'tip'",
            [r#"{"_branched_from":"different-parent"}"#],
        )
        .unwrap();
        assert_eq!(read_messages(&path, "tip").unwrap().len(), 3);
        db.execute(
            "UPDATE sessions SET parent_session_id = 'missing' WHERE id = 'tip'",
            [],
        )
        .unwrap();
        assert!(read_messages(&path, "tip").is_err());
    }

    #[test]
    fn legacy_display_rows_dedupe_copies_without_timestamp_sorting() {
        let (dir, db) = fixture();
        db.execute_batch("UPDATE messages SET display_order = NULL;
            UPDATE messages SET active = 0, compacted = 1 WHERE id = 1;
            INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s', 'user', 'Hello', 100);
            INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s', 'user', 'Clock regressed', 99);").unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "s").unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].content, Some(json!("Clock regressed")));
    }

    #[test]
    fn sentinel_content_decodes_but_authored_json_text_is_not_reinterpreted() {
        let (dir, db) = fixture();
        db.execute(
            "UPDATE messages SET content = ?1 WHERE id = 1",
            ["\0json:[{\"type\":\"text\",\"text\":\"Multimodal\"}]"],
        )
        .unwrap();
        let path = dir.path().join("state.db");
        assert_eq!(
            read_messages(&path, "s").unwrap()[0]
                .content
                .as_ref()
                .unwrap()[0]["text"],
            "Multimodal"
        );
        db.execute(
            "UPDATE messages SET content = ?1 WHERE id = 1",
            ["[\"authored JSON\"]"],
        )
        .unwrap();
        assert_eq!(
            read_messages(&path, "s").unwrap()[0].content,
            Some(json!("[\"authored JSON\"]"))
        );
        db.execute(
            "UPDATE messages SET content = ?1 WHERE id = 1",
            ["\0json:{bad"],
        )
        .unwrap();
        assert!(read_messages(&path, "s").is_err());
    }

    #[test]
    fn discovers_profiles_and_preserves_native_lifecycle_flags() {
        let (dir, db) = fixture();
        let profile = dir.path().join("profiles/work");
        std::fs::create_dir_all(&profile).unwrap();
        std::fs::write(profile.join("state.db"), []).unwrap();
        assert_eq!(stores_in_home(dir.path()).len(), 2);
        db.execute_batch(
            "ALTER TABLE sessions ADD COLUMN hidden INTEGER;
            ALTER TABLE sessions ADD COLUMN archived INTEGER;
            ALTER TABLE sessions ADD COLUMN auto_archived INTEGER;
            ALTER TABLE sessions ADD COLUMN pinned INTEGER;
            UPDATE sessions SET hidden = 1, auto_archived = 1, pinned = 1;",
        )
        .unwrap();
        assert_eq!(
            lifecycle(&selector(&dir.path().join("state.db"), "session", "s")).unwrap(),
            (true, true, true)
        );
    }

    #[test]
    fn compression_segments_keep_global_insertion_order_when_rows_interleave() {
        let (dir, db) = fixture();
        db.execute_batch("UPDATE sessions SET end_reason = 'compression' WHERE id = 's';
            INSERT INTO sessions (id, source, started_at, parent_session_id) VALUES ('tip', 'cli', 103, 's');
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('tip', 'user', 'Child first', 103, 1);
            INSERT INTO messages (session_id, role, content, timestamp, display_order) VALUES ('s', 'assistant', 'Parent late', 104, 3);").unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "tip").unwrap();
        assert_eq!(messages[2].content, Some(json!("Child first")));
        assert_eq!(messages[3].content, Some(json!("Parent late")));
    }

    #[test]
    fn long_valid_compression_lineages_remain_discoverable() {
        let (dir, db) = fixture();
        db.execute_batch("UPDATE sessions SET end_reason = 'compression' WHERE id = 's';
            WITH RECURSIVE seq(n) AS (VALUES(1) UNION ALL SELECT n + 1 FROM seq WHERE n < 180)
            INSERT INTO sessions (id, source, started_at, parent_session_id, end_reason)
            SELECT 'c' || n, 'cli', 100 + n, CASE WHEN n = 1 THEN 's' ELSE 'c' || (n - 1) END, 'compression' FROM seq;
            INSERT INTO messages (session_id, role, content, timestamp) VALUES ('c180', 'user', 'Long continuation', 280);").unwrap();
        let path = dir.path().join("state.db");
        assert_eq!(read_messages(&path, "c180").unwrap().len(), 3);
        assert_eq!(scan_store(&path).unwrap()[0].session_count, 181);
        db.execute(
            "UPDATE sessions SET parent_session_id = 'c180' WHERE id = 's'",
            [],
        )
        .unwrap();
        assert!(read_messages(&path, "c180").is_err());
    }

    #[test]
    fn native_tool_aliases_and_occurrence_uids_pair_repeated_wire_ids() {
        let (dir, db) = fixture();
        db.execute_batch("ALTER TABLE messages ADD COLUMN tool_call_uid TEXT;
            ALTER TABLE messages ADD COLUMN tool_call_uids TEXT;
            UPDATE messages SET content = '', tool_call_uids = '{\"call\":\"uid-one\"}' WHERE id = 2;").unwrap();
        let call = r#"[{"id":"call|fc-one","call_id":"call","response_item_id":"fc-one","function":{"name":"terminal","arguments":"{}"}}]"#;
        db.execute("UPDATE messages SET tool_calls = ?1 WHERE id = 2", [call])
            .unwrap();
        db.execute_batch("INSERT INTO messages (session_id, role, content, timestamp, display_order, tool_call_id, tool_call_uid) VALUES ('s', 'tool', 'First', 103, 3, 'fc-one', 'uid-one');").unwrap();
        db.execute("INSERT INTO messages (session_id, role, content, timestamp, display_order, tool_calls, tool_call_uids) VALUES ('s', 'assistant', '', 104, 4, ?1, '{\"call\":\"uid-two\"}')", [call]).unwrap();
        db.execute_batch("INSERT INTO messages (session_id, role, content, timestamp, display_order, tool_call_id, tool_call_uid) VALUES ('s', 'tool', 'Second', 105, 5, 'call|fc-one', 'uid-two');").unwrap();
        let path = dir.path().join("state.db");
        let messages = read_messages(&path, "s").unwrap();
        assert_eq!(
            messages[1].content.as_ref().unwrap()[0]["id"],
            "hermes-tool:uid-one"
        );
        assert_eq!(
            messages[2].content.as_ref().unwrap()[0]["tool_use_id"],
            "hermes-tool:uid-one"
        );
        assert_eq!(
            messages[3].content.as_ref().unwrap()[0]["id"],
            "hermes-tool:uid-two"
        );
        assert_eq!(
            messages[4].content.as_ref().unwrap()[0]["tool_use_id"],
            "hermes-tool:uid-two"
        );
        db.execute_batch("UPDATE messages SET tool_call_uid = NULL, tool_call_uids = NULL;")
            .unwrap();
        let messages = read_messages(&path, "s").unwrap();
        assert_eq!(
            messages[1].content.as_ref().unwrap()[0]["id"],
            messages[2].content.as_ref().unwrap()[0]["tool_use_id"]
        );
        assert_eq!(
            messages[3].content.as_ref().unwrap()[0]["id"],
            messages[4].content.as_ref().unwrap()[0]["tool_use_id"]
        );
        assert_ne!(
            messages[1].content.as_ref().unwrap()[0]["id"],
            messages[3].content.as_ref().unwrap()[0]["id"]
        );
    }

    #[test]
    fn structured_reasoning_and_images_survive_without_exporting_ciphertext() {
        let (dir, db) = fixture();
        db.execute_batch("ALTER TABLE messages ADD COLUMN reasoning_details TEXT;
            UPDATE messages SET reasoning_details = '[{\"type\":\"reasoning.text\",\"text\":\"Plan\"},{\"type\":\"reasoning.encrypted\",\"data\":\"opaque-test-ciphertext\"}]' WHERE id = 2;").unwrap();
        db.execute("UPDATE messages SET content = ?1 WHERE id = 1", ["\0json:[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,aGVsbG8=\"}}]"]).unwrap();
        let messages = read_messages(&dir.path().join("state.db"), "s").unwrap();
        assert_eq!(
            messages[0].content.as_ref().unwrap()[0]["source"]["type"],
            "base64"
        );
        assert_eq!(messages[1].content.as_ref().unwrap()[0]["thinking"], "Plan");
        assert_eq!(
            messages[1].content.as_ref().unwrap()[1]["type"],
            "redacted_thinking"
        );
        assert!(!serde_json::to_string(&messages)
            .unwrap()
            .contains("opaque-test-ciphertext"));
        assert!(open_db(&dir.path().join("missing.db")).is_err());
        assert!(!dir.path().join("missing.db").exists());
    }

    #[test]
    fn home_override_expands_native_environment_and_current_user_syntax() {
        let base = std::env::current_dir().unwrap();
        let lookup = |key: &str| {
            if key == "DATA" {
                Some(base.to_string_lossy().into_owned())
            } else {
                None
            }
        };
        assert_eq!(
            expand_home("${DATA}/hermes", lookup, Some(&base), &base),
            base.join("hermes")
        );
        assert_eq!(
            expand_home("~/profile", |_| None, Some(&base), &base),
            base.join("profile")
        );
        assert_eq!(
            expand_home("relative", |_| None, Some(&base), &base),
            base.join("relative")
        );
        if cfg!(windows) {
            assert_eq!(
                expand_home("%DATA%/hermes", lookup, Some(&base), &base),
                base.join("hermes")
            );
        }
    }
}
