//! Claude for PowerPoint provider.
//!
//! The Office add-in persists chat history in the WebView2 profile's Chromium
//! IndexedDB store (`https_pivot.claude.ai_0`). Values are V8 structured-clone
//! payloads. Chromium may Snappy-compress them and moves large values to the
//! adjacent `.indexeddb.blob` directory.

use crate::models::{ClaudeMessage, ClaudeProject, ClaudeSession};
use crate::providers::ProviderInfo;
use crate::utils::build_provider_message;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{SecondsFormat, TimeZone, Utc};
use rusty_leveldb::{LdbIterator, Options, DB};
use serde_json::{json, Map, Number, Value};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use walkdir::WalkDir;

const PROVIDER: &str = "powerpoint";
const SCHEME: &str = "powerpoint://";
const ENTRYPOINT: &str = "claude-powerpoint";
const STORE_DIR: &str = "https_pivot.claude.ai_0.indexeddb.leveldb";
const BLOB_DIR: &str = "https_pivot.claude.ai_0.indexeddb.blob";
const WRAPPER_MIME: &str = "application/vnd.blink-idb-value-wrapper";
const PATH_OVERRIDE: &str = "CLAUDE_POWERPOINT_INDEXEDDB_PATH";

#[derive(Clone, Debug)]
struct ChatSession {
    id: String,
    title: String,
    messages: Vec<ClaudeMessage>,
    message_count: usize,
    first: String,
    last: String,
    has_tool_use: bool,
    has_errors: bool,
}

#[derive(Clone, Debug)]
struct ExternalObject {
    blob_number: u64,
    mime_type: String,
    size: u64,
}

pub fn detect() -> Option<ProviderInfo> {
    let stores = find_stores();
    let first = stores.first()?;
    Some(ProviderInfo {
        id: PROVIDER.to_string(),
        display_name: "Claude for PowerPoint".to_string(),
        base_path: first.to_string_lossy().to_string(),
        is_available: true,
    })
}

pub fn get_base_path() -> Option<String> {
    find_stores()
        .first()
        .map(|path| path.to_string_lossy().to_string())
}

pub fn scan_projects() -> Result<Vec<ClaudeProject>, String> {
    let mut projects = Vec::new();
    for path in find_stores() {
        let sessions = match read_store(&path) {
            Ok(sessions) => sessions,
            Err(error) => {
                log::warn!(
                    "PowerPoint IndexedDB scan failed for {}: {error}",
                    path.display()
                );
                continue;
            }
        };
        if sessions.is_empty() {
            continue;
        }
        let message_count = sessions.iter().map(|session| session.message_count).sum();
        let last_modified = sessions
            .iter()
            .map(|session| session.last.as_str())
            .max()
            .unwrap_or_default()
            .to_string();
        projects.push(ClaudeProject {
            name: "Claude for PowerPoint".to_string(),
            path: path.to_string_lossy().to_string(),
            // PowerPoint chats are presentation-scoped by an opaque file id;
            // the local store does not retain a filesystem presentation path.
            actual_path: "Claude for PowerPoint".to_string(),
            session_count: sessions.len(),
            message_count,
            last_modified,
            git_info: None,
            provider: Some(PROVIDER.to_string()),
            storage_type: Some("indexeddb".to_string()),
            custom_directory_label: None,
        });
    }
    Ok(projects)
}

pub fn load_sessions(
    project_path: &str,
    _exclude_sidechain: bool,
) -> Result<Vec<ClaudeSession>, String> {
    let path = Path::new(project_path);
    let mut sessions = read_store(path)?;
    sessions.sort_by(|a, b| b.last.cmp(&a.last));
    Ok(sessions
        .into_iter()
        .map(|session| {
            let file_path = session_path(path, &session.id);
            ClaudeSession {
                session_id: file_path.clone(),
                actual_session_id: session.id,
                file_path,
                project_name: "Claude for PowerPoint".to_string(),
                message_count: session.message_count,
                first_message_time: session.first,
                last_message_time: session.last.clone(),
                last_modified: session.last,
                has_tool_use: session.has_tool_use,
                has_errors: session.has_errors,
                summary: Some(session.title),
                is_renamed: false,
                provider: Some(PROVIDER.to_string()),
                storage_type: Some("indexeddb".to_string()),
                entrypoint: Some(ENTRYPOINT.to_string()),
            }
        })
        .collect())
}

pub fn load_messages(session_path: &str) -> Result<Vec<ClaudeMessage>, String> {
    let (path, id) = parse_session_path(session_path)?;
    read_store(&path)?
        .into_iter()
        .find(|session| session.id == id)
        .map(|session| session.messages)
        .ok_or_else(|| format!("PowerPoint session not found: {id}"))
}

fn find_stores() -> Vec<PathBuf> {
    if let Ok(value) = std::env::var(PATH_OVERRIDE) {
        let value = value.trim();
        if !value.is_empty() {
            let path = PathBuf::from(value);
            let store = if path.file_name().and_then(|name| name.to_str()) == Some(STORE_DIR) {
                path
            } else {
                path.join(STORE_DIR)
            };
            return store.is_dir().then_some(store).into_iter().collect();
        }
    }

    let Some(local) = dirs::data_local_dir() else {
        return Vec::new();
    };
    let office = local.join("Microsoft").join("Office");
    if !office.is_dir() {
        return Vec::new();
    }
    let mut stores = WalkDir::new(office)
        .max_depth(14)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_dir() && entry.file_name() == STORE_DIR)
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    stores.sort();
    stores.dedup();
    stores
}

/// Copy the live LevelDB and blob directory before opening it. Chromium keeps a
/// lock while PowerPoint is running; a snapshot avoids contending for that lock
/// and guarantees that this read-only provider never mutates Office's store.
fn snapshot_store(store: &Path) -> Result<(TempDir, PathBuf, PathBuf), String> {
    let temp = tempfile::Builder::new()
        .prefix("claude-powerpoint-indexeddb-")
        .tempdir()
        .map_err(|e| format!("Failed to create PowerPoint DB snapshot: {e}"))?;
    let leveldb = temp.path().join("leveldb");
    fs::create_dir(&leveldb).map_err(|e| e.to_string())?;
    for entry in
        fs::read_dir(store).map_err(|e| format!("Failed to read {}: {e}", store.display()))?
    {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_file() || entry.file_name() == "LOCK" {
            continue;
        }
        fs::copy(entry.path(), leveldb.join(entry.file_name()))
            .map_err(|e| format!("Failed to snapshot {}: {e}", entry.path().display()))?;
    }

    let blob_source = store.with_file_name(BLOB_DIR);
    let blob_dest = temp.path().join("blob");
    if blob_source.is_dir() {
        copy_tree(&blob_source, &blob_dest)?;
    }
    Ok((temp, leveldb, blob_dest))
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    for entry in WalkDir::new(source).follow_links(false) {
        let entry = entry.map_err(|e| e.to_string())?;
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|e| e.to_string())?;
        let target = destination.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target).map_err(|e| e.to_string())?;
        } else if entry.file_type().is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::copy(entry.path(), &target).map_err(|e| {
                format!(
                    "Failed to snapshot PowerPoint blob {}: {e}",
                    entry.path().display()
                )
            })?;
        }
    }
    Ok(())
}

fn read_store(store: &Path) -> Result<Vec<ChatSession>, String> {
    let (_temp, leveldb, blobs) = snapshot_store(store)?;
    let mut options = Options::default();
    options.create_if_missing = false;
    let mut db = DB::open(&leveldb, options)
        .map_err(|e| format!("Failed to open PowerPoint IndexedDB snapshot: {e}"))?;
    let mut iter = db
        .new_iter()
        .map_err(|e| format!("Failed to iterate PowerPoint IndexedDB: {e}"))?;
    iter.seek_to_first();
    let mut records = Vec::new();
    while let Some((key, value)) = iter.next() {
        records.push((key, value));
    }
    drop(iter);

    let mut sessions = Vec::new();
    let mut seen = HashSet::new();
    for (key, stored_value) in records {
        let Some(prefix) = KeyPrefix::decode(&key) else {
            continue;
        };
        if prefix.database_id == 0 || prefix.object_store_id == 0 || prefix.index_id != 1 {
            continue;
        }
        let wire = match resolve_value(&mut db, &blobs, &key, prefix, &stored_value) {
            Ok(wire) => wire,
            Err(_) => continue,
        };
        let value = match V8Reader::deserialize(&wire) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(session) = parse_chat_session(&value) else {
            continue;
        };
        if seen.insert(session.id.clone()) {
            sessions.push(session);
        }
    }
    Ok(sessions)
}

#[derive(Clone, Copy, Debug)]
struct KeyPrefix {
    database_id: u64,
    object_store_id: u64,
    index_id: u64,
    index_offset: usize,
    index_len: usize,
}

impl KeyPrefix {
    fn decode(key: &[u8]) -> Option<Self> {
        let header = *key.first()?;
        let database_len = ((header >> 5) & 0x07) as usize + 1;
        let object_store_len = ((header >> 2) & 0x07) as usize + 1;
        let index_len = (header & 0x03) as usize + 1;
        let object_store_offset = 1 + database_len;
        let index_offset = object_store_offset + object_store_len;
        if key.len() < index_offset + index_len {
            return None;
        }
        Some(Self {
            database_id: little_uint(&key[1..object_store_offset]),
            object_store_id: little_uint(&key[object_store_offset..index_offset]),
            index_id: little_uint(&key[index_offset..index_offset + index_len]),
            index_offset,
            index_len,
        })
    }

    fn external_object_key(self, object_key: &[u8]) -> Option<Vec<u8>> {
        let mut key = object_key.to_vec();
        let encoded = 3u64.to_le_bytes();
        if self.index_len > encoded.len() {
            return None;
        }
        key[self.index_offset..self.index_offset + self.index_len]
            .copy_from_slice(&encoded[..self.index_len]);
        Some(key)
    }
}

fn little_uint(bytes: &[u8]) -> u64 {
    bytes.iter().enumerate().fold(0, |value, (index, byte)| {
        value | ((*byte as u64) << (index * 8))
    })
}

fn resolve_value(
    db: &mut DB,
    blobs: &Path,
    key: &[u8],
    prefix: KeyPrefix,
    stored: &[u8],
) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(stored);
    let _record_version = cursor.varint()?;
    let wire = cursor.remaining();
    if wire.starts_with(&[0xff, 0x11, 0x02]) {
        return snap::raw::Decoder::new()
            .decompress_vec(&wire[3..])
            .map_err(|e| format!("Failed to decompress PowerPoint value: {e}"));
    }
    if !wire.starts_with(&[0xff, 0x11, 0x01]) {
        return Ok(wire.to_vec());
    }

    let mut wrapper = Cursor::new(&wire[3..]);
    let expected_size = wrapper.varint()?;
    let blob_index = wrapper.varint()? as usize;
    let external_key = prefix
        .external_object_key(key)
        .ok_or("Invalid PowerPoint external-object key")?;
    let encoded = db
        .get(&external_key)
        .ok_or("PowerPoint wrapper blob metadata is missing")?;
    let objects = decode_external_objects(&encoded)?;
    let object = objects.get(blob_index).ok_or_else(|| {
        format!(
            "PowerPoint wrapper blob index {blob_index} is missing ({} external objects)",
            objects.len()
        )
    })?;
    if object.mime_type != WRAPPER_MIME {
        return Err(format!(
            "PowerPoint external object {blob_index} has MIME type {:?}",
            object.mime_type
        ));
    }
    if object.size != expected_size {
        return Err("PowerPoint wrapper blob size does not match metadata".to_string());
    }
    let path = blob_path(blobs, prefix.database_id, object.blob_number)?;
    let bytes = fs::read(&path).map_err(|e| {
        format!(
            "Failed to read PowerPoint wrapper blob {}: {e}",
            path.display()
        )
    })?;
    if bytes.starts_with(&[0xff, 0x11, 0x02]) {
        snap::raw::Decoder::new()
            .decompress_vec(&bytes[3..])
            .map_err(|e| format!("Failed to decompress PowerPoint wrapper blob: {e}"))
    } else {
        Ok(bytes)
    }
}

fn decode_external_objects(encoded: &[u8]) -> Result<Vec<ExternalObject>, String> {
    let mut cursor = Cursor::new(encoded);
    let mut objects = Vec::new();
    while !cursor.remaining().is_empty() {
        let is_file = cursor.byte()? != 0;
        let blob_number = cursor.varint()?;
        let mime_type = cursor.utf16_string()?;
        let size = cursor.varint()?;
        if is_file {
            let _filename = cursor.utf16_string()?;
            let _last_modified = cursor.varint()?;
        }
        objects.push(ExternalObject {
            blob_number,
            mime_type,
            size,
        });
    }
    Ok(objects)
}

fn blob_path(root: &Path, database_id: u64, blob_number: u64) -> Result<PathBuf, String> {
    let direct = root
        .join(database_id.to_string())
        .join(format!("{:02x}", blob_number >> 8))
        .join(format!("{:x}", blob_number & 0xff));
    if direct.is_file() {
        return Ok(direct);
    }
    let filename = format!("{:x}", blob_number & 0xff);
    WalkDir::new(root.join(database_id.to_string()))
        .into_iter()
        .filter_map(Result::ok)
        .find(|entry| {
            entry.file_type().is_file() && entry.file_name().to_string_lossy() == filename
        })
        .map(|entry| entry.into_path())
        .ok_or_else(|| format!("PowerPoint wrapper blob {database_id}/{blob_number} is missing"))
}

fn session_path(store: &Path, id: &str) -> String {
    let encoded = URL_SAFE_NO_PAD.encode(store.to_string_lossy().as_bytes());
    format!("{SCHEME}{encoded}/{id}")
}

fn parse_session_path(value: &str) -> Result<(PathBuf, String), String> {
    let encoded = value
        .strip_prefix(SCHEME)
        .ok_or("Invalid PowerPoint session path")?;
    let (path, id) = encoded
        .rsplit_once('/')
        .ok_or("Invalid PowerPoint session path")?;
    let path = URL_SAFE_NO_PAD
        .decode(path)
        .map_err(|e| format!("Invalid PowerPoint session path: {e}"))?;
    let path = String::from_utf8(path).map_err(|e| format!("Invalid PowerPoint path: {e}"))?;
    Ok((PathBuf::from(path), id.to_string()))
}

fn parse_chat_session(root: &Value) -> Option<ChatSession> {
    let id = root.get("id")?.as_str()?.to_string();
    let raw_messages = find_messages(root)?;
    let message_count = raw_messages.len();
    let mut messages = Vec::new();
    let mut parent = None;
    let mut fallback_time = root
        .get("createdAt")
        .and_then(timestamp_value)
        .unwrap_or_default();
    let mut has_tool_use = false;
    let mut has_errors = false;

    for (index, raw) in raw_messages.iter().enumerate() {
        let Some(object) = raw.as_object() else {
            continue;
        };
        let Some(kind) = string_field(object, &["type", "role"]) else {
            continue;
        };
        let uuid = string_field(object, &["uuid", "id", "apiId"])
            .map(str::to_string)
            .unwrap_or_else(|| format!("{id}-{index}"));
        let timestamp = object
            .get("timestamp")
            .and_then(timestamp_value)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| fallback_time.clone());
        if !timestamp.is_empty() {
            fallback_time.clone_from(&timestamp);
        }
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        let content = object.get("content");
        if kind == "tool" {
            has_tool_use = true;
            has_errors |= tool_error(object);
            let use_uuid = format!("{uuid}-use");
            let mut use_message = build_provider_message(
                PROVIDER,
                use_uuid.clone(),
                &id,
                timestamp.clone(),
                "assistant",
                Some("assistant"),
                Some(Value::Array(vec![tool_use_block(object)])),
                model.clone(),
            );
            use_message.parent_uuid = parent.clone();
            parent = Some(use_uuid);
            messages.push(use_message);

            if object.get("result").is_some() || object.get("error").is_some() {
                let result_uuid = format!("{uuid}-result");
                let mut result_message = build_provider_message(
                    PROVIDER,
                    result_uuid.clone(),
                    &id,
                    timestamp,
                    "user",
                    Some("user"),
                    Some(Value::Array(vec![tool_result_block(object)])),
                    None,
                );
                result_message.parent_uuid = parent.clone();
                parent = Some(result_uuid);
                messages.push(result_message);
            }
            continue;
        }
        let (role, blocks) = match kind {
            "user" => ("user", text_blocks(content)),
            "assistant" | "text" => ("assistant", assistant_blocks(content)),
            "thinking" => ("assistant", thinking_blocks(content)),
            "tool_use" => {
                has_tool_use = true;
                ("assistant", vec![tool_use_block(object)])
            }
            "tool_result" => ("user", vec![tool_result_block(object)]),
            value if value.contains("error") => {
                has_errors = true;
                ("assistant", text_blocks(content))
            }
            _ => continue,
        };
        if blocks.is_empty() {
            continue;
        }
        let mut message = build_provider_message(
            PROVIDER,
            uuid.clone(),
            &id,
            timestamp,
            role,
            Some(role),
            Some(Value::Array(blocks)),
            model,
        );
        message.parent_uuid = parent.clone();
        parent = Some(uuid);
        messages.push(message);
    }
    if messages.is_empty() {
        return None;
    }
    let first = messages
        .iter()
        .map(|message| message.timestamp.as_str())
        .filter(|value| !value.is_empty())
        .min()
        .unwrap_or_default()
        .to_string();
    let last = messages
        .iter()
        .map(|message| message.timestamp.as_str())
        .filter(|value| !value.is_empty())
        .max()
        .unwrap_or(first.as_str())
        .to_string();
    let title = root
        .get("title")
        .or_else(|| root.get("name"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| first_user_text(&messages))
        .unwrap_or_else(|| id.clone());
    Some(ChatSession {
        id,
        title: one_line(&title, 120),
        messages,
        message_count,
        first,
        last,
        has_tool_use,
        has_errors,
    })
}

fn find_messages(value: &Value) -> Option<&Vec<Value>> {
    fn visit<'a>(value: &'a Value, best: &mut Option<&'a Vec<Value>>) {
        match value {
            Value::Object(object) => {
                if let Some(messages) = object.get("messages").and_then(Value::as_array) {
                    let plausible = messages.iter().any(|message| {
                        message.as_object().is_some_and(|item| {
                            item.contains_key("type") && item.contains_key("content")
                        })
                    });
                    if plausible
                        && best
                            .as_ref()
                            .map(|current| messages.len() > current.len())
                            .unwrap_or(true)
                    {
                        *best = Some(messages);
                    }
                }
                for child in object.values() {
                    visit(child, best);
                }
            }
            Value::Array(array) => {
                for child in array {
                    visit(child, best);
                }
            }
            _ => {}
        }
    }
    let mut best = None;
    visit(value, &mut best);
    best
}

fn string_field<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_str))
}

fn timestamp_value(value: &Value) -> Option<String> {
    if let Some(value) = value.as_str() {
        return Some(value.to_string());
    }
    let millis = value.as_f64()?;
    Utc.timestamp_millis_opt(millis as i64)
        .single()
        .map(|date| date.to_rfc3339_opts(SecondsFormat::Millis, true))
}

fn text_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) if !text.is_empty() => {
            vec![json!({ "type": "text", "text": text })]
        }
        Some(Value::Array(items)) => items
            .iter()
            .flat_map(|item| text_blocks(Some(item)))
            .collect(),
        Some(Value::Object(object)) => object
            .get("text")
            .or_else(|| object.get("content"))
            .map(|value| text_blocks(Some(value)))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn assistant_blocks(content: Option<&Value>) -> Vec<Value> {
    let Some(content) = content else {
        return Vec::new();
    };
    if let Some(items) = content.as_array() {
        return items
            .iter()
            .flat_map(|item| assistant_blocks(Some(item)))
            .collect();
    }
    if let Some(object) = content.as_object() {
        match object.get("type").and_then(Value::as_str) {
            Some("thinking") => return thinking_blocks(Some(content)),
            Some("tool_use") => return vec![tool_use_block(object)],
            _ => {}
        }
    }
    text_blocks(Some(content))
}

fn thinking_blocks(content: Option<&Value>) -> Vec<Value> {
    let text = match content {
        Some(Value::String(text)) => text.as_str(),
        Some(Value::Object(object)) => object
            .get("thinking")
            .or_else(|| object.get("text"))
            .or_else(|| object.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default(),
        _ => "",
    };
    (!text.is_empty())
        .then(|| json!({ "type": "thinking", "thinking": text, "signature": "" }))
        .into_iter()
        .collect()
}

fn tool_use_block(object: &Map<String, Value>) -> Value {
    json!({
        "type": "tool_use",
        "id": string_field(object, &["id", "apiId", "uuid"]).unwrap_or_default(),
        "name": string_field(object, &["name", "toolName"]).unwrap_or("unknown"),
        "input": object.get("input").cloned().unwrap_or_else(|| json!({}))
    })
}

fn tool_result_block(object: &Map<String, Value>) -> Value {
    let content = object
        .get("content")
        .or_else(|| object.get("result"))
        .or_else(|| object.get("error"))
        .map(value_text)
        .unwrap_or_default();
    let is_error = object
        .get("is_error")
        .or_else(|| object.get("isError"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| tool_error(object));
    json!({
        "type": "tool_result",
        "tool_use_id": string_field(object, &["tool_use_id", "toolUseId", "apiId", "id", "uuid"]).unwrap_or_default(),
        "content": content,
        "is_error": is_error
    })
}

fn tool_error(object: &Map<String, Value>) -> bool {
    object.get("error").is_some_and(|error| match error {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        _ => true,
    })
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => values.iter().map(value_text).collect::<Vec<_>>().join("\n"),
        Value::Object(object) => object
            .get("text")
            .or_else(|| object.get("content"))
            .map(value_text)
            .unwrap_or_else(|| value.to_string()),
        _ => value.to_string(),
    }
}

fn first_user_text(messages: &[ClaudeMessage]) -> Option<String> {
    messages.iter().find_map(|message| {
        if message.role.as_deref() != Some("user") {
            return None;
        }
        message
            .content
            .as_ref()?
            .as_array()?
            .iter()
            .find_map(|block| {
                (block.get("type")?.as_str()? == "text")
                    .then(|| block.get("text")?.as_str().map(str::to_string))
                    .flatten()
            })
    })
}

fn one_line(value: &str, max_chars: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let prefix = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn byte(&mut self) -> Result<u8, String> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or("Unexpected end of PowerPoint data")?;
        self.position += 1;
        Ok(byte)
    }

    fn bytes(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(length)
            .ok_or("PowerPoint value is too large")?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or("Unexpected end of PowerPoint data")?;
        self.position = end;
        Ok(bytes)
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("Invalid PowerPoint varint".to_string())
    }

    fn utf16_string(&mut self) -> Result<String, String> {
        let length = self.varint()? as usize;
        let bytes = self.bytes(
            length
                .checked_mul(2)
                .ok_or("PowerPoint string is too large")?,
        )?;
        let units = bytes
            .chunks_exact(2)
            // Chromium's IndexedDB String encoding is byte-swapped (network
            // order), unlike V8's two-byte strings below.
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        String::from_utf16(&units).map_err(|e| format!("Invalid PowerPoint UTF-16: {e}"))
    }

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }
}

/// Minimal V8 ValueSerializer reader for the plain JS data types emitted by the
/// PowerPoint chat store. Unsupported host objects fail the individual record;
/// other IndexedDB object stores are then simply skipped.
struct V8Reader<'a> {
    cursor: Cursor<'a>,
    references: Vec<Value>,
}

impl<'a> V8Reader<'a> {
    fn deserialize(bytes: &'a [u8]) -> Result<Value, String> {
        let payload = find_v8_payload(bytes).ok_or("V8 payload header is missing")?;
        let mut reader = Self {
            cursor: Cursor::new(payload),
            references: Vec::new(),
        };
        if reader.cursor.byte()? != 0xff {
            return Err("V8 payload header is invalid".to_string());
        }
        let _version = reader.cursor.varint()?;
        reader.value()
    }

    fn tag(&mut self) -> Result<u8, String> {
        loop {
            let tag = self.cursor.byte()?;
            if tag != 0 {
                return Ok(tag);
            }
        }
    }

    fn peek_tag(&self) -> Option<u8> {
        self.cursor
            .remaining()
            .iter()
            .copied()
            .find(|tag| *tag != 0)
    }

    fn reserve_reference(&mut self) -> usize {
        let id = self.references.len();
        self.references.push(Value::Null);
        id
    }

    fn set_reference(&mut self, id: usize, value: &Value) {
        self.references[id] = value.clone();
    }

    fn value(&mut self) -> Result<Value, String> {
        let tag = self.tag()?;
        match tag {
            b'?' => {
                let _count = self.cursor.varint()?;
                self.value()
            }
            b'-' | b'_' | b'0' => Ok(Value::Null),
            b'T' => Ok(Value::Bool(true)),
            b'F' => Ok(Value::Bool(false)),
            b'I' => {
                let encoded = self.cursor.varint()? as u32;
                let value = ((encoded >> 1) as i32) ^ (-((encoded & 1) as i32));
                Ok(Value::Number(Number::from(value)))
            }
            b'U' => Ok(Value::Number(Number::from(self.cursor.varint()?))),
            b'N' => self.double_value(),
            b'D' => {
                let id = self.reserve_reference();
                let value = self.double_value()?;
                self.set_reference(id, &value);
                Ok(value)
            }
            b'S' => {
                let length = self.cursor.varint()? as usize;
                let bytes = self.cursor.bytes(length)?;
                Ok(Value::String(String::from_utf8_lossy(bytes).into_owned()))
            }
            b'"' => {
                let length = self.cursor.varint()? as usize;
                let text = self
                    .cursor
                    .bytes(length)?
                    .iter()
                    .map(|byte| char::from(*byte))
                    .collect();
                Ok(Value::String(text))
            }
            b'c' => {
                let length = self.cursor.varint()? as usize;
                let bytes = self.cursor.bytes(length)?;
                if bytes.len() % 2 != 0 {
                    return Err("Invalid V8 two-byte string".to_string());
                }
                let units = bytes
                    .chunks_exact(2)
                    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                    .collect::<Vec<_>>();
                Ok(Value::String(String::from_utf16_lossy(&units)))
            }
            b'^' => {
                let id = self.cursor.varint()? as usize;
                self.references
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("Invalid V8 object reference: {id}"))
            }
            b'o' => self.object(),
            b'A' => self.dense_array(),
            b'a' => self.sparse_array(),
            b';' => self.map(),
            b'\'' => self.set(),
            b'y' => self.boxed(Value::Bool(true)),
            b'x' => self.boxed(Value::Bool(false)),
            b'n' => {
                let value = self.double_value()?;
                self.boxed(value)
            }
            b's' => {
                let length = self.cursor.varint()? as usize;
                let bytes = self.cursor.bytes(length)?;
                let value = Value::String(String::from_utf8_lossy(bytes).into_owned());
                self.boxed(value)
            }
            b'B' | b'C' => self.array_buffer(false),
            b'~' => self.array_buffer(true),
            other => Err(format!("Unsupported V8 tag: 0x{other:02x}")),
        }
    }

    fn double_value(&mut self) -> Result<Value, String> {
        let bytes = self.cursor.bytes(8)?;
        let value = f64::from_le_bytes(bytes.try_into().map_err(|_| "Invalid V8 double")?);
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| "Non-finite V8 number".to_string())
    }

    fn object(&mut self) -> Result<Value, String> {
        let id = self.reserve_reference();
        let mut object = Map::new();
        while self.peek_tag() != Some(b'{') {
            let key = self.value()?;
            let value = self.value()?;
            let key = match key {
                Value::String(key) => key,
                Value::Number(key) => key.to_string(),
                _ => return Err("Invalid V8 object key".to_string()),
            };
            object.insert(key, value);
        }
        if self.tag()? != b'{' {
            return Err("Invalid V8 object end".to_string());
        }
        let _property_count = self.cursor.varint()?;
        let value = Value::Object(object);
        self.set_reference(id, &value);
        Ok(value)
    }

    fn dense_array(&mut self) -> Result<Value, String> {
        let length = self.cursor.varint()? as usize;
        let id = self.reserve_reference();
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(self.value()?);
        }
        while self.peek_tag() != Some(b'$') {
            let _key = self.value()?;
            let _value = self.value()?;
        }
        if self.tag()? != b'$' {
            return Err("Invalid V8 array end".to_string());
        }
        let _property_count = self.cursor.varint()?;
        let stored_length = self.cursor.varint()? as usize;
        if stored_length != length {
            return Err("V8 array length mismatch".to_string());
        }
        let value = Value::Array(values);
        self.set_reference(id, &value);
        Ok(value)
    }

    fn sparse_array(&mut self) -> Result<Value, String> {
        let length = self.cursor.varint()? as usize;
        let id = self.reserve_reference();
        let mut values = vec![Value::Null; length];
        while self.peek_tag() != Some(b'@') {
            let key = self.value()?;
            let value = self.value()?;
            if let Some(index) = key.as_u64().or_else(|| key.as_str()?.parse().ok()) {
                if let Some(slot) = values.get_mut(index as usize) {
                    *slot = value;
                }
            }
        }
        if self.tag()? != b'@' {
            return Err("Invalid V8 sparse-array end".to_string());
        }
        let _property_count = self.cursor.varint()?;
        let _stored_length = self.cursor.varint()?;
        let value = Value::Array(values);
        self.set_reference(id, &value);
        Ok(value)
    }

    fn map(&mut self) -> Result<Value, String> {
        let id = self.reserve_reference();
        let mut object = Map::new();
        let mut fallback = Vec::new();
        while self.peek_tag() != Some(b':') {
            let key = self.value()?;
            let value = self.value()?;
            if let Some(key) = key.as_str() {
                object.insert(key.to_string(), value);
            } else {
                fallback.push(json!({ "key": key, "value": value }));
            }
        }
        if self.tag()? != b':' {
            return Err("Invalid V8 map end".to_string());
        }
        let _entry_value_count = self.cursor.varint()?;
        let value = if fallback.is_empty() {
            Value::Object(object)
        } else {
            for (key, value) in object {
                fallback.push(json!({ "key": key, "value": value }));
            }
            Value::Array(fallback)
        };
        self.set_reference(id, &value);
        Ok(value)
    }

    fn set(&mut self) -> Result<Value, String> {
        let id = self.reserve_reference();
        let mut values = Vec::new();
        while self.peek_tag() != Some(b',') {
            values.push(self.value()?);
        }
        if self.tag()? != b',' {
            return Err("Invalid V8 set end".to_string());
        }
        let _length = self.cursor.varint()?;
        let value = Value::Array(values);
        self.set_reference(id, &value);
        Ok(value)
    }

    fn boxed(&mut self, value: Value) -> Result<Value, String> {
        let id = self.reserve_reference();
        self.set_reference(id, &value);
        Ok(value)
    }

    fn array_buffer(&mut self, resizable: bool) -> Result<Value, String> {
        let id = self.reserve_reference();
        let length = self.cursor.varint()? as usize;
        if resizable {
            let _max_length = self.cursor.varint()?;
        }
        let _bytes = self.cursor.bytes(length)?;
        let value = Value::Null;
        self.set_reference(id, &value);
        Ok(value)
    }
}

fn find_v8_payload(bytes: &[u8]) -> Option<&[u8]> {
    let limit = bytes.len().min(64);
    for offset in 0..limit {
        if bytes[offset] != 0xff {
            continue;
        }
        let mut cursor = Cursor::new(&bytes[offset + 1..]);
        let Ok(version) = cursor.varint() else {
            continue;
        };
        let Some(tag) = cursor.remaining().iter().copied().find(|tag| *tag != 0) else {
            continue;
        };
        if version <= 32 && matches!(tag, b'o' | b'A' | b'a' | b'"' | b'c' | b'S') {
            return Some(&bytes[offset..]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_byte(value: &str, out: &mut Vec<u8>) {
        out.push(b'"');
        out.push(value.len() as u8);
        out.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn decodes_v8_object_array_and_date() {
        let mut bytes = vec![0xff, 0x15, 0xfe, 0, 0, 0, 0xff, 0x10, b'o'];
        one_byte("id", &mut bytes);
        one_byte("session-1", &mut bytes);
        one_byte("messages", &mut bytes);
        bytes.extend_from_slice(&[b'A', 1, b'o']);
        one_byte("type", &mut bytes);
        one_byte("user", &mut bytes);
        one_byte("content", &mut bytes);
        one_byte("hello", &mut bytes);
        one_byte("timestamp", &mut bytes);
        bytes.push(b'D');
        bytes.extend_from_slice(&1_700_000_000_000f64.to_le_bytes());
        bytes.extend_from_slice(&[b'{', 3, b'$', 0, 1, b'{', 2]);

        let value = V8Reader::deserialize(&bytes).unwrap();
        assert_eq!(value["id"], "session-1");
        assert_eq!(value["messages"][0]["content"], "hello");
        assert_eq!(value["messages"][0]["timestamp"], 1_700_000_000_000f64);
    }

    #[test]
    fn parses_powerpoint_messages_into_normalized_shape() {
        let root = json!({
            "id": "session-1",
            "messages": [
                { "uuid": "u1", "timestamp": 1_700_000_000_000f64, "type": "user", "content": "make a deck" },
                { "uuid": "a1", "timestamp": 1_700_000_001_000f64, "type": "thinking", "content": "planning" },
                { "uuid": "a2", "timestamp": 1_700_000_002_000f64, "type": "assistant", "content": "done", "model": "claude-test" }
            ]
        });
        let session = parse_chat_session(&root).unwrap();
        assert_eq!(session.title, "make a deck");
        assert_eq!(session.message_count, 3);
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[0].role.as_deref(), Some("user"));
        assert_eq!(
            session.messages[1].content.as_ref().unwrap()[0]["type"],
            "thinking"
        );
        assert_eq!(session.messages[2].model.as_deref(), Some("claude-test"));
        assert_eq!(session.messages[2].parent_uuid.as_deref(), Some("a1"));
    }

    #[test]
    fn decodes_external_wrapper_metadata() {
        let mut encoded = vec![0, 0xc5, 1, 39];
        for unit in WRAPPER_MIME.encode_utf16() {
            encoded.extend_from_slice(&unit.to_be_bytes());
        }
        encoded.extend_from_slice(&[0xf5, 0xcb, 0xde, 0x03]);
        let objects = decode_external_objects(&encoded).unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].blob_number, 197);
        assert_eq!(objects[0].mime_type, WRAPPER_MIME);
        assert_eq!(objects[0].size, 7_841_269);
    }

    #[test]
    fn session_path_round_trips_windows_store_path() {
        let path = Path::new(r"C:\Users\Example\Office\IndexedDB\store.leveldb");
        let encoded = session_path(path, "session-1");
        let (decoded, id) = parse_session_path(&encoded).unwrap();
        assert_eq!(decoded, path);
        assert_eq!(id, "session-1");
    }
}
