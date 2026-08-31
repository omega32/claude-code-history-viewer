use super::{ProviderInfo, SessionSnapshotLoad};
use crate::commands::multi_provider::finalize_loaded_messages;
use crate::models::{
    ClaudeMessage, ClaudeProject, ClaudeSession, InferenceCost, InferenceMetadata, InferenceUsage,
    SubagentProvenance, TokenUsage,
};
use crate::utils::{
    build_provider_message, estimate_message_count_from_size, find_line_ranges,
    search_json_value_case_insensitive,
};
use base64::prelude::{Engine as _, BASE64_STANDARD, BASE64_URL_SAFE_NO_PAD};
use chrono::{DateTime, Utc};
use fs2::FileExt;
use memchr::{memchr_iter, memmem};
use memmap2::Mmap;
use quick_xml::de::from_str as from_xml_str;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::commands::session::NativeRenameResult;

const STATE_DB_FILENAME: &str = "state_5.sqlite";
const SESSION_INDEX_FILENAME: &str = "session_index.jsonl";
const EXTERNAL_AGENT_IMPORTS_FILENAME: &str = "external_agent_session_imports.json";
const SESSION_METADATA_CACHE_FILENAME: &str = ".claude-code-history-viewer-session-cache.json";
const SESSION_METADATA_CACHE_VERSION: u32 = 1;
const AUTHORED_USER_SUBTYPE: &str = "authored_user";
const INJECTED_CONTEXT_SUBTYPE: &str = "injected_context";
const HOOK_PROMPT_SUBTYPE: &str = "hook_prompt";
const STEER_SUBTYPE: &str = "steer";
const SNAPSHOT_CURSOR_VERSION: u32 = 13;
/// Snapshot date of the published Codex `ChatGPT` credit rate card used below.
const CODEX_CREDIT_RATE_CARD_VERSION: &str = "2026-07-31";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CodexParserState {
    session_id: String,
    meta_seen: bool,
    #[serde(default)]
    forked_from_session_id: Option<String>,
    #[serde(default)]
    fork_replay_seen: bool,
    #[serde(default)]
    fork_transition_seen: bool,
    current_inference: InferenceMetadata,
    prev_input_tokens: u32,
    prev_output_tokens: u32,
    prev_cached_tokens: u32,
    prev_cache_write_tokens: u32,
    prev_reasoning_tokens: u32,
    #[serde(default)]
    pending_usage: CodexTokenUsage,
    msg_counter: u64,
}

impl CodexParserState {
    fn initial(path: &Path) -> Self {
        Self {
            session_id: session_id_from_rollout_filename(path).unwrap_or_default(),
            meta_seen: false,
            forked_from_session_id: None,
            fork_replay_seen: false,
            fork_transition_seen: false,
            current_inference: InferenceMetadata::default(),
            prev_input_tokens: 0,
            prev_output_tokens: 0,
            prev_cached_tokens: 0,
            prev_cache_write_tokens: 0,
            prev_reasoning_tokens: 0,
            pending_usage: CodexTokenUsage::default(),
            msg_counter: 0,
        }
    }
}

#[derive(Debug)]
struct PendingForkRollback {
    message_index: usize,
    replacement_task_started: bool,
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
    diagnostics: Vec<CodexAuthorshipDiagnostic>,
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

#[derive(Debug)]
struct PendingCodexUserMessage {
    message_index: usize,
    message_id: String,
    source_line: usize,
    response_text: Option<String>,
    authored_turn_id: Option<String>,
    precedes_input_boundary: bool,
    terminal_evidence: TerminalContextEvidence,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CodexAuthorshipDiagnostic {
    pub(crate) kind: String,
    pub(crate) message_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) provider_turn_id: Option<String>,
    pub(crate) source_line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) discriminator: Option<String>,
}

#[derive(Debug)]
pub(crate) struct CodexAuthorshipAuditProjection {
    pub(crate) session_id: String,
    pub(crate) messages: Vec<ClaudeMessage>,
    pub(crate) diagnostics: Vec<CodexAuthorshipDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalContextEvidence {
    AwaitingBoundary,
    AwaitingSameTurnActivity,
    SameTurnActivitySeen,
    Invalid,
}

impl TerminalContextEvidence {
    fn observe_boundary(&mut self) {
        *self = match self {
            Self::AwaitingBoundary | Self::AwaitingSameTurnActivity => {
                Self::AwaitingSameTurnActivity
            }
            Self::SameTurnActivitySeen | Self::Invalid => Self::Invalid,
        };
    }

    fn observe_same_turn_activity(&mut self) {
        *self = match self {
            Self::AwaitingSameTurnActivity | Self::SameTurnActivitySeen => {
                Self::SameTurnActivitySeen
            }
            Self::AwaitingBoundary | Self::Invalid => Self::Invalid,
        };
    }

    fn invalidate(&mut self) {
        *self = Self::Invalid;
    }
}

fn bounded_discriminator(value: Option<&str>) -> Option<String> {
    const MAX_DISCRIMINATOR_CHARS: usize = 128;
    value
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(MAX_DISCRIMINATOR_CHARS).collect())
}

fn invalidate_pending_authorship(
    pending: &mut [PendingCodexUserMessage],
    diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
    kind: &str,
    source_line: usize,
    discriminator: Option<&str>,
) {
    for candidate in pending {
        if candidate.terminal_evidence == TerminalContextEvidence::Invalid {
            continue;
        }
        candidate.terminal_evidence.invalidate();
        diagnostics.push(CodexAuthorshipDiagnostic {
            kind: kind.to_string(),
            message_id: candidate.message_id.clone(),
            provider_turn_id: candidate.authored_turn_id.clone(),
            source_line,
            discriminator: bounded_discriminator(discriminator),
        });
    }
}

fn diagnose_unresolved_candidate(
    candidate: &PendingCodexUserMessage,
    diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
    kind: &str,
    source_line: usize,
) {
    if candidate.terminal_evidence != TerminalContextEvidence::Invalid {
        diagnostics.push(CodexAuthorshipDiagnostic {
            kind: kind.to_string(),
            message_id: candidate.message_id.clone(),
            provider_turn_id: candidate.authored_turn_id.clone(),
            source_line,
            discriminator: None,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CodexAuthorshipLaneKey {
    Turn(String),
    Unscoped,
}

impl CodexAuthorshipLaneKey {
    fn turn_id(&self) -> Option<&str> {
        match self {
            Self::Turn(turn_id) => Some(turn_id),
            Self::Unscoped => None,
        }
    }
}

#[derive(Debug, Default)]
struct CodexAuthorshipLane {
    active: bool,
    authored_user_count: usize,
    pending_user_messages: Vec<PendingCodexUserMessage>,
}

#[derive(Debug, Default)]
struct CodexAuthorshipTracker {
    lanes: HashMap<CodexAuthorshipLaneKey, CodexAuthorshipLane>,
}

impl CodexAuthorshipTracker {
    fn start_turn(
        &mut self,
        turn_id: &str,
        diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
        source_line: usize,
    ) {
        for (lane_key, lane) in &mut self.lanes {
            let preserve_active_explicit_lane =
                matches!(lane_key, CodexAuthorshipLaneKey::Turn(_)) && lane.active;
            if !preserve_active_explicit_lane {
                invalidate_pending_authorship(
                    &mut lane.pending_user_messages,
                    diagnostics,
                    "unresolved-before-task-start",
                    source_line,
                    Some(turn_id),
                );
                lane.pending_user_messages.clear();
            }
        }
        let key = CodexAuthorshipLaneKey::Turn(turn_id.to_string());
        if let Some(previous) = self.lanes.remove(&key) {
            for candidate in &previous.pending_user_messages {
                diagnose_unresolved_candidate(
                    candidate,
                    diagnostics,
                    "unresolved-before-task-restart",
                    source_line,
                );
            }
        }
        self.lanes.insert(
            key,
            CodexAuthorshipLane {
                active: true,
                ..CodexAuthorshipLane::default()
            },
        );
    }

    fn push_candidate(&mut self, candidate: PendingCodexUserMessage) {
        let key = candidate
            .authored_turn_id
            .as_deref()
            .map(|turn_id| CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
            .or_else(|| {
                self.sole_active_turn_id()
                    .map(|turn_id| CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
            })
            .unwrap_or(CodexAuthorshipLaneKey::Unscoped);
        self.lanes
            .entry(key)
            .or_default()
            .pending_user_messages
            .push(candidate);
    }

    fn most_recent_pending_key(&self) -> Option<CodexAuthorshipLaneKey> {
        self.lanes
            .iter()
            .filter_map(|(key, lane)| {
                lane.pending_user_messages
                    .last()
                    .map(|candidate| (key, candidate.source_line))
            })
            .max_by_key(|(_, source_line)| *source_line)
            .map(|(key, _)| key.clone())
    }

    fn has_active_turns(&self) -> bool {
        self.lanes.values().any(|lane| lane.active)
    }

    fn is_quiescent(&self) -> bool {
        !self.has_active_turns()
            && self
                .lanes
                .values()
                .all(|lane| lane.pending_user_messages.is_empty())
    }

    fn active_turn_ids(&self) -> impl Iterator<Item = &str> {
        self.lanes
            .iter()
            .filter_map(|(key, lane)| if lane.active { key.turn_id() } else { None })
    }

    fn sole_active_turn_id(&self) -> Option<&str> {
        let mut turn_ids = self.active_turn_ids();
        let turn_id = turn_ids.next()?;
        turn_ids.next().is_none().then_some(turn_id)
    }

    fn invalidate_all(
        &mut self,
        diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
        kind: &str,
        source_line: usize,
        discriminator: Option<&str>,
    ) {
        for lane in self.lanes.values_mut() {
            invalidate_pending_authorship(
                &mut lane.pending_user_messages,
                diagnostics,
                kind,
                source_line,
                discriminator,
            );
        }
    }
}

fn codex_user_response_text(payload: &Value) -> Option<String> {
    let text = payload
        .get("content")?
        .as_array()?
        .iter()
        .filter_map(|item| {
            matches!(
                item.get("type").and_then(Value::as_str),
                Some("input_text" | "text")
            )
            .then(|| item.get("text").and_then(Value::as_str))
            .flatten()
        })
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn codex_authored_turn_id(payload: &Value) -> Option<String> {
    payload
        .get("internal_chat_message_metadata_passthrough")
        .and_then(|metadata| metadata.get("turn_id"))
        .and_then(Value::as_str)
        .filter(|turn_id| !turn_id.is_empty())
        .map(str::to_string)
}

fn non_empty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

fn merge_codex_message_provenance(
    message: &mut ClaudeMessage,
    provider_turn_id: Option<&str>,
    client_message_id: Option<&str>,
) {
    let provider_turn_id = provider_turn_id.filter(|value| !value.is_empty());
    let client_message_id = client_message_id.filter(|value| !value.is_empty());
    if provider_turn_id.is_none() && client_message_id.is_none() {
        return;
    }

    let data = message
        .data
        .get_or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(data) = data.as_object_mut() else {
        return;
    };
    if let Some(provider_turn_id) = provider_turn_id {
        data.entry("providerTurnId")
            .or_insert_with(|| Value::String(provider_turn_id.to_string()));
    }
    if let Some(client_message_id) = client_message_id {
        data.entry("clientMessageId")
            .or_insert_with(|| Value::String(client_message_id.to_string()));
    }
}

fn clear_inferred_codex_turn_provenance(message: &mut ClaudeMessage) {
    let Some(data) = message.data.as_mut().and_then(Value::as_object_mut) else {
        return;
    };
    data.remove("providerTurnId");
    data.remove("imageArtifacts");
    if data.is_empty() {
        message.data = None;
    }
}

struct CanonicalCodexUserMessage {
    id: Option<String>,
    client_id: Option<String>,
    turn_id: Option<String>,
    content: Option<Value>,
    response_text: Option<String>,
}

struct CanonicalCodexUserProjectionContext<'a> {
    session_id: &'a str,
    line_timestamp: &'a str,
    counter: &'a mut u64,
    fallback_turn_id: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
#[serde(rename = "hook_prompt")]
struct CodexHookPromptXml {
    #[serde(rename = "@hook_run_id")]
    hook_run_id: String,
    #[serde(rename = "$text")]
    text: String,
}

fn codex_hook_prompt_fragments(payload: &Value) -> Option<Vec<CodexHookPromptXml>> {
    let content = payload.get("content")?.as_array()?;
    if content.is_empty() {
        return None;
    }
    content
        .iter()
        .map(|item| {
            if item.get("type").and_then(Value::as_str) != Some("input_text") {
                return None;
            }
            let text = item.get("text").and_then(Value::as_str)?.trim();
            let fragment = from_xml_str::<CodexHookPromptXml>(text).ok()?;
            (!fragment.hook_run_id.trim().is_empty()).then_some(fragment)
        })
        .collect()
}

fn legacy_user_event_content(event_payload: &Value) -> Option<Value> {
    let message = event_payload.get("message").and_then(Value::as_str)?;
    let mut content = vec![serde_json::json!({ "type": "text", "text": message })];

    for field in ["images", "local_images"] {
        for image in event_payload
            .get(field)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|image| !image.is_empty())
        {
            content.push(serde_json::json!({
                "type": "image",
                "source": { "type": "url", "url": image }
            }));
        }
    }

    Some(Value::Array(content))
}

fn canonical_codex_user_message(event_payload: &Value) -> Option<CanonicalCodexUserMessage> {
    match event_payload.get("type").and_then(Value::as_str)? {
        "user_message" => {
            let message = event_payload.get("message").and_then(Value::as_str)?;
            Some(CanonicalCodexUserMessage {
                id: None,
                client_id: non_empty_string(event_payload.get("client_id")).map(str::to_string),
                turn_id: None,
                content: legacy_user_event_content(event_payload),
                response_text: Some(message.to_string()),
            })
        }
        "item_completed" => {
            let item = event_payload.get("item")?;
            if item.get("type").and_then(Value::as_str) != Some("UserMessage") {
                return None;
            }
            Some(CanonicalCodexUserMessage {
                id: non_empty_string(item.get("id")).map(str::to_string),
                client_id: non_empty_string(item.get("client_id")).map(str::to_string),
                turn_id: non_empty_string(event_payload.get("turn_id")).map(str::to_string),
                content: convert_codex_content_array(item.get("content"), None),
                response_text: codex_user_response_text(item),
            })
        }
        _ => None,
    }
}

fn project_canonical_user_event(
    messages: &mut Vec<ClaudeMessage>,
    tracker: &mut CodexAuthorshipTracker,
    event_payload: &Value,
    context: &mut CanonicalCodexUserProjectionContext<'_>,
    diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
    source_line: usize,
) {
    let Some(canonical) = canonical_codex_user_message(event_payload) else {
        if event_payload.get("type").and_then(Value::as_str) != Some("user_message") {
            return;
        }
        tracker.invalidate_all(
            diagnostics,
            "missing-user-message-text",
            source_line,
            Some("user_message"),
        );
        for lane in tracker.lanes.values_mut() {
            lane.pending_user_messages.clear();
        }
        return;
    };

    let pending_lane_key = if let Some(turn_id) = canonical.turn_id.as_ref() {
        let key = CodexAuthorshipLaneKey::Turn(turn_id.clone());
        tracker
            .lanes
            .get(&key)
            .is_some_and(|lane| !lane.pending_user_messages.is_empty())
            .then_some(key)
    } else {
        tracker.most_recent_pending_key()
    };
    let matched_text = pending_lane_key
        .as_ref()
        .and_then(|lane_key| tracker.lanes.get(lane_key))
        .and_then(|lane| lane.pending_user_messages.last())
        .and_then(|candidate| candidate.response_text.as_deref());
    let exact_match = canonical
        .response_text
        .as_deref()
        .is_some_and(|canonical_text| matched_text == Some(canonical_text));
    if pending_lane_key.is_some() && !exact_match {
        tracker.invalidate_all(
            diagnostics,
            "pair-text-mismatch",
            source_line,
            Some("user_message"),
        );
        for lane in tracker.lanes.values_mut() {
            lane.pending_user_messages.clear();
        }
    }

    let lane_key = pending_lane_key
        .clone()
        .or_else(|| {
            canonical
                .turn_id
                .as_ref()
                .map(|turn_id| CodexAuthorshipLaneKey::Turn(turn_id.clone()))
        })
        .or_else(|| {
            context
                .fallback_turn_id
                .map(|turn_id| CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
        })
        .or_else(|| {
            tracker
                .sole_active_turn_id()
                .map(|turn_id| CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
        })
        .unwrap_or(CodexAuthorshipLaneKey::Unscoped);
    let confirmed_turn_id = canonical
        .turn_id
        .as_deref()
        .or_else(|| lane_key.turn_id())
        .map(str::to_string);
    let matched_index = exact_match
        .then(|| {
            tracker
                .lanes
                .get(&lane_key)
                .and_then(|lane| lane.pending_user_messages.last())
                .map(|candidate| candidate.message_index)
        })
        .flatten();
    let lane = tracker.lanes.entry(lane_key).or_default();
    let subtype = if lane.authored_user_count > 0 && (lane.active || canonical.turn_id.is_some()) {
        STEER_SUBTYPE
    } else {
        AUTHORED_USER_SUBTYPE
    };

    if let Some(message_index) = matched_index {
        messages[message_index].subtype = Some(subtype.to_string());
        if let Some(id) = canonical.id.as_ref() {
            messages[message_index].uuid.clone_from(id);
        }
        merge_codex_message_provenance(
            &mut messages[message_index],
            confirmed_turn_id.as_deref(),
            canonical.client_id.as_deref(),
        );
    } else {
        *context.counter += 1;
        let mut message = build_codex_message(
            canonical
                .id
                .clone()
                .unwrap_or_else(|| format!("codex-event-{}", context.counter)),
            context.session_id,
            context.line_timestamp.to_string(),
            "user",
            Some("user"),
            canonical.content,
            None,
        );
        message.subtype = Some(subtype.to_string());
        merge_codex_message_provenance(
            &mut message,
            confirmed_turn_id.as_deref(),
            canonical.client_id.as_deref(),
        );
        messages.push(message);
    }
    lane.pending_user_messages.clear();
    lane.authored_user_count += 1;
}

fn codex_response_is_same_turn_agent_activity(payload: &Value, active_turn_id: &str) -> bool {
    if codex_authored_turn_id(payload).as_deref() != Some(active_turn_id) {
        return false;
    }

    match payload.get("type").and_then(Value::as_str) {
        Some("message") => payload.get("role").and_then(Value::as_str) == Some("assistant"),
        Some(
            "reasoning"
            | "local_shell_call"
            | "function_call"
            | "function_call_output"
            | "custom_tool_call"
            | "custom_tool_call_output"
            | "web_search_call",
        ) => true,
        _ => false,
    }
}

fn codex_response_is_interleaved_context_instruction(
    payload: &Value,
    active_turn_id: &str,
    pending: &[PendingCodexUserMessage],
) -> bool {
    payload.get("type").and_then(Value::as_str) == Some("message")
        && payload.get("role").and_then(Value::as_str) == Some("developer")
        && codex_authored_turn_id(payload).as_deref() == Some(active_turn_id)
        && !pending.is_empty()
        && pending.iter().all(|candidate| {
            candidate.authored_turn_id.as_deref() == Some(active_turn_id)
                && !candidate.precedes_input_boundary
                && candidate.terminal_evidence == TerminalContextEvidence::AwaitingBoundary
        })
}

fn codex_event_is_known_terminal_bookkeeping(payload: &Value, active_turn_id: &str) -> bool {
    match payload.get("type").and_then(Value::as_str) {
        Some("token_count" | "agent_message" | "agent_reasoning") => true,
        Some(
            "exec_command_begin"
            | "exec_command_end"
            | "mcp_tool_call_begin"
            | "mcp_tool_call_end"
            | "patch_apply_begin"
            | "patch_apply_end"
            | "web_search_begin"
            | "web_search_end",
        ) => payload.get("turn_id").and_then(Value::as_str) == Some(active_turn_id),
        _ => false,
    }
}

fn observe_pending_terminal_record(
    tracker: &mut CodexAuthorshipTracker,
    diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
    line_type: &str,
    event_type: &str,
    payload: Option<&Value>,
    source_line: usize,
) {
    let Some(latest_pending_key) = tracker.most_recent_pending_key() else {
        return;
    };

    match line_type {
        "world_state" => {
            let payload_is_object = payload.is_some_and(Value::is_object);
            let explicit_turn_id = payload
                .and_then(|payload| payload.get("turn_id"))
                .and_then(Value::as_str)
                .filter(|turn_id| !turn_id.is_empty());
            let lane_key = if let Some(turn_id) = explicit_turn_id {
                let key = CodexAuthorshipLaneKey::Turn(turn_id.to_string());
                if tracker.lanes.contains_key(&key) {
                    key
                } else {
                    let lane = tracker
                        .lanes
                        .get_mut(&latest_pending_key)
                        .expect("the latest pending lane must still exist");
                    invalidate_pending_authorship(
                        &mut lane.pending_user_messages,
                        diagnostics,
                        "invalid-input-boundary",
                        source_line,
                        Some(turn_id),
                    );
                    return;
                }
            } else {
                latest_pending_key
            };
            let has_active_turns = tracker.has_active_turns();
            let lane = tracker
                .lanes
                .get_mut(&lane_key)
                .expect("the selected world-state lane must still exist");
            let boundary_is_valid = payload_is_object
                && (!has_active_turns || lane.active || explicit_turn_id.is_none());
            for candidate in &mut lane.pending_user_messages {
                if boundary_is_valid {
                    candidate.precedes_input_boundary = true;
                    candidate.terminal_evidence.observe_boundary();
                } else {
                    invalidate_pending_authorship(
                        std::slice::from_mut(candidate),
                        diagnostics,
                        "invalid-input-boundary",
                        source_line,
                        None,
                    );
                }
            }
        }
        "turn_context" => {
            let explicit_turn_id = payload
                .and_then(|payload| payload.get("turn_id"))
                .and_then(Value::as_str)
                .filter(|turn_id| !turn_id.is_empty());
            let Some(turn_id) = explicit_turn_id else {
                tracker.invalidate_all(diagnostics, "invalid-input-boundary", source_line, None);
                return;
            };
            let key = CodexAuthorshipLaneKey::Turn(turn_id.to_string());
            let has_active_turns = tracker.has_active_turns();
            if let Some(lane) = tracker.lanes.get_mut(&key) {
                let boundary_is_valid =
                    payload.is_some_and(Value::is_object) && (!has_active_turns || lane.active);
                for candidate in &mut lane.pending_user_messages {
                    if boundary_is_valid {
                        candidate.precedes_input_boundary = true;
                        candidate.terminal_evidence.observe_boundary();
                    } else {
                        invalidate_pending_authorship(
                            std::slice::from_mut(candidate),
                            diagnostics,
                            "invalid-input-boundary",
                            source_line,
                            Some(turn_id),
                        );
                    }
                }
            } else {
                let lane = tracker
                    .lanes
                    .get_mut(&latest_pending_key)
                    .expect("the latest pending lane must still exist");
                invalidate_pending_authorship(
                    &mut lane.pending_user_messages,
                    diagnostics,
                    "invalid-input-boundary",
                    source_line,
                    Some(turn_id),
                );
            }
        }
        "response_item" => {
            let Some(payload) = payload else {
                tracker.invalidate_all(diagnostics, "malformed-response-item", source_line, None);
                return;
            };
            let is_user_message = payload.get("type").and_then(Value::as_str) == Some("message")
                && payload.get("role").and_then(Value::as_str) == Some("user");
            if is_user_message {
                return;
            }
            let Some(turn_id) = codex_authored_turn_id(payload) else {
                tracker.invalidate_all(
                    diagnostics,
                    "unknown-or-cross-turn-response-item",
                    source_line,
                    payload.get("type").and_then(Value::as_str),
                );
                return;
            };
            let key = CodexAuthorshipLaneKey::Turn(turn_id.clone());
            let Some(lane) = tracker.lanes.get_mut(&key) else {
                let latest = tracker
                    .lanes
                    .get_mut(&latest_pending_key)
                    .expect("the latest pending lane must still exist");
                invalidate_pending_authorship(
                    &mut latest.pending_user_messages,
                    diagnostics,
                    "unknown-or-cross-turn-response-item",
                    source_line,
                    payload.get("type").and_then(Value::as_str),
                );
                return;
            };
            if codex_response_is_interleaved_context_instruction(
                payload,
                &turn_id,
                &lane.pending_user_messages,
            ) {
                return;
            }
            let is_same_turn_activity =
                codex_response_is_same_turn_agent_activity(payload, &turn_id);
            for candidate in &mut lane.pending_user_messages {
                if is_same_turn_activity {
                    candidate.terminal_evidence.observe_same_turn_activity();
                } else {
                    let discriminator = payload.get("type").and_then(Value::as_str);
                    invalidate_pending_authorship(
                        std::slice::from_mut(candidate),
                        diagnostics,
                        "unknown-or-cross-turn-response-item",
                        source_line,
                        discriminator,
                    );
                }
            }
        }
        "event_msg" => {
            if matches!(event_type, "user_message" | "task_started") {
                return;
            }
            if matches!(event_type, "task_complete" | "turn_aborted") {
                let completion_turn_id = payload
                    .and_then(|payload| payload.get("turn_id"))
                    .and_then(Value::as_str);
                if completion_turn_id.is_some_and(|turn_id| {
                    tracker
                        .lanes
                        .get(&CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
                        .is_some_and(|lane| lane.active)
                }) {
                    return;
                }
                let latest = tracker
                    .lanes
                    .get_mut(&latest_pending_key)
                    .expect("the latest pending lane must still exist");
                invalidate_pending_authorship(
                    &mut latest.pending_user_messages,
                    diagnostics,
                    "mismatched-task-completion",
                    source_line,
                    completion_turn_id,
                );
                return;
            }
            if matches!(
                event_type,
                "token_count"
                    | "agent_message"
                    | "agent_reasoning"
                    | "image_generation_start"
                    | "image_generation_end"
            ) {
                return;
            }
            let event_turn_id = payload
                .and_then(|payload| payload.get("turn_id"))
                .and_then(Value::as_str)
                .filter(|turn_id| !turn_id.is_empty());
            if let Some(turn_id) = event_turn_id {
                let key = CodexAuthorshipLaneKey::Turn(turn_id.to_string());
                if let Some(lane) = tracker.lanes.get_mut(&key) {
                    if payload.is_some_and(|payload| {
                        codex_event_is_known_terminal_bookkeeping(payload, turn_id)
                    }) {
                        return;
                    }
                    invalidate_pending_authorship(
                        &mut lane.pending_user_messages,
                        diagnostics,
                        "unknown-event-type",
                        source_line,
                        (!event_type.is_empty()).then_some(event_type),
                    );
                    return;
                }
            }
            tracker.invalidate_all(
                diagnostics,
                "unknown-event-type",
                source_line,
                (!event_type.is_empty()).then_some(event_type),
            );
        }
        _ => {
            let kind = if line_type.is_empty() {
                "missing-record-type"
            } else {
                "unknown-top-level-record"
            };
            let scoped_turn_id = payload
                .and_then(|payload| payload.get("turn_id"))
                .and_then(Value::as_str)
                .filter(|turn_id| !turn_id.is_empty());
            if let Some(turn_id) = scoped_turn_id {
                if let Some(lane) = tracker
                    .lanes
                    .get_mut(&CodexAuthorshipLaneKey::Turn(turn_id.to_string()))
                {
                    invalidate_pending_authorship(
                        &mut lane.pending_user_messages,
                        diagnostics,
                        kind,
                        source_line,
                        Some(line_type),
                    );
                    return;
                }
            }
            tracker.invalidate_all(
                diagnostics,
                kind,
                source_line,
                (!line_type.is_empty()).then_some(line_type),
            );
        }
    }
}

fn classify_pending_terminal_context(
    messages: &mut [ClaudeMessage],
    pending: &[PendingCodexUserMessage],
    authored_user_count: usize,
    active_turn_id: &str,
    terminal_payload: &Value,
    diagnostics: &mut Vec<CodexAuthorshipDiagnostic>,
    source_line: usize,
) {
    // A context refresh injected while a task is already running has no companion
    // user_message event. Resolve it only when the surrounding provider sequence is
    // complete and unambiguous; every incomplete or reordered shape remains raw-only.
    if authored_user_count == 0
        || terminal_payload.get("turn_id").and_then(Value::as_str) != Some(active_turn_id)
    {
        for candidate in pending {
            diagnose_unresolved_candidate(
                candidate,
                diagnostics,
                "terminal-context-missing-prior-authorship",
                source_line,
            );
        }
        return;
    }

    let [candidate] = pending else {
        for candidate in pending {
            diagnose_unresolved_candidate(
                candidate,
                diagnostics,
                "competing-pending-candidates",
                source_line,
            );
        }
        return;
    };
    if candidate.authored_turn_id.as_deref() == Some(active_turn_id)
        && candidate.precedes_input_boundary
        && candidate.terminal_evidence == TerminalContextEvidence::SameTurnActivitySeen
    {
        messages[candidate.message_index].subtype = Some(INJECTED_CONTEXT_SUBTYPE.to_string());
    } else {
        let kind = if candidate.authored_turn_id.as_deref() != Some(active_turn_id) {
            "cross-turn-terminal-context"
        } else if !candidate.precedes_input_boundary {
            "missing-input-boundary"
        } else {
            "incomplete-terminal-sequence"
        };
        diagnose_unresolved_candidate(candidate, diagnostics, kind, source_line);
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

fn metadata_cache_path(base_path: &Path) -> PathBuf {
    base_path.join(SESSION_METADATA_CACHE_FILENAME)
}

fn metadata_cache_lock_path(base_path: &Path) -> PathBuf {
    base_path.join(format!("{SESSION_METADATA_CACHE_FILENAME}.lock"))
}

fn load_session_metadata_cache(base_path: &Path) -> CodexSessionMetadataCache {
    let Ok(content) = fs::read_to_string(metadata_cache_path(base_path)) else {
        return CodexSessionMetadataCache::default();
    };
    let Ok(cache) = serde_json::from_str::<CodexSessionMetadataCache>(&content) else {
        return CodexSessionMetadataCache::default();
    };
    if cache.version == SESSION_METADATA_CACHE_VERSION {
        cache
    } else {
        CodexSessionMetadataCache::default()
    }
}

fn save_session_metadata_cache(base_path: &Path, cache: &CodexSessionMetadataCache) {
    let Ok(lock_file) = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(metadata_cache_lock_path(base_path))
    else {
        return;
    };
    if FileExt::lock_exclusive(&lock_file).is_err() {
        return;
    }
    write_session_metadata_cache(base_path, cache);
}

fn write_session_metadata_cache(base_path: &Path, cache: &CodexSessionMetadataCache) {
    let Ok(content) = serde_json::to_vec(cache) else {
        return;
    };
    let path = metadata_cache_path(base_path);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let temp = path.with_extension(format!("json.{nonce}.tmp"));
    if fs::write(&temp, content).is_err() {
        return;
    }
    #[cfg(target_os = "windows")]
    if path.exists() {
        let _ = fs::remove_file(&path);
    }
    if fs::rename(&temp, &path).is_err() {
        let _ = fs::remove_file(temp);
    }
}

fn merge_session_metadata_cache_entry(
    base_path: &Path,
    rollout_path: &Path,
    key: String,
    entry: CachedSessionInfo,
) {
    let Ok(lock_file) = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(metadata_cache_lock_path(base_path))
    else {
        return;
    };
    if FileExt::lock_exclusive(&lock_file).is_err()
        || session_info_fingerprint(rollout_path).as_ref() != Some(&entry.fingerprint)
        || !is_discoverable_rollout(rollout_path)
    {
        return;
    }
    let mut latest = load_session_metadata_cache(base_path);
    latest.version = SESSION_METADATA_CACHE_VERSION;
    latest.entries.insert(key, entry);
    write_session_metadata_cache(base_path, &latest);
}

fn session_info_fingerprint(path: &Path) -> Option<SessionInfoFingerprint> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(SessionInfoFingerprint {
        modified_secs: modified.as_secs(),
        modified_nanos: modified.subsec_nanos(),
        file_size: metadata.len(),
    })
}

fn session_metadata_cache_key(base_path: &Path, rollout_path: &Path) -> Option<String> {
    rollout_path
        .strip_prefix(base_path)
        .ok()
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
}

fn probe_cached_session_info(
    base_path: &Path,
    rollout_path: &Path,
    old: &CodexSessionMetadataCache,
    next: &mut CodexSessionMetadataCache,
) -> Result<(SessionInfo, bool), String> {
    let key = session_metadata_cache_key(base_path, rollout_path);
    let before = session_info_fingerprint(rollout_path);
    if let (Some(key), Some(fingerprint)) = (key.as_ref(), before) {
        if let Some(hit) = old.entries.get(key) {
            if hit.fingerprint == fingerprint {
                next.entries.insert(key.clone(), hit.clone());
                let mut info = hit.info.clone();
                info.file_path = rollout_path.to_string_lossy().to_string();
                return Ok((info, false));
            }
        }
    }

    let info = extract_session_info(rollout_path)?;
    let after = session_info_fingerprint(rollout_path);
    let mut cache_updated = false;
    if let (Some(key), Some(before), Some(after)) = (key, before, after) {
        if before == after {
            next.entries.insert(
                key,
                CachedSessionInfo {
                    fingerprint: after,
                    info: info.clone(),
                },
            );
            cache_updated = true;
        }
    }
    Ok((info, cache_updated))
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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SessionInfo {
    pub(crate) session_id: String,
    pub(crate) cwd: Option<String>,
    /// Provider-qualified source from the rollout's first `session_meta`.
    /// User-facing surfaces are `codex-cli` / `codex-vscode`; structured
    /// subagent provenance is normalized separately as `codex-subagent`.
    pub(crate) entrypoint: Option<String>,
    /// Authoritative parent id from the rollout's first `session_meta`.
    pub(crate) forked_from_id: Option<String>,
    /// Authenticated child-agent identity and fork boundary from the rollout's
    /// first `session_meta` only.
    pub(crate) subagent_provenance: Option<SubagentProvenance>,
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    pub(crate) message_count: usize,
    pub(crate) first_message_time: String,
    pub(crate) last_message_time: String,
    pub(crate) last_modified: String,
    #[serde(skip)]
    pub(crate) file_path: String,
    pub(crate) has_tool_use: bool,
    pub(crate) summary: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SessionInfoFingerprint {
    modified_secs: u64,
    modified_nanos: u32,
    file_size: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CachedSessionInfo {
    fingerprint: SessionInfoFingerprint,
    info: SessionInfo,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct CodexSessionMetadataCache {
    version: u32,
    entries: HashMap<String, CachedSessionInfo>,
}

pub(crate) struct CodexSessionListing {
    pub(crate) session: ClaudeSession,
    pub(crate) project_path: String,
    pub(crate) is_archived: bool,
}

struct CodexProjectListingGroup {
    last_modified: String,
    sessions: Vec<CodexSessionListing>,
}

#[cfg(test)]
static SESSION_INFO_PARSE_COUNT: AtomicUsize = AtomicUsize::new(0);

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

fn codex_subagent_provenance(
    timestamp: Option<&Value>,
    source: Option<&Value>,
) -> Option<SubagentProvenance> {
    let spawned_at = timestamp?.as_str()?.trim();
    if spawned_at.is_empty() || DateTime::parse_from_rfc3339(spawned_at).is_err() {
        return None;
    }

    let thread_spawn = source?
        .as_object()?
        .get("subagent")?
        .as_object()?
        .get("thread_spawn")?
        .as_object()?;
    let agent_path = thread_spawn.get("agent_path")?.as_str()?.trim();
    if agent_path.is_empty() {
        return None;
    }
    let agent_nickname = thread_spawn
        .get("agent_nickname")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|nickname| !nickname.is_empty())
        .map(String::from);

    Some(SubagentProvenance {
        spawned_at: spawned_at.to_string(),
        agent_path: agent_path.to_string(),
        agent_nickname,
    })
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

fn session_from_info(
    info: SessionInfo,
    project_cwd: &str,
    title_index: &HashMap<String, NativeTitle>,
) -> ClaudeSession {
    let native_title = title_index
        .get(&info.session_id)
        .map(|native| (native.title.clone(), native.is_renamed));
    ClaudeSession {
        session_id: info.file_path.clone(),
        actual_session_id: info.session_id,
        file_path: info.file_path,
        project_name: Path::new(project_cwd)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default(),
        message_count: info.message_count,
        first_message_time: info.first_message_time,
        last_message_time: info.last_message_time,
        last_modified: info.last_modified,
        has_tool_use: info.has_tool_use,
        has_errors: false,
        summary: native_title
            .as_ref()
            .map(|(title, _)| title.clone())
            .or(info.summary),
        is_renamed: native_title.is_some_and(|(_, is_renamed)| is_renamed),
        provider: Some("codex".to_string()),
        storage_type: None,
        entrypoint: info.entrypoint,
        forked_from_id: info.forked_from_id,
        subagent_provenance: info.subagent_provenance,
    }
}

/// Load every live Codex session in one rollout-tree pass. Rollout-derived
/// metadata is cached across processes; independently mutable native titles,
/// import state, and archive provenance remain live overlays outside the cache.
pub(crate) fn load_all_sessions() -> Result<Vec<CodexSessionListing>, String> {
    let base_path_string = get_base_path().ok_or("Codex base path not found")?;
    crate::utils::require_absolute_path(&base_path_string, "Codex base path")?;
    let base_path = PathBuf::from(base_path_string);
    let title_index = load_native_title_index(&base_path.to_string_lossy());
    let old_cache = load_session_metadata_cache(&base_path);
    let mut next_cache = CodexSessionMetadataCache {
        version: SESSION_METADATA_CACHE_VERSION,
        entries: HashMap::new(),
    };
    let roots = [
        (base_path.join("sessions"), false),
        (base_path.join("archived_sessions"), true),
    ];
    let mut groups: HashMap<String, CodexProjectListingGroup> = HashMap::new();

    for (root, is_archived) in roots {
        let is_directory = fs::symlink_metadata(&root)
            .map(|metadata| metadata.file_type().is_dir())
            .unwrap_or(false);
        if !is_directory {
            continue;
        }
        for entry in WalkDir::new(root)
            .min_depth(1)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| is_discoverable_rollout(entry.path()))
        {
            let rollout_path = entry.path();
            let Ok((info, _)) =
                probe_cached_session_info(&base_path, rollout_path, &old_cache, &mut next_cache)
            else {
                continue;
            };
            let project_path = info.cwd.clone().unwrap_or_else(|| "unknown".to_string());
            let project_last_modified = file_modified_rfc3339(rollout_path);
            let session = session_from_info(info, &project_path, &title_index);
            let group =
                groups
                    .entry(project_path.clone())
                    .or_insert_with(|| CodexProjectListingGroup {
                        last_modified: project_last_modified.clone(),
                        sessions: Vec::new(),
                    });
            if project_last_modified > group.last_modified {
                group.last_modified = project_last_modified;
            }
            group.sessions.push(CodexSessionListing {
                session,
                project_path,
                is_archived,
            });
        }
    }

    save_session_metadata_cache(&base_path, &next_cache);
    let mut groups: Vec<CodexProjectListingGroup> = groups.into_values().collect();
    groups.sort_by(|left, right| right.last_modified.cmp(&left.last_modified));
    let mut sessions = Vec::new();
    for mut group in groups {
        group
            .sessions
            .sort_by(|left, right| right.session.last_modified.cmp(&left.session.last_modified));
        sessions.extend(group.sessions);
    }
    Ok(sessions)
}

/// Load one Codex listing row from an exact live rollout path without walking
/// any other rollout. Lexical and canonical confinement plus archive provenance
/// are validated against the active/archive roots, while the exact provider path
/// is retained for cache identity and serialized parity with `load_all_sessions`.
pub(crate) fn load_session_metadata_by_path(
    session_path: &str,
) -> Result<Option<CodexSessionListing>, String> {
    let base_path_string = get_base_path().ok_or("Codex base path not found")?;
    crate::utils::require_absolute_path(&base_path_string, "Codex base path")?;
    let base_path = PathBuf::from(&base_path_string);
    let rollout_path = Path::new(session_path);
    if !rollout_path.is_absolute() {
        return Err("Codex session path must be a non-empty absolute path".to_string());
    }
    let mut root_provenance = None;
    for (root, is_archived) in [
        (base_path.join("sessions"), false),
        (base_path.join("archived_sessions"), true),
    ] {
        let Ok(relative_path) = rollout_path.strip_prefix(&root) else {
            continue;
        };
        let components: Vec<_> = relative_path.components().collect();
        if components.is_empty()
            || components
                .iter()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!(
                "Session path is not an exact Codex rollout path: {session_path}"
            ));
        }
        let mut exact_path = root.clone();
        for component in &components {
            let Component::Normal(name) = component else {
                unreachable!("relative components were validated above");
            };
            exact_path.push(name);
        }
        if exact_path.as_os_str() != rollout_path.as_os_str() {
            return Err(format!(
                "Session path is not an exact Codex rollout path: {session_path}"
            ));
        }

        let root_metadata = match fs::symlink_metadata(&root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "Failed to inspect Codex session directory: {error}"
                ));
            }
        };
        if !root_metadata.file_type().is_dir() || is_symlink_or_reparse(&root_metadata) {
            return Err(format!(
                "Codex session directory is not a direct directory: {}",
                root.display()
            ));
        }

        let mut current = root.clone();
        for (index, component) in components.iter().enumerate() {
            let Component::Normal(name) = component else {
                unreachable!("relative components were validated above");
            };
            current.push(name);
            let metadata = match fs::symlink_metadata(&current) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => {
                    return Err(format!("Failed to inspect session path: {error}"));
                }
            };
            if is_symlink_or_reparse(&metadata) {
                return Err(format!(
                    "Session path contains a symbolic link or reparse point: {session_path}"
                ));
            }
            let is_final = index + 1 == components.len();
            if (is_final && !metadata.file_type().is_file())
                || (!is_final && !metadata.file_type().is_dir())
            {
                return Ok(None);
            }
        }

        let canonical_root = root
            .canonicalize()
            .map_err(|error| format!("Failed to resolve Codex session directory: {error}"))?;
        let canonical_rollout = rollout_path
            .canonicalize()
            .map_err(|error| format!("Failed to resolve session path: {error}"))?;
        if !canonical_rollout.starts_with(&canonical_root) {
            return Err(format!(
                "Session path is outside Codex session directories: {session_path}"
            ));
        }
        root_provenance = Some(is_archived);
        break;
    }
    let Some(is_archived) = root_provenance else {
        return Err(format!(
            "Session path is outside Codex session directories: {session_path}"
        ));
    };
    if !is_discoverable_rollout(rollout_path) {
        return Ok(None);
    }
    let old_cache = load_session_metadata_cache(&base_path);
    let mut next_cache = old_cache.clone();
    next_cache.version = SESSION_METADATA_CACHE_VERSION;
    let (info, cache_updated) =
        probe_cached_session_info(&base_path, rollout_path, &old_cache, &mut next_cache)?;
    if cache_updated {
        if let Some(key) = session_metadata_cache_key(&base_path, rollout_path) {
            if let Some(entry) = next_cache.entries.remove(&key) {
                merge_session_metadata_cache_entry(&base_path, rollout_path, key, entry);
            }
        }
    }
    let project_path = info.cwd.clone().unwrap_or_else(|| "unknown".to_string());
    let title_index = load_native_title_index(&base_path_string);
    let session = session_from_info(info, &project_path, &title_index);
    Ok(Some(CodexSessionListing {
        session,
        project_path,
        is_archived,
    }))
}

fn is_symlink_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
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
                let session_cwd = info.cwd.as_deref().unwrap_or("unknown");
                if session_cwd != target_cwd {
                    continue;
                }
                sessions.push(session_from_info(info, target_cwd, &title_index));
            }
        }
    }

    sessions.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(sessions)
}

/// Derive listing metadata from one already-confined immutable rollout without
/// consulting `CODEX_HOME`, the native title index, or any current live state.
pub(crate) fn load_offline_session_metadata(
    rollout_path: &Path,
) -> Result<(ClaudeSession, Option<String>), String> {
    let info = extract_session_info(rollout_path)?;
    let project_path = info.cwd.clone();
    let project_name = project_path
        .as_deref()
        .and_then(|cwd| Path::new(cwd).file_name())
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Ok((
        ClaudeSession {
            session_id: info.file_path.clone(),
            actual_session_id: info.session_id,
            file_path: info.file_path,
            project_name,
            message_count: info.message_count,
            first_message_time: info.first_message_time,
            last_message_time: info.last_message_time,
            last_modified: info.last_modified,
            has_tool_use: info.has_tool_use,
            has_errors: false,
            summary: info.summary,
            is_renamed: false,
            provider: Some("codex".to_string()),
            storage_type: None,
            entrypoint: info.entrypoint,
            forked_from_id: info.forked_from_id,
            subagent_provenance: info.subagent_provenance,
        },
        project_path,
    ))
}

/// Parse an immutable rollout after the headless offline boundary has confined
/// it to the selected backup payload.
pub(crate) fn load_offline_messages(rollout_path: &Path) -> Result<Vec<ClaudeMessage>, String> {
    if !is_discoverable_rollout(rollout_path) {
        return Err("Offline Codex session is not a supported rollout carrier".to_string());
    }
    parse_rollout_file(rollout_path)
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

pub(crate) fn validate_authorship_audit_path(session_path: &Path) -> Result<PathBuf, String> {
    if !session_path.is_absolute() {
        return Err("Codex audit rollout path must be absolute".to_string());
    }
    let base_path = PathBuf::from(get_base_path().ok_or("Codex base path not found")?);
    for root in [
        base_path.join("sessions"),
        base_path.join("archived_sessions"),
    ] {
        let Ok(relative) = session_path.strip_prefix(&root) else {
            continue;
        };
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err("Codex audit rollout path is not an exact provider path".to_string());
        }
        let root_metadata = fs::symlink_metadata(&root)
            .map_err(|error| format!("Failed to inspect Codex session directory: {error}"))?;
        if !root_metadata.is_dir() || is_symlink_or_reparse(&root_metadata) {
            return Err("Codex audit session directory is not a direct directory".to_string());
        }
        let mut current = root.clone();
        let component_count = relative.components().count();
        for (index, component) in relative.components().enumerate() {
            let Component::Normal(name) = component else {
                unreachable!("relative path components were validated");
            };
            current.push(name);
            let metadata = fs::symlink_metadata(&current)
                .map_err(|error| format!("Failed to inspect Codex audit rollout: {error}"))?;
            if is_symlink_or_reparse(&metadata) {
                return Err(
                    "Codex audit rollout path contains a symbolic link or reparse point"
                        .to_string(),
                );
            }
            let is_final = index + 1 == component_count;
            if (is_final && !metadata.is_file()) || (!is_final && !metadata.is_dir()) {
                return Err("Codex audit rollout path is not a regular file".to_string());
            }
        }
        let canonical_root = root
            .canonicalize()
            .map_err(|error| format!("Failed to resolve Codex session directory: {error}"))?;
        let canonical_path = session_path
            .canonicalize()
            .map_err(|error| format!("Failed to resolve Codex audit rollout: {error}"))?;
        if !canonical_path.starts_with(canonical_root) {
            return Err(
                "Codex audit rollout path is outside provider session directories".to_string(),
            );
        }
        if !is_discoverable_rollout(&canonical_path) {
            return Err("Codex audit path is not a supported rollout carrier".to_string());
        }
        return Ok(canonical_path);
    }
    Err("Codex audit rollout path is outside provider session directories".to_string())
}

pub(crate) fn parse_authorship_audit(
    session_path: &Path,
) -> Result<CodexAuthorshipAuditProjection, String> {
    let canonical_path = validate_authorship_audit_path(session_path)?;
    let bytes = read_rollout_bytes(&canonical_path)?;
    let ranges = find_line_ranges(&bytes);
    let state = CodexParserState::initial(&canonical_path);
    let checkpoint = CodexParserCheckpoint {
        byte_offset: 0,
        replace_from: 0,
        state: state.clone(),
    };
    let outcome = parse_rollout_slice(&bytes, &ranges, state, checkpoint, false)
        .map_err(|()| "Codex audit parse unexpectedly crossed its prefix".to_string())?;
    let session_id = outcome
        .messages
        .first()
        .map(|message| message.session_id.clone())
        .filter(|session_id| !session_id.is_empty() && session_id != "unknown")
        .ok_or("Codex audit rollout did not expose a stable thread id")?;
    Ok(CodexAuthorshipAuditProjection {
        session_id,
        messages: finalize_loaded_messages(outcome.messages),
        diagnostics: outcome.diagnostics,
    })
}

fn parse_rollout_slice(
    bytes: &[u8],
    ranges: &[(usize, usize)],
    mut state: CodexParserState,
    mut checkpoint: CodexParserCheckpoint,
    resumed: bool,
) -> Result<CodexParseOutcome, ()> {
    let mut messages: Vec<ClaudeMessage> = Vec::new();
    let mut diagnostics = Vec::new();
    let slice_base_replace_from = checkpoint.replace_from;
    let mut accepted_len = usize::try_from(checkpoint.byte_offset).map_err(|_| ())?;
    let mut active_turn_id: Option<String> = None;
    let mut active_turn_message_start = 0usize;
    let mut active_turn_message_starts = HashMap::<String, usize>::new();
    let mut active_turn_order = Vec::<String>::new();
    let mut overlap_ambiguous_turns = HashSet::<String>::new();
    let mut inferred_provider_turn_messages = HashSet::<usize>::new();
    let mut authorship_tracker = CodexAuthorshipTracker::default();
    let mut pending_compacted_notification = false;
    let mut pending_fork_rollback: Option<PendingForkRollback> = None;

    for (range_index, &(start, end)) in ranges.iter().enumerate() {
        if start < usize::try_from(checkpoint.byte_offset).map_err(|_| ())? {
            continue;
        }
        let line = &bytes[start..end];
        let mut buf = line.to_vec();
        let source_line = range_index + 1;
        let val: Value = if let Ok(value) = simd_json::from_slice(&mut buf) {
            value
        } else {
            authorship_tracker.invalidate_all(
                &mut diagnostics,
                "malformed-json",
                source_line,
                None,
            );
            continue;
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
        observe_pending_terminal_record(
            &mut authorship_tracker,
            &mut diagnostics,
            line_type,
            event_type,
            val.get("payload"),
            source_line,
        );

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
            // The first session_meta owns the rollout identity. Later metas are
            // replayed history, except that the first return to the child id
            // closes the raw `codex fork` transition and can classify its
            // immediately preceding rollback without re-tagging messages.
            "session_meta" => {
                if let Some(payload) = val.get("payload") {
                    let meta_session_id = non_empty_string(payload.get("id"));
                    if !state.meta_seen {
                        state.meta_seen = true;
                        state.session_id = meta_session_id.unwrap_or("unknown").to_string();
                        state.forked_from_session_id =
                            non_empty_string(payload.get("forked_from_id")).map(str::to_string);
                    } else if state.forked_from_session_id.is_some() && !state.fork_transition_seen
                    {
                        match meta_session_id {
                            Some(id) if id == state.session_id && state.fork_replay_seen => {
                                if let Some(candidate) = pending_fork_rollback.take() {
                                    if candidate.replacement_task_started {
                                        let data = messages[candidate.message_index]
                                            .data
                                            .get_or_insert_with(|| {
                                                Value::Object(serde_json::Map::new())
                                            });
                                        if let Some(data) = data.as_object_mut() {
                                            data.insert(
                                                "rollbackOrigin".to_string(),
                                                Value::String("fork".to_string()),
                                            );
                                        }
                                    }
                                }
                                state.fork_transition_seen = true;
                            }
                            Some(id)
                                if !state.fork_replay_seen
                                    && state.forked_from_session_id.as_deref() == Some(id) =>
                            {
                                state.fork_replay_seen = true;
                                pending_fork_rollback = None;
                            }
                            Some(id) if state.fork_replay_seen && id != state.session_id => {
                                pending_fork_rollback = None;
                            }
                            _ => {}
                        }
                    }
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
                    let hook_prompt_fragments = is_user_message
                        .then(|| codex_hook_prompt_fragments(payload))
                        .flatten();
                    if let Some(msg) = convert_codex_item(
                        payload,
                        &state.session_id,
                        state.current_inference.model.as_ref(),
                        &line_timestamp,
                        &mut state.msg_counter,
                    ) {
                        let mut msg = msg;
                        let explicit_provider_turn_id = codex_authored_turn_id(payload);
                        let exact_provider_turn_id =
                            explicit_provider_turn_id.as_deref().or_else(|| {
                                (active_turn_order.len() == 1
                                    && !overlap_ambiguous_turns.contains(&active_turn_order[0]))
                                .then(|| active_turn_order[0].as_str())
                            });
                        let provider_turn_id_was_inferred =
                            explicit_provider_turn_id.is_none() && exact_provider_turn_id.is_some();
                        merge_codex_message_provenance(&mut msg, exact_provider_turn_id, None);
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
                        if provider_turn_id_was_inferred {
                            inferred_provider_turn_messages.insert(messages.len() - 1);
                        }
                        if is_user_message {
                            let message_index = messages.len() - 1;
                            if let Some(fragments) = hook_prompt_fragments {
                                messages[message_index].subtype =
                                    Some(HOOK_PROMPT_SUBTYPE.to_string());
                                messages[message_index].content = Some(Value::Array(
                                    fragments
                                        .iter()
                                        .map(|fragment| {
                                            serde_json::json!({
                                                "type": "text",
                                                "text": fragment.text
                                            })
                                        })
                                        .collect(),
                                ));
                                let data = messages[message_index]
                                    .data
                                    .get_or_insert_with(|| Value::Object(serde_json::Map::new()));
                                if let Some(data) = data.as_object_mut() {
                                    data.insert(
                                        "hookPromptFragments".to_string(),
                                        Value::Array(
                                            fragments
                                                .into_iter()
                                                .map(|fragment| {
                                                    serde_json::json!({
                                                        "text": fragment.text,
                                                        "hookRunId": fragment.hook_run_id
                                                    })
                                                })
                                                .collect(),
                                        ),
                                    );
                                }
                            } else {
                                // Ordinary user-role response items are model-input records,
                                // not canonical visible history. A matching canonical user
                                // event may reuse the row to preserve richer content and
                                // artifacts.
                                messages[message_index].subtype =
                                    Some(INJECTED_CONTEXT_SUBTYPE.to_string());
                                authorship_tracker.push_candidate(PendingCodexUserMessage {
                                    message_index,
                                    message_id: messages[message_index].uuid.clone(),
                                    source_line,
                                    response_text: codex_user_response_text(payload),
                                    authored_turn_id: explicit_provider_turn_id,
                                    precedes_input_boundary: false,
                                    terminal_evidence: TerminalContextEvidence::AwaitingBoundary,
                                });
                            }
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

                    // Canonical user events define visible history. Reuse an exact raw
                    // response-item match when available, otherwise synthesize the visible
                    // message directly from the canonical event.
                    let is_completed_user_item = event_type == "item_completed"
                        && payload
                            .get("item")
                            .and_then(|item| item.get("type"))
                            .and_then(Value::as_str)
                            == Some("UserMessage");
                    if event_type == "user_message" || is_completed_user_item {
                        project_canonical_user_event(
                            &mut messages,
                            &mut authorship_tracker,
                            payload,
                            &mut CanonicalCodexUserProjectionContext {
                                session_id: &state.session_id,
                                line_timestamp: &line_timestamp,
                                counter: &mut state.msg_counter,
                                fallback_turn_id: active_turn_id.as_deref(),
                            },
                            &mut diagnostics,
                            source_line,
                        );
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
                        if let Some(turn_id) = active_turn_id.as_deref() {
                            let overlapping_turns = active_turn_order
                                .iter()
                                .filter(|active_turn_id| active_turn_id.as_str() != turn_id)
                                .cloned()
                                .collect::<Vec<_>>();
                            if !overlapping_turns.is_empty() {
                                for message_index in inferred_provider_turn_messages.iter().copied()
                                {
                                    let Some(message) = messages.get_mut(message_index) else {
                                        continue;
                                    };
                                    let belongs_to_overlapping_turn = message
                                        .data
                                        .as_ref()
                                        .and_then(|data| data.get("providerTurnId"))
                                        .and_then(Value::as_str)
                                        .is_some_and(|provider_turn_id| {
                                            overlapping_turns
                                                .iter()
                                                .any(|turn_id| turn_id == provider_turn_id)
                                        });
                                    if belongs_to_overlapping_turn {
                                        clear_inferred_codex_turn_provenance(message);
                                    }
                                }
                                overlap_ambiguous_turns.insert(turn_id.to_string());
                                overlap_ambiguous_turns.extend(overlapping_turns);
                            }
                            authorship_tracker.start_turn(turn_id, &mut diagnostics, source_line);
                            active_turn_message_starts
                                .insert(turn_id.to_string(), active_turn_message_start);
                            active_turn_order.retain(|active_turn_id| active_turn_id != turn_id);
                            active_turn_order.push(turn_id.to_string());
                        }
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
                    }

                    if event_type == "image_generation_end" {
                        let explicit_provider_turn_id = non_empty_string(payload.get("turn_id"));
                        let exact_provider_turn_id =
                            explicit_provider_turn_id.map(str::to_string).or_else(|| {
                                (active_turn_order.len() == 1
                                    && !overlap_ambiguous_turns.contains(&active_turn_order[0]))
                                .then(|| active_turn_order[0].clone())
                            });
                        if let Some(provider_turn_id) = exact_provider_turn_id {
                            if let Some(msg) = convert_codex_image_generation_event(
                                payload,
                                &state.session_id,
                                &line_timestamp,
                                &mut state.msg_counter,
                                &provider_turn_id,
                            ) {
                                messages.push(msg);
                                if explicit_provider_turn_id.is_none() {
                                    inferred_provider_turn_messages.insert(messages.len() - 1);
                                }
                            }
                        }
                    } else if event_type == "token_count" {
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

                        // A cumulative snapshot can repeat without any new model output,
                        // especially in the bookkeeping sequence around compaction. It
                        // does not describe a zero-token visible assistant response. Keep
                        // any input/cache delta pending until output advances, then attach
                        // the combined invocation usage to the actual assistant record.
                        state.pending_usage.accumulate(delta);
                        if delta.output == 0 {
                            continue;
                        }

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
                            state.pending_usage = CodexTokenUsage::default();
                            continue;
                        };
                        let delta = std::mem::take(&mut state.pending_usage);

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
                        let plan_type = payload
                            .get("rate_limits")
                            .and_then(|limits| limits.get("plan_type"))
                            .and_then(Value::as_str);
                        let cost = codex_credit_estimate(inference, delta, plan_type);
                        inference.cost = cost;
                    } else if let Some(mut msg) = convert_codex_event(
                        payload,
                        &state.session_id,
                        &line_timestamp,
                        &mut state.msg_counter,
                    ) {
                        merge_codex_message_provenance(
                            &mut msg,
                            non_empty_string(payload.get("turn_id")),
                            None,
                        );
                        if msg.message_type == "assistant" {
                            if msg.model.is_none() {
                                msg.model.clone_from(&state.current_inference.model);
                            }
                            msg.inference = (!state.current_inference.is_empty())
                                .then(|| state.current_inference.clone());
                        }
                        messages.push(msg);
                        if event_type == "thread_rolled_back"
                            && state.forked_from_session_id.is_some()
                            && state.fork_replay_seen
                            && !state.fork_transition_seen
                        {
                            pending_fork_rollback = Some(PendingForkRollback {
                                message_index: messages.len() - 1,
                                replacement_task_started: false,
                            });
                        }
                    }

                    if event_type == "task_started"
                        && non_empty_string(payload.get("turn_id")).is_some()
                    {
                        if let Some(candidate) = pending_fork_rollback.as_mut() {
                            candidate.replacement_task_started = true;
                        }
                    } else if matches!(event_type, "task_complete" | "turn_aborted") {
                        pending_fork_rollback = None;
                    }

                    if matches!(event_type, "task_complete" | "turn_aborted") {
                        if let Some(completed_turn_id) = payload
                            .get("turn_id")
                            .and_then(Value::as_str)
                            .filter(|turn_id| !turn_id.is_empty())
                        {
                            let completion_allows_turnless_fallback =
                                !overlap_ambiguous_turns.contains(completed_turn_id);
                            let key = CodexAuthorshipLaneKey::Turn(completed_turn_id.to_string());
                            if let Some(lane) = authorship_tracker.lanes.remove(&key) {
                                if lane.active {
                                    classify_pending_terminal_context(
                                        &mut messages,
                                        &lane.pending_user_messages,
                                        lane.authored_user_count,
                                        completed_turn_id,
                                        payload,
                                        &mut diagnostics,
                                        source_line,
                                    );
                                }
                            }
                            let completed_message_start =
                                active_turn_message_starts.remove(completed_turn_id);
                            active_turn_order.retain(|turn_id| turn_id != completed_turn_id);
                            if event_type == "task_complete" {
                                if let Some(completed_message_start) = completed_message_start {
                                    if let Some(last_msg) = messages[completed_message_start..]
                                        .iter_mut()
                                        .rev()
                                        .find(|message| {
                                            message.message_type == "assistant"
                                                && message
                                                    .data
                                                    .as_ref()
                                                    .and_then(|data| data.get("providerTurnId"))
                                                    .and_then(Value::as_str)
                                                    .map_or(
                                                        completion_allows_turnless_fallback,
                                                        |turn_id| turn_id == completed_turn_id,
                                                    )
                                        })
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
                                }
                            }
                            if active_turn_id.as_deref() == Some(completed_turn_id) {
                                active_turn_id = active_turn_order.last().cloned();
                                active_turn_message_start = active_turn_id
                                    .as_deref()
                                    .and_then(|turn_id| active_turn_message_starts.get(turn_id))
                                    .copied()
                                    .unwrap_or(messages.len());
                            }
                        }
                        if authorship_tracker.is_quiescent() && pending_fork_rollback.is_none() {
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

    for lane in authorship_tracker.lanes.values() {
        for candidate in &lane.pending_user_messages {
            diagnose_unresolved_candidate(
                candidate,
                &mut diagnostics,
                "unresolved-at-eof",
                candidate.source_line,
            );
        }
    }

    Ok(CodexParseOutcome {
        messages,
        diagnostics,
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
    #[cfg(test)]
    SESSION_INFO_PARSE_COUNT.fetch_add(1, Ordering::SeqCst);
    let mmap = read_rollout_bytes(rollout_path)?;
    let ranges = find_line_ranges(&mmap);

    let mut session_id = String::new();
    let mut meta_seen = false;
    let mut cwd = None;
    let mut source = None;
    let mut forked_from_id = None;
    let mut subagent_provenance = None;
    let mut turn_context_cwd = None;
    let mut model = None;
    let mut message_count = 0usize;
    let mut first_time = String::new();
    let mut last_time = String::new();
    let mut has_tool_use = false;
    let mut authored_summary = None;
    let mut fallback_summary = None;

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
                    subagent_provenance =
                        codex_subagent_provenance(val.get("timestamp"), payload.get("source"));
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
            "event_msg" => {
                if authored_summary.is_none() {
                    let payload = val.get("payload");
                    if payload
                        .and_then(|item| item.get("type"))
                        .and_then(Value::as_str)
                        == Some("user_message")
                    {
                        authored_summary = payload
                            .and_then(|item| item.get("message"))
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                            .map(truncate_preview_text);
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

                        // Event-less legacy or incomplete rollouts cannot prove
                        // authorship. Keep their first user-role item as a fail-open
                        // fallback; a structurally authored event wins when present.
                        if fallback_summary.is_none() {
                            if let Some(role) = payload.get("role").and_then(|r| r.as_str()) {
                                if role == "user" {
                                    if let Some(text) = extract_text_from_content(payload) {
                                        fallback_summary = Some(text);
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
        subagent_provenance,
        model,
        message_count,
        first_message_time: first_time,
        last_message_time: last_time,
        last_modified,
        file_path: rollout_path.to_string_lossy().to_string(),
        has_tool_use,
        summary: authored_summary.or(fallback_summary),
    })
}

fn truncate_preview_text(text: &str) -> String {
    match text.char_indices().nth(200) {
        Some((idx, _)) => format!("{}...", &text[..idx]),
        None => text.to_string(),
    }
}

fn extract_text_from_content(item: &Value) -> Option<String> {
    let content = item.get("content")?.as_array()?;
    for c in content {
        let ctype = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if ctype == "input_text" || ctype == "output_text" || ctype == "text" {
            if let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                return Some(truncate_preview_text(text));
            }
        }
    }
    None
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
                "text": "[interrupted]"
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
            msg.subtype = Some("interruption".to_string());
            msg.level = Some("warning".to_string());
            msg.duration_ms = payload.get("duration_ms").and_then(Value::as_u64);
            let mut data = serde_json::Map::new();
            if !turn_id.is_empty() {
                data.insert(
                    "providerTurnId".to_string(),
                    Value::String(turn_id.to_string()),
                );
            }
            data.insert("reason".to_string(), Value::String(reason.to_string()));
            if let Some(started_at) = payload.get("started_at").and_then(Value::as_u64) {
                data.insert("startedAt".to_string(), Value::Number(started_at.into()));
            }
            if let Some(completed_at) = payload.get("completed_at").and_then(Value::as_u64) {
                data.insert(
                    "completedAt".to_string(),
                    Value::Number(completed_at.into()),
                );
            }
            if let Some(duration_ms) = msg.duration_ms {
                data.insert("durationMs".to_string(), Value::Number(duration_ms.into()));
            }
            msg.data = Some(Value::Object(data));
            Some(msg)
        }
        // Unsupported/duplicated Codex events are intentionally ignored.
        _ => None,
    }
}

fn convert_codex_image_generation_event(
    payload: &Value,
    session_id: &str,
    line_timestamp: &str,
    counter: &mut u64,
    provider_turn_id: &str,
) -> Option<ClaudeMessage> {
    if payload.get("status").and_then(Value::as_str) != Some("completed") {
        return None;
    }
    let call_id = non_empty_string(payload.get("call_id"))?;
    let encoded = non_empty_string(payload.get("result"))?;
    let bytes = BASE64_STANDARD.decode(encoded).ok()?;
    let media_type = detected_image_media_type(&bytes)?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));

    *counter += 1;
    let mut message = build_codex_message(
        format!("codex-image-generation-{counter}"),
        session_id,
        line_timestamp.to_string(),
        "progress",
        None,
        Some(serde_json::json!([{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded
            }
        }])),
        None,
    );
    message.subtype = Some("provider_rendered_image".to_string());
    merge_codex_message_provenance(&mut message, Some(provider_turn_id), None);
    let source_message_uuid = message.uuid.clone();
    append_codex_image_artifacts(
        &mut message,
        vec![serde_json::json!({
            "version": 1,
            "artifactId": format!("codex:{call_id}:canvas:0:{sha256}"),
            "providerTurnId": provider_turn_id,
            "sourceMessageUuid": source_message_uuid,
            "toolCallId": call_id,
            "sourceContentIndex": 0,
            "sourceKind": "provider_rendered_image",
            "presentationKind": "canvas",
            "mediaType": media_type,
            "byteLength": bytes.len(),
            "sha256": sha256
        })],
    );
    Some(message)
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct CodexTokenUsage {
    input: u32,
    output: u32,
    cached: Option<u32>,
    cache_write: Option<u32>,
    reasoning: Option<u32>,
}

impl CodexTokenUsage {
    fn accumulate(&mut self, usage: Self) {
        let accumulate_optional = |current: &mut Option<u32>, incoming: Option<u32>| {
            if let Some(incoming) = incoming {
                *current = Some(current.unwrap_or(0).saturating_add(incoming));
            }
        };

        self.input = self.input.saturating_add(usage.input);
        self.output = self.output.saturating_add(usage.output);
        accumulate_optional(&mut self.cached, usage.cached);
        accumulate_optional(&mut self.cache_write, usage.cache_write);
        accumulate_optional(&mut self.reasoning, usage.reasoning);
    }
}

#[derive(Debug, Clone, Copy)]
struct CodexCreditRates {
    input: f64,
    cached_input: f64,
    output: f64,
    fast_multiplier: f64,
}

fn codex_credit_rates(model: &str) -> Option<CodexCreditRates> {
    let rates = match model {
        "gpt-5.6-sol" | "gpt-5.5" => (125.0, 12.5, 750.0, 2.5),
        "gpt-5.6-terra" => (50.0, 5.0, 300.0, 2.5),
        "gpt-5.6-luna" => (5.0, 0.5, 30.0, 2.5),
        "gpt-5.4" => (62.5, 6.25, 375.0, 2.0),
        "gpt-5.4-mini" => (18.75, 1.875, 113.0, 2.0),
        _ => return None,
    };
    Some(CodexCreditRates {
        input: rates.0,
        cached_input: rates.1,
        output: rates.2,
        fast_multiplier: rates.3,
    })
}

/// Estimate `ChatGPT` credits for one Codex inference only when the rollout
/// records every billing discriminator required by the published rate card.
/// API-key/custom providers and unknown models or tiers deliberately remain
/// unset instead of receiving a guessed value.
fn codex_credit_estimate(
    inference: &InferenceMetadata,
    usage: CodexTokenUsage,
    plan_type: Option<&str>,
) -> Option<InferenceCost> {
    plan_type.filter(|plan| !plan.trim().is_empty())?;
    if inference.model_provider.as_deref() != Some("openai") {
        return None;
    }
    let rates = codex_credit_rates(inference.model.as_deref()?)?;
    let multiplier = match inference.service_tier.as_deref()? {
        "default" | "standard" => 1.0,
        // Codex configuration's `fast` tier is recorded on requests as
        // `priority`; accept both representations at the normalization edge.
        "fast" | "priority" => rates.fast_multiplier,
        _ => return None,
    };

    // OpenAI input_tokens includes cached_input_tokens as a subset. Reasoning
    // tokens likewise remain a detail of output_tokens and must not be charged
    // a second time.
    let cached = usage.cached.unwrap_or(0).min(usage.input);
    let non_cached = usage.input.saturating_sub(cached);
    let value = (f64::from(non_cached) * rates.input
        + f64::from(cached) * rates.cached_input
        + f64::from(usage.output) * rates.output)
        * multiplier
        / 1_000_000.0;
    Some(InferenceCost {
        value,
        unit: "credits".to_string(),
        kind: "estimated".to_string(),
        rate_card_version: CODEX_CREDIT_RATE_CARD_VERSION.to_string(),
    })
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
        if let Some(tool_use) = matching_tool_use(prev, &tool_use_id).cloned() {
            let tool_name = tool_use
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if tool_name == "apply_patch" && failed_apply_patch_result(&tool_result_block) {
                if let Some(result) = tool_result_block.as_object_mut() {
                    result.insert("is_error".to_string(), Value::Bool(true));
                }
            }
            let image_artifacts = codex_tool_result_image_artifacts(
                prev,
                &tool_use_id,
                &tool_use,
                &tool_result_block,
            );
            append_codex_image_artifacts(prev, image_artifacts);
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

fn matching_tool_use<'a>(msg: &'a ClaudeMessage, tool_use_id: &str) -> Option<&'a Value> {
    let arr = msg.content.as_ref().and_then(Value::as_array)?;
    arr.iter().find(|item| {
        item.get("type").and_then(Value::as_str) == Some("tool_use")
            && item.get("id").and_then(Value::as_str) == Some(tool_use_id)
    })
}

fn codex_tool_result_image_artifacts(
    message: &ClaudeMessage,
    tool_use_id: &str,
    tool_use: &Value,
    tool_result: &Value,
) -> Vec<Value> {
    let Some(provider_turn_id) = message
        .data
        .as_ref()
        .and_then(|data| data.get("providerTurnId"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        // Tool records in overlapping tasks can lack an authoritative owner.
        // Retain their original structural payload, but do not expose an image
        // artifact that a consumer could attribute to the wrong turn.
        return Vec::new();
    };
    let Some(content) = tool_result.get("content").and_then(Value::as_array) else {
        return Vec::new();
    };
    let image_blocks = content
        .iter()
        .enumerate()
        .filter_map(|(content_index, block)| {
            let block_type = block.get("type").and_then(Value::as_str)?;
            if !matches!(block_type, "input_image" | "image" | "local_image") {
                return None;
            }
            Some((content_index, block))
        })
        .collect::<Vec<_>>();
    let correlations = codex_view_image_locators(tool_use, image_blocks.len());

    image_blocks
        .into_iter()
        .enumerate()
        .filter_map(|(image_index, (content_index, block))| {
            let source = block
                .get("image_url")
                .or_else(|| block.get("url"))
                .and_then(Value::as_str)
                .or_else(|| {
                    block
                        .get("source")
                        .and_then(|source| source.get("url"))
                        .and_then(Value::as_str)
                })?;
            let (media_type, bytes) = decode_image_data_url(source)?;
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            let mut artifact = serde_json::json!({
                "version": 1,
                "artifactId": format!("codex:{tool_use_id}:{content_index}:{sha256}"),
                "providerTurnId": provider_turn_id,
                "sourceMessageUuid": message.uuid,
                "toolCallId": tool_use_id,
                "toolResultContentIndex": content_index,
                "sourceKind": "tool_result_image",
                "mediaType": media_type,
                "byteLength": bytes.len(),
                "sha256": sha256
            });
            if let Some(locator) = correlations.get(image_index).and_then(Option::as_ref) {
                artifact.as_object_mut()?.insert(
                    "correlation".to_string(),
                    serde_json::json!({ "kind": "local_path", "value": locator }),
                );
            }
            Some(artifact)
        })
        .collect()
}

fn append_codex_image_artifacts(message: &mut ClaudeMessage, artifacts: Vec<Value>) {
    if artifacts.is_empty() {
        return;
    }
    let data = message
        .data
        .get_or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(data) = data.as_object_mut() else {
        return;
    };
    let target = data
        .entry("imageArtifacts")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(target) = target.as_array_mut() else {
        return;
    };
    for artifact in artifacts {
        let id = artifact.get("artifactId").and_then(Value::as_str);
        if id.is_some_and(|id| {
            target
                .iter()
                .any(|existing| existing.get("artifactId").and_then(Value::as_str) == Some(id))
        }) {
            continue;
        }
        target.push(artifact);
    }
}

fn decode_image_data_url(source: &str) -> Option<(&str, Vec<u8>)> {
    let (metadata, encoded) = source.strip_prefix("data:")?.split_once(',')?;
    let mut parts = metadata.split(';');
    let declared_media_type = parts.next()?.to_ascii_lowercase();
    if !declared_media_type.starts_with("image/") || !parts.any(|part| part == "base64") {
        return None;
    }
    let bytes = BASE64_STANDARD.decode(encoded).ok()?;
    let media_type = detected_image_media_type(&bytes)?;
    let declared_matches = declared_media_type == media_type
        || (declared_media_type == "image/jpg" && media_type == "image/jpeg");
    declared_matches.then_some((media_type, bytes))
}

fn detected_image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

#[cfg(test)]
fn codex_view_image_locator(tool_use: &Value) -> Option<String> {
    codex_view_image_locators(tool_use, 1)
        .into_iter()
        .next()
        .flatten()
}

fn codex_view_image_locators(tool_use: &Value, image_count: usize) -> Vec<Option<String>> {
    let unmatched = || vec![None; image_count];
    if image_count == 0 {
        return Vec::new();
    }
    let Some(name) = tool_use.get("name").and_then(Value::as_str) else {
        return unmatched();
    };
    let Some(input) = tool_use.get("input") else {
        return unmatched();
    };
    if name == "view_image" {
        if image_count != 1 {
            return unmatched();
        }
        return vec![input
            .get("path")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)];
    }
    if name != "exec" {
        return unmatched();
    }
    let Some(source) = input.get("input").and_then(Value::as_str) else {
        return unmatched();
    };
    if let Some(locators) = codex_batched_view_image_locators(source) {
        return if locators.len() == image_count {
            locators.into_iter().map(Some).collect()
        } else {
            unmatched()
        };
    }
    if image_count != 1 {
        return unmatched();
    }
    vec![codex_single_exec_view_image_locator(source)]
}

fn codex_single_exec_view_image_locator(source: &str) -> Option<String> {
    const CALL: &str = "tools.view_image";
    let mut calls = source.match_indices(CALL);
    let (call_start, matched) = calls.next()?;
    if calls.next().is_some() {
        return None;
    }
    let call = &source[call_start + matched.len()..];
    let object_start = call.find('{')?;
    let object = &call[object_start + 1..];
    let (key_start, key_len) = object
        .find("\"path\"")
        .map(|index| (index, "\"path\"".len()))
        .or_else(|| object.find("path").map(|index| (index, "path".len())))?;
    let after_key = object[key_start + key_len..].trim_start();
    let after_colon = after_key.strip_prefix(':')?.trim_start();
    let literal_end = json_string_literal_end(after_colon)?;
    serde_json::from_str::<String>(&after_colon[..literal_end])
        .ok()
        .filter(|value| !value.is_empty())
}

fn codex_batched_view_image_locators(source: &str) -> Option<Vec<String>> {
    const PREFIX: &str = "const [";
    const PROMISE: &str = "] = await Promise.all([";
    let source = source.trim();
    let bindings = source.strip_prefix(PREFIX)?;
    let promise_index = bindings.find(PROMISE)?;
    let names = bindings[..promise_index]
        .split(',')
        .map(str::trim)
        .map(|name| is_javascript_identifier(name).then_some(name))
        .collect::<Option<Vec<_>>>()?;
    if names.len() < 2 {
        return None;
    }

    let array_and_tail = &bindings[promise_index + PROMISE.len()..];
    let array_end = matching_square_bracket(array_and_tail)?;
    let calls = split_top_level_commas(&array_and_tail[..array_end])?;
    if calls.len() != names.len() {
        return None;
    }
    let paths = calls
        .iter()
        .map(|call| {
            let call = call.trim();
            let invocation = call.strip_prefix("tools.view_image")?.trim_start();
            if !invocation.starts_with('(') || !invocation.ends_with(')') {
                return None;
            }
            codex_view_image_path_from_call(invocation)
        })
        .collect::<Option<Vec<_>>>()?;

    let mut tail = array_and_tail[array_end + 1..].trim_start();
    tail = tail.strip_prefix(");")?.trim_start();
    let mut emitted = Vec::new();
    while !tail.is_empty() {
        let after_image = tail.strip_prefix("image")?.trim_start();
        let after_open = after_image.strip_prefix('(')?;
        let close = after_open.find(')')?;
        let expression = after_open[..close].trim();
        let name = expression.strip_suffix(".image_url")?.trim();
        if !is_javascript_identifier(name) {
            return None;
        }
        emitted.push(name);
        tail = after_open[close + 1..]
            .trim_start()
            .strip_prefix(';')?
            .trim_start();
    }
    if emitted.len() != names.len() {
        return None;
    }

    let mut locators = Vec::with_capacity(emitted.len());
    let mut seen = std::collections::HashSet::new();
    for emitted_name in emitted {
        if !seen.insert(emitted_name) {
            return None;
        }
        let binding_index = names.iter().position(|name| *name == emitted_name)?;
        locators.push(paths[binding_index].clone());
    }
    (seen.len() == names.len()).then_some(locators)
}

fn codex_view_image_path_from_call(call: &str) -> Option<String> {
    let invocation = call.strip_prefix('(')?.strip_suffix(')')?.trim();
    let object = invocation.strip_prefix('{')?.strip_suffix('}')?;
    let mut path = None;
    for property in split_top_level_commas(object)? {
        let property = property.trim();
        let after_key = if let Some(value) = property.strip_prefix("\"path\"") {
            value
        } else if let Some(value) = property.strip_prefix("path") {
            value
        } else {
            continue;
        };
        let Some(after_colon) = after_key.trim_start().strip_prefix(':') else {
            continue;
        };
        let value = after_colon.trim_start();
        let literal_end = json_string_literal_end(value)?;
        if !value[literal_end..].trim().is_empty() {
            return None;
        }
        let value = serde_json::from_str::<String>(&value[..literal_end])
            .ok()
            .filter(|value| !value.is_empty())?;
        if path.replace(value).is_some() {
            return None;
        }
    }
    path
}

fn is_javascript_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first == '$' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
}

fn matching_square_bracket(value: &str) -> Option<usize> {
    let mut depth = 1usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in value.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
            continue;
        }
        match ch {
            '"' => quoted = true,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_commas(value: &str) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut round = 0usize;
    let mut curly = 0usize;
    let mut square = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in value.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
            continue;
        }
        match ch {
            '"' => quoted = true,
            '(' => round += 1,
            ')' => round = round.checked_sub(1)?,
            '{' => curly += 1,
            '}' => curly = curly.checked_sub(1)?,
            '[' => square += 1,
            ']' => square = square.checked_sub(1)?,
            ',' if round == 0 && curly == 0 && square == 0 => {
                parts.push(&value[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if quoted || round != 0 || curly != 0 || square != 0 {
        return None;
    }
    parts.push(&value[start..]);
    parts
        .iter()
        .all(|part| !part.trim().is_empty())
        .then_some(parts)
}

fn json_string_literal_end(value: &str) -> Option<usize> {
    let bytes = value.as_bytes();
    if bytes.first() != Some(&b'"') {
        return None;
    }
    let mut escaped = false;
    for (index, byte) in bytes.iter().enumerate().skip(1) {
        if escaped {
            escaped = false;
        } else if *byte == b'\\' {
            escaped = true;
        } else if *byte == b'"' {
            return Some(index + 1);
        }
    }
    None
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
                "input_image" | "image" | "local_image" => {
                    let image_url = item
                        .get("image_url")
                        .or_else(|| item.get("url"))
                        .or_else(|| item.get("path"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
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
    use std::sync::{atomic::Ordering, Arc, Barrier};
    use tempfile::TempDir;

    #[cfg(unix)]
    fn try_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn try_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
    }

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

    fn message_data_str<'a>(message: &'a ClaudeMessage, key: &str) -> Option<&'a str> {
        message
            .data
            .as_ref()
            .and_then(|data| data.get(key))
            .and_then(Value::as_str)
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

    #[test]
    #[serial]
    fn all_sessions_rejects_relative_codex_home() {
        let current_dir = std::env::current_dir().expect("current dir should resolve");
        let temp = tempfile::tempdir_in(current_dir).expect("relative temp dir should be created");
        let relative_home = Path::new(
            temp.path()
                .file_name()
                .expect("relative temp dir should have a name"),
        );
        let _guard = EnvVarGuard::set("CODEX_HOME", relative_home);

        let error = match load_all_sessions() {
            Ok(_) => panic!("relative Codex home should be rejected"),
            Err(error) => error,
        };
        assert_eq!(error, "Codex base path must be a non-empty absolute path");
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_rejects_a_rollout_outside_codex_home() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        fs::create_dir_all(codex_home.join("sessions")).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let outside = write_codex_rollout(
            tmp.path(),
            "rollout-outside.jsonl",
            "outside",
            "C:/Outside",
            "outside prompt",
        );

        let error = match load_session_metadata_by_path(outside.to_str().unwrap()) {
            Ok(_) => panic!("outside rollout should be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("outside Codex session directories"));
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_rejects_parent_directory_aliases() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout = write_codex_rollout(
            &sessions_dir,
            "rollout-parent-alias.jsonl",
            "parent-alias",
            "C:/Repo",
            "parent alias prompt",
        );
        let aliased = sessions_dir
            .join("..")
            .join("sessions")
            .join(rollout.file_name().unwrap());

        let error = match load_session_metadata_by_path(aliased.to_str().unwrap()) {
            Ok(_) => panic!("parent-directory alias should be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("not an exact Codex rollout path"));
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_rejects_dot_and_redundant_separator_aliases() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        let nested_dir = sessions_dir.join("2026");
        fs::create_dir_all(&nested_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout = write_codex_rollout(
            &nested_dir,
            "rollout-lexical-alias.jsonl",
            "lexical-alias",
            "C:/Repo",
            "lexical alias prompt",
        );
        let dot_alias = sessions_dir
            .join("2026")
            .join(".")
            .join(rollout.file_name().unwrap());
        let redundant_separator_alias = format!(
            "{}{}{}2026{}{}",
            sessions_dir.display(),
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR,
            rollout.file_name().unwrap().to_string_lossy()
        );

        for alias in [
            dot_alias.to_string_lossy().to_string(),
            redundant_separator_alias,
        ] {
            let error = match load_session_metadata_by_path(&alias) {
                Ok(_) => panic!("lexical alias should be rejected: {alias}"),
                Err(error) => error,
            };
            assert!(error.contains("not an exact Codex rollout path"));
        }
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_rejects_an_outside_symlink_alias() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout = write_codex_rollout(
            &sessions_dir,
            "rollout-outside-alias.jsonl",
            "outside-alias",
            "C:/Repo",
            "outside alias prompt",
        );
        let alias = tmp.path().join("sessions-alias");
        if try_symlink_dir(&sessions_dir, &alias).is_err() {
            return;
        }
        let aliased = alias.join(rollout.file_name().unwrap());

        let error = match load_session_metadata_by_path(aliased.to_str().unwrap()) {
            Ok(_) => panic!("outside symlink alias should be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("outside Codex session directories"));
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_rejects_an_intermediate_symlink() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        let real_dir = sessions_dir.join("real");
        fs::create_dir_all(&real_dir).expect("real sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout = write_codex_rollout(
            &real_dir,
            "rollout-intermediate-alias.jsonl",
            "intermediate-alias",
            "C:/Repo",
            "intermediate alias prompt",
        );
        let alias = sessions_dir.join("alias");
        if try_symlink_dir(&real_dir, &alias).is_err() {
            return;
        }
        let aliased = alias.join(rollout.file_name().unwrap());

        let error = match load_session_metadata_by_path(aliased.to_str().unwrap()) {
            Ok(_) => panic!("intermediate symlink should be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("symbolic link or reparse point"));
    }

    #[test]
    #[serial]
    fn all_session_metadata_cache_reuses_stable_rollouts_and_refreshes_changes() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home
            .join("sessions")
            .join("2026")
            .join("08")
            .join("07");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let alpha = write_codex_rollout(
            &sessions_dir,
            "rollout-alpha.jsonl",
            "alpha",
            "C:/Repo",
            "alpha prompt",
        );
        let beta = write_codex_rollout(
            &sessions_dir,
            "rollout-beta.jsonl",
            "beta",
            "C:/Other",
            "beta prompt",
        );
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);

        let cold = load_all_sessions().expect("cold listing should succeed");
        assert_eq!(cold.len(), 2);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 2);
        assert!(metadata_cache_path(&codex_home).is_file());

        let warm = load_all_sessions().expect("warm listing should succeed");
        assert_eq!(warm.len(), 2);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 2);

        append_rollout_lines(
            &alpha,
            &[json!({
                "timestamp": "2026-08-07T10:00:01Z",
                "type": "response_item",
                "payload": { "type": "function_call", "name": "shell", "arguments": "{}" }
            })],
        );
        let changed = load_all_sessions().expect("changed listing should succeed");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 3);
        let alpha_row = changed
            .iter()
            .find(|row| row.session.actual_session_id == "alpha")
            .expect("changed alpha row should remain listed");
        assert_eq!(alpha_row.session.message_count, 2);
        assert!(alpha_row.session.has_tool_use);

        fs::remove_file(beta).expect("beta rollout should be removed");
        let after_delete = load_all_sessions().expect("listing after delete should succeed");
        assert_eq!(after_delete.len(), 1);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 3);
        let cache: CodexSessionMetadataCache = serde_json::from_slice(
            &fs::read(metadata_cache_path(&codex_home)).expect("cache should be readable"),
        )
        .expect("cache should deserialize");
        assert_eq!(cache.entries.len(), 1);

        let archived_dir = codex_home.join("archived_sessions");
        fs::create_dir_all(&archived_dir).expect("archive dir should be created");
        let archived_alpha = archived_dir.join("rollout-alpha.jsonl");
        fs::rename(alpha, &archived_alpha).expect("alpha should move to archive root");
        assert!(load_session_metadata_by_path(
            sessions_dir.join("rollout-alpha.jsonl").to_str().unwrap()
        )
        .expect("old active path should be a clean miss")
        .is_none());
        let targeted_archive = load_session_metadata_by_path(archived_alpha.to_str().unwrap())
            .expect("archived target should load")
            .expect("archived target should remain listed");
        assert!(targeted_archive.is_archived);
        let after_archive = load_all_sessions().expect("archived listing should succeed");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 4);
        assert_eq!(after_archive.len(), 1);
        assert!(after_archive[0].is_archived);
        assert_eq!(
            after_archive[0].session.file_path,
            archived_alpha.to_string_lossy()
        );
        let cache: CodexSessionMetadataCache = serde_json::from_slice(
            &fs::read(metadata_cache_path(&codex_home)).expect("cache should be readable"),
        )
        .expect("cache should deserialize");
        assert_eq!(cache.entries.len(), 1);
        assert!(cache
            .entries
            .keys()
            .all(|key| key.starts_with("archived_sessions/")));
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_refreshes_only_its_rollout_and_preserves_the_cache() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let alpha = write_codex_rollout(
            &sessions_dir,
            "rollout-alpha.jsonl",
            "alpha",
            "C:/Repo",
            "alpha prompt",
        );
        write_codex_rollout(
            &sessions_dir,
            "rollout-beta.jsonl",
            "beta",
            "C:/Other",
            "beta prompt",
        );
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);

        assert_eq!(load_all_sessions().unwrap().len(), 2);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 2);
        append_rollout_lines(
            &alpha,
            &[json!({
                "timestamp": "2026-08-07T10:00:01Z",
                "type": "response_item",
                "payload": { "type": "function_call", "name": "shell", "arguments": "{}" }
            })],
        );

        let targeted = load_session_metadata_by_path(alpha.to_str().unwrap())
            .expect("targeted metadata should load")
            .expect("targeted session should remain listed");
        assert_eq!(targeted.session.actual_session_id, "alpha");
        assert_eq!(targeted.session.message_count, 2);
        assert!(targeted.session.has_tool_use);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 3);

        let cache: CodexSessionMetadataCache = serde_json::from_slice(
            &fs::read(metadata_cache_path(&codex_home)).expect("cache should be readable"),
        )
        .expect("cache should deserialize");
        assert_eq!(cache.entries.len(), 2);

        assert_eq!(load_all_sessions().unwrap().len(), 2);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_cache_merges_concurrent_entries() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let alpha = write_codex_rollout(
            &sessions_dir,
            "rollout-alpha.jsonl",
            "alpha",
            "C:/Repo",
            "alpha prompt",
        );
        let beta = write_codex_rollout(
            &sessions_dir,
            "rollout-beta.jsonl",
            "beta",
            "C:/Other",
            "beta prompt",
        );
        let barrier = Arc::new(Barrier::new(2));
        let threads = [alpha, beta].map(|rollout| {
            let base_path = codex_home.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let key = session_metadata_cache_key(&base_path, &rollout)
                    .expect("rollout should have a cache key");
                let entry = CachedSessionInfo {
                    fingerprint: session_info_fingerprint(&rollout)
                        .expect("rollout should have a fingerprint"),
                    info: extract_session_info(&rollout).expect("rollout should parse"),
                };
                barrier.wait();
                merge_session_metadata_cache_entry(&base_path, &rollout, key, entry);
            })
        });
        for thread in threads {
            thread.join().expect("cache merge thread should finish");
        }

        let cache = load_session_metadata_cache(&codex_home);
        assert_eq!(cache.version, SESSION_METADATA_CACHE_VERSION);
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.entries.contains_key("sessions/rollout-alpha.jsonl"));
        assert!(cache.entries.contains_key("sessions/rollout-beta.jsonl"));
    }

    #[test]
    #[serial]
    fn targeted_session_metadata_keeps_native_titles_live() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout = write_codex_rollout(
            &sessions_dir,
            "rollout-title.jsonl",
            "title-session",
            "C:/Repo",
            "Original prompt",
        );
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);

        let cold = load_all_sessions().expect("cold listing should succeed");
        assert_eq!(cold[0].session.summary.as_deref(), Some("Original prompt"));
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 1);

        create_codex_state_db(
            &codex_home,
            &[("title-session", "Renamed title", "Original prompt")],
        );
        write_session_index(
            &codex_home,
            &[
                json!({"id":"title-session","thread_name":"Original prompt"}),
                json!({"id":"title-session","thread_name":"Renamed title"}),
            ],
        );
        let warm = load_session_metadata_by_path(rollout.to_str().unwrap())
            .expect("targeted metadata should load")
            .expect("targeted session should remain listed");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(warm.session.summary.as_deref(), Some("Renamed title"));
        assert!(warm.session.is_renamed);
    }

    #[test]
    #[serial]
    fn all_session_listing_preserves_recorded_path_casing() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        write_codex_rollout(
            &sessions_dir,
            "rollout-upper.jsonl",
            "upper",
            "E:/Work/Repo",
            "upper prompt",
        );
        write_codex_rollout(
            &sessions_dir,
            "rollout-lower.jsonl",
            "lower",
            "e:/Work/Repo",
            "lower prompt",
        );

        let rows = load_all_sessions().expect("listing should succeed");
        let paths: std::collections::HashSet<&str> =
            rows.iter().map(|row| row.project_path.as_str()).collect();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains("E:/Work/Repo"));
        assert!(paths.contains("e:/Work/Repo"));
    }

    #[test]
    #[serial]
    fn all_session_listing_matches_legacy_project_loading() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let active_dir = codex_home.join("sessions");
        let archived_dir = codex_home.join("archived_sessions");
        fs::create_dir_all(&active_dir).expect("active dir should be created");
        fs::create_dir_all(&archived_dir).expect("archive dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        write_codex_rollout(
            &active_dir,
            "rollout-upper.jsonl",
            "upper",
            "E:/Work/Repo",
            "upper prompt",
        );
        write_codex_rollout(
            &active_dir,
            "rollout-lower.jsonl",
            "lower",
            "e:/Work/Repo",
            "lower prompt",
        );
        write_codex_rollout(
            &archived_dir,
            "rollout-archived.jsonl",
            "archived",
            "E:/Work/Other",
            "archived prompt",
        );
        create_codex_state_db(
            &codex_home,
            &[("upper", "Current upper title", "upper prompt")],
        );

        let mut legacy = std::collections::BTreeMap::new();
        for project in scan_projects().expect("legacy project scan should succeed") {
            for session in
                load_sessions(&project.path, false).expect("legacy project load should succeed")
            {
                let is_archived = is_archived_session_path(Path::new(&session.file_path));
                legacy.insert(
                    session.file_path.clone(),
                    json!({
                        "session": session,
                        "project_path": project.actual_path.clone(),
                        "is_archived": is_archived,
                    }),
                );
            }
        }

        let current = load_all_sessions().expect("one-pass listing should succeed");
        let current: std::collections::BTreeMap<String, Value> = current
            .into_iter()
            .map(|listed| {
                (
                    listed.session.file_path.clone(),
                    json!({
                        "session": listed.session,
                        "project_path": listed.project_path,
                        "is_archived": listed.is_archived,
                    }),
                )
            })
            .collect();
        assert_eq!(current, legacy);
    }

    #[test]
    #[serial]
    fn all_session_metadata_cache_recovers_from_corruption_and_stale_versions() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        write_codex_rollout(
            &sessions_dir,
            "rollout-cache-recovery.jsonl",
            "cache-recovery",
            "C:/Repo",
            "cache recovery",
        );
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);

        load_all_sessions().expect("cold listing should seed cache");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 1);
        fs::write(metadata_cache_path(&codex_home), b"{invalid")
            .expect("corrupt cache should be written");
        load_all_sessions().expect("corrupt cache should fall back to parsing");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 2);

        let mut stale: CodexSessionMetadataCache = serde_json::from_slice(
            &fs::read(metadata_cache_path(&codex_home)).expect("repaired cache should be readable"),
        )
        .expect("repaired cache should deserialize");
        stale.version = SESSION_METADATA_CACHE_VERSION + 1;
        fs::write(
            metadata_cache_path(&codex_home),
            serde_json::to_vec(&stale).expect("stale cache should serialize"),
        )
        .expect("stale cache should be written");
        load_all_sessions().expect("stale cache should fall back to parsing");
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    #[serial]
    fn all_session_metadata_cache_replaces_compressed_entry_with_plain_twin() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let lines = [
            session_meta_line_with("2026-08-07T10:00:00Z", "cached-zst", "C:/Repo"),
            user_message_line("2026-08-07T10:00:01Z", "compressed prompt"),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let zst_path = sessions_dir.join("rollout-cached.jsonl.zst");
        fs::write(
            &zst_path,
            zstd::encode_all(body.as_bytes(), 3).expect("fixture should compress"),
        )
        .expect("compressed rollout should be written");
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);

        let compressed = load_all_sessions().expect("compressed listing should succeed");
        assert_eq!(compressed.len(), 1);
        assert!(compressed[0].session.file_path.ends_with(".jsonl.zst"));
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 1);

        let plain_path = zst_path.with_extension("");
        fs::write(&plain_path, format!("{body}\n")).expect("plain twin should be written");
        assert!(load_session_metadata_by_path(zst_path.to_str().unwrap())
            .expect("suppressed compressed twin should be a clean miss")
            .is_none());
        let targeted_plain = load_session_metadata_by_path(plain_path.to_str().unwrap())
            .expect("plain twin target should load")
            .expect("plain twin should remain listed");
        assert_eq!(targeted_plain.session.actual_session_id, "cached-zst");
        let plain = load_all_sessions().expect("plain-twin listing should succeed");
        assert_eq!(plain.len(), 1);
        assert_eq!(plain[0].session.file_path, plain_path.to_string_lossy());
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 2);
        let cache: CodexSessionMetadataCache = serde_json::from_slice(
            &fs::read(metadata_cache_path(&codex_home)).expect("cache should be readable"),
        )
        .expect("cache should deserialize");
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.entries.keys().all(|key| Path::new(key)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))));
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
    fn snapshot_replays_an_unconfirmed_fork_rollback_until_child_metadata_arrives() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_rollout_lines(
            &sessions_dir,
            "rollout-2026-08-29T10-00-00-snapshot-fork.jsonl",
            &[
                forked_session_meta_line_with(
                    "2026-08-29T10:00:00Z",
                    "snapshot-fork",
                    "C:/repo",
                    json!("snapshot-parent"),
                ),
                session_meta_line_with("2026-08-28T10:00:00Z", "snapshot-parent", "C:/repo"),
                json!({
                    "timestamp": "2026-08-28T10:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "parent-final" }
                }),
                json!({
                    "timestamp": "2026-08-28T10:00:02Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "parent-final" }
                }),
            ],
        );
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
                    "timestamp": "2026-08-29T10:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "thread_rolled_back", "num_turns": 2 }
                }),
                json!({
                    "timestamp": "2026-08-29T10:00:02Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "fork-first" }
                }),
            ],
        );
        let cursor = match load_session_snapshot(&path_text, Some(&cursor))
            .expect("unconfirmed fork delta")
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
            _ => panic!("the unconfirmed rollback should remain replayable"),
        };
        assert_eq!(
            cached
                .iter()
                .find(|message| message.subtype.as_deref() == Some("thread_rolled_back"))
                .and_then(|message| message_data_str(message, "rollbackOrigin")),
            None
        );

        append_rollout_lines(
            &path,
            &[
                session_meta_line_with("2026-08-29T10:00:03Z", "snapshot-fork", "C:/repo"),
                json!({
                    "timestamp": "2026-08-29T10:00:04Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "fork-first" }
                }),
            ],
        );
        match load_session_snapshot(&path_text, Some(&cursor)).expect("confirmed fork delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("the confirmed rollback should replace the unclassified suffix"),
        }

        assert_snapshot_matches_fresh(&cached, &path);
        assert_eq!(
            cached
                .iter()
                .find(|message| message.subtype.as_deref() == Some("thread_rolled_back"))
                .and_then(|message| message_data_str(message, "rollbackOrigin")),
            Some("fork")
        );
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
        cursor.version = SNAPSHOT_CURSOR_VERSION - 1;
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
    fn zero_output_token_snapshots_wait_for_real_model_output() {
        let tmp = TempDir::new().expect("temp dir");
        let rollout_path = tmp
            .path()
            .join("rollout-2026-08-09-zero-token-snapshot.jsonl");
        let lines = [
            json!({"timestamp":"2026-08-09T10:00:00Z","type":"session_meta","payload":{"id":"zero-token-snapshot"}}),
            json!({"timestamp":"2026-08-09T10:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-1"}}),
            json!({"timestamp":"2026-08-09T10:00:02Z","type":"response_item","payload":{"id":"intro","type":"message","role":"assistant","content":[{"type":"output_text","text":"I will inspect it."}]}}),
            json!({"timestamp":"2026-08-09T10:00:03Z","type":"response_item","payload":{"id":"tool-call","type":"custom_tool_call","name":"exec","call_id":"call-1","input":"pwd"}}),
            json!({"timestamp":"2026-08-09T10:00:04Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"call-1","output":"done"}}),
            json!({"timestamp":"2026-08-09T10:00:05Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":20}}}}),
            json!({"timestamp":"2026-08-09T10:00:06Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1"}}),
            json!({"timestamp":"2026-08-09T10:00:07Z","type":"compacted","payload":{"replacement_history":[{"type":"summary"}]}}),
            json!({"timestamp":"2026-08-09T10:00:08Z","type":"world_state","payload":{}}),
            json!({"timestamp":"2026-08-09T10:00:09Z","type":"turn_context","payload":{"model":"gpt-5.6-sol"}}),
            json!({"timestamp":"2026-08-09T10:00:10Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":20}}}}),
            json!({"timestamp":"2026-08-09T10:00:11Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-2"}}),
            json!({"timestamp":"2026-08-09T10:00:12Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":130,"cached_input_tokens":35,"output_tokens":20}}}}),
            json!({"timestamp":"2026-08-09T10:00:13Z","type":"response_item","payload":{"id":"final","type":"message","role":"assistant","content":[{"type":"output_text","text":"Done."}]}}),
            json!({"timestamp":"2026-08-09T10:00:14Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":180,"cached_input_tokens":45,"output_tokens":30}}}}),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("write rollout");

        let messages = parse_rollout_file(&rollout_path).expect("parse rollout");
        let intro = messages
            .iter()
            .find(|message| message.uuid == "intro")
            .expect("intro message");
        assert!(intro.usage.is_none());

        let tool = messages
            .iter()
            .find(|message| message.uuid == "tool-call")
            .expect("tool call");
        assert_eq!(
            tool.usage.as_ref().and_then(|usage| usage.output_tokens),
            Some(20)
        );

        let final_message = messages
            .iter()
            .find(|message| message.uuid == "final")
            .expect("final message");
        let usage = final_message.usage.as_ref().expect("combined usage");
        assert_eq!(usage.input_tokens, Some(55));
        assert_eq!(usage.cache_read_input_tokens, Some(25));
        assert_eq!(usage.output_tokens, Some(10));
        let inference_usage = final_message
            .inference
            .as_ref()
            .and_then(|inference| inference.usage.as_ref())
            .expect("detailed combined usage");
        assert_eq!(inference_usage.input_tokens, Some(80));
        assert_eq!(inference_usage.cached_input_tokens, Some(25));
        assert_eq!(inference_usage.output_tokens, Some(10));
        assert!(messages.iter().all(|message| {
            message.usage.as_ref().and_then(|usage| usage.output_tokens) != Some(0)
        }));
    }

    #[test]
    #[serial]
    fn snapshot_cursor_preserves_pending_zero_output_usage() {
        let tmp = TempDir::new().expect("temp dir");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir
            .join("rollout-2026-08-09T10-00-00-snapshot-pending-zero-output-usage.jsonl");
        fs::write(&rollout_path, "").expect("empty rollout");
        append_rollout_lines(
            &rollout_path,
            &[
                json!({"timestamp":"2026-08-09T10:00:00Z","type":"session_meta","payload":{"id":"snapshot-pending-zero-output-usage"}}),
                json!({"timestamp":"2026-08-09T10:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-1"}}),
                json!({"timestamp":"2026-08-09T10:00:02Z","type":"response_item","payload":{"id":"first","type":"message","role":"assistant","content":[{"type":"output_text","text":"First."}]}}),
                json!({"timestamp":"2026-08-09T10:00:03Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":20}}}}),
                json!({"timestamp":"2026-08-09T10:00:04Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1"}}),
                json!({"timestamp":"2026-08-09T10:00:05Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-2"}}),
                json!({"timestamp":"2026-08-09T10:00:06Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":130,"cached_input_tokens":35,"output_tokens":20}}}}),
                json!({"timestamp":"2026-08-09T10:00:07Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-2"}}),
            ],
        );
        let path_text = rollout_path.to_string_lossy();
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
            &rollout_path,
            &[
                json!({"timestamp":"2026-08-09T10:00:08Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-3"}}),
                json!({"timestamp":"2026-08-09T10:00:09Z","type":"response_item","payload":{"id":"final","type":"message","role":"assistant","content":[{"type":"output_text","text":"Final."}]}}),
                json!({"timestamp":"2026-08-09T10:00:10Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":180,"cached_input_tokens":45,"output_tokens":30}}}}),
                json!({"timestamp":"2026-08-09T10:00:11Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-3"}}),
            ],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("usage delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("appended usage should produce a replacement"),
        }
        assert_snapshot_matches_fresh(&cached, &rollout_path);
        let final_message = cached
            .iter()
            .find(|message| message.uuid == "final")
            .expect("final message");
        let usage = final_message.usage.as_ref().expect("combined usage");
        assert_eq!(usage.input_tokens, Some(55));
        assert_eq!(usage.cache_read_input_tokens, Some(25));
        assert_eq!(usage.output_tokens, Some(10));
    }

    #[test]
    fn estimates_chatgpt_credits_from_supported_codex_usage() {
        let inference = InferenceMetadata {
            model: Some("gpt-5.6-sol".to_string()),
            model_provider: Some("openai".to_string()),
            service_tier: Some("default".to_string()),
            ..InferenceMetadata::default()
        };
        let usage = CodexTokenUsage {
            input: 1_000,
            cached: Some(800),
            output: 100,
            ..CodexTokenUsage::default()
        };

        let standard = codex_credit_estimate(&inference, usage, Some("prolite"))
            .expect("supported ChatGPT estimate");
        assert_eq!(standard.unit, "credits");
        assert_eq!(standard.kind, "estimated");
        assert_eq!(standard.rate_card_version, CODEX_CREDIT_RATE_CARD_VERSION);
        assert!((standard.value - 0.11).abs() < 1e-12);

        let priority = codex_credit_estimate(
            &InferenceMetadata {
                service_tier: Some("priority".to_string()),
                ..inference
            },
            usage,
            Some("prolite"),
        )
        .expect("supported Fast estimate");
        assert!((priority.value - 0.275).abs() < 1e-12);
    }

    #[test]
    fn omits_codex_credit_estimates_without_complete_billing_evidence() {
        let inference = InferenceMetadata {
            model: Some("gpt-5.6-sol".to_string()),
            model_provider: Some("openai".to_string()),
            service_tier: Some("default".to_string()),
            ..InferenceMetadata::default()
        };
        let usage = CodexTokenUsage {
            input: 1_000,
            output: 100,
            ..CodexTokenUsage::default()
        };

        assert_eq!(codex_credit_estimate(&inference, usage, None), None);
        assert_eq!(
            codex_credit_estimate(
                &InferenceMetadata {
                    model: Some("future-model".to_string()),
                    ..inference.clone()
                },
                usage,
                Some("prolite"),
            ),
            None
        );
        assert_eq!(
            codex_credit_estimate(
                &InferenceMetadata {
                    model_provider: Some("custom".to_string()),
                    ..inference.clone()
                },
                usage,
                Some("prolite"),
            ),
            None
        );
        assert_eq!(
            codex_credit_estimate(
                &InferenceMetadata {
                    service_tier: Some("future-tier".to_string()),
                    ..inference
                },
                usage,
                Some("prolite"),
            ),
            None
        );
    }

    #[test]
    fn normalizes_authoritative_turn_inference_metadata() {
        let tmp = TempDir::new().expect("temp dir");
        let rollout_path = tmp.path().join("rollout-2026-07-22-test-session.jsonl");
        let lines = [
            json!({"timestamp":"2026-07-22T10:00:00Z","type":"session_meta","payload":{"id":"test-session"}}),
            json!({"timestamp":"2026-07-22T10:00:01Z","type":"event_msg","payload":{"type":"thread_settings_applied","thread_settings":{
                "model":"gpt-5.6-sol","model_provider_id":"openai","service_tier":"default",
                "reasoning_effort":"high","reasoning_summary":"none","personality":"pragmatic",
                "collaboration_mode":{"mode":"default"}
            }}}),
            json!({"timestamp":"2026-07-22T10:00:02Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-1","model_context_window":258400,"collaboration_mode_kind":"default"}}),
            json!({"timestamp":"2026-07-22T10:00:03Z","type":"turn_context","payload":{"turn_id":"turn-1","model":"gpt-5.6-sol","effort":"high","summary":"auto"}}),
            json!({"timestamp":"2026-07-22T10:00:04Z","type":"response_item","payload":{"id":"a1","type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}),
            json!({"timestamp":"2026-07-22T10:00:05Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{
                "input_tokens":100,"cached_input_tokens":40,"cache_write_input_tokens":3,
                "output_tokens":20,"reasoning_output_tokens":7
            }},"rate_limits":{"plan_type":"prolite"}}}),
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
        assert_eq!(inference.model.as_deref(), Some("gpt-5.6-sol"));
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
        let cost = inference.cost.as_ref().expect("credit estimate");
        assert_eq!(cost.unit, "credits");
        assert_eq!(cost.kind, "estimated");
        assert_eq!(cost.rate_card_version, CODEX_CREDIT_RATE_CARD_VERSION);
        assert!((cost.value - 0.023).abs() < 1e-12);
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
        let mut message = convert_codex_item(
            &json!({ "type": "message", "role": "user", "content": content }),
            "session-1",
            None,
            "2026-07-21T12:00:00Z",
            &mut counter,
        )
        .expect("user message should be converted");
        merge_codex_message_provenance(
            &mut message,
            Some("turn-with-pasted-note"),
            Some("client-with-pasted-note"),
        );
        assert_eq!(
            message.data,
            Some(json!({
                "promptArtifacts": [{
                    "kind": "note",
                    "name": "pasted-text.txt",
                    "label": "Build output"
                }],
                "providerTurnId": "turn-with-pasted-note",
                "clientMessageId": "client-with-pasted-note"
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
    fn merge_indexes_authoritative_view_image_payload_without_copying_it() {
        let path = r"C:\Users\Example\AppData\Local\Temp\result.png";
        let source = "data:image/png;base64,iVBORw0KGgo=";
        let mut messages = vec![build_codex_message(
            "assistant-image".to_string(),
            "session-1",
            "2026-08-17T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call-image",
                "name": "exec",
                "input": {
                    "input": format!(
                        "const r = await tools.view_image({{path:{}}}); image(r.image_url);",
                        serde_json::to_string(path).unwrap()
                    )
                }
            }])),
            None,
        )];
        merge_codex_message_provenance(&mut messages[0], Some("turn-33"), None);
        let result_msg = build_codex_message(
            "user-image".to_string(),
            "session-1",
            "2026-08-17T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call-image",
                "content": [
                    { "type": "input_text", "text": "Image Size: 10x10." },
                    { "type": "input_image", "image_url": source }
                ]
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let artifacts = messages[0]
            .data
            .as_ref()
            .and_then(|data| data.get("imageArtifacts"))
            .and_then(Value::as_array)
            .expect("image artifact index should be present");
        assert_eq!(artifacts.len(), 1);
        let artifact = &artifacts[0];
        assert_eq!(artifact["version"], 1);
        assert_eq!(artifact["providerTurnId"], "turn-33");
        assert_eq!(artifact["sourceMessageUuid"], "assistant-image");
        assert_eq!(artifact["toolCallId"], "call-image");
        assert_eq!(artifact["toolResultContentIndex"], 1);
        assert_eq!(artifact["mediaType"], "image/png");
        assert_eq!(artifact["byteLength"], 8);
        assert_eq!(
            artifact["sha256"],
            format!("{:x}", Sha256::digest(b"\x89PNG\r\n\x1a\n"))
        );
        assert_eq!(artifact["correlation"]["kind"], "local_path");
        assert_eq!(artifact["correlation"]["value"], path);
        assert!(artifact.get("source").is_none());
        assert!(artifact.get("image_url").is_none());

        let merged = messages[0]
            .content
            .as_ref()
            .and_then(Value::as_array)
            .expect("merged content should remain available");
        assert_eq!(merged[1]["content"][1]["image_url"], source);
    }

    #[test]
    fn image_artifact_requires_exact_provider_turn_ownership() {
        let mut messages = vec![build_codex_message(
            "assistant-image".to_string(),
            "session-1",
            "2026-08-17T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call-image",
                "name": "view_image",
                "input": { "path": "C:\\Temp\\result.png" }
            }])),
            None,
        )];
        let result_msg = build_codex_message(
            "user-image".to_string(),
            "session-1",
            "2026-08-17T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call-image",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo="
                }]
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        assert!(messages[0]
            .data
            .as_ref()
            .and_then(|data| data.get("imageArtifacts"))
            .is_none());
        assert_eq!(
            messages[0]
                .content
                .as_ref()
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn view_image_locator_rejects_ambiguous_or_non_literal_exec_calls() {
        assert_eq!(
            codex_view_image_locator(&json!({
                "name": "view_image",
                "input": { "path": "C:\\Temp\\direct.png" }
            })),
            Some(r"C:\Temp\direct.png".to_string())
        );
        assert!(codex_view_image_locator(&json!({
            "name": "exec",
            "input": { "input": "tools.view_image({path:first}); tools.view_image({path:\"second.png\"});" }
        }))
        .is_none());
        assert!(codex_view_image_locator(&json!({
            "name": "exec",
            "input": { "input": "tools.view_image({path:someVariable});" }
        }))
        .is_none());
    }

    #[test]
    fn batched_view_image_locator_requires_a_complete_literal_binding_and_emission_bijection() {
        let reordered = r#"const [first, second] = await Promise.all([tools.view_image({path:"first.png"}), tools.view_image({path:"second.png"})]); image(second.image_url); image(first.image_url);"#;
        assert_eq!(
            codex_batched_view_image_locators(reordered),
            Some(vec!["second.png".to_string(), "first.png".to_string()])
        );

        for rejected in [
            r#"const [first, second] = await Promise.all([tools.view_image({path:firstPath}), tools.view_image({path:"second.png"})]); image(first.image_url); image(second.image_url);"#,
            r#"const [first, second] = await Promise.all([tools.view_image({path:"first" + suffix}), tools.view_image({path:"second.png"})]); image(first.image_url); image(second.image_url);"#,
            r#"const [first, second] = await Promise.all([tools.view_image({meta:{path:"wrong.png"}}), tools.view_image({path:"second.png"})]); image(first.image_url); image(second.image_url);"#,
            r#"const [first, second] = await Promise.all([tools.view_image({path:"first.png"}), tools.view_image({path:"second.png"})]); image(first.image_url);"#,
            r#"const [first, second] = await Promise.all([tools.view_image({path:"first.png"}), tools.view_image({path:"second.png"})]); image(first.image_url); image(first.image_url);"#,
            r#"const [first, second] = await Promise.all([tools.view_image({path:"first.png"}), tools.view_image({path:"second.png"})]); image(first.image_url); image(second.image_url); text("extra");"#,
        ] {
            assert!(codex_batched_view_image_locators(rejected).is_none());
        }
    }

    #[test]
    fn merge_correlates_structurally_provable_batched_view_images_per_output_block() {
        let first_path = r"C:\Temp\comparison-full-size.png";
        let second_path = r"C:\Temp\comparison-16px.png";
        let first_source = format!(
            "data:image/png;base64,{}",
            BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\nfirst")
        );
        let second_source = format!(
            "data:image/png;base64,{}",
            BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\nsecond")
        );
        let source = format!(
            "const [full, small] = await Promise.all([\n  tools.view_image({{path:{}, detail:\"original\"}}),\n  tools.view_image({{path:{}, detail:\"original\"}})\n]);\nimage(full.image_url);\nimage(small.image_url);",
            serde_json::to_string(first_path).unwrap(),
            serde_json::to_string(second_path).unwrap()
        );
        let mut messages = vec![build_codex_message(
            "assistant-batch".to_string(),
            "session-1",
            "2026-08-27T12:00:00Z".to_string(),
            "assistant",
            Some("assistant"),
            Some(json!([{
                "type": "tool_use",
                "id": "call-batch",
                "name": "exec",
                "input": { "input": source }
            }])),
            None,
        )];
        merge_codex_message_provenance(&mut messages[0], Some("turn-8"), None);
        let result_msg = build_codex_message(
            "user-batch".to_string(),
            "session-1",
            "2026-08-27T12:00:01Z".to_string(),
            "user",
            Some("user"),
            Some(json!([{
                "type": "tool_result",
                "tool_use_id": "call-batch",
                "content": [
                    { "type": "input_text", "text": "Two images." },
                    { "type": "input_image", "image_url": first_source },
                    { "type": "input_image", "image_url": second_source }
                ]
            }])),
            None,
        );

        assert!(try_merge_tool_result_into_previous(
            &mut messages,
            &result_msg
        ));
        let artifacts = messages[0].data.as_ref().unwrap()["imageArtifacts"]
            .as_array()
            .expect("batched artifacts should be indexed");
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0]["toolResultContentIndex"], 1);
        assert_eq!(artifacts[0]["correlation"]["value"], first_path);
        assert_eq!(artifacts[1]["toolResultContentIndex"], 2);
        assert_eq!(artifacts[1]["correlation"]["value"], second_path);
    }

    #[test]
    fn rollout_indexes_completed_image_generation_events_as_presented_canvas_artifacts() {
        let temp = TempDir::new().expect("temp directory should be created");
        let first = BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\ncanvas-first");
        let second = BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\ncanvas-second");
        let ignored = BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\nfailed");
        let path = write_rollout_lines(
            temp.path(),
            "rollout-2026-08-27T12-00-00-00000000-0000-0000-0000-000000000008.jsonl",
            &[
                json!({
                    "timestamp": "2026-08-27T12:00:00Z",
                    "type": "session_meta",
                    "payload": { "id": "00000000-0000-0000-0000-000000000008" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-8" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:02Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "image_generation_end",
                        "call_id": "exec-canvas-first",
                        "status": "completed",
                        "result": first,
                        "saved_path": "C:\\Users\\Example\\.codex\\generated_images\\session\\exec-canvas-first.png"
                    }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:03Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "image_generation_end",
                        "call_id": "exec-canvas-failed",
                        "status": "failed",
                        "result": ignored,
                        "saved_path": "C:\\Users\\Example\\.codex\\generated_images\\session\\exec-canvas-failed.png"
                    }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:04Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "image_generation_end",
                        "call_id": "exec-canvas-second",
                        "status": "completed",
                        "result": second,
                        "saved_path": "C:\\Users\\Example\\.codex\\generated_images\\session\\exec-canvas-second.png"
                    }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:05Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-8" }
                }),
            ],
        );

        let messages = parse_rollout_file(&path).expect("rollout should parse");
        let canvas_messages = messages
            .iter()
            .filter(|message| {
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("imageArtifacts"))
                    .is_some_and(|artifacts| {
                        artifacts.as_array().is_some_and(|artifacts| {
                            artifacts.iter().any(|artifact| {
                                artifact.get("sourceKind").and_then(Value::as_str)
                                    == Some("provider_rendered_image")
                            })
                        })
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(canvas_messages.len(), 2);
        for (index, message) in canvas_messages.iter().enumerate() {
            assert_eq!(
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("providerTurnId"))
                    .and_then(Value::as_str),
                Some("turn-8")
            );
            let artifact = &message.data.as_ref().unwrap()["imageArtifacts"][0];
            assert_eq!(artifact["version"], 1);
            assert_eq!(artifact["sourceMessageUuid"], message.uuid);
            assert_eq!(artifact["sourceContentIndex"], 0);
            assert_eq!(artifact["sourceKind"], "provider_rendered_image");
            assert_eq!(artifact["presentationKind"], "canvas");
            assert_eq!(
                artifact["toolCallId"],
                if index == 0 {
                    "exec-canvas-first"
                } else {
                    "exec-canvas-second"
                }
            );
            assert_eq!(message.message_type, "progress");
            assert_eq!(message.content.as_ref().unwrap()[0]["type"], "image");
        }
    }

    #[test]
    fn rollout_omits_canvas_artifacts_when_task_ownership_overlaps() {
        let temp = TempDir::new().expect("temp directory should be created");
        let result = BASE64_STANDARD.encode(b"\x89PNG\r\n\x1a\nambiguous");
        let path = write_rollout_lines(
            temp.path(),
            "rollout-2026-08-27T12-00-00-00000000-0000-0000-0000-000000000009.jsonl",
            &[
                json!({
                    "timestamp": "2026-08-27T12:00:00Z",
                    "type": "session_meta",
                    "payload": { "id": "00000000-0000-0000-0000-000000000009" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-a" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:02Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-b" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:03Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "image_generation_end",
                        "call_id": "exec-canvas-ambiguous",
                        "status": "completed",
                        "result": result,
                        "saved_path": "C:\\Users\\Example\\.codex\\generated_images\\session\\exec-canvas-ambiguous.png"
                    }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:04Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-b" }
                }),
                json!({
                    "timestamp": "2026-08-27T12:00:05Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-a" }
                }),
            ],
        );

        let messages = parse_rollout_file(&path).expect("rollout should parse");
        assert!(messages.iter().all(|message| {
            message
                .data
                .as_ref()
                .and_then(|data| data.get("imageArtifacts"))
                .is_none()
        }));
    }

    #[test]
    fn image_data_url_requires_matching_supported_image_bytes() {
        assert_eq!(
            decode_image_data_url("data:image/png;base64,iVBORw0KGgo=")
                .map(|(media_type, bytes)| (media_type, bytes.len())),
            Some(("image/png", 8))
        );
        assert!(decode_image_data_url("data:image/jpeg;base64,iVBORw0KGgo=").is_none());
        assert!(decode_image_data_url("data:image/png;base64,aW1hZ2UtYnl0ZXM=").is_none());
        assert!(decode_image_data_url("data:text/plain;base64,aW1hZ2U=").is_none());
    }

    #[test]
    fn rollout_indexes_view_image_artifact_under_the_single_active_turn() {
        let temp = TempDir::new().expect("temp directory should be created");
        let path = write_rollout_lines(
            temp.path(),
            "rollout-2026-08-17T12-00-00-00000000-0000-0000-0000-000000000033.jsonl",
            &[
                json!({
                    "timestamp": "2026-08-17T12:00:00Z",
                    "type": "session_meta",
                    "payload": { "id": "00000000-0000-0000-0000-000000000033" }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-33" }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:02Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{ "type": "input_text", "text": "Inspect it." }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-33" }
                    }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:03Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "exec",
                        "call_id": "call-image",
                        "input": "const r = await tools.view_image({path:\"C:\\\\Temp\\\\result.png\"}); image(r.image_url);"
                    }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:04Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call-image",
                        "output": [
                            { "type": "input_text", "text": "Image Size: 10x10." },
                            { "type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=" }
                        ]
                    }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:05Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "![Result](</C:/Temp/result.png>)"
                        }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-33" }
                    }
                }),
                json!({
                    "timestamp": "2026-08-17T12:00:06Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-33" }
                }),
            ],
        );

        let messages = parse_rollout_file(&path).expect("rollout should parse");
        let tool_message = messages
            .iter()
            .find(|message| {
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("imageArtifacts"))
                    .is_some()
            })
            .expect("tool message should carry the image artifact index");
        assert_eq!(
            tool_message
                .data
                .as_ref()
                .and_then(|data| data.get("providerTurnId")),
            Some(&Value::String("turn-33".to_string()))
        );
        let artifact = &tool_message.data.as_ref().unwrap()["imageArtifacts"][0];
        assert_eq!(artifact["correlation"]["value"], r"C:\Temp\result.png");
        assert_eq!(artifact["byteLength"], 8);
        assert!(messages.iter().any(|message| {
            message.message_type == "assistant"
                && message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("providerTurnId"))
                    .and_then(Value::as_str)
                    == Some("turn-33")
                && message
                    .content
                    .as_ref()
                    .and_then(Value::as_array)
                    .is_some_and(|content| {
                        content.iter().any(|block| {
                            block.get("text").and_then(Value::as_str)
                                == Some("![Result](</C:/Temp/result.png>)")
                        })
                    })
        }));
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
    fn load_messages_projects_legacy_user_event_without_response_item() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("event-only-user-message.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "event-only-user-message" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-1" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:02Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-1",
                    "message": "canonical prompt"
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 1);
        assert_eq!(users[0].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[0], "providerTurnId"),
            Some("turn-1")
        );
        assert_eq!(
            message_data_str(&users[0], "clientMessageId"),
            Some("client-1")
        );
        assert_eq!(
            users[0]
                .content
                .as_ref()
                .and_then(Value::as_array)
                .and_then(|content| content.first())
                .and_then(|block| block.get("text"))
                .and_then(Value::as_str),
            Some("canonical prompt")
        );
    }

    #[test]
    fn load_messages_keeps_unmatched_response_hidden_and_projects_canonical_event() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("mismatched-canonical-user-message.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "mismatched-canonical-user-message" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-1" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "raw-context",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "provider context" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:03Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-1",
                    "message": "canonical prompt"
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 2);
        assert_eq!(users[0].uuid, "raw-context");
        assert_eq!(users[0].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(users[1].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[1], "clientMessageId"),
            Some("client-1")
        );
    }

    #[test]
    fn load_messages_projects_paginated_completed_user_item() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("paginated-user-message.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "paginated-user-message" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "thread_id": "paginated-user-message",
                    "turn_id": "turn-1",
                    "item": {
                        "type": "UserMessage",
                        "id": "canonical-user-1",
                        "client_id": "client-1",
                        "content": [{ "type": "text", "text": "canonical paginated prompt" }]
                    }
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 1);
        assert_eq!(users[0].uuid, "canonical-user-1");
        assert_eq!(users[0].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[0], "providerTurnId"),
            Some("turn-1")
        );
        assert_eq!(
            message_data_str(&users[0], "clientMessageId"),
            Some("client-1")
        );
    }

    #[test]
    fn load_messages_preserves_paginated_user_images() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("paginated-user-images.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "paginated-user-images" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": {
                        "type": "UserMessage",
                        "id": "canonical-user-1",
                        "content": [
                            { "type": "image", "image_url": "data:image/png;base64,remote" },
                            { "type": "local_image", "path": "C:\\screenshots\\local.png" }
                        ]
                    }
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 1);
        assert_eq!(users[0].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            users[0].content,
            Some(json!([
                {
                    "type": "image",
                    "source": { "type": "url", "url": "data:image/png;base64,remote" }
                },
                {
                    "type": "image",
                    "source": { "type": "url", "url": "C:\\screenshots\\local.png" }
                }
            ]))
        );
    }

    #[test]
    fn load_messages_projects_only_exact_hook_prompt_response_items() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("hook-prompt.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "hook-prompt" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "response_item",
                "payload": {
                    "id": "hook-1",
                    "type": "message",
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": "<hook_prompt hook_run_id=\"run-1\">Retry &amp; test.</hook_prompt>" },
                        { "type": "input_text", "text": "<hook_prompt hook_run_id=\"run-2\">Then finish.</hook_prompt>" }
                    ]
                }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "lookalike",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "prefix <hook_prompt hook_run_id=\"run-3\">not exact</hook_prompt>" }]
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 2);
        assert_eq!(users[0].subtype.as_deref(), Some(HOOK_PROMPT_SUBTYPE));
        assert_eq!(users[1].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(
            users[0]
                .content
                .as_ref()
                .and_then(Value::as_array)
                .map(|content| {
                    content
                        .iter()
                        .filter_map(|block| block.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                }),
            Some(vec!["Retry & test.", "Then finish."])
        );
        assert_eq!(
            users[0]
                .data
                .as_ref()
                .and_then(|data| data.get("hookPromptFragments"))
                .and_then(Value::as_array)
                .and_then(|fragments| fragments.first())
                .and_then(|fragment| fragment.get("hookRunId"))
                .and_then(Value::as_str),
            Some("run-1")
        );
    }

    #[test]
    fn paginated_user_items_in_one_turn_mark_later_input_as_steer() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("paginated-steer.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "paginated-steer" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:01Z",
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": { "type": "UserMessage", "id": "user-1", "content": [{ "type": "text", "text": "first" }] }
                }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:02Z",
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": { "type": "UserMessage", "id": "user-2", "content": [{ "type": "text", "text": "second" }] }
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 2);
        assert_eq!(users[0].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(users[1].subtype.as_deref(), Some(STEER_SUBTYPE));
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
    fn convert_turn_aborted_to_provider_neutral_interruption() {
        let mut counter = 0u64;
        let msg = convert_codex_event(
            &json!({
                "type": "turn_aborted",
                "turn_id": "turn-a",
                "reason": "interrupted",
                "started_at": 1_786_646_968,
                "completed_at": 1_786_647_014,
                "duration_ms": 46_359
            }),
            "session-1",
            "2026-08-13T18:50:14Z",
            &mut counter,
        )
        .expect("abort event should be retained");

        assert_eq!(msg.message_type, "system");
        assert_eq!(msg.subtype.as_deref(), Some("interruption"));
        assert_eq!(msg.duration_ms, Some(46_359));
        assert_eq!(
            msg.content,
            Some(json!([{ "type": "text", "text": "[interrupted]" }]))
        );
        assert_eq!(
            msg.data,
            Some(json!({
                "providerTurnId": "turn-a",
                "reason": "interrupted",
                "startedAt": 1_786_646_968,
                "completedAt": 1_786_647_014,
                "durationMs": 46_359
            }))
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
            // no matching user_message event. Its content intentionally uses an unknown
            // future wrapper so provenance cannot depend on a tag-name allowlist.
            json!({
                "timestamp": "2026-07-14T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "context-1", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<future_host_context>opaque</future_host_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:02.5Z",
                "type": "world_state",
                "payload": {}
            }),
            json!({
                "timestamp": "2026-07-14T10:00:03Z",
                "type": "response_item",
                "payload": {
                    "id": "u1", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "start the work" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:03Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-turn-1-primary",
                    "message": "start the work"
                }
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
                    "content": [{ "type": "input_text", "text": "focus on tests first" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:05Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-turn-1-steer",
                    "message": "focus on tests first"
                }
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
                    "content": [{ "type": "input_text", "text": "ordinary follow-up" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-2" }
                }
            }),
            json!({
                "timestamp": "2026-07-14T10:00:08Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-turn-2-primary",
                    "message": "ordinary follow-up"
                }
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
        assert_eq!(
            users[0].subtype.as_deref(),
            Some("injected_context"),
            "unmatched user-role context receives provider provenance"
        );
        assert_eq!(
            users[1].subtype.as_deref(),
            Some(AUTHORED_USER_SUBTYPE),
            "the turn's first prompt is positively authored"
        );
        assert_eq!(users[2].subtype.as_deref(), Some(STEER_SUBTYPE));
        assert_eq!(
            users[3].subtype.as_deref(),
            Some(AUTHORED_USER_SUBTYPE),
            "the next task's first prompt resets steer detection and remains authored"
        );
        assert_eq!(message_data_str(users[0], "providerTurnId"), Some("turn-1"));
        assert_eq!(message_data_str(users[0], "clientMessageId"), None);
        assert_eq!(message_data_str(users[1], "providerTurnId"), Some("turn-1"));
        assert_eq!(
            message_data_str(users[1], "clientMessageId"),
            Some("client-turn-1-primary")
        );
        assert_eq!(message_data_str(users[2], "providerTurnId"), Some("turn-1"));
        assert_eq!(
            message_data_str(users[2], "clientMessageId"),
            Some("client-turn-1-steer")
        );
        assert_eq!(message_data_str(users[3], "providerTurnId"), Some("turn-2"));
        assert_eq!(
            message_data_str(users[3], "clientMessageId"),
            Some("client-turn-2-primary")
        );
    }

    #[test]
    fn load_messages_marks_interleaved_agent_context_as_injected() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("interleaved-agent-context.jsonl");
        let turn_id = "turn-resumed";
        let lines = [
            json!({
                "timestamp": "2026-08-12T19:57:11Z",
                "type": "session_meta",
                "payload": { "id": "interleaved-agent-context" }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:11Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": turn_id }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:11Z",
                "type": "session_meta",
                "payload": { "id": "interleaved-agent-context" }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "response_item",
                "payload": {
                    "id": "agent-context", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "# AGENTS.md instructions\n\n<INSTRUCTIONS>opaque</INSTRUCTIONS>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "response_item",
                "payload": {
                    "id": "developer-after-agent-context", "type": "message", "role": "developer",
                    "content": [{ "type": "input_text", "text": "developer context" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "response_item",
                "payload": {
                    "id": "environment-context", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>opaque</environment_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "response_item",
                "payload": {
                    "id": "developer-after-environment-context", "type": "message", "role": "developer",
                    "content": [{ "type": "input_text", "text": "more developer context" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "world_state",
                "payload": { "full": true }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "turn_context",
                "payload": { "turn_id": turn_id }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "response_item",
                "payload": {
                    "id": "authored", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "Proceed with the fix." }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                }
            }),
            json!({
                "timestamp": "2026-08-12T19:57:12Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message", "client_id": "client-authored",
                    "message": "Proceed with the fix."
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("rollout should parse");
        let users = messages
            .iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 3);
        assert_eq!(users[0].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(users[1].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(users[2].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(users[2], "clientMessageId"),
            Some("client-authored")
        );

        let mut cross_turn_developer = lines.to_vec();
        cross_turn_developer[4]["payload"]["internal_chat_message_metadata_passthrough"]
            ["turn_id"] = Value::String("other-turn".to_string());
        let cross_turn_path = tmp.path().join("cross-turn-developer-context.jsonl");
        write_terminal_context_fixture(&cross_turn_path, &cross_turn_developer, false);
        let cross_turn_users = parse_rollout_file(&cross_turn_path)
            .expect("cross-turn rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();
        assert_eq!(
            cross_turn_users[0].subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a cross-turn developer record remains raw-only model input"
        );
        assert_eq!(
            cross_turn_users[1].subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a broken corridor cannot expose later raw candidates"
        );

        let mut post_boundary_developer = lines.to_vec();
        let developer = post_boundary_developer.remove(6);
        post_boundary_developer.insert(8, developer);
        let post_boundary_path = tmp.path().join("post-boundary-developer-context.jsonl");
        write_terminal_context_fixture(&post_boundary_path, &post_boundary_developer, false);
        let post_boundary_users = parse_rollout_file(&post_boundary_path)
            .expect("post-boundary rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();
        assert_eq!(
            post_boundary_users[0].subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "developer records after the input boundary remain raw-only"
        );
        assert_eq!(
            post_boundary_users[1].subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "developer records after the input boundary cannot expose pending candidates"
        );
    }

    #[test]
    fn load_messages_classifies_overlapping_task_corridors_per_turn() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("overlapping-task-context.jsonl");
        let authored = "keep our chat in English";
        let lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "overlapping-task-context" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.4Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:24.2Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-b" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33Z",
                "type": "response_item",
                "payload": {
                    "id": "developer-b", "type": "message", "role": "developer",
                    "content": [{ "type": "input_text", "text": "permissions for B" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-b" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33.1Z",
                "type": "response_item",
                "payload": {
                    "id": "context-b", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>B</environment_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-b" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33.2Z",
                "type": "world_state",
                "payload": { "full": false }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33.3Z",
                "type": "turn_context",
                "payload": { "turn_id": "turn-b" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33.4Z",
                "type": "response_item",
                "payload": {
                    "id": "authored-b", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": authored }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-b" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:33.5Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "client_id": "client-b", "message": authored }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34Z",
                "type": "response_item",
                "payload": {
                    "id": "developer-a", "type": "message", "role": "developer",
                    "content": [{ "type": "input_text", "text": "permissions for A" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34.1Z",
                "type": "response_item",
                "payload": {
                    "id": "context-a", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>A</environment_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34.2Z",
                "type": "world_state",
                "payload": { "full": false }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34.3Z",
                "type": "turn_context",
                "payload": { "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34.4Z",
                "type": "response_item",
                "payload": {
                    "id": "authored-a", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": authored }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:34.5Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "client_id": "client-a", "message": authored }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:35Z",
                "type": "response_item",
                "payload": {
                    "id": "steer-a", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "steer A" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:35.1Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "client_id": "client-a-steer", "message": "steer A" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:36Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-b" }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let users = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(users.len(), 5);
        assert_eq!(users[0].uuid, "context-b");
        assert_eq!(users[0].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(users[1].uuid, "authored-b");
        assert_eq!(users[1].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[1], "clientMessageId"),
            Some("client-b")
        );
        assert_eq!(users[2].uuid, "context-a");
        assert_eq!(users[2].subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(users[3].uuid, "authored-a");
        assert_eq!(users[3].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[3], "clientMessageId"),
            Some("client-a")
        );
        assert_eq!(users[4].uuid, "steer-a");
        assert_eq!(users[4].subtype.as_deref(), Some(STEER_SUBTYPE));
        assert_eq!(
            message_data_str(&users[4], "clientMessageId"),
            Some("client-a-steer")
        );
    }

    #[test]
    #[serial]
    fn snapshot_checkpoints_after_abort_and_does_not_taint_the_next_task() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-abort-checkpoint.jsonl");
        let initial_lines = [
            json!({
                "timestamp": "2026-08-13T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "abort-checkpoint" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:00.1Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:00.2Z",
                "type": "response_item",
                "payload": {
                    "id": "prompt-a", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "prompt A" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:00.3Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "prompt A" }
            }),
            json!({
                "timestamp": "2026-08-13T10:00:00.4Z",
                "type": "event_msg",
                "payload": { "type": "turn_aborted", "turn_id": "turn-a", "reason": "interrupted" }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &initial_lines, false);
        let path_text = rollout_path.to_string_lossy();
        let (initial_messages, cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("initial snapshot should carry a cursor"),
            };
        assert!(initial_messages
            .iter()
            .any(|message| message.subtype.as_deref() == Some("interruption")));

        append_rollout_lines(
            &rollout_path,
            &[
                json!({
                    "timestamp": "2026-08-13T10:00:01Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-b" }
                }),
                json!({
                    "timestamp": "2026-08-13T10:00:01.1Z",
                    "type": "response_item",
                    "payload": {
                        "id": "assistant-b", "type": "message", "role": "assistant",
                        "content": [{ "type": "output_text", "text": "answer B" }]
                    }
                }),
                json!({
                    "timestamp": "2026-08-13T10:00:01.2Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-b", "duration_ms": 222 }
                }),
            ],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("abort delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                assert_eq!(replace_from, initial_messages.len());
                assert_eq!(
                    messages
                        .iter()
                        .find(|message| message.uuid == "assistant-b")
                        .and_then(|message| message.inference.as_ref())
                        .and_then(|inference| inference.duration_ms),
                    Some(222),
                    "an aborted prior task must not make the next task overlap-ambiguous"
                );
            }
            _ => panic!("an abort should close its lane and checkpoint the accepted prefix"),
        }
    }

    #[test]
    #[serial]
    fn snapshot_keeps_an_overlapping_open_task_in_the_replaceable_suffix() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-overlapping-snapshot.jsonl");
        let initial_lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "overlapping-snapshot" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.1Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.2Z",
                "type": "response_item",
                "payload": {
                    "id": "context-a", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>A</environment_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.3Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-b" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.4Z",
                "type": "response_item",
                "payload": {
                    "id": "authored-b", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "prompt B" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-b" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.5Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "client_id": "client-b", "message": "prompt B" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.6Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-b" }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &initial_lines, false);
        let path_text = rollout_path.to_string_lossy();
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
            &rollout_path,
            &[
                json!({
                    "timestamp": "2026-07-30T16:22:24Z",
                    "type": "world_state",
                    "payload": {}
                }),
                json!({
                    "timestamp": "2026-07-30T16:22:24.1Z",
                    "type": "turn_context",
                    "payload": { "turn_id": "turn-a" }
                }),
                json!({
                    "timestamp": "2026-07-30T16:22:24.2Z",
                    "type": "response_item",
                    "payload": {
                        "id": "authored-a", "type": "message", "role": "user",
                        "content": [{ "type": "input_text", "text": "prompt A" }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                    }
                }),
                json!({
                    "timestamp": "2026-07-30T16:22:24.3Z",
                    "type": "event_msg",
                    "payload": { "type": "user_message", "client_id": "client-a", "message": "prompt A" }
                }),
                json!({
                    "timestamp": "2026-07-30T16:22:24.4Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-a" }
                }),
            ],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("overlap delta") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("the open overlapping task should keep its corridor replaceable"),
        }
        assert_snapshot_matches_fresh(&cached, &rollout_path);
        let context = cached
            .iter()
            .find(|message| message.uuid == "context-a")
            .expect("context A should be retained");
        assert_eq!(context.subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
    }

    #[test]
    fn unknown_task_completion_does_not_stamp_the_current_assistant_duration() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("unknown-overlap-completion.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "unknown-overlap-completion" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.1Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.2Z",
                "type": "response_item",
                "payload": {
                    "id": "assistant-a", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "working" }]
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.3Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete", "turn_id": "unknown-turn",
                    "duration_ms": 999, "time_to_first_token_ms": 111
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let assistant = parse_rollout_file(&rollout_path)
            .expect("rollout should parse")
            .into_iter()
            .find(|message| message.uuid == "assistant-a")
            .expect("assistant should be retained");
        assert_eq!(
            assistant
                .inference
                .as_ref()
                .and_then(|inference| inference.duration_ms),
            None
        );
        assert_eq!(
            assistant
                .inference
                .as_ref()
                .and_then(|inference| inference.time_to_first_token_ms),
            None
        );
    }

    #[test]
    fn overlapping_completion_does_not_stamp_a_turnless_assistant_from_another_task() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("overlapping-turnless-assistants.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "overlapping-turnless-assistants" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.1Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.2Z",
                "type": "response_item",
                "payload": {
                    "id": "assistant-a", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "working A" }]
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.3Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-b" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.4Z",
                "type": "response_item",
                "payload": {
                    "id": "assistant-b", "type": "message", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "working B" }]
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.5Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-a", "duration_ms": 111 }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.6Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-b", "duration_ms": 222 }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("rollout should parse");
        let assistant_a = messages
            .iter()
            .find(|message| message.uuid == "assistant-a")
            .expect("assistant A should be retained");
        let assistant_b = messages
            .iter()
            .find(|message| message.uuid == "assistant-b")
            .expect("assistant B should be retained");
        assert_eq!(
            message_data_str(assistant_a, "providerTurnId"),
            None,
            "later overlap must revoke inferred ownership from earlier records"
        );
        assert_eq!(message_data_str(assistant_b, "providerTurnId"), None);
        assert_eq!(
            assistant_a
                .inference
                .as_ref()
                .and_then(|inference| inference.duration_ms),
            None,
            "A's ambiguous completion must not claim either turnless assistant"
        );
        assert_eq!(
            assistant_b
                .inference
                .as_ref()
                .and_then(|inference| inference.duration_ms),
            None,
            "a task that overlapped earlier must not regain the turnless fallback"
        );

        let mut reverse_lines = lines.clone();
        reverse_lines[5]["payload"]["turn_id"] = Value::String("turn-b".to_string());
        reverse_lines[5]["payload"]["duration_ms"] = Value::from(222);
        reverse_lines[6]["payload"]["turn_id"] = Value::String("turn-a".to_string());
        reverse_lines[6]["payload"]["duration_ms"] = Value::from(111);
        let reverse_path = tmp
            .path()
            .join("overlapping-turnless-assistants-reverse.jsonl");
        write_terminal_context_fixture(&reverse_path, &reverse_lines, false);
        let reverse_messages =
            parse_rollout_file(&reverse_path).expect("reverse rollout should parse");
        for assistant_id in ["assistant-a", "assistant-b"] {
            assert_eq!(
                reverse_messages
                    .iter()
                    .find(|message| message.uuid == assistant_id)
                    .and_then(|message| message.inference.as_ref())
                    .and_then(|inference| inference.duration_ms),
                None,
                "neither completion order may attribute duration without exact turn provenance"
            );
        }
    }

    #[test]
    fn malformed_or_pre_task_events_cannot_rehabilitate_pending_authorship() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let malformed_event_path = tmp.path().join("malformed-then-valid-user-event.jsonl");
        let malformed_event_lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "malformed-then-valid-user-event" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.1Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.2Z",
                "type": "response_item",
                "payload": {
                    "id": "candidate", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "prompt" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-a" }
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.3Z",
                "type": "event_msg",
                "payload": { "type": "user_message" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.4Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "prompt" }
            }),
        ];
        write_terminal_context_fixture(&malformed_event_path, &malformed_event_lines, false);
        let malformed_candidate = parse_rollout_file(&malformed_event_path)
            .expect("malformed-event rollout should parse")
            .into_iter()
            .find(|message| message.uuid == "candidate")
            .expect("candidate should be retained");
        assert_eq!(
            malformed_candidate.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE)
        );

        let pre_task_path = tmp.path().join("unscoped-before-task-start.jsonl");
        let pre_task_lines = [
            json!({
                "timestamp": "2026-07-30T16:22:23Z",
                "type": "session_meta",
                "payload": { "id": "unscoped-before-task-start" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.1Z",
                "type": "response_item",
                "payload": {
                    "id": "unscoped-candidate", "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "prompt" }]
                }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.2Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-a" }
            }),
            json!({
                "timestamp": "2026-07-30T16:22:23.3Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": "prompt" }
            }),
        ];
        write_terminal_context_fixture(&pre_task_path, &pre_task_lines, false);
        let unscoped_candidate = parse_rollout_file(&pre_task_path)
            .expect("pre-task rollout should parse")
            .into_iter()
            .find(|message| message.uuid == "unscoped-candidate")
            .expect("unscoped candidate should be retained");
        assert_eq!(
            unscoped_candidate.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE)
        );
    }

    #[test]
    fn load_messages_matches_captured_app_server_authorship_shape() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("app-server-authorship-oracle.jsonl");
        let authored_per_turn = [1usize, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1];
        let mut lines = vec![json!({
            "timestamp": "2026-08-11T05:00:00Z",
            "type": "session_meta",
            "payload": { "id": "app-server-authorship-oracle" }
        })];
        let mut expected_client_ids = Vec::new();

        for (turn_index, authored_count) in authored_per_turn.into_iter().enumerate() {
            let turn_id = format!("turn-{}", turn_index + 1);
            lines.push(json!({
                "timestamp": "2026-08-11T05:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": turn_id }
            }));
            for authored_index in 0..authored_count {
                let text = format!("turn {} input {}", turn_index + 1, authored_index + 1);
                let client_id = format!("client-{}-{}", turn_index + 1, authored_index + 1);
                expected_client_ids.push(client_id.clone());
                lines.extend([
                    json!({
                        "timestamp": "2026-08-11T05:00:02Z",
                        "type": "response_item",
                        "payload": {
                            "id": format!("item-{}-{}", turn_index + 1, authored_index + 1),
                            "type": "message",
                            "role": "user",
                            "content": [{ "type": "input_text", "text": text }],
                            "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                        }
                    }),
                    json!({
                        "timestamp": "2026-08-11T05:00:03Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "user_message",
                            "client_id": client_id,
                            "message": text
                        }
                    }),
                ]);
            }
            if turn_index + 1 == authored_per_turn.len() {
                lines.extend([
                    json!({
                        "timestamp": "2026-08-11T05:00:04Z",
                        "type": "response_item",
                        "payload": {
                            "id": "terminal-context",
                            "type": "message",
                            "role": "user",
                            "content": [{ "type": "input_text", "text": "<environment_context>refresh</environment_context>" }],
                            "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                        }
                    }),
                    json!({
                        "timestamp": "2026-08-11T05:00:05Z",
                        "type": "world_state",
                        "payload": {}
                    }),
                    json!({
                        "timestamp": "2026-08-11T05:00:06Z",
                        "type": "response_item",
                        "payload": {
                            "type": "reasoning",
                            "summary": [],
                            "encrypted_content": "opaque",
                            "internal_chat_message_metadata_passthrough": { "turn_id": turn_id }
                        }
                    }),
                ]);
            }
            lines.push(json!({
                "timestamp": "2026-08-11T05:00:07Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": turn_id }
            }));
        }
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("oracle fixture should parse");
        let authored = messages
            .iter()
            .filter(|message| {
                matches!(
                    message.subtype.as_deref(),
                    Some(AUTHORED_USER_SUBTYPE | STEER_SUBTYPE)
                )
            })
            .collect::<Vec<_>>();
        let steers = authored
            .iter()
            .filter(|message| message.subtype.as_deref() == Some(STEER_SUBTYPE))
            .count();
        let client_ids = authored
            .iter()
            .filter_map(|message| {
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("clientMessageId"))
                    .and_then(Value::as_str)
            })
            .collect::<Vec<_>>();
        let mut provider_turn_ids = authored
            .iter()
            .filter_map(|message| {
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("providerTurnId"))
                    .and_then(Value::as_str)
            })
            .collect::<Vec<_>>();
        provider_turn_ids.sort_unstable();
        provider_turn_ids.dedup();

        assert_eq!(authored.len(), 18);
        assert_eq!(steers, 3);
        assert_eq!(client_ids, expected_client_ids);
        assert_eq!(provider_turn_ids.len(), 15);
        let context = messages
            .iter()
            .find(|message| message.uuid == "terminal-context")
            .expect("terminal context should remain projected");
        assert_eq!(context.subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        assert_eq!(
            context
                .data
                .as_ref()
                .and_then(|data| data.get("providerTurnId"))
                .and_then(Value::as_str),
            Some("turn-15")
        );
        assert!(context
            .data
            .as_ref()
            .map_or(true, |data| data.get("clientMessageId").is_none()));
    }

    #[test]
    fn load_messages_rejects_cross_turn_boundary_in_ordinary_pairing() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("cross-turn-ordinary-boundary.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-11T05:30:00Z",
                "type": "session_meta",
                "payload": { "id": "cross-turn-ordinary-boundary" }
            }),
            json!({
                "timestamp": "2026-08-11T05:30:01Z",
                "type": "response_item",
                "payload": {
                    "id": "context",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "<environment_context>refresh</environment_context>" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-11T05:30:02Z",
                "type": "turn_context",
                "payload": { "turn_id": "turn-2" }
            }),
            json!({
                "timestamp": "2026-08-11T05:30:03Z",
                "type": "response_item",
                "payload": {
                    "id": "authored",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "real prompt" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-11T05:30:04Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "client-real",
                    "message": "real prompt"
                }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("rollout should parse");
        let context = messages
            .iter()
            .find(|message| message.uuid == "context")
            .expect("context candidate should remain projected");
        let authored = messages
            .iter()
            .find(|message| message.uuid == "authored")
            .expect("authored message should remain projected");

        assert_eq!(
            context.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a boundary from another physical turn cannot expose raw model input"
        );
        assert_eq!(authored.subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(message_data_str(authored, "providerTurnId"), Some("turn-1"));
        assert_eq!(
            message_data_str(authored, "clientMessageId"),
            Some("client-real")
        );

        let stale_boundary_path = tmp.path().join("stale-ordinary-boundary.jsonl");
        let stale_boundary_lines = [
            lines[0].clone(),
            lines[1].clone(),
            json!({
                "timestamp": "2026-08-11T05:30:01.5Z",
                "type": "turn_context",
                "payload": { "turn_id": "turn-1" }
            }),
            lines[2].clone(),
            lines[3].clone(),
            lines[4].clone(),
        ];
        write_terminal_context_fixture(&stale_boundary_path, &stale_boundary_lines, false);
        let stale_boundary_messages =
            parse_rollout_file(&stale_boundary_path).expect("rollout should parse");
        let stale_boundary_context = stale_boundary_messages
            .iter()
            .find(|message| message.uuid == "context")
            .expect("context candidate should remain projected");
        assert_eq!(
            stale_boundary_context.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a later cross-turn boundary cannot expose an earlier raw candidate"
        );

        let missing_candidate_turn_path = tmp.path().join("missing-candidate-turn.jsonl");
        let mut missing_candidate_turn_lines = lines.clone();
        missing_candidate_turn_lines[1]
            .get_mut("payload")
            .and_then(Value::as_object_mut)
            .expect("candidate payload should be an object")
            .remove("internal_chat_message_metadata_passthrough");
        write_terminal_context_fixture(
            &missing_candidate_turn_path,
            &missing_candidate_turn_lines,
            false,
        );
        let missing_candidate_turn_messages =
            parse_rollout_file(&missing_candidate_turn_path).expect("rollout should parse");
        let missing_candidate_turn_context = missing_candidate_turn_messages
            .iter()
            .find(|message| message.uuid == "context")
            .expect("context candidate should remain projected");
        assert_eq!(
            missing_candidate_turn_context.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a candidate without a physical turn remains raw-only"
        );
    }

    #[test]
    fn duplicate_canonical_client_id_does_not_hide_a_visible_steer() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("duplicate-client-id.jsonl");
        let lines = [
            json!({
                "timestamp": "2026-08-11T06:00:00Z",
                "type": "session_meta",
                "payload": { "id": "duplicate-client-id" }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-1" }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:02Z",
                "type": "response_item",
                "payload": {
                    "id": "primary",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "primary" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:03Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "duplicate-client",
                    "message": "primary"
                }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:04Z",
                "type": "response_item",
                "payload": {
                    "id": "duplicate",
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "duplicate" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:05Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "client_id": "duplicate-client",
                    "message": "duplicate"
                }
            }),
            json!({
                "timestamp": "2026-08-11T06:00:06Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-1" }
            }),
        ];
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("fixture should parse");
        let primary = messages
            .iter()
            .find(|message| message.uuid == "primary")
            .expect("primary input should be projected");
        let duplicate = messages
            .iter()
            .find(|message| message.uuid == "duplicate")
            .expect("duplicate canonical input should remain projected");
        assert_eq!(primary.subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            message_data_str(primary, "clientMessageId"),
            Some("duplicate-client")
        );
        assert_eq!(duplicate.subtype.as_deref(), Some(STEER_SUBTYPE));
        assert_eq!(
            message_data_str(duplicate, "clientMessageId"),
            Some("duplicate-client")
        );
    }

    #[test]
    #[serial]
    fn load_messages_preserves_authored_injection_lookalike() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join("codex-home");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions dir should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let rollout_path = sessions_dir.join("rollout-authored-lookalike.jsonl");
        let authored =
            "# AGENTS.md instructions\n<INSTRUCTIONS>\nKeep this authored text.\n</INSTRUCTIONS>";
        let lines = [
            json!({
                "timestamp": "2026-08-07T10:00:00Z",
                "type": "session_meta",
                "payload": { "id": "authored-lookalike" }
            }),
            json!({
                "timestamp": "2026-08-07T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "turn-1" }
            }),
            json!({
                "timestamp": "2026-08-07T10:00:01.5Z",
                "type": "response_item",
                "payload": {
                    "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "authored event was lost" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "other-turn" }
                }
            }),
            json!({
                "timestamp": "2026-08-07T10:00:01.75Z",
                "type": "world_state",
                "payload": {}
            }),
            json!({
                "timestamp": "2026-08-07T10:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": authored }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                }
            }),
            json!({
                "timestamp": "2026-08-07T10:00:03Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": authored }
            }),
            json!({
                "timestamp": "2026-08-07T10:00:04Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-1" }
            }),
        ];
        let content = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{content}\n"))
            .expect("rollout fixture should be written");

        let messages = load_messages(rollout_path.to_str().unwrap()).expect("rollout should parse");
        let users = messages
            .iter()
            .filter(|message| message.message_type == "user")
            .collect::<Vec<_>>();

        assert_eq!(
            users[0].subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "a cross-turn row with a missing canonical event remains raw-only"
        );
        assert_eq!(users[1].subtype.as_deref(), Some(AUTHORED_USER_SUBTYPE));
        assert_eq!(
            users[1]
                .content
                .as_ref()
                .and_then(Value::as_array)
                .and_then(|content| content.first())
                .and_then(|item| item.get("text"))
                .and_then(Value::as_str),
            Some(authored)
        );
    }

    fn terminal_context_fixture(
        active_turn_id: &str,
        prior_authored_turn_id: Option<&str>,
        context_turn_id: &str,
        include_boundary: bool,
        activity_turn_id: Option<&str>,
        include_second_pending: bool,
        terminal_turn_id: &str,
    ) -> Vec<Value> {
        let mut lines = vec![
            json!({
                "timestamp": "2026-08-11T04:00:00Z",
                "type": "session_meta",
                "payload": { "id": "terminal-context" }
            }),
            json!({
                "timestamp": "2026-08-11T04:00:01Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": active_turn_id }
            }),
        ];
        if let Some(prior_authored_turn_id) = prior_authored_turn_id {
            lines.extend([
                json!({
                    "timestamp": "2026-08-11T04:00:02Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{ "type": "input_text", "text": "implement the change" }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": prior_authored_turn_id }
                    }
                }),
                json!({
                    "timestamp": "2026-08-11T04:00:03Z",
                    "type": "event_msg",
                    "payload": { "type": "user_message", "message": "implement the change" }
                }),
            ]);
        }
        lines.push(json!({
            "timestamp": "2026-08-11T04:00:04Z",
            "type": "response_item",
            "payload": {
                "type": "message", "role": "user",
                "content": [{ "type": "input_text", "text": "<environment_context>refresh</environment_context>" }],
                "internal_chat_message_metadata_passthrough": { "turn_id": context_turn_id }
            }
        }));
        if include_second_pending {
            lines.push(json!({
                "timestamp": "2026-08-11T04:00:04.5Z",
                "type": "response_item",
                "payload": {
                    "type": "message", "role": "user",
                    "content": [{ "type": "input_text", "text": "second unresolved record" }],
                    "internal_chat_message_metadata_passthrough": { "turn_id": active_turn_id }
                }
            }));
        }
        if include_boundary {
            lines.push(json!({
                "timestamp": "2026-08-11T04:00:05Z",
                "type": "world_state",
                "payload": { "full": false }
            }));
        }
        if let Some(activity_turn_id) = activity_turn_id {
            lines.push(json!({
                "timestamp": "2026-08-11T04:00:06Z",
                "type": "response_item",
                "payload": {
                    "type": "reasoning", "summary": [], "encrypted_content": "opaque",
                    "internal_chat_message_metadata_passthrough": { "turn_id": activity_turn_id }
                }
            }));
        }
        lines.push(json!({
            "timestamp": "2026-08-11T04:00:07Z",
            "type": "event_msg",
            "payload": { "type": "task_complete", "turn_id": terminal_turn_id }
        }));
        lines
    }

    fn write_terminal_context_fixture(path: &Path, lines: &[Value], malformed_gap: bool) {
        let mut serialized = lines.iter().map(Value::to_string).collect::<Vec<_>>();
        if malformed_gap {
            serialized.insert(serialized.len() - 1, "{malformed".to_string());
        }
        fs::write(path, format!("{}\n", serialized.join("\n")))
            .expect("terminal context fixture should be written");
    }

    #[test]
    fn load_messages_marks_completed_same_turn_context_refresh_as_injected() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("terminal-context.jsonl");
        let mut lines = terminal_context_fixture(
            "turn-1",
            Some("turn-1"),
            "turn-1",
            true,
            Some("turn-1"),
            false,
            "turn-1",
        );
        lines.splice(
            lines.len() - 1..lines.len() - 1,
            [
                json!({
                    "timestamp": "2026-08-11T04:00:06.1Z",
                    "type": "event_msg",
                    "payload": { "type": "token_count" }
                }),
                json!({
                    "timestamp": "2026-08-11T04:00:06.2Z",
                    "type": "event_msg",
                    "payload": { "type": "agent_message", "message": "duplicate activity" }
                }),
                json!({
                    "timestamp": "2026-08-11T04:00:06.3Z",
                    "type": "event_msg",
                    "payload": { "type": "patch_apply_end", "turn_id": "turn-1" }
                }),
                json!({
                    "timestamp": "2026-08-11T04:00:06.4Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "done",
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                    }
                }),
            ],
        );
        write_terminal_context_fixture(&rollout_path, &lines, false);

        let messages = parse_rollout_file(&rollout_path).expect("rollout should parse");
        let context = messages
            .iter()
            .find(|message| {
                message.message_type == "user"
                    && message
                        .content
                        .as_ref()
                        .is_some_and(|content| content.to_string().contains("environment_context"))
            })
            .expect("context refresh should remain in the base projection");
        assert_eq!(context.subtype.as_deref(), Some(INJECTED_CONTEXT_SUBTYPE));
        let authored = messages
            .iter()
            .find(|message| message.subtype.as_deref() == Some(AUTHORED_USER_SUBTYPE))
            .expect("authored prompt should remain projected");
        assert_eq!(message_data_str(authored, "providerTurnId"), Some("turn-1"));
        assert_eq!(message_data_str(authored, "clientMessageId"), None);
    }

    #[test]
    fn load_messages_terminal_context_inference_fails_open_without_each_proof() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let unexpected_valid_record = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines.insert(
                lines.len() - 1,
                json!({
                    "timestamp": "2026-08-11T04:00:06.5Z",
                    "type": "future_protocol_record",
                    "payload": { "turn_id": "turn-1" }
                }),
            );
            lines
        };
        let unexpected_event = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines.insert(
                lines.len() - 1,
                json!({
                    "timestamp": "2026-08-11T04:00:06.5Z",
                    "type": "event_msg",
                    "payload": { "type": "future_event", "turn_id": "turn-1" }
                }),
            );
            lines
        };
        let unexpected_response_item = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines.insert(
                lines.len() - 1,
                json!({
                    "timestamp": "2026-08-11T04:00:06.5Z",
                    "type": "response_item",
                    "payload": {
                        "type": "future_agent_activity",
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                    }
                }),
            );
            lines
        };
        let boundary_after_activity = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines.insert(
                lines.len() - 1,
                json!({
                    "timestamp": "2026-08-11T04:00:06.5Z",
                    "type": "turn_context",
                    "payload": { "turn_id": "turn-1" }
                }),
            );
            lines
        };
        let activity_before_boundary = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines.swap(5, 6);
            lines.insert(
                lines.len() - 1,
                json!({
                    "timestamp": "2026-08-11T04:00:06.5Z",
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": "opaque-again",
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-1" }
                    }
                }),
            );
            lines
        };
        let mismatched_then_matching_completion = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "other-turn",
            );
            lines.push(json!({
                "timestamp": "2026-08-11T04:00:08Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-1" }
            }));
            lines
        };
        let cross_turn_turn_context = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines[5] = json!({
                "timestamp": "2026-08-11T04:00:05Z",
                "type": "turn_context",
                "payload": { "turn_id": "other-turn" }
            });
            lines
        };
        let cross_turn_world_state = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines[5] = json!({
                "timestamp": "2026-08-11T04:00:05Z",
                "type": "world_state",
                "payload": { "turn_id": "other-turn" }
            });
            lines
        };
        let turn_context_without_turn_id = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines[5] = json!({
                "timestamp": "2026-08-11T04:00:05Z",
                "type": "turn_context",
                "payload": {}
            });
            lines
        };
        let malformed_world_state = {
            let mut lines = terminal_context_fixture(
                "turn-1",
                Some("turn-1"),
                "turn-1",
                true,
                Some("turn-1"),
                false,
                "turn-1",
            );
            lines[5] = json!({
                "timestamp": "2026-08-11T04:00:05Z",
                "type": "world_state",
                "payload": "malformed"
            });
            lines
        };
        let cases = [
            (
                "no-prior-authored-pair",
                terminal_context_fixture(
                    "turn-1",
                    None,
                    "turn-1",
                    true,
                    Some("turn-1"),
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "cross-turn-prior-authored-pair",
                terminal_context_fixture(
                    "turn-1",
                    Some("other-turn"),
                    "turn-1",
                    true,
                    Some("turn-1"),
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "cross-turn-context",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "other-turn",
                    true,
                    Some("turn-1"),
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "no-input-boundary",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    false,
                    Some("turn-1"),
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "no-agent-activity",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    true,
                    None,
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "cross-turn-agent-activity",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    true,
                    Some("other-turn"),
                    false,
                    "turn-1",
                ),
                false,
            ),
            (
                "multiple-pending-records",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    true,
                    Some("turn-1"),
                    true,
                    "turn-1",
                ),
                false,
            ),
            (
                "mismatched-terminal",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    true,
                    Some("turn-1"),
                    false,
                    "other-turn",
                ),
                false,
            ),
            (
                "malformed-gap",
                terminal_context_fixture(
                    "turn-1",
                    Some("turn-1"),
                    "turn-1",
                    true,
                    Some("turn-1"),
                    false,
                    "turn-1",
                ),
                true,
            ),
            ("unexpected-valid-record", unexpected_valid_record, false),
            ("unexpected-event", unexpected_event, false),
            ("unexpected-response-item", unexpected_response_item, false),
            ("boundary-after-activity", boundary_after_activity, false),
            ("activity-before-boundary", activity_before_boundary, false),
            (
                "mismatched-then-matching-completion",
                mismatched_then_matching_completion,
                false,
            ),
            ("cross-turn-turn-context", cross_turn_turn_context, false),
            ("cross-turn-world-state", cross_turn_world_state, false),
            (
                "turn-context-without-turn-id",
                turn_context_without_turn_id,
                false,
            ),
            ("malformed-world-state", malformed_world_state, false),
        ];

        for (name, lines, malformed_gap) in cases {
            let rollout_path = tmp.path().join(format!("{name}.jsonl"));
            write_terminal_context_fixture(&rollout_path, &lines, malformed_gap);
            let messages = parse_rollout_file(&rollout_path).expect("rollout should parse");
            let unresolved = messages
                .iter()
                .find(|message| {
                    message.message_type == "user"
                        && message.content.as_ref().is_some_and(|content| {
                            content.to_string().contains("environment_context")
                        })
                })
                .expect("candidate should remain visible");
            assert_eq!(
                unresolved.subtype.as_deref(),
                Some(INJECTED_CONTEXT_SUBTYPE),
                "{name} must not expose raw model input"
            );
        }
    }

    #[test]
    #[serial]
    fn snapshot_reclassifies_same_turn_context_only_after_matching_completion() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-terminal-context");
        let path_text = path.to_string_lossy();

        let mut active_lines = terminal_context_fixture(
            "turn-2",
            Some("turn-2"),
            "turn-2",
            true,
            Some("turn-2"),
            false,
            "turn-2",
        );
        active_lines.remove(0);
        active_lines.pop();
        append_rollout_lines(&path, &active_lines);

        let (mut cached, cursor) =
            match load_session_snapshot(&path_text, None).expect("unresolved snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("initial snapshot should carry a cursor"),
            };
        let unresolved = cached
            .iter()
            .find(|message| {
                message.message_type == "user"
                    && message
                        .content
                        .as_ref()
                        .is_some_and(|content| content.to_string().contains("environment_context"))
            })
            .expect("context refresh should be visible before completion");
        assert_eq!(
            unresolved.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE)
        );

        append_rollout_lines(
            &path,
            &[json!({
                "timestamp": "2026-08-11T04:00:07Z",
                "type": "event_msg",
                "payload": { "type": "task_complete", "turn_id": "turn-2" }
            })],
        );
        match load_session_snapshot(&path_text, Some(&cursor)).expect("resolved replacement") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("matching completion should replace the active suffix"),
        }
        assert_snapshot_matches_fresh(&cached, &path);
        assert!(cached.iter().any(|message| {
            message.subtype.as_deref() == Some(INJECTED_CONTEXT_SUBTYPE)
                && message
                    .content
                    .as_ref()
                    .is_some_and(|content| content.to_string().contains("environment_context"))
        }));
    }

    #[test]
    #[serial]
    fn snapshot_reclassifies_unresolved_context_after_authored_event_arrives() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions_dir = codex_home.join("sessions");
        fs::create_dir_all(&sessions_dir).expect("sessions directory should be created");
        let _guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        let path = write_snapshot_fixture(&sessions_dir, "snapshot-injected-context");
        let path_text = path.to_string_lossy();

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-08-07T10:01:00Z",
                    "type": "event_msg",
                    "payload": { "type": "task_started", "turn_id": "turn-2" }
                }),
                json!({
                    "timestamp": "2026-08-07T10:01:01Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{ "type": "input_text", "text": "<future_context>pending</future_context>" }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-2" }
                    }
                }),
            ],
        );

        let (mut cached, cursor) =
            match load_session_snapshot(&path_text, None).expect("initial snapshot") {
                SessionSnapshotLoad::Full {
                    messages,
                    cursor: Some(cursor),
                    ..
                } => (messages, cursor),
                _ => panic!("initial snapshot should carry a cursor"),
            };
        let unresolved = cached
            .iter()
            .find(|message| {
                message.message_type == "user"
                    && message
                        .content
                        .as_ref()
                        .is_some_and(|content| content.to_string().contains("future_context"))
            })
            .expect("unresolved context should remain visible");
        assert_eq!(
            unresolved.subtype.as_deref(),
            Some(INJECTED_CONTEXT_SUBTYPE),
            "EOF without a canonical event must keep raw model input hidden"
        );

        append_rollout_lines(
            &path,
            &[
                json!({
                    "timestamp": "2026-08-07T10:01:01.5Z",
                    "type": "world_state",
                    "payload": {}
                }),
                json!({
                    "timestamp": "2026-08-07T10:01:02Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{ "type": "input_text", "text": "authored prompt" }],
                        "internal_chat_message_metadata_passthrough": { "turn_id": "turn-2" }
                    }
                }),
                json!({
                    "timestamp": "2026-08-07T10:01:03Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "client_id": "incremental-authored-client",
                        "message": "authored prompt"
                    }
                }),
                json!({
                    "timestamp": "2026-08-07T10:01:04Z",
                    "type": "event_msg",
                    "payload": { "type": "task_complete", "turn_id": "turn-2" }
                }),
            ],
        );

        match load_session_snapshot(&path_text, Some(&cursor)).expect("resolved replacement") {
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                ..
            } => {
                cached.truncate(replace_from);
                cached.extend(messages);
            }
            _ => panic!("the active suffix should be replaced after append"),
        }
        assert_snapshot_matches_fresh(&cached, &path);
        assert!(cached.iter().any(|message| {
            message.subtype.as_deref() == Some("injected_context")
                && message
                    .content
                    .as_ref()
                    .is_some_and(|content| content.to_string().contains("future_context"))
        }));
        let authored = cached
            .iter()
            .find(|message| {
                message_data_str(message, "clientMessageId") == Some("incremental-authored-client")
            })
            .expect("appended authored input should be reclassified");
        assert_eq!(message_data_str(authored, "providerTurnId"), Some("turn-2"));
        assert_eq!(
            message_data_str(authored, "clientMessageId"),
            Some("incremental-authored-client")
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
    /// `SessionInfo`. Used by the summary-provenance tests below.
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
    /// Without authored events, even a known-looking wrapper is ambiguous and
    /// must remain visible as the legacy summary fallback.
    fn extract_session_info_fails_open_for_eventless_context_lookalike() {
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line("2026-05-13T08:00:01Z", ENV_CONTEXT_BLOCK),
            user_message_line(
                "2026-05-13T08:00:02Z",
                "Please review my PR for the Antigravity provider.",
            ),
        ]);

        assert_eq!(info.summary.as_deref(), Some(ENV_CONTEXT_BLOCK));
        // message_count still counts every response_item type=message.
        assert_eq!(info.message_count, 2);
    }

    #[test]
    fn extract_session_info_prefers_structurally_authored_summary() {
        let authored = "# AGENTS.md instructions\n<INSTRUCTIONS>authored</INSTRUCTIONS>";
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line(
                "2026-08-07T08:00:01Z",
                "<future_context>injected</future_context>",
            ),
            user_message_line("2026-08-07T08:00:02Z", authored),
            json!({
                "timestamp": "2026-08-07T08:00:03Z",
                "type": "event_msg",
                "payload": { "type": "user_message", "message": authored }
            }),
        ]);

        assert_eq!(info.summary.as_deref(), Some(authored));
        assert_eq!(
            info.message_count, 2,
            "authorship classification must not change activity counts"
        );
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
        let provenance = info
            .subagent_provenance
            .expect("valid thread_spawn metadata should expose provenance");
        assert_eq!(provenance.spawned_at, "2026-05-13T08:00:00Z");
        assert_eq!(provenance.agent_path, "/root/research");
        assert_eq!(provenance.agent_nickname.as_deref(), Some("Parfit"));
        assert_eq!(codex_entrypoint(Some(&json!({ "subagent": {} }))), None);
        assert_eq!(codex_entrypoint(Some(&json!({ "future": {} }))), None);
    }

    #[test]
    fn extract_session_info_omits_malformed_subagent_provenance() {
        let invalid_meta = [
            json!({
                "timestamp": "",
                "type": "session_meta",
                "payload": {
                    "id": "missing-time",
                    "cwd": "/tmp/proj",
                    "source": { "subagent": { "thread_spawn": {
                        "agent_path": "/root/research",
                        "agent_nickname": "Parfit"
                    } } }
                }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "missing-path",
                    "cwd": "/tmp/proj",
                    "source": { "subagent": { "thread_spawn": {
                        "agent_nickname": "Parfit"
                    } } }
                }
            }),
            json!({
                "timestamp": "2026-05-13T08:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "unknown-source",
                    "cwd": "/tmp/proj",
                    "source": { "future": {} }
                }
            }),
        ];

        for meta in invalid_meta {
            let info = run_extract_session_info_on_lines(vec![meta]);
            assert_eq!(info.subagent_provenance, None);
        }

        let info = run_extract_session_info_on_lines(vec![json!({
            "timestamp": "2026-05-13T08:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "malformed-nickname",
                "cwd": "/tmp/proj",
                "source": { "subagent": { "thread_spawn": {
                    "agent_path": "/root/research",
                    "agent_nickname": 42
                } } }
            }
        })]);
        assert_eq!(
            info.subagent_provenance.unwrap().agent_nickname,
            None,
            "a malformed optional nickname should be omitted without losing valid provenance"
        );
    }

    #[test]
    fn extract_session_info_keeps_first_subagent_provenance() {
        let info = run_extract_session_info_on_lines(vec![
            json!({
                "timestamp": "2026-05-13T08:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-child",
                    "cwd": "/tmp/proj",
                    "source": { "subagent": { "thread_spawn": {
                        "agent_path": "/root/implement",
                        "agent_nickname": "Singer"
                    } } }
                }
            }),
            json!({
                "timestamp": "2026-05-12T08:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-parent",
                    "cwd": "/tmp/older",
                    "source": { "subagent": { "thread_spawn": {
                        "agent_path": "/root/replayed",
                        "agent_nickname": "Wrong"
                    } } }
                }
            }),
        ]);

        let provenance = info.subagent_provenance.unwrap();
        assert_eq!(provenance.spawned_at, "2026-05-13T08:00:00Z");
        assert_eq!(provenance.agent_path, "/root/implement");
        assert_eq!(provenance.agent_nickname.as_deref(), Some("Singer"));
    }

    #[test]
    /// An event-less wrapper-only session is ambiguous, so its content remains
    /// the fail-open summary instead of being hidden by a tag-name heuristic.
    fn extract_session_info_env_context_only_fails_open() {
        let info = run_extract_session_info_on_lines(vec![
            session_meta_line(),
            user_message_line("2026-05-13T08:00:01Z", ENV_CONTEXT_BLOCK),
        ]);

        assert_eq!(info.summary.as_deref(), Some(ENV_CONTEXT_BLOCK));
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

    #[test]
    fn parse_rollout_file_marks_only_the_rollback_that_creates_a_fork() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("rollout-2026-08-29.jsonl");
        let lines = [
            forked_session_meta_line_with(
                "2026-08-29T10:00:00Z",
                "sess-fork-new",
                "/tmp/proj",
                json!("sess-orig"),
            ),
            session_meta_line_with("2026-08-27T10:00:00Z", "foreign-session", "/tmp/proj"),
            json!({
                "timestamp": "2026-08-27T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "thread_rolled_back", "num_turns": 4 }
            }),
            json!({
                "timestamp": "2026-08-27T10:00:02Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "foreign-replacement" }
            }),
            session_meta_line_with("2026-08-27T10:00:03Z", "sess-fork-new", "/tmp/proj"),
            session_meta_line_with("2026-08-28T10:00:00Z", "sess-orig", "/tmp/proj"),
            json!({
                "timestamp": "2026-08-28T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "thread_rolled_back", "num_turns": 1 }
            }),
            json!({
                "timestamp": "2026-08-28T10:00:02Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "parent-replacement" }
            }),
            session_meta_line_with("2026-08-28T10:00:03Z", "sess-orig", "/tmp/proj"),
            json!({
                "timestamp": "2026-08-29T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "thread_rolled_back", "num_turns": 3 }
            }),
            json!({
                "timestamp": "2026-08-29T10:00:02Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "fork-first-turn" }
            }),
            session_meta_line_with("2026-08-29T10:00:03Z", "sess-fork-new", "/tmp/proj"),
            json!({
                "timestamp": "2026-08-29T10:00:04Z",
                "type": "event_msg",
                "payload": { "type": "thread_rolled_back", "num_turns": 2 }
            }),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");

        let rollbacks = parse_rollout_file(&rollout_path)
            .expect("parse_rollout_file should succeed")
            .into_iter()
            .filter(|message| message.subtype.as_deref() == Some("thread_rolled_back"))
            .collect::<Vec<_>>();

        assert_eq!(rollbacks.len(), 4);
        assert_eq!(message_data_str(&rollbacks[0], "rollbackOrigin"), None);
        assert_eq!(message_data_str(&rollbacks[1], "rollbackOrigin"), None);
        assert_eq!(
            message_data_str(&rollbacks[2], "rollbackOrigin"),
            Some("fork")
        );
        assert_eq!(message_data_str(&rollbacks[3], "rollbackOrigin"), None);
    }

    #[test]
    fn parse_rollout_file_requires_a_valid_replacement_task_to_classify_a_fork_rollback() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let rollout_path = tmp.path().join("rollout-2026-08-29-invalid-task.jsonl");
        let lines = [
            forked_session_meta_line_with(
                "2026-08-29T10:00:00Z",
                "sess-fork-new",
                "/tmp/proj",
                json!("sess-orig"),
            ),
            session_meta_line_with("2026-08-28T10:00:00Z", "sess-orig", "/tmp/proj"),
            json!({
                "timestamp": "2026-08-29T10:00:01Z",
                "type": "event_msg",
                "payload": { "type": "thread_rolled_back", "num_turns": 2 }
            }),
            json!({
                "timestamp": "2026-08-29T10:00:02Z",
                "type": "event_msg",
                "payload": { "type": "task_started", "turn_id": "" }
            }),
            session_meta_line_with("2026-08-29T10:00:03Z", "sess-fork-new", "/tmp/proj"),
        ];
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&rollout_path, format!("{body}\n")).expect("rollout fixture should be written");

        let rollback = parse_rollout_file(&rollout_path)
            .expect("parse_rollout_file should succeed")
            .into_iter()
            .find(|message| message.subtype.as_deref() == Some("thread_rolled_back"))
            .expect("rollback should be retained");

        assert_eq!(message_data_str(&rollback, "rollbackOrigin"), None);
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
