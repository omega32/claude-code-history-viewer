use super::{
    append_rollout_lineage, find_line_ranges, get_base_path, non_empty_string,
    parse_logical_rollout_bytes, parse_rollout_header, parse_rollout_slice_capturing,
    read_rollout_bytes_bounded, source_stayed_stable, structural_provider_turn_id,
    validate_session_path, verify_captured_history_sources, CodexLineageCapture, CodexParseOutcome,
    CodexParserCheckpoint, CodexParserState, CodexRolloutHeader,
};
use crate::commands::multi_provider::finalize_loaded_messages;
use crate::models::ClaudeMessage;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_DECODED_BYTES: usize = 128 * 1024 * 1024;
const MAX_RECORDS: usize = 250_000;
const MAX_TASKS: usize = 4096;
const MAX_MESSAGES: usize = 16_384;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SubagentActivity {
    schema_version: u32,
    provider: &'static str,
    session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_path: Option<String>,
    boundary: ActivityBoundary,
    inherited: Vec<ClaudeMessage>,
    tasks: Vec<ActivityTask>,
    unassigned: Vec<ClaudeMessage>,
    unassigned_messages: Vec<AgentMessage>,
    diagnostics: Vec<ActivityDiagnostic>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ActivityBoundary {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_ordinal: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ActivityTask {
    task_id: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed_at: Option<String>,
    start_ordinal: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_ordinal_exclusive: Option<u64>,
    records: Vec<ClaudeMessage>,
    messages: Vec<AgentMessage>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    ordinal: u64,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recipient: Option<String>,
    direction: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    trigger_turn: Option<bool>,
    text: String,
    encrypted_payload_unavailable: bool,
}

#[derive(Debug, Serialize)]
struct ActivityDiagnostic {
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ordinal: Option<u64>,
}

fn diagnose(activity: &mut SubagentActivity, code: &'static str, ordinal: Option<u64>) {
    activity
        .diagnostics
        .push(ActivityDiagnostic { code, ordinal });
}

fn parse_outcome(path: &Path, bytes: &[u8]) -> Result<CodexParseOutcome, String> {
    let state = CodexParserState::initial(path);
    let checkpoint = CodexParserCheckpoint {
        byte_offset: 0,
        replace_from: 0,
        source_line: 0,
        state: state.clone(),
        detached_active_lanes: Vec::new(),
    };
    parse_rollout_slice_capturing(
        bytes,
        &find_line_ranges(bytes),
        state,
        checkpoint,
        false,
        0,
        true,
    )
    .map_err(|()| "Codex activity parser could not prove the complete projection".to_string())
}

fn prove_boundary(
    records: &[Value],
    session_id: &str,
    parent: Option<&str>,
) -> Result<u64, &'static str> {
    let Some(first) = records.first() else {
        return Err("missing-child-metadata");
    };
    if first.get("type").and_then(Value::as_str) != Some("session_meta") || session_id.is_empty() {
        return Err("missing-child-metadata");
    }
    let Some(parent) = parent.filter(|parent| *parent != session_id) else {
        return Err("missing-spawn-parent");
    };
    let payload = &first["payload"];
    if ["parent_thread_id", "forked_from_id"]
        .into_iter()
        .any(|key| {
            payload
                .get(key)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(parent))
        })
    {
        return Err("parent-metadata-disagreement");
    }
    let boundary = payload
        .get("subagent_history_start_ordinal")
        .and_then(Value::as_u64)
        .filter(|ordinal| *ordinal > 0 && *ordinal <= records.len() as u64)
        .ok_or("invalid-history-boundary")?;
    if records
        .iter()
        .enumerate()
        .any(|(index, value)| value.get("ordinal").and_then(Value::as_u64) != Some(index as u64))
    {
        return Err("noncontiguous-physical-ordinals");
    }
    let mut inherited_parent_seen = boundary == 1;
    for (ordinal, value) in records.iter().enumerate().skip(1) {
        if value.get("type").and_then(Value::as_str) != Some("session_meta") {
            continue;
        }
        let id = value["payload"].get("id").and_then(Value::as_str);
        if (ordinal as u64) < boundary {
            if id != Some(parent) {
                return Err("inherited-metadata-disagreement");
            }
            inherited_parent_seen = true;
        } else if id != Some(session_id) {
            return Err("child-metadata-disagreement");
        }
    }
    if !inherited_parent_seen {
        return Err("missing-inherited-parent-metadata");
    }
    Ok(boundary)
}

fn routed_message(
    value: &Value,
    previous: Option<&Value>,
    agent_path: Option<&str>,
    physical_index: usize,
) -> Option<AgentMessage> {
    if value.get("type").and_then(Value::as_str) != Some("response_item")
        || value["payload"].get("type").and_then(Value::as_str) != Some("agent_message")
    {
        return None;
    }
    let payload = &value["payload"];
    let author = non_empty_string(payload.get("author")).map(str::to_string);
    let recipient = non_empty_string(payload.get("recipient")).map(str::to_string);
    let direction = match (
        agent_path.is_some() && recipient.as_deref() == agent_path,
        agent_path.is_some() && author.as_deref() == agent_path,
    ) {
        (true, false) => "inbound",
        (false, true) => "outbound",
        _ => "unknown",
    };
    let content = payload.get("content").and_then(Value::as_array);
    let text = content
        .into_iter()
        .flatten()
        .filter(|block| {
            matches!(
                block.get("type").and_then(Value::as_str),
                Some("input_text" | "output_text" | "text")
            )
        })
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<String>();
    let encrypted_payload_unavailable = content
        .into_iter()
        .flatten()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("encrypted_content"))
        || payload.get("encrypted_content").is_some();
    let trigger_turn = previous
        .filter(|previous| {
            previous.get("type").and_then(Value::as_str)
                == Some("inter_agent_communication_metadata")
        })
        .and_then(|previous| previous["payload"].get("trigger_turn"))
        .and_then(Value::as_bool);
    Some(AgentMessage {
        id: non_empty_string(payload.get("id")).map(str::to_string),
        ordinal: physical_index as u64,
        timestamp: value
            .get("timestamp")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        author,
        recipient,
        direction,
        trigger_turn,
        text,
        encrypted_payload_unavailable,
    })
}

fn collect_routed_messages(
    records: &[Value],
    agent_path: Option<&str>,
) -> Result<Vec<(usize, AgentMessage)>, String> {
    let mut routed = Vec::new();
    let mut ids = HashSet::new();
    for (index, value) in records.iter().enumerate() {
        if let Some(message) = routed_message(
            value,
            index
                .checked_sub(1)
                .and_then(|previous| records.get(previous)),
            agent_path,
            index,
        ) {
            if routed.len() >= MAX_MESSAGES {
                return Err("Codex activity exceeds routed-message limit".to_string());
            }
            if let Some(id) = message.id.as_ref() {
                if !ids.insert(id.clone()) {
                    return Err("Codex activity contains duplicate routed-message IDs".to_string());
                }
            }
            routed.push((index, message));
        }
    }
    Ok(routed)
}

fn empty_activity(header: &CodexRolloutHeader, first: Option<&Value>) -> SubagentActivity {
    SubagentActivity {
        schema_version: 1,
        provider: "codex",
        session_id: header.session_id.clone(),
        parent_session_id: header
            .subagent_provenance
            .as_ref()
            .and_then(|provenance| provenance.parent_session_id.clone()),
        agent_path: first
            .and_then(|first| {
                non_empty_string(
                    first["payload"]["source"]["subagent"]["thread_spawn"].get("agent_path"),
                )
            })
            .map(str::to_string),
        boundary: ActivityBoundary {
            status: "unavailable",
            start_ordinal: None,
            reason: None,
        },
        inherited: Vec::new(),
        tasks: Vec::new(),
        unassigned: Vec::new(),
        unassigned_messages: Vec::new(),
        diagnostics: Vec::new(),
    }
}

fn validate_normalized_records(messages: &[ClaudeMessage], session_id: &str) -> Result<(), String> {
    if messages.len() > MAX_RECORDS {
        return Err("Codex activity exceeds normalized-record limit".to_string());
    }
    let mut ids = HashSet::new();
    if messages
        .iter()
        .any(|message| !ids.insert(message.uuid.as_str()))
    {
        return Err("Codex activity contains duplicate normalized record IDs".to_string());
    }
    if messages.iter().any(|message| {
        message.session_id != session_id || message.provider.as_deref() != Some("codex")
    }) {
        return Err(
            "Codex activity normalized record identity disagrees with the selected session"
                .to_string(),
        );
    }
    Ok(())
}

fn load_paginated_unassigned(
    base_path: &Path,
    selected_path: &Path,
    decoded_budget: usize,
) -> Result<SubagentActivity, String> {
    let mut bytes = Vec::new();
    let mut segments = Vec::new();
    let mut capture = CodexLineageCapture {
        carriers: Vec::new(),
        remaining_decoded_bytes: decoded_budget,
    };
    append_rollout_lineage(
        base_path,
        selected_path,
        None,
        &mut HashSet::new(),
        &mut bytes,
        &mut segments,
        Some(&mut capture),
    )?;
    let physical_records = capture
        .carriers
        .iter()
        .try_fold(0usize, |count, carrier| {
            count.checked_add(find_line_ranges(&carrier.bytes).len())
        })
        .ok_or("Codex activity native-record count overflowed")?;
    if physical_records > MAX_RECORDS {
        return Err("Codex activity exceeds native-record limit".to_string());
    }
    let selected = capture
        .carriers
        .last()
        .ok_or("Codex activity lineage is empty")?;
    let first = find_line_ranges(&selected.bytes)
        .first()
        .and_then(|&(start, end)| {
            serde_json::from_slice::<Value>(&selected.bytes[start..end]).ok()
        });
    let mut activity = empty_activity(&selected.header, first.as_ref());
    let records = find_line_ranges(&bytes)
        .into_iter()
        .map(|(start, end)| {
            serde_json::from_slice::<Value>(&bytes[start..end]).unwrap_or(Value::Null)
        })
        .collect::<Vec<_>>();
    activity.unassigned_messages =
        collect_routed_messages(&records, activity.agent_path.as_deref())?
            .into_iter()
            .map(|(_, message)| message)
            .collect();
    activity.unassigned = finalize_loaded_messages(parse_logical_rollout_bytes(
        selected_path,
        &selected.header,
        &bytes,
    )?);
    validate_normalized_records(&activity.unassigned, &activity.session_id)?;
    activity.boundary.reason = Some("paginated-boundary-unavailable");
    diagnose(&mut activity, "paginated-boundary-unavailable", None);
    verify_captured_history_sources(base_path, &capture.carriers)?;
    Ok(activity)
}

fn parse_activity(path: &Path, bytes: &[u8]) -> Result<SubagentActivity, String> {
    if bytes.len() > MAX_DECODED_BYTES {
        return Err("Codex activity exceeds decoded-byte limit".to_string());
    }
    let ranges = find_line_ranges(bytes);
    if ranges.len() > MAX_RECORDS {
        return Err("Codex activity exceeds native-record limit".to_string());
    }
    let records = ranges
        .iter()
        .map(|&(start, end)| {
            serde_json::from_slice::<Value>(&bytes[start..end]).unwrap_or(Value::Null)
        })
        .collect::<Vec<_>>();
    let header = parse_rollout_header(bytes)?.ok_or("Codex activity has no session metadata")?;
    let mut activity = empty_activity(&header, records.first());
    let outcome = parse_outcome(path, bytes)?;
    let raw_len = outcome.messages.len();
    let messages = finalize_loaded_messages(outcome.messages);
    validate_normalized_records(&messages, &activity.session_id)?;
    let routed = collect_routed_messages(&records, activity.agent_path.as_deref())?;
    let proof = if header.history_base.is_some() {
        Err("paginated-boundary-unavailable")
    } else if raw_len != messages.len() {
        Err("post-normalization-source-ambiguity")
    } else if outcome.source_spans.len() != raw_len {
        Err("missing-normalized-source-span")
    } else {
        prove_boundary(
            &records,
            &activity.session_id,
            activity.parent_session_id.as_deref(),
        )
    };
    let boundary = match proof {
        Ok(boundary) => boundary,
        Err(reason) => {
            activity.boundary.reason = Some(reason);
            diagnose(&mut activity, reason, None);
            activity.unassigned = messages;
            activity.unassigned_messages = routed.into_iter().map(|(_, message)| message).collect();
            return Ok(activity);
        }
    };
    activity.boundary = ActivityBoundary {
        status: "verified",
        start_ordinal: Some(boundary),
        reason: None,
    };
    let mut task_indices = HashMap::<String, usize>::new();
    let mut ambiguous = HashSet::<String>::new();
    let mut active = HashSet::<String>::new();
    let mut owners = vec![None::<String>; records.len()];
    let mut unidentified_lifecycle = false;
    for (index, value) in records.iter().enumerate().skip(boundary as usize) {
        let payload = &value["payload"];
        let event = (value.get("type").and_then(Value::as_str) == Some("event_msg"))
            .then(|| payload.get("type").and_then(Value::as_str))
            .flatten();
        let lifecycle = matches!(
            event,
            Some("task_started" | "task_complete" | "turn_aborted")
        );
        let direct_turn_id = non_empty_string(payload.get("turn_id"));
        let nested_turn_id =
            non_empty_string(payload["internal_chat_message_metadata_passthrough"].get("turn_id"));
        let conflicting_turn_ids = direct_turn_id
            .zip(nested_turn_id)
            .is_some_and(|(direct, nested)| direct != nested);
        let task_id = if lifecycle {
            direct_turn_id
        } else {
            structural_provider_turn_id(value)
        };
        if conflicting_turn_ids {
            diagnose(&mut activity, "conflicting-task-owner", Some(index as u64));
        }
        if lifecycle && (task_id.is_none() || conflicting_turn_ids) {
            unidentified_lifecycle = true;
            ambiguous.extend(active.iter().cloned());
        }
        if event == Some("task_started") {
            if let Some(id) = task_id {
                if task_indices.contains_key(id) {
                    ambiguous.insert(id.to_string());
                    diagnose(&mut activity, "duplicate-task-id", Some(index as u64));
                } else {
                    if activity.tasks.len() >= MAX_TASKS {
                        return Err("Codex activity exceeds task limit".to_string());
                    }
                    task_indices.insert(id.to_string(), activity.tasks.len());
                    activity.tasks.push(ActivityTask {
                        task_id: id.to_string(),
                        status: "active",
                        started_at: non_empty_string(value.get("timestamp")).map(str::to_string),
                        completed_at: None,
                        start_ordinal: index as u64,
                        end_ordinal_exclusive: None,
                        records: Vec::new(),
                        messages: Vec::new(),
                    });
                }
                active.insert(id.to_string());
                if unidentified_lifecycle {
                    ambiguous.insert(id.to_string());
                }
                if active.len() > 1 {
                    ambiguous.extend(active.iter().cloned());
                    diagnose(
                        &mut activity,
                        "overlapping-task-ownership",
                        Some(index as u64),
                    );
                }
            } else {
                diagnose(&mut activity, "missing-task-id", Some(index as u64));
            }
        }
        owners[index] = match task_id.filter(|_| !conflicting_turn_ids) {
            Some(id) if active.contains(id) => Some(id.to_string()),
            Some(_) => {
                diagnose(&mut activity, "unknown-task-owner", Some(index as u64));
                None
            }
            None if active.len() == 1
                && !conflicting_turn_ids
                && !lifecycle
                && !unidentified_lifecycle =>
            {
                active.iter().next().cloned()
            }
            None => None,
        };
        if matches!(event, Some("task_complete" | "turn_aborted")) {
            if let Some(id) = task_id {
                if active.remove(id) {
                    if let Some(task_index) = task_indices.get(id) {
                        let task = &mut activity.tasks[*task_index];
                        task.status = if event == Some("task_complete") {
                            "complete"
                        } else {
                            "interrupted"
                        };
                        task.completed_at =
                            non_empty_string(value.get("timestamp")).map(str::to_string);
                        task.end_ordinal_exclusive = Some(index as u64 + 1);
                    }
                } else {
                    diagnose(&mut activity, "unmatched-task-terminal", Some(index as u64));
                }
            } else {
                diagnose(&mut activity, "missing-task-id", Some(index as u64));
            }
        }
    }
    for (record, &(start, end)) in messages.into_iter().zip(&outcome.source_spans) {
        let start = start
            .checked_sub(1)
            .ok_or("Codex activity has invalid source position")?;
        let end = end
            .checked_sub(1)
            .ok_or("Codex activity has invalid source position")?;
        if (end as u64) < boundary {
            activity.inherited.push(record);
            continue;
        }
        if (start as u64) < boundary {
            diagnose(
                &mut activity,
                "cross-history-record-contribution",
                Some(start as u64),
            );
            activity.unassigned.push(record);
            continue;
        }
        let owner = owners.get(start).and_then(Option::as_ref).filter(|owner| {
            !ambiguous.contains(*owner)
                && owners.get(start..=end).is_some_and(|range| {
                    range
                        .iter()
                        .all(|candidate| candidate.as_ref() == Some(*owner))
                })
        });
        if let Some(task_index) = owner.and_then(|owner| task_indices.get(owner)) {
            activity.tasks[*task_index].records.push(record);
        } else {
            diagnose(
                &mut activity,
                "unassigned-record-ownership",
                Some(start as u64),
            );
            activity.unassigned.push(record);
        }
    }
    for (source_index, message) in routed
        .into_iter()
        .filter(|(source_index, _)| (*source_index as u64) >= boundary)
    {
        if let Some(index) = owners[source_index]
            .as_ref()
            .filter(|owner| !ambiguous.contains(*owner))
            .and_then(|owner| task_indices.get(owner))
        {
            activity.tasks[*index].messages.push(message);
        } else {
            activity.unassigned_messages.push(message);
        }
    }
    activity
        .tasks
        .retain(|task| !ambiguous.contains(&task.task_id));
    Ok(activity)
}

pub(crate) fn load(session_path: &str) -> Result<SubagentActivity, String> {
    let path = validate_session_path(Path::new(session_path), session_path)?;
    let before = fs::metadata(&path).map_err(|error| error.to_string())?;
    let bytes = read_rollout_bytes_bounded(&path, MAX_DECODED_BYTES)?;
    let digest = blake3::hash(&bytes);
    let activity =
        if parse_rollout_header(&bytes)?.is_some_and(|header| header.history_base.is_some()) {
            let base = PathBuf::from(get_base_path().ok_or("Codex base path not found")?);
            load_paginated_unassigned(&base, &path, MAX_DECODED_BYTES)?
        } else {
            parse_activity(&path, &bytes)?
        };
    let after = fs::metadata(&path).map_err(|error| error.to_string())?;
    let verified_bytes = read_rollout_bytes_bounded(&path, MAX_DECODED_BYTES)?;
    if !source_stayed_stable(&before, &after, before.len() as usize)
        || blake3::hash(&verified_bytes) != digest
    {
        return Err("Codex activity source changed during read".to_string());
    }
    if serde_json::to_vec_pretty(&activity)
        .map_err(|error| error.to_string())?
        .len()
        >= MAX_DECODED_BYTES
    {
        return Err("Codex activity exceeds envelope-byte limit".to_string());
    }
    Ok(activity)
}

#[cfg(test)]
mod tests {
    use super::super::parse_rollout_bytes;
    use super::super::tests::EnvVarGuard;
    use super::*;
    use std::fmt::Write as _;

    fn fixture() -> Vec<Value> {
        vec![
            serde_json::json!({"type":"session_meta","payload":{"id":"child","source":{"subagent":{"thread_spawn":{"parent_thread_id":"parent","agent_path":"/root/audit"}}},"forked_from_id":"parent","subagent_history_start_ordinal":3}}),
            serde_json::json!({"type":"session_meta","payload":{"id":"parent"}}),
            serde_json::json!({"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Inherited answer"}]}}),
            serde_json::json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"task-one"}}),
            serde_json::json!({"type":"inter_agent_communication_metadata","payload":{"trigger_turn":true}}),
            serde_json::json!({"type":"response_item","payload":{"type":"agent_message","id":"assignment-one","author":"/root","recipient":"/root/audit","content":[{"type":"input_text","text":"Assignment header"},{"type":"encrypted_content","encrypted_content":"secret-ciphertext"}],"internal_chat_message_metadata_passthrough":{"turn_id":"task-one"}}}),
            serde_json::json!({"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Own answer"}],"internal_chat_message_metadata_passthrough":{"turn_id":"task-one"}}}),
            serde_json::json!({"type":"event_msg","payload":{"type":"task_complete","turn_id":"task-one"}}),
        ]
    }

    fn bytes(mut records: Vec<Value>) -> Vec<u8> {
        records
            .iter_mut()
            .enumerate()
            .fold(String::new(), |mut body, (i, value)| {
                value["ordinal"] = serde_json::json!(i);
                value["timestamp"] = serde_json::json!(format!("2026-10-07T20:00:{i:02}Z"));
                writeln!(body, "{value}").unwrap();
                body
            })
            .into_bytes()
    }

    #[test]
    #[serial_test::serial]
    fn subagent_activity_paginated_fallback_bounds_and_revalidates_the_complete_projection() {
        let temp = tempfile::TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("CODEX_HOME", temp.path());
        let sessions = temp.path().join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        let thread = "01a10000-0000-7000-8000-000000000001";
        let ancestor_id = "01a10000-0000-7000-8000-000000000002";
        let selected_id = "01a10000-0000-7000-8000-000000000003";
        let ancestor = sessions.join(format!(
            "rollout-2026-10-07T20-00-00-{thread}_{ancestor_id}.jsonl"
        ));
        let selected = sessions.join(format!(
            "rollout-2026-10-07T20-00-01-{thread}_{selected_id}.jsonl"
        ));
        let ancestor_bytes = bytes(vec![
            serde_json::json!({"type":"session_meta","payload":{"id":thread,"history_mode":"paginated"}}),
            serde_json::json!({"type":"response_item","payload":{"type":"message","id":"ancestor-answer","role":"assistant","content":[{"type":"output_text","text":"Before pagination"}]}}),
        ]);
        fs::write(&ancestor, &ancestor_bytes).unwrap();
        let mut continuation = fixture()[0].clone();
        continuation["ordinal"] = serde_json::json!(2);
        continuation["timestamp"] = serde_json::json!("2026-10-07T20:00:02Z");
        continuation["payload"]["id"] = serde_json::json!(thread);
        continuation["payload"]["history_mode"] = serde_json::json!("paginated");
        continuation["payload"]["history_base"] = serde_json::json!({"thread_id":ancestor_id,"end_byte_offset":ancestor_bytes.len(),"end_ordinal_exclusive":2});
        let answer = serde_json::json!({"timestamp":"2026-10-07T20:00:03Z","ordinal":3,"type":"response_item","payload":{"type":"message","id":"selected-answer","role":"assistant","content":[{"type":"output_text","text":"After pagination"}]}});
        fs::write(&selected, format!("{continuation}\n{answer}\n")).unwrap();
        let activity =
            load_paginated_unassigned(temp.path(), &selected, MAX_DECODED_BYTES).unwrap();
        assert_eq!(
            activity.boundary.reason,
            Some("paginated-boundary-unavailable")
        );
        assert!(activity.tasks.is_empty());
        assert_eq!(activity.unassigned.len(), 2);
        assert!(activity.unassigned.iter().all(
            |record| record.session_id == thread && record.provider.as_deref() == Some("codex")
        ));
        assert!(load_paginated_unassigned(temp.path(), &selected, 1).is_err());
        let mut capture = CodexLineageCapture {
            carriers: Vec::new(),
            remaining_decoded_bytes: MAX_DECODED_BYTES,
        };
        append_rollout_lineage(
            temp.path(),
            &selected,
            None,
            &mut HashSet::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            Some(&mut capture),
        )
        .unwrap();
        fs::write(&ancestor, b"changed ancestor\n").unwrap();
        assert!(verify_captured_history_sources(temp.path(), &capture.carriers).is_err());
    }

    #[test]
    fn subagent_activity_verified_boundary_does_not_label_inherited_routes_as_child_activity() {
        let mut records = fixture();
        let mut inherited_route = records[5].clone();
        inherited_route["payload"]["id"] = serde_json::json!("inherited-assignment");
        records.insert(3, inherited_route);
        records[0]["payload"]["subagent_history_start_ordinal"] = serde_json::json!(4);
        let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
        assert_eq!(activity.boundary.status, "verified");
        assert!(activity.unassigned_messages.is_empty());
        assert_eq!(activity.tasks[0].messages.len(), 1);
        assert_eq!(
            activity.tasks[0].messages[0].id.as_deref(),
            Some("assignment-one")
        );
    }

    #[test]
    fn subagent_activity_unavailable_boundary_retains_unassigned_routed_evidence() {
        let mut records = fixture();
        records[0]["payload"]["subagent_history_start_ordinal"] = Value::Null;
        let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
        assert!(activity.tasks.is_empty());
        assert_eq!(activity.unassigned_messages.len(), 1);
        assert!(activity.unassigned_messages[0].encrypted_payload_unavailable);
        assert_eq!(activity.unassigned_messages[0].text, "Assignment header");
        let malformed = String::from_utf8(bytes(fixture()))
            .unwrap()
            .replace("\"ordinal\":5", "\"ordinal\":\"invalid\"");
        let activity = parse_activity(Path::new("child.jsonl"), malformed.as_bytes()).unwrap();
        assert_eq!(
            activity.boundary.reason,
            Some("noncontiguous-physical-ordinals")
        );
        assert_eq!(activity.unassigned_messages[0].ordinal, 5);
        assert!(activity.unassigned_messages[0].encrypted_payload_unavailable);
    }

    #[test]
    fn subagent_activity_unidentified_lifecycle_never_reuses_the_previous_task_lane() {
        for terminal in ["task_started", "task_complete", "turn_aborted"] {
            let mut records = fixture();
            records.insert(
                6,
                serde_json::json!({"type":"event_msg","payload":{"type":terminal}}),
            );
            records[7]["payload"]
                .as_object_mut()
                .unwrap()
                .remove("internal_chat_message_metadata_passthrough");
            let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
            assert!(
                activity.tasks.is_empty(),
                "unidentified {terminal} must quarantine the earlier lane"
            );
            assert!(activity.unassigned.iter().any(|record| record
                .content
                .as_ref()
                .is_some_and(|content| content.to_string().contains("Own answer"))));
        }
    }

    #[test]
    fn subagent_activity_separates_history_tasks_and_encrypted_assignments() {
        let bytes = bytes(fixture());
        let activity = parse_activity(Path::new("rollout-child.jsonl"), &bytes).unwrap();
        assert_eq!(activity.boundary.status, "verified");
        assert_eq!(activity.boundary.start_ordinal, Some(3));
        assert_eq!(activity.inherited.len(), 1);
        assert_eq!(activity.tasks.len(), 1);
        assert_eq!(activity.tasks[0].status, "complete");
        assert_eq!(activity.tasks[0].end_ordinal_exclusive, Some(8));
        assert_eq!(activity.tasks[0].messages[0].trigger_turn, Some(true));
        assert!(activity.tasks[0].messages[0].encrypted_payload_unavailable);
        let encoded = serde_json::to_string(&activity).unwrap();
        assert!(!encoded.contains("secret-ciphertext"));
        assert!(!activity.tasks[0]
            .records
            .iter()
            .any(|record| record.message_type == "user"));
        let ordinary = finalize_loaded_messages(
            parse_rollout_bytes(Path::new("rollout-child.jsonl"), &bytes).unwrap(),
        );
        let mut grouped = activity.inherited;
        grouped.extend(activity.tasks.into_iter().flat_map(|task| task.records));
        grouped.extend(activity.unassigned);
        assert_eq!(
            serde_json::to_value(grouped).unwrap(),
            serde_json::to_value(ordinary).unwrap()
        );
    }

    #[test]
    fn subagent_activity_invalid_boundary_keeps_ordinary_records_unassigned() {
        for bad in [
            Value::Null,
            serde_json::json!(-1),
            serde_json::json!(80),
            serde_json::json!("3"),
        ] {
            let mut records = fixture();
            records[0]["payload"]["subagent_history_start_ordinal"] = bad;
            let activity =
                parse_activity(Path::new("rollout-child.jsonl"), &bytes(records)).unwrap();
            assert_eq!(activity.boundary.status, "unavailable");
            assert!(activity.inherited.is_empty());
            assert!(activity.tasks.is_empty());
            assert!(!activity.unassigned.is_empty());
        }
    }

    #[test]
    fn subagent_activity_uses_native_reactivation_and_interruption_without_quoted_triggers() {
        let mut records = fixture();
        records.extend([
            serde_json::json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"task-two"}}),
            serde_json::json!({"type":"response_item","payload":{"type":"agent_message","id":"followup","author":"/root","recipient":"/root/audit","content":[{"type":"input_text","text":"Message Type: NEW_TASK"}],"internal_chat_message_metadata_passthrough":{"turn_id":"task-two"}}}),
            serde_json::json!({"type":"event_msg","payload":{"type":"turn_aborted","turn_id":"task-two"}}),
        ]);
        let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
        assert_eq!(activity.tasks.len(), 2);
        assert_eq!(activity.tasks[1].status, "interrupted");
        assert_eq!(activity.tasks[1].messages[0].trigger_turn, None);
        assert_eq!(
            activity.tasks[1].records.last().unwrap().subtype.as_deref(),
            Some("interruption")
        );
    }

    #[test]
    fn subagent_activity_retains_overlapping_and_duplicate_task_records_unassigned() {
        for task_id in ["task-two", "task-one"] {
            let mut records = fixture();
            records.insert(6, serde_json::json!({"type":"event_msg","payload":{"type":"task_started","turn_id":task_id}}));
            let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
            assert!(activity.tasks.is_empty());
            assert!(!activity.unassigned.is_empty());
            assert!(!activity.unassigned_messages.is_empty());
        }
    }

    #[test]
    fn subagent_activity_does_not_split_merged_tool_results_at_the_history_boundary() {
        let mut records = fixture();
        records[2] = serde_json::json!({"type":"response_item","payload":{"type":"function_call","name":"exec_command","call_id":"call-one","arguments":"{}"}});
        records.insert(6, serde_json::json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"call-one","output":"child tool result"}}));
        let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
        assert!(activity.inherited.is_empty());
        assert!(activity
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "cross-history-record-contribution"));
        assert!(activity.unassigned.iter().any(|record| record
            .content
            .as_ref()
            .is_some_and(|content| content.to_string().contains("child tool result"))));
    }

    #[test]
    fn subagent_activity_rejects_duplicate_routed_ids_and_invalid_physical_proof() {
        let mut records = fixture();
        records.insert(6, records[5].clone());
        assert!(parse_activity(Path::new("child.jsonl"), &bytes(records))
            .unwrap_err()
            .contains("duplicate routed-message"));
        let mut records = fixture();
        records[0]["payload"]["parent_thread_id"] = serde_json::json!("wrong-parent");
        let activity = parse_activity(Path::new("child.jsonl"), &bytes(records)).unwrap();
        assert_eq!(
            activity.boundary.reason,
            Some("parent-metadata-disagreement")
        );
        let mut records: Vec<Value> = String::from_utf8(bytes(fixture()))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        records[2]["ordinal"] = serde_json::json!(90);
        let body = records.into_iter().fold(String::new(), |mut body, record| {
            writeln!(body, "{record}").unwrap();
            body
        });
        let activity = parse_activity(Path::new("child.jsonl"), body.as_bytes()).unwrap();
        assert_eq!(
            activity.boundary.reason,
            Some("noncontiguous-physical-ordinals")
        );
    }

    #[test]
    fn subagent_activity_append_refresh_preserves_native_task_identity_and_ordinary_projection() {
        let prefix =
            parse_activity(Path::new("child.jsonl"), &bytes(fixture()[..7].to_vec())).unwrap();
        let complete = parse_activity(Path::new("child.jsonl"), &bytes(fixture())).unwrap();
        assert_eq!(prefix.tasks[0].task_id, complete.tasks[0].task_id);
        assert_eq!(prefix.tasks[0].status, "active");
        assert_eq!(complete.tasks[0].status, "complete");
        assert_eq!(
            prefix.tasks[0].start_ordinal,
            complete.tasks[0].start_ordinal
        );
    }

    #[test]
    fn subagent_activity_redacted_reported_sessions_have_distinct_child_reply_counts() {
        let cases = [
            (
                include_bytes!(
                    "../../tests/fixtures/codex-subagent-activity/docs-and-device.jsonl"
                )
                .as_slice(),
                21,
                4,
            ),
            (
                include_bytes!("../../tests/fixtures/codex-subagent-activity/cursor-fix.jsonl")
                    .as_slice(),
                19,
                8,
            ),
            (
                include_bytes!("../../tests/fixtures/codex-subagent-activity/window-fix.jsonl")
                    .as_slice(),
                20,
                7,
            ),
            (
                include_bytes!(
                    "../../tests/fixtures/codex-subagent-activity/navigation-audit.jsonl"
                )
                .as_slice(),
                13,
                1,
            ),
            (
                include_bytes!("../../tests/fixtures/codex-subagent-activity/scroll-audit.jsonl")
                    .as_slice(),
                12,
                1,
            ),
        ];
        for (bytes, boundary, replies) in cases {
            let activity = parse_activity(Path::new("fixture.jsonl"), bytes).unwrap();
            assert_eq!(activity.boundary.start_ordinal, Some(boundary));
            assert_eq!(
                activity
                    .tasks
                    .iter()
                    .flat_map(|task| &task.records)
                    .filter(|record| record.message_type == "assistant")
                    .count(),
                replies
            );
            assert!(!activity.tasks.is_empty());
            assert!(
                activity.unassigned.is_empty(),
                "unexpected ownership diagnostics: {:?}",
                activity.diagnostics
            );
            assert!(activity
                .inherited
                .iter()
                .any(|record| record.message_type == "assistant"));
        }
    }
}
