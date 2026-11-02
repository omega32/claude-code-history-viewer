use crate::models::ClaudeMessage;
use std::collections::{HashMap, HashSet};

// Advance for result-affecting native prompt changes, not unrelated releases.
// The owning contract is specs/2026-11-02-prompt-semantics-revision/spec.md.
pub(crate) const PROMPT_SEMANTICS_REVISION: &str = "1";

pub(crate) fn valid_record_ref(record: &crate::models::RecordRef) -> bool {
    let mut parts = record.scheme.split('-');
    record.version == 1
        && record.scope == "session"
        && record.scheme.len() <= 64
        && record
            .scheme
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_lowercase)
        && parts.all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
        && record.id.encode_utf16().count() <= 512
        && record
            .id
            .chars()
            .any(|character| !ecmascript_trim_character(character))
        && !record
            .id
            .chars()
            .any(|character| character <= '\u{001f}' || character == '\u{007f}')
}

// Rust's whitespace set differs from JavaScript trim (notably BOM and NEL).
fn ecmascript_trim_character(character: char) -> bool {
    matches!(character, '\u{0009}'..='\u{000d}' | '\u{0020}' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}

pub(crate) fn select_candidates(
    mut messages: Vec<ClaudeMessage>,
    provider: &str,
    task_ids: &HashSet<String>,
) -> Vec<ClaudeMessage> {
    let mut counts = HashMap::new();
    for record in messages
        .iter()
        .filter_map(|message| message.record_ref.as_ref())
        .filter(|record| valid_record_ref(record))
    {
        *counts
            .entry((record.scheme.clone(), record.id.clone()))
            .or_insert(0usize) += 1;
    }
    // Omitted records still invalidate a candidate's identity in the full dump.
    for message in &mut messages {
        if message.record_ref.as_ref().is_some_and(|record| {
            valid_record_ref(record)
                && counts
                    .get(&(record.scheme.clone(), record.id.clone()))
                    .is_some_and(|count| *count > 1)
        }) {
            message.record_ref = None;
        }
    }
    messages
        .into_iter()
        .filter(|message| {
            message.message_type == "user"
                && (provider != "codex" || message.subtype.as_deref() == Some("authored_user"))
                && message
                    .data
                    .as_ref()
                    .and_then(|data| {
                        data.get(if provider == "codex" {
                            "providerTurnId"
                        } else {
                            "providerPromptId"
                        })
                    })
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|task_id| task_ids.contains(task_id))
        })
        .collect()
}
