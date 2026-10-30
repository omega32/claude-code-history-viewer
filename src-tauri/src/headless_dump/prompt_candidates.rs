use super::{block_on, emit_read_only_audit_json, resolve_session_path};
use crate::cli_args::extract_flag_value;
use crate::commands::multi_provider::load_provider_messages;
use crate::models::ClaudeMessage;
use std::collections::{HashMap, HashSet};
use std::path::Path;

const COMMAND: &str = "--dump-session-prompt-candidates";
const USAGE: &str = "Usage: --dump-session-prompt-candidates <absolute-carrier-path|canonical-session-uuid> --provider <codex|claude> --task-id <canonical-uuid> [--task-id <canonical-uuid> ...] [--format json] [--output <new-file>]";

struct Options {
    session: String,
    provider: String,
    task_ids: HashSet<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, &'static str> {
        let mut singletons = HashMap::new();
        let mut task_ids = HashSet::new();
        let mut index = 1;
        while let Some(argument) = args.get(index) {
            if argument.len() > 32_768 || argument.chars().any(char::is_control) {
                return Err("Invalid bounded argument");
            }
            let flag = argument.split('=').next().unwrap_or_default();
            if !matches!(
                flag,
                COMMAND | "--provider" | "--task-id" | "--format" | "--output"
            ) {
                return Err("Unknown or conflicting option");
            }
            let value = extract_flag_value(&args[index..], flag).ok_or("Missing option value")?;
            if value.is_empty() || value.len() > 32_768 || value.chars().any(char::is_control) {
                return Err("Invalid bounded option value");
            }
            if flag == "--task-id" {
                if !canonical_uuid(&value) || !task_ids.insert(value) || task_ids.len() > 8 {
                    return Err("Require one to eight distinct canonical task UUIDs");
                }
            } else if singletons.insert(flag, value).is_some() {
                return Err("Repeated singleton option");
            }
            index += if argument == flag { 2 } else { 1 };
        }
        let session = singletons
            .remove(COMMAND)
            .ok_or("Missing session selector")?;
        if !Path::new(&session).is_absolute() && !canonical_uuid(&session) {
            return Err("Require an absolute carrier path or canonical session UUID");
        }
        let provider = singletons.remove("--provider").ok_or("Missing provider")?;
        if !matches!(provider.as_str(), "codex" | "claude") {
            return Err("Unsupported prompt-candidate provider");
        }
        if task_ids.is_empty() {
            return Err("Missing task UUID");
        }
        if singletons
            .get("--format")
            .is_some_and(|format| format != "json")
        {
            return Err("Only JSON format is supported");
        }
        Ok(Self {
            session,
            provider,
            task_ids,
        })
    }
}

fn canonical_uuid(value: &str) -> bool {
    value.len() == 36
        && uuid::Uuid::parse_str(value).is_ok_and(|id| id.hyphenated().to_string() == value)
}

pub fn run_dump_session_prompt_candidates(args: &[String]) -> i32 {
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return 2;
        }
    };
    let result = block_on(async {
        let path = resolve_session_path(&options.provider, &options.session).await?;
        load_provider_messages(options.provider.clone(), path).await
    });
    match result {
        Ok(messages) => emit_read_only_audit_json(
            args,
            &select_candidates(messages, &options.provider, &options.task_ids),
        ),
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

fn valid_record_ref(record: &crate::models::RecordRef) -> bool {
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

fn select_candidates(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_dump::run_dump_session;
    use serde_json::{json, Value};
    use tempfile::TempDir;

    const TASK_ONE: &str = "00000000-0000-4000-8000-000000000001";
    const TASK_TWO: &str = "00000000-0000-4000-8000-000000000002";

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn message(patch: Value) -> ClaudeMessage {
        let mut row = json!({
            "uuid":"record", "sessionId":"session", "timestamp":"2026-10-30T00:00:00Z",
            "type":"user", "subtype":"authored_user", "role":"user", "content":"verbatim",
            "data":{"providerTurnId":TASK_ONE},
            "recordRef":{"version":1,"scope":"session","scheme":"native","id":"record"}
        });
        row.as_object_mut()
            .unwrap()
            .extend(patch.as_object().unwrap().clone());
        serde_json::from_value(row).unwrap()
    }

    #[test]
    fn prompt_candidates_count_all_valid_refs_before_narrowing() {
        let rows = vec![
            message(json!({"uuid":"authored"})),
            message(json!({"uuid":"omitted-steer","subtype":"steer"})),
            message(json!({"uuid":"no-ref","recordRef":null})),
            message(
                json!({"uuid":"unique","isSidechain":true,"recordRef":{"version":1,"scope":"session","scheme":"native","id":"other"},"data":{"providerTurnId":TASK_TWO,"keep":[1,2]}}),
            ),
            message(
                json!({"uuid":"invalid-omitted","type":"assistant","recordRef":{"version":2,"scope":"session","scheme":"native","id":"other"}}),
            ),
            message(json!({"uuid":"unbound","data":null})),
        ];
        let selected = select_candidates(
            rows,
            "codex",
            &HashSet::from([TASK_ONE.to_string(), TASK_TWO.to_string()]),
        );
        assert_eq!(
            selected
                .iter()
                .map(|row| row.uuid.as_str())
                .collect::<Vec<_>>(),
            ["authored", "no-ref", "unique"]
        );
        assert!(
            selected[0].record_ref.is_none(),
            "omitted steer still owns a colliding reference"
        );
        assert!(selected[1].record_ref.is_none());
        assert_eq!(
            selected[2].record_ref.as_ref().unwrap().id,
            "other",
            "invalid envelopes must not collide"
        );
        assert_eq!(selected[2].is_sidechain, Some(true));
        assert_eq!(selected[2].data.as_ref().unwrap()["keep"], json!([1, 2]));
    }

    #[test]
    fn prompt_candidates_keep_duplicate_bindings_and_original_claude_records() {
        let mut first = message(
            json!({"uuid":"first","subtype":null,"data":{"providerPromptId":TASK_ONE},"recordRef":null}),
        );
        first.content = Some(
            json!([{"type":"text","text":"<ide_opened_file>literal</ide_opened_file>"},{"type":"image","source":{"data":"opaque"}}]),
        );
        let mut second = first.clone();
        second.uuid = "second".to_string();
        second.is_sidechain = Some(true);
        let third =
            message(json!({"uuid":"third","data":{"providerPromptId":TASK_TWO},"recordRef":null}));
        let expected =
            serde_json::to_value(vec![first.clone(), second.clone(), third.clone()]).unwrap();
        let rows = vec![
            first,
            message(
                json!({"uuid":"assistant","type":"assistant","data":{"providerPromptId":TASK_ONE}}),
            ),
            second,
            third,
        ];
        let selected = select_candidates(
            rows,
            "claude",
            &HashSet::from([TASK_TWO.to_string(), TASK_ONE.to_string()]),
        );
        assert_eq!(serde_json::to_value(selected).unwrap(), expected);
    }

    #[test]
    fn prompt_candidates_reject_malformed_or_unbounded_options() {
        let base = args(&[
            "viewer",
            "--dump-session-prompt-candidates",
            TASK_ONE,
            "--provider",
            "codex",
            "--task-id",
            TASK_ONE,
        ]);
        for suffix in [
            args(&["--provider", "claude"]),
            args(&["--format", "yaml"]),
            args(&["--format="]),
            args(&["--output"]),
            args(&["--output", ""]),
            args(&["--cursor", "opaque"]),
            args(&["--dump-session", TASK_ONE]),
            args(&["--export", TASK_ONE]),
            args(&["--task-id", TASK_ONE]),
            args(&["--task-id", "short"]),
        ] {
            let mut invalid = base.clone();
            invalid.extend(suffix);
            assert_eq!(run_dump_session_prompt_candidates(&invalid), 2);
        }
        for (flag, value) in [
            ("--provider", "hermes"),
            ("--task-id", "ABCDEF00-0000-4000-8000-000000000001"),
            ("--dump-session-prompt-candidates", "prefix"),
        ] {
            let mut invalid = base.clone();
            let index = invalid.iter().position(|arg| arg == flag).unwrap();
            invalid[index + 1] = value.to_string();
            assert_eq!(run_dump_session_prompt_candidates(&invalid), 2);
        }
        let mut invalid = base;
        for index in 2..=9 {
            invalid.extend(args(&[
                "--task-id",
                &format!("00000000-0000-4000-8000-{index:012}"),
            ]));
        }
        assert_eq!(run_dump_session_prompt_candidates(&invalid), 2);
        assert_eq!(
            run_dump_session_prompt_candidates(&args(&[
                "viewer",
                "--dump-session-prompt-candidates",
                TASK_ONE,
                "--provider",
                "codex"
            ])),
            2
        );
    }

    #[test]
    fn prompt_candidates_reference_validator_matches_javascript_boundaries() {
        for (id, expected) in [
            ("x".repeat(512), true),
            ("x".repeat(513), false),
            ("😀".repeat(256), true),
            ("😀".repeat(257), false),
            ("\u{feff}".to_string(), false),
            ("\u{0085}".to_string(), true),
            ("\u{2000}\u{3000}".to_string(), false),
            ("x\u{001f}".to_string(), false),
            ("x\u{007f}".to_string(), false),
            (String::new(), false),
        ] {
            assert_eq!(
                valid_record_ref(&crate::models::RecordRef::session("native", id)),
                expected
            );
        }
        for (scheme, expected) in [
            ("a".repeat(64), true),
            ("a".repeat(65), false),
            ("native-1".to_string(), true),
            ("Native".to_string(), false),
            ("native--1".to_string(), false),
            ("native-".to_string(), false),
            ("-native".to_string(), false),
            ("1native".to_string(), false),
        ] {
            assert_eq!(
                valid_record_ref(&crate::models::RecordRef::session(&scheme, "id")),
                expected
            );
        }
        let mut record = crate::models::RecordRef::session("native", "id");
        record.version = 2;
        assert!(!valid_record_ref(&record));
        record.version = 1;
        record.scope = "other".to_string();
        assert!(!valid_record_ref(&record));
    }

    #[test]
    fn prompt_candidates_options_accept_equals_and_eight_distinct_selectors() {
        let mut argv = args(&[
            "viewer",
            &format!("{COMMAND}={TASK_ONE}"),
            "--provider=claude",
            "--format=json",
        ]);
        for index in 1..=8 {
            argv.push(format!("--task-id=00000000-0000-4000-8000-{index:012}"));
        }
        assert_eq!(Options::parse(&argv).unwrap().task_ids.len(), 8);
        for suffix in [
            args(&["--format=json"]),
            args(&["--output="]),
            args(&["--unknown=1"]),
            args(&["unexpected"]),
        ] {
            let mut invalid = argv.clone();
            invalid.extend(suffix);
            assert!(Options::parse(&invalid).is_err());
        }
    }

    fn assert_dump_parity(provider: &str, lines: Vec<Value>, expected_count: usize) {
        let profile = crate::profile_paths::TestProfile::new().unwrap();
        let temp = TempDir::new().unwrap();
        let source_directory = if provider == "codex" {
            profile.path().join(".codex/sessions")
        } else {
            temp.path().to_path_buf()
        };
        std::fs::create_dir_all(&source_directory).unwrap();
        let input = source_directory.join("input.jsonl");
        let ordinary_output = temp.path().join("ordinary.json");
        let output = temp.path().join("candidates.json");
        let source = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&input, &source).unwrap();
        assert_eq!(
            run_dump_session(&args(&[
                "viewer",
                "--dump-session",
                input.to_str().unwrap(),
                "--provider",
                provider,
                "--output",
                ordinary_output.to_str().unwrap()
            ])),
            0
        );
        let ordinary: Vec<Value> =
            serde_json::from_slice(&std::fs::read(ordinary_output).unwrap()).unwrap();
        let expected: Vec<_> = ordinary
            .into_iter()
            .filter(|row| {
                let binding = if provider == "codex" {
                    "providerTurnId"
                } else {
                    "providerPromptId"
                };
                row["type"] == "user"
                    && (provider != "codex" || row["subtype"] == "authored_user")
                    && matches!(row["data"][binding].as_str(), Some(TASK_ONE | TASK_TWO))
            })
            .collect();
        assert_eq!(expected.len(), expected_count);
        assert!(
            expected[0].get("recordRef").is_none(),
            "complete normalization must invalidate the omitted-row collision"
        );
        let argv = args(&[
            "viewer",
            "--dump-session-prompt-candidates",
            input.to_str().unwrap(),
            "--provider",
            provider,
            "--task-id",
            TASK_TWO,
            "--task-id",
            TASK_ONE,
            "--output",
            output.to_str().unwrap(),
        ]);
        assert_eq!(run_dump_session_prompt_candidates(&argv), 0);
        let actual: Vec<Value> = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            actual, expected,
            "narrow transport must retain exact ordinary normalized candidates"
        );
        assert_eq!(
            run_dump_session_prompt_candidates(&argv),
            1,
            "output is create-new"
        );
        assert_eq!(std::fs::read_to_string(input).unwrap(), source);
    }

    #[test]
    fn prompt_candidates_codex_command_matches_complete_normalization() {
        let user = |id: &str, task: &str, text: &str| json!({"type":"event_msg","timestamp":"2026-10-30T00:00:01Z","payload":{"type":"item_completed","turn_id":task,"item":{"type":"UserMessage","id":id,"content":[{"type":"text","text":text}]}}});
        assert_dump_parity(
            "codex",
            vec![
                json!({"type":"session_meta","timestamp":"2026-10-30T00:00:00Z","payload":{"id":"session","cwd":"C:/Repo"}}),
                user("collision", TASK_ONE, "original"),
                user("collision", TASK_ONE, "steer"),
                user("unique", TASK_TWO, "second prompt"),
                json!({"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"omit"}]}}),
            ],
            2,
        );
    }

    #[test]
    fn prompt_candidates_claude_command_matches_complete_normalization() {
        let user = |uuid: &str, prompt: Option<&str>, text: &str| json!({"type":"user","uuid":uuid,"sessionId":"session","timestamp":"2026-10-30T00:00:00Z","promptId":prompt,"promptSource":"user","message":{"role":"user","content":text}});
        assert_dump_parity(
            "claude",
            vec![
                user("collision", Some(TASK_ONE), "original"),
                user("collision", None, "omitted unbound record"),
                user("second", Some(TASK_TWO), "second prompt"),
                user("third", Some(TASK_TWO), "duplicate binding"),
                json!({"type":"assistant","uuid":"assistant","sessionId":"session","timestamp":"2026-10-30T00:00:01Z","message":{"role":"assistant","content":"omit"}}),
            ],
            3,
        );
    }
}
