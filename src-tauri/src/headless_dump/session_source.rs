use super::emit_read_only_audit_json;
use crate::cli_args::extract_flag_value;
use crate::providers::codex;
use std::collections::HashMap;

const COMMAND: &str = "--session-source";
const USAGE: &str = "Usage: --session-source <canonical-session-uuid> --provider codex [--format json] [--output <new-file>]";

struct Options {
    session_id: String,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, &'static str> {
        let mut values = HashMap::new();
        let mut index = 1;
        while let Some(argument) = args.get(index) {
            if argument.len() > 32_768 || argument.chars().any(char::is_control) {
                return Err("Invalid bounded argument");
            }
            let flag = argument.split('=').next().unwrap_or_default();
            if !matches!(flag, COMMAND | "--provider" | "--format" | "--output") {
                return Err("Unknown or conflicting option");
            }
            let value = extract_flag_value(&args[index..], flag).ok_or("Missing option value")?;
            if value.is_empty() || value.len() > 32_768 || value.chars().any(char::is_control) {
                return Err("Invalid bounded option value");
            }
            if values.insert(flag, value).is_some() {
                return Err("Repeated singleton option");
            }
            index += if argument == flag { 2 } else { 1 };
        }
        let session_id = values.remove(COMMAND).ok_or("Missing session UUID")?;
        if !codex::is_canonical_session_id(&session_id) {
            return Err("Require one canonical session UUID");
        }
        if values.get("--provider").map(String::as_str) != Some("codex") {
            return Err("Session source requires provider codex");
        }
        if values.get("--format").is_some_and(|value| value != "json") {
            return Err("Only JSON format is supported");
        }
        Ok(Self { session_id })
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Envelope {
    schema_version: u32,
    kind: &'static str,
    source: Option<codex::CodexSessionSource>,
}

pub fn run_session_source(args: &[String]) -> i32 {
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return 2;
        }
    };
    match codex::load_session_source(&options.session_id) {
        Ok(source) => emit_read_only_audit_json(
            args,
            &Envelope {
                schema_version: 1,
                kind: "session-source",
                source,
            },
        ),
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_paths::TestProfile;
    use serde_json::{json, Value};
    use std::fs;

    const ID: &str = "01a0bb2f-24a5-7712-8e2d-dbf43a5057d6";

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn session_source_options_require_one_exact_codex_uuid() {
        for argv in [
            args(&["viewer", COMMAND, ID, "--provider", "codex"]),
            args(&[
                "viewer",
                &format!("{COMMAND}={ID}"),
                "--provider=codex",
                "--format=json",
                "--output=new.json",
            ]),
        ] {
            assert_eq!(Options::parse(&argv).unwrap().session_id, ID);
        }
        let base = args(&["viewer", COMMAND, ID, "--provider", "codex"]);
        for suffix in [
            args(&["--provider", "codex"]),
            args(&[COMMAND, ID]),
            args(&["--format", "yaml"]),
            args(&["--format="]),
            args(&["--output"]),
            args(&["--output", ""]),
            args(&["--output=\n"]),
            args(&["--output", &"x".repeat(32_769)]),
            args(&["--cursor", "opaque"]),
            args(&["--session-metadata", ID]),
            args(&["--dump-session", ID]),
            args(&["--task-id", ID]),
            args(&["--prompt-state-cache"]),
            args(&["unexpected"]),
        ] {
            let mut invalid = base.clone();
            invalid.extend(suffix);
            assert_eq!(run_session_source(&invalid), 2);
        }
        for selector in [
            "prefix",
            "../session",
            "/absolute.jsonl",
            "01A0BB2F-24A5-7712-8E2D-DBF43A5057D6",
            "",
        ] {
            assert_eq!(
                run_session_source(&args(&["viewer", COMMAND, selector, "--provider", "codex"])),
                2
            );
        }
        for provider in ["claude", "hermes", "Codex", ""] {
            assert_eq!(
                run_session_source(&args(&["viewer", COMMAND, ID, "--provider", provider])),
                2
            );
        }
        assert_eq!(run_session_source(&args(&["viewer", COMMAND, ID])), 2);
    }

    #[test]
    fn session_source_command_emits_versioned_five_field_envelope_and_create_new_output() {
        let profile = TestProfile::new().unwrap();
        let root = profile.path().join(".codex").join("sessions");
        fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("rollout-2026-10-30T10-00-00-{ID}.jsonl"));
        let content = format!("{}\n", json!({"type":"session_meta","payload":{"id":ID}}));
        fs::write(&path, &content).unwrap();
        let output = profile.path().join("source.json");
        let argv = args(&[
            "viewer",
            COMMAND,
            ID,
            "--provider",
            "codex",
            "--output",
            output.to_str().unwrap(),
        ]);
        assert_eq!(run_session_source(&argv), 0);
        let value: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            value,
            json!({"schemaVersion":1,"kind":"session-source","source":{"provider":"codex","actual_session_id":ID,"file_path":path.to_string_lossy(),"is_hidden":false,"is_archived":false}})
        );
        assert_eq!(run_session_source(&argv), 1);
        assert_eq!(fs::read_to_string(&path).unwrap(), content);
        let null_output = profile.path().join("missing.json");
        assert_eq!(
            run_session_source(&args(&[
                "viewer",
                COMMAND,
                "00000000-0000-4000-8000-000000000000",
                "--provider",
                "codex",
                "--output",
                null_output.to_str().unwrap()
            ])),
            0
        );
        let missing: Value = serde_json::from_slice(&fs::read(null_output).unwrap()).unwrap();
        assert_eq!(
            missing,
            json!({"schemaVersion":1,"kind":"session-source","source":null})
        );
    }

    #[test]
    fn session_source_capability_advertises_command_and_provider_specific_contract() {
        let profile = TestProfile::new().unwrap();
        let output = profile.path().join("capabilities.json");
        assert_eq!(
            super::super::run_capabilities(&args(&[
                "viewer",
                "--capabilities",
                "--output",
                output.to_str().unwrap()
            ])),
            0
        );
        let value: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(value["commands"]
            .as_array()
            .unwrap()
            .contains(&json!("session-source")));
        assert!(value["features"]
            .as_array()
            .unwrap()
            .contains(&json!("codex-session-source-v1")));
    }
}
