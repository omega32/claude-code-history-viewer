use crate::providers::codex::{self, CodexAuthorshipDiagnostic};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

const CAPTURE_SCHEMA_VERSION: u32 = 1;
const AUDIT_SCHEMA_VERSION: u32 = 1;
const MAX_APP_SERVER_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_IDENTIFIER_CHARS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CodexAuthorshipAuditStatus {
    Match,
    Mismatch,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CodexAuthorshipAuditCounts {
    physical_turns: usize,
    authored_messages: usize,
    steers: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cli_version_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum CodexAuthorshipDifference {
    PhysicalTurnSequence {
        parser: Vec<String>,
        app_server: Vec<String>,
    },
    MissingFromParser {
        client_id: String,
        app_server_turn_id: String,
    },
    ExtraInParser {
        client_id: String,
        parser_turn_id: String,
    },
    TurnMismatch {
        client_id: String,
        parser_turn_id: String,
        app_server_turn_id: String,
    },
    OrderMismatch {
        parser: Vec<String>,
        app_server: Vec<String>,
    },
    ParserSubtypeMismatch {
        client_id: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum CodexAuditDiagnostic {
    Parser {
        diagnostic: CodexAuthorshipDiagnostic,
    },
    UnboundCapture,
    SourceBindingMismatch,
    DuplicateTurnId {
        turn_id: String,
    },
    MissingTurnItems {
        turn_id: String,
    },
    IncompleteTurnItems {
        turn_id: String,
        items_view: String,
    },
    MissingOrMalformedItemsView {
        turn_id: String,
    },
    ActiveTurn {
        turn_id: String,
    },
    UnknownTurnStatus {
        turn_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        discriminator: Option<String>,
    },
    MissingClientId {
        turn_id: String,
        item_id: String,
    },
    DuplicateClientId {
        client_id: String,
    },
    UnknownAppServerItemType {
        turn_id: String,
        item_id: String,
        discriminator: String,
    },
    MissingParserTurnId {
        message_id: String,
    },
    MissingParserClientId {
        message_id: String,
    },
    MissingParserPhysicalTurnId,
    InvalidParserIdentifier {
        field: String,
    },
    DuplicateParserClientId {
        client_id: String,
    },
    UnresolvedParserAuthorship {
        message_id: String,
    },
    MissingAppServerCliVersion,
    InvalidAppServerCliVersion,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CodexAuthorshipAuditReport {
    pub(crate) schema_version: u32,
    pub(crate) status: CodexAuthorshipAuditStatus,
    pub(crate) thread_id: String,
    pub(crate) parser: CodexAuthorshipAuditCounts,
    pub(crate) app_server: CodexAuthorshipAuditCounts,
    pub(crate) differences: Vec<CodexAuthorshipDifference>,
    pub(crate) diagnostics: Vec<CodexAuditDiagnostic>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CaptureEnvelope {
    schema_version: u32,
    rollout: CaptureRolloutBinding,
    app_server: CaptureAppServer,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CaptureRolloutBinding {
    thread_id: String,
    sha256: String,
    length: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CaptureAppServer {
    #[serde(default)]
    cli_version: Option<String>,
    response: Value,
}

#[derive(Debug, Clone)]
struct AuthoredIdentity {
    turn_id: String,
    client_id: String,
    subtype: String,
}

#[derive(Debug)]
struct ParsedOracle {
    thread_id: String,
    turns: Vec<String>,
    authored: Vec<AuthoredIdentity>,
    diagnostics: Vec<CodexAuditDiagnostic>,
}

fn bounded_identifier(value: &str, label: &str) -> Result<String, String> {
    if value.is_empty() || value.chars().count() > MAX_IDENTIFIER_CHARS {
        return Err(format!(
            "{label} must contain 1 to {MAX_IDENTIFIER_CHARS} characters"
        ));
    }
    Ok(value.to_string())
}

fn bounded_discriminator(value: &str) -> String {
    opaque_identity(&value.chars().take(MAX_IDENTIFIER_CHARS).collect::<String>())
}

fn opaque_identity(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}

fn validated_cli_version(value: &str) -> Option<String> {
    let version = value.strip_prefix("codex-cli ")?;
    (!version.is_empty()
        && version.starts_with(|character: char| character.is_ascii_digit())
        && version
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || ".+-".contains(character)))
    .then(|| value.to_string())
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

fn validate_response_path(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("Codex app-server response path must be absolute".to_string());
    }

    let mut current = PathBuf::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        match component {
            Component::Prefix(_) | Component::RootDir => current.push(component.as_os_str()),
            Component::Normal(_) => {
                current.push(component.as_os_str());
                let metadata = fs::symlink_metadata(&current).map_err(|error| {
                    format!("Failed to inspect Codex app-server response: {error}")
                })?;
                if is_symlink_or_reparse(&metadata) {
                    return Err(
                        "Codex app-server response path must not contain symlinks or reparse points"
                            .to_string(),
                    );
                }
                let is_last = components.peek().is_none();
                if (is_last && !metadata.is_file()) || (!is_last && !metadata.is_dir()) {
                    return Err(
                        "Codex app-server response must be a regular non-symlink file".to_string(),
                    );
                }
            }
            Component::CurDir | Component::ParentDir => {
                return Err(
                    "Codex app-server response path must not contain relative components"
                        .to_string(),
                );
            }
        }
    }
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Failed to inspect Codex app-server response: {error}"))?;
    if metadata.len() > MAX_APP_SERVER_RESPONSE_BYTES {
        return Err(format!(
            "Codex app-server response exceeds the {MAX_APP_SERVER_RESPONSE_BYTES}-byte limit"
        ));
    }
    path.canonicalize()
        .map_err(|error| format!("Failed to resolve Codex app-server response: {error}"))
}

fn source_fingerprint(path: &Path) -> Result<(u64, String), String> {
    let bytes = fs::read(path).map_err(|error| format!("Failed to read Codex rollout: {error}"))?;
    let length =
        u64::try_from(bytes.len()).map_err(|_| "Codex rollout is too large".to_string())?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    Ok((length, hash))
}

fn thread_from_response(response: &Value) -> Option<&Value> {
    response.get("thread").or_else(|| {
        response
            .get("result")
            .and_then(|result| result.get("thread"))
    })
}

fn known_app_server_item_type(item_type: &str) -> bool {
    matches!(
        item_type,
        "userMessage"
            | "agentMessage"
            | "collabAgentToolCall"
            | "commandExecution"
            | "contextCompaction"
            | "dynamicToolCall"
            | "enteredReviewMode"
            | "exitedReviewMode"
            | "fileChange"
            | "hookPrompt"
            | "imageGeneration"
            | "imageView"
            | "mcpToolCall"
            | "plan"
            | "reasoning"
            | "sleep"
            | "subAgentActivity"
            | "webSearch"
    )
}

fn parse_oracle(response: &Value) -> Result<ParsedOracle, String> {
    let thread = thread_from_response(response)
        .and_then(Value::as_object)
        .ok_or("Codex capture response must contain one thread/read result")?;
    let thread_id = bounded_identifier(
        thread
            .get("id")
            .and_then(Value::as_str)
            .ok_or("Codex thread/read response is missing thread.id")?,
        "Codex thread id",
    )?;
    let turns = thread
        .get("turns")
        .and_then(Value::as_array)
        .ok_or("Codex thread/read response is missing thread.turns")?;
    let mut turn_ids = Vec::with_capacity(turns.len());
    let mut authored = Vec::new();
    let mut diagnostics = Vec::new();
    let mut seen_turn_ids = HashSet::new();
    let mut seen_client_ids = HashSet::new();

    for turn in turns {
        let turn = turn
            .as_object()
            .ok_or("Codex thread/read turn must be an object")?;
        let turn_id = bounded_identifier(
            turn.get("id")
                .and_then(Value::as_str)
                .ok_or("Codex thread/read turn is missing id")?,
            "Codex turn id",
        )?;
        if !seen_turn_ids.insert(turn_id.clone()) {
            diagnostics.push(CodexAuditDiagnostic::DuplicateTurnId {
                turn_id: opaque_identity(&turn_id),
            });
        }
        turn_ids.push(turn_id.clone());
        match turn.get("status").and_then(Value::as_str) {
            Some("completed" | "interrupted" | "failed") => {}
            Some("inProgress") => diagnostics.push(CodexAuditDiagnostic::ActiveTurn {
                turn_id: opaque_identity(&turn_id),
            }),
            status => diagnostics.push(CodexAuditDiagnostic::UnknownTurnStatus {
                turn_id: opaque_identity(&turn_id),
                discriminator: status.map(bounded_discriminator),
            }),
        }
        match turn.get("itemsView").and_then(Value::as_str) {
            Some("full") => {}
            Some(items_view) => diagnostics.push(CodexAuditDiagnostic::IncompleteTurnItems {
                turn_id: opaque_identity(&turn_id),
                items_view: bounded_discriminator(items_view),
            }),
            None => diagnostics.push(CodexAuditDiagnostic::MissingOrMalformedItemsView {
                turn_id: opaque_identity(&turn_id),
            }),
        }
        let Some(items) = turn.get("items").and_then(Value::as_array) else {
            diagnostics.push(CodexAuditDiagnostic::MissingTurnItems {
                turn_id: opaque_identity(&turn_id),
            });
            continue;
        };
        let mut authored_index = 0usize;
        for item in items {
            let Some(item) = item.as_object() else {
                return Err("Codex thread/read item must be an object".to_string());
            };
            let item_id = bounded_identifier(
                item.get("id")
                    .and_then(Value::as_str)
                    .ok_or("Codex thread/read item is missing id")?,
                "Codex item id",
            )?;
            let item_type = item
                .get("type")
                .and_then(Value::as_str)
                .ok_or("Codex thread/read item is missing type")?;
            if item_type != "userMessage" {
                if !known_app_server_item_type(item_type) {
                    diagnostics.push(CodexAuditDiagnostic::UnknownAppServerItemType {
                        turn_id: opaque_identity(&turn_id),
                        item_id: opaque_identity(&item_id),
                        discriminator: bounded_discriminator(item_type),
                    });
                }
                continue;
            }
            let Some(client_id) = item
                .get("clientId")
                .and_then(Value::as_str)
                .filter(|client_id| !client_id.is_empty())
            else {
                diagnostics.push(CodexAuditDiagnostic::MissingClientId {
                    turn_id: opaque_identity(&turn_id),
                    item_id: opaque_identity(&item_id),
                });
                continue;
            };
            let client_id = bounded_identifier(client_id, "Codex client id")?;
            if !seen_client_ids.insert(client_id.clone()) {
                diagnostics.push(CodexAuditDiagnostic::DuplicateClientId {
                    client_id: opaque_identity(&client_id),
                });
            }
            authored.push(AuthoredIdentity {
                turn_id: turn_id.clone(),
                client_id,
                subtype: if authored_index == 0 {
                    "authored_user"
                } else {
                    "steer"
                }
                .to_string(),
            });
            authored_index += 1;
        }
    }

    Ok(ParsedOracle {
        thread_id,
        turns: turn_ids,
        authored,
        diagnostics,
    })
}

fn parser_projection(
    messages: &[crate::models::ClaudeMessage],
    diagnostics: &mut Vec<CodexAuditDiagnostic>,
) -> (Vec<String>, Vec<AuthoredIdentity>) {
    let mut turns = Vec::new();
    for message in messages {
        let Some(data) = message.data.as_ref() else {
            continue;
        };
        if message.message_type != "progress"
            || data.get("type").and_then(Value::as_str) != Some("waiting_for_task")
            || data.get("status").and_then(Value::as_str) != Some("started")
        {
            continue;
        }
        let Some(turn_id) = data
            .get("taskId")
            .and_then(Value::as_str)
            .filter(|turn_id| !turn_id.is_empty())
        else {
            diagnostics.push(CodexAuditDiagnostic::MissingParserPhysicalTurnId);
            continue;
        };
        match bounded_identifier(turn_id, "Parser physical turn id") {
            Ok(turn_id) => turns.push(turn_id),
            Err(_) => diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                field: "physical-turn-id".to_string(),
            }),
        }
    }

    let mut authored = Vec::new();
    let mut seen_client_ids = HashSet::new();
    for message in messages
        .iter()
        .filter(|message| matches!(message.subtype.as_deref(), Some("authored_user" | "steer")))
    {
        let turn_id = message
            .data
            .as_ref()
            .and_then(|data| data.get("providerTurnId"))
            .and_then(Value::as_str)
            .filter(|turn_id| !turn_id.is_empty());
        let client_id = message
            .data
            .as_ref()
            .and_then(|data| data.get("clientMessageId"))
            .and_then(Value::as_str)
            .filter(|client_id| !client_id.is_empty());
        let Some(turn_id) = turn_id else {
            match bounded_identifier(&message.uuid, "Parser message id") {
                Ok(message_id) => {
                    diagnostics.push(CodexAuditDiagnostic::MissingParserTurnId {
                        message_id: opaque_identity(&message_id),
                    });
                }
                Err(_) => diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                    field: "message-id".to_string(),
                }),
            }
            continue;
        };
        let Some(client_id) = client_id else {
            match bounded_identifier(&message.uuid, "Parser message id") {
                Ok(message_id) => {
                    diagnostics.push(CodexAuditDiagnostic::MissingParserClientId {
                        message_id: opaque_identity(&message_id),
                    });
                }
                Err(_) => diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                    field: "message-id".to_string(),
                }),
            }
            continue;
        };
        let turn_id = if let Ok(turn_id) = bounded_identifier(turn_id, "Parser turn id") {
            turn_id
        } else {
            diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                field: "turn-id".to_string(),
            });
            continue;
        };
        let client_id = if let Ok(client_id) = bounded_identifier(client_id, "Parser client id") {
            client_id
        } else {
            diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                field: "client-id".to_string(),
            });
            continue;
        };
        if !seen_client_ids.insert(client_id.clone()) {
            diagnostics.push(CodexAuditDiagnostic::DuplicateParserClientId {
                client_id: opaque_identity(&client_id),
            });
        }
        authored.push(AuthoredIdentity {
            turn_id,
            client_id,
            subtype: message.subtype.clone().unwrap_or_default(),
        });
    }
    (turns, authored)
}

fn counts(
    turns: &[String],
    authored: &[AuthoredIdentity],
    cli_version: Option<String>,
) -> CodexAuthorshipAuditCounts {
    CodexAuthorshipAuditCounts {
        physical_turns: turns.len(),
        authored_messages: authored.len(),
        steers: authored
            .iter()
            .filter(|identity| identity.subtype == "steer")
            .count(),
        cli_version_digest: cli_version,
    }
}

pub(crate) fn audit_paths(
    rollout_path: &Path,
    app_server_response_path: &Path,
) -> Result<CodexAuthorshipAuditReport, String> {
    let canonical_rollout_path = codex::validate_authorship_audit_path(rollout_path)?;
    let response_path = validate_response_path(app_server_response_path)?;
    let response_bytes = fs::read(&response_path)
        .map_err(|error| format!("Failed to read Codex app-server response: {error}"))?;
    if u64::try_from(response_bytes.len())
        .map_or(true, |length| length > MAX_APP_SERVER_RESPONSE_BYTES)
    {
        return Err(format!(
            "Codex app-server response exceeds the {MAX_APP_SERVER_RESPONSE_BYTES}-byte limit"
        ));
    }
    let capture_value: Value = serde_json::from_slice(&response_bytes)
        .map_err(|error| format!("Invalid Codex app-server capture JSON: {error}"))?;
    let capture = serde_json::from_value::<CaptureEnvelope>(capture_value.clone()).ok();

    let (response, binding, raw_cli_version) = if let Some(capture) = capture {
        if capture.schema_version != CAPTURE_SCHEMA_VERSION {
            return Err(format!(
                "Unsupported Codex capture schema version {}",
                capture.schema_version
            ));
        }
        (
            capture.app_server.response,
            Some(capture.rollout),
            capture.app_server.cli_version,
        )
    } else {
        (capture_value, None, None)
    };
    let oracle = parse_oracle(&response)?;
    let (before_length, before_sha256) = source_fingerprint(&canonical_rollout_path)?;
    let projection = codex::parse_authorship_audit(rollout_path)?;
    let (after_length, after_sha256) = source_fingerprint(&canonical_rollout_path)?;
    if (before_length, &before_sha256) != (after_length, &after_sha256) {
        return Err("Codex rollout changed while the authorship audit was running".to_string());
    }
    if oracle.thread_id != projection.session_id {
        return Err("Codex app-server thread id does not match the rollout thread id".to_string());
    }

    let mut diagnostics = Vec::new();
    let mut diagnosed_message_ids = HashSet::new();
    for diagnostic in projection.diagnostics {
        let message_id_is_valid =
            bounded_identifier(&diagnostic.message_id, "Parser message id").is_ok();
        let provider_turn_id_is_valid = diagnostic
            .provider_turn_id
            .as_deref()
            .map_or(true, |turn_id| {
                bounded_identifier(turn_id, "Parser turn id").is_ok()
            });
        if message_id_is_valid && provider_turn_id_is_valid {
            diagnosed_message_ids.insert(diagnostic.message_id.clone());
            diagnostics.push(CodexAuditDiagnostic::Parser {
                diagnostic: CodexAuthorshipDiagnostic {
                    kind: diagnostic.kind,
                    message_id: opaque_identity(&diagnostic.message_id),
                    provider_turn_id: diagnostic.provider_turn_id.as_deref().map(opaque_identity),
                    source_line: diagnostic.source_line,
                    discriminator: diagnostic
                        .discriminator
                        .as_deref()
                        .map(bounded_discriminator),
                },
            });
        } else {
            diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                field: if message_id_is_valid {
                    "turn-id"
                } else {
                    "message-id"
                }
                .to_string(),
            });
        }
    }
    for message in projection
        .messages
        .iter()
        .filter(|message| message.subtype.as_deref() == Some("authorship_unknown"))
    {
        if diagnosed_message_ids.contains(&message.uuid) {
            continue;
        }
        match bounded_identifier(&message.uuid, "Parser message id") {
            Ok(message_id) => {
                diagnostics.push(CodexAuditDiagnostic::UnresolvedParserAuthorship {
                    message_id: opaque_identity(&message_id),
                });
            }
            Err(_) => diagnostics.push(CodexAuditDiagnostic::InvalidParserIdentifier {
                field: "message-id".to_string(),
            }),
        }
    }
    let capture_is_bound = binding.is_some();
    match binding {
        Some(binding) => {
            let binding_thread_id = bounded_identifier(&binding.thread_id, "Capture thread id")?;
            if binding_thread_id != oracle.thread_id
                || binding.length != before_length
                || !binding.sha256.eq_ignore_ascii_case(&before_sha256)
            {
                diagnostics.push(CodexAuditDiagnostic::SourceBindingMismatch);
            }
        }
        None => diagnostics.push(CodexAuditDiagnostic::UnboundCapture),
    }
    let cli_version = match raw_cli_version {
        Some(version) if !version.is_empty() => {
            if bounded_identifier(&version, "Codex CLI version").is_ok() {
                if let Some(version) = validated_cli_version(&version) {
                    Some(opaque_identity(&version))
                } else {
                    diagnostics.push(CodexAuditDiagnostic::InvalidAppServerCliVersion);
                    None
                }
            } else {
                diagnostics.push(CodexAuditDiagnostic::InvalidAppServerCliVersion);
                None
            }
        }
        _ if capture_is_bound => {
            diagnostics.push(CodexAuditDiagnostic::MissingAppServerCliVersion);
            None
        }
        _ => None,
    };
    diagnostics.extend(oracle.diagnostics);

    let (parser_turns, parser_authored) = parser_projection(&projection.messages, &mut diagnostics);
    let mut differences = Vec::new();
    if parser_turns != oracle.turns {
        differences.push(CodexAuthorshipDifference::PhysicalTurnSequence {
            parser: parser_turns
                .iter()
                .map(|turn_id| opaque_identity(turn_id))
                .collect(),
            app_server: oracle
                .turns
                .iter()
                .map(|turn_id| opaque_identity(turn_id))
                .collect(),
        });
    }

    let parser_by_client = parser_authored
        .iter()
        .map(|identity| (identity.client_id.as_str(), identity))
        .collect::<HashMap<_, _>>();
    let oracle_by_client = oracle
        .authored
        .iter()
        .map(|identity| (identity.client_id.as_str(), identity))
        .collect::<HashMap<_, _>>();
    for identity in &oracle.authored {
        match parser_by_client.get(identity.client_id.as_str()) {
            None => differences.push(CodexAuthorshipDifference::MissingFromParser {
                client_id: opaque_identity(&identity.client_id),
                app_server_turn_id: opaque_identity(&identity.turn_id),
            }),
            Some(parser) if parser.turn_id != identity.turn_id => {
                differences.push(CodexAuthorshipDifference::TurnMismatch {
                    client_id: opaque_identity(&identity.client_id),
                    parser_turn_id: opaque_identity(&parser.turn_id),
                    app_server_turn_id: opaque_identity(&identity.turn_id),
                });
            }
            Some(parser) if parser.subtype != identity.subtype => {
                differences.push(CodexAuthorshipDifference::ParserSubtypeMismatch {
                    client_id: opaque_identity(&identity.client_id),
                    expected: identity.subtype.clone(),
                    actual: parser.subtype.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for identity in &parser_authored {
        if !oracle_by_client.contains_key(identity.client_id.as_str()) {
            differences.push(CodexAuthorshipDifference::ExtraInParser {
                client_id: opaque_identity(&identity.client_id),
                parser_turn_id: opaque_identity(&identity.turn_id),
            });
        }
    }
    let parser_order = parser_authored
        .iter()
        .map(|identity| identity.client_id.clone())
        .collect::<Vec<_>>();
    let oracle_order = oracle
        .authored
        .iter()
        .map(|identity| identity.client_id.clone())
        .collect::<Vec<_>>();
    if parser_order != oracle_order {
        differences.push(CodexAuthorshipDifference::OrderMismatch {
            parser: parser_order
                .iter()
                .map(|identity| opaque_identity(identity))
                .collect(),
            app_server: oracle_order
                .iter()
                .map(|identity| opaque_identity(identity))
                .collect(),
        });
    }

    let status = if !diagnostics.is_empty() {
        CodexAuthorshipAuditStatus::Inconclusive
    } else if differences.is_empty() {
        CodexAuthorshipAuditStatus::Match
    } else {
        CodexAuthorshipAuditStatus::Mismatch
    };
    Ok(CodexAuthorshipAuditReport {
        schema_version: AUDIT_SCHEMA_VERSION,
        status,
        thread_id: opaque_identity(&projection.session_id),
        parser: counts(&parser_turns, &parser_authored, None),
        app_server: counts(&oracle.turns, &oracle.authored, cli_version),
        differences,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serial_test::serial;
    use std::ffi::OsString;
    use tempfile::TempDir;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &Path) -> Self {
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

    fn write_jsonl(path: &Path, lines: &[Value]) {
        let body = lines
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, format!("{body}\n")).expect("rollout fixture should be written");
    }

    fn rollout_lines() -> Vec<Value> {
        vec![
            json!({"timestamp":"2026-08-12T10:00:00Z","type":"session_meta","payload":{"id":"session-1","cwd":"/private/project"}}),
            json!({"timestamp":"2026-08-12T10:00:01Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-1"}}),
            json!({"timestamp":"2026-08-12T10:00:02Z","type":"response_item","payload":{"id":"raw-1","type":"message","role":"user","content":[{"type":"input_text","text":"PRIVATE PROMPT SENTINEL"}],"internal_chat_message_metadata_passthrough":{"turn_id":"turn-1"}}}),
            json!({"timestamp":"2026-08-12T10:00:03Z","type":"event_msg","payload":{"type":"user_message","client_id":"client-1","message":"PRIVATE PROMPT SENTINEL"}}),
            json!({"timestamp":"2026-08-12T10:00:04Z","type":"response_item","payload":{"id":"raw-2","type":"message","role":"user","content":[{"type":"input_text","text":"steer"}],"internal_chat_message_metadata_passthrough":{"turn_id":"turn-1"}}}),
            json!({"timestamp":"2026-08-12T10:00:05Z","type":"event_msg","payload":{"type":"user_message","client_id":"client-2","message":"steer"}}),
            json!({"timestamp":"2026-08-12T10:00:06Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-1"}}),
            json!({"timestamp":"2026-08-12T10:00:07Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-2"}}),
            json!({"timestamp":"2026-08-12T10:00:08Z","type":"response_item","payload":{"id":"raw-3","type":"message","role":"user","content":[{"type":"input_text","text":"follow-up"}],"internal_chat_message_metadata_passthrough":{"turn_id":"turn-2"}}}),
            json!({"timestamp":"2026-08-12T10:00:09Z","type":"event_msg","payload":{"type":"user_message","client_id":"client-3","message":"follow-up"}}),
            json!({"timestamp":"2026-08-12T10:00:10Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-2"}}),
        ]
    }

    fn oracle_response() -> Value {
        json!({
            "thread": {
                "id": "session-1",
                "turns": [
                    {"id":"turn-1","itemsView":"full","status":"completed","items":[
                        {"id":"item-1","type":"userMessage","clientId":"client-1","content":[]},
                        {"id":"item-2","type":"agentMessage","text":"PRIVATE ASSISTANT SENTINEL"},
                        {"id":"item-3","type":"userMessage","clientId":"client-2","content":[]}
                    ]},
                    {"id":"turn-2","itemsView":"full","status":"completed","items":[
                        {"id":"item-4","type":"userMessage","clientId":"client-3","content":[]}
                    ]}
                ]
            }
        })
    }

    fn write_bound_capture(rollout: &Path, oracle: &Path, response: Value) {
        let (length, sha256) = source_fingerprint(rollout).expect("rollout should hash");
        fs::write(
            oracle,
            serde_json::to_vec_pretty(&json!({
                "schemaVersion": 1,
                "rollout": {"threadId":"session-1","sha256":sha256,"length":length},
                "appServer": {"cliVersion":"codex-cli 0.146.0","response":response}
            }))
            .expect("capture should serialize"),
        )
        .expect("capture should be written");
    }

    fn setup() -> (TempDir, PathBuf, PathBuf, EnvVarGuard) {
        let tmp = TempDir::new().expect("temp dir should be created");
        let codex_home = tmp.path().join(".codex");
        let sessions = codex_home.join("sessions");
        fs::create_dir_all(&sessions).expect("sessions directory should be created");
        let rollout = sessions.join("rollout-audit.jsonl");
        let oracle = tmp.path().join("thread-read.json");
        write_jsonl(&rollout, &rollout_lines());
        let guard = EnvVarGuard::set("CODEX_HOME", &codex_home);
        (tmp, rollout, oracle, guard)
    }

    #[test]
    #[serial]
    fn captured_thread_read_matches_normalized_authorship_by_client_and_turn() {
        let (_tmp, rollout, oracle, _guard) = setup();
        write_bound_capture(&rollout, &oracle, oracle_response());

        let audit = audit_paths(&rollout, &oracle).expect("audit should succeed");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Match);
        assert_eq!(audit.thread_id, opaque_identity("session-1"));
        assert_eq!(audit.parser.authored_messages, 3);
        assert_eq!(audit.parser.physical_turns, 2);
        assert_eq!(audit.parser.steers, 1);
        assert!(audit.differences.is_empty());
        assert!(audit.diagnostics.is_empty());
        let serialized = serde_json::to_string(&audit).expect("audit should serialize");
        assert!(!serialized.contains("PRIVATE PROMPT SENTINEL"));
        assert!(!serialized.contains("PRIVATE ASSISTANT SENTINEL"));
        assert!(!serialized.contains(rollout.as_os_str().to_string_lossy().as_ref()));
        assert!(!serialized.contains(oracle.as_os_str().to_string_lossy().as_ref()));
    }

    #[test]
    #[serial]
    fn bare_response_and_source_hash_mismatch_are_inconclusive() {
        let (_tmp, rollout, oracle, _guard) = setup();
        fs::write(
            &oracle,
            serde_json::to_vec_pretty(&oracle_response()).unwrap(),
        )
        .unwrap();
        let bare = audit_paths(&rollout, &oracle).expect("bare response should be auditable");
        assert_eq!(bare.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(matches!(
            bare.diagnostics.as_slice(),
            [CodexAuditDiagnostic::UnboundCapture]
        ));

        fs::write(
            &oracle,
            serde_json::to_vec_pretty(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "result": oracle_response()
            }))
            .unwrap(),
        )
        .unwrap();
        let json_rpc = audit_paths(&rollout, &oracle).expect("JSON-RPC response should parse");
        assert_eq!(json_rpc.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(matches!(
            json_rpc.diagnostics.as_slice(),
            [CodexAuditDiagnostic::UnboundCapture]
        ));

        write_bound_capture(&rollout, &oracle, oracle_response());
        let mut capture: Value =
            serde_json::from_slice(&fs::read(&oracle).unwrap()).expect("capture should parse");
        capture["rollout"]["sha256"] = Value::String("00".repeat(32));
        fs::write(&oracle, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
        let stale = audit_paths(&rollout, &oracle).expect("stale capture should report");
        assert_eq!(stale.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(stale
            .diagnostics
            .iter()
            .any(|diagnostic| matches!(diagnostic, CodexAuditDiagnostic::SourceBindingMismatch)));
    }

    #[test]
    #[serial]
    fn mismatch_and_protocol_drift_are_typed_without_content() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let mut response = oracle_response();
        response["thread"]["turns"][0]["items"][0]["clientId"] =
            Value::String("different-client".to_string());
        response["thread"]["turns"][1]["items"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"future-item","type":"futureItem","secret":"PRIVATE ITEM"}));
        response["thread"]["turns"][1]["status"] = Value::String("futureStatus".to_string());
        write_bound_capture(&rollout, &oracle, response);

        let audit = audit_paths(&rollout, &oracle).expect("audit should report drift");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(!audit.differences.is_empty());
        assert!(audit.diagnostics.iter().any(|diagnostic| matches!(
            diagnostic,
            CodexAuditDiagnostic::UnknownAppServerItemType { discriminator, .. }
                if discriminator == &opaque_identity("futureItem")
        )));
        assert!(audit.diagnostics.iter().any(|diagnostic| matches!(
            diagnostic,
            CodexAuditDiagnostic::UnknownTurnStatus { discriminator, .. }
                if discriminator.as_deref() == Some(opaque_identity("futureStatus").as_str())
        )));
        let serialized = serde_json::to_string(&audit).unwrap();
        assert!(!serialized.contains("PRIVATE ITEM"));
        assert!(!serialized.contains("PRIVATE PROMPT SENTINEL"));
    }

    #[test]
    #[serial]
    fn incomplete_duplicate_cross_turn_and_reordered_oracles_never_match() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let mut cases = Vec::new();

        let mut missing_client_id = oracle_response();
        missing_client_id["thread"]["turns"][0]["items"][0]
            .as_object_mut()
            .unwrap()
            .remove("clientId");
        cases.push(missing_client_id);

        let mut duplicate_client_id = oracle_response();
        duplicate_client_id["thread"]["turns"][0]["items"][2]["clientId"] =
            Value::String("client-1".to_string());
        cases.push(duplicate_client_id);

        let mut reordered = oracle_response();
        reordered["thread"]["turns"][0]["items"]
            .as_array_mut()
            .unwrap()
            .swap(0, 2);
        cases.push(reordered);

        let mut cross_turn = oracle_response();
        let moved = cross_turn["thread"]["turns"][0]["items"]
            .as_array_mut()
            .unwrap()
            .remove(2);
        cross_turn["thread"]["turns"][1]["items"]
            .as_array_mut()
            .unwrap()
            .push(moved);
        cases.push(cross_turn);

        let mut missing_items = oracle_response();
        missing_items["thread"]["turns"][1]
            .as_object_mut()
            .unwrap()
            .remove("items");
        cases.push(missing_items);

        let mut incomplete_items = oracle_response();
        incomplete_items["thread"]["turns"][0]["itemsView"] = Value::String("summary".to_string());
        cases.push(incomplete_items);

        let mut missing_items_view = oracle_response();
        missing_items_view["thread"]["turns"][0]
            .as_object_mut()
            .unwrap()
            .remove("itemsView");
        cases.push(missing_items_view);

        let mut malformed_items_view = oracle_response();
        malformed_items_view["thread"]["turns"][0]["itemsView"] = json!({"private":"text"});
        cases.push(malformed_items_view);

        let mut active_turn = oracle_response();
        active_turn["thread"]["turns"][1]["status"] = Value::String("inProgress".to_string());
        cases.push(active_turn);

        let mut duplicate_turn = oracle_response();
        duplicate_turn["thread"]["turns"][1]["id"] = Value::String("turn-1".to_string());
        cases.push(duplicate_turn);

        for response in cases {
            write_bound_capture(&rollout, &oracle, response);
            let audit = audit_paths(&rollout, &oracle).expect("invalid oracle should report");
            assert_ne!(audit.status, CodexAuthorshipAuditStatus::Match);
        }
    }

    #[test]
    #[serial]
    fn physical_turn_without_authored_input_remains_in_the_comparison() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let mut lines = rollout_lines();
        lines.push(json!({"timestamp":"2026-08-12T10:00:11Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-3"}}));
        lines.push(json!({"timestamp":"2026-08-12T10:00:12Z","type":"event_msg","payload":{"type":"task_complete","turn_id":"turn-3"}}));
        write_jsonl(&rollout, &lines);
        let mut response = oracle_response();
        response["thread"]["turns"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"turn-3","itemsView":"full","status":"completed","items":[]}));
        write_bound_capture(&rollout, &oracle, response);

        let audit = audit_paths(&rollout, &oracle).expect("zero-input turn should compare");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Match);
        assert_eq!(audit.parser.physical_turns, 3);
        assert_eq!(audit.app_server.physical_turns, 3);
    }

    #[test]
    #[serial]
    fn parser_sidecar_reports_unknown_record_and_unresolved_authorship() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let mut lines = rollout_lines();
        lines.insert(
            3,
            json!({"timestamp":"2026-08-12T10:00:02.5Z","type":"future_protocol_record","payload":{"secret":"PRIVATE DRIFT"}}),
        );
        write_jsonl(&rollout, &lines);
        write_bound_capture(&rollout, &oracle, oracle_response());

        let audit = audit_paths(&rollout, &oracle).expect("audit should report parser drift");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(audit.diagnostics.iter().any(|diagnostic| matches!(
            diagnostic,
            CodexAuditDiagnostic::Parser { diagnostic }
                if diagnostic.kind == "unknown-top-level-record"
                    && diagnostic.source_line == 4
                    && diagnostic.discriminator.as_deref()
                        == Some(opaque_identity("future_protocol_record").as_str())
        )));
        let serialized = serde_json::to_string(&audit).unwrap();
        assert!(!serialized.contains("PRIVATE DRIFT"));
        assert!(!serialized.contains("PRIVATE PROMPT SENTINEL"));
    }

    #[test]
    #[serial]
    fn oversized_parser_identity_is_replaced_by_a_typed_diagnostic() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let mut lines = rollout_lines();
        let private_identity = "PRIVATE IDENTITY ".repeat(32);
        lines[2]["payload"]["id"] = Value::String(private_identity.clone());
        lines.insert(
            3,
            json!({"timestamp":"2026-08-12T10:00:02.5Z","type":"future_protocol_record","payload":{}}),
        );
        write_jsonl(&rollout, &lines);
        write_bound_capture(&rollout, &oracle, oracle_response());

        let audit = audit_paths(&rollout, &oracle).expect("audit should report bounded drift");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Inconclusive);
        assert!(audit.diagnostics.iter().any(|diagnostic| matches!(
            diagnostic,
            CodexAuditDiagnostic::InvalidParserIdentifier { field }
                if field == "message-id"
        )));
        let serialized = serde_json::to_string(&audit).unwrap();
        assert!(!serialized.contains(&private_identity));
        assert!(!serialized.contains("PRIVATE PROMPT SENTINEL"));
    }

    #[test]
    #[serial]
    fn short_provider_controlled_strings_are_never_emitted_verbatim() {
        let (_tmp, rollout, oracle, _guard) = setup();
        let private = "C:\\private\\prompt.txt";
        let mut response = oracle_response();
        response["thread"]["turns"][0]["id"] = Value::String(private.to_string());
        response["thread"]["turns"][0]["items"][0]["id"] = Value::String(private.to_string());
        response["thread"]["turns"][0]["items"][0]["clientId"] = Value::String(private.to_string());
        response["thread"]["turns"][1]["items"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":private,"type":private}));
        write_bound_capture(&rollout, &oracle, response);
        let mut capture: Value = serde_json::from_slice(&fs::read(&oracle).unwrap()).unwrap();
        capture["appServer"]["cliVersion"] = Value::String(private.to_string());
        fs::write(&oracle, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();

        let audit = audit_paths(&rollout, &oracle).expect("private identities should be redacted");

        assert_eq!(audit.status, CodexAuthorshipAuditStatus::Inconclusive);
        let serialized = serde_json::to_string(&audit).unwrap();
        assert!(!serialized.contains(private));
        assert!(!serialized.contains("PRIVATE PROMPT SENTINEL"));
        assert!(!serialized.contains("PRIVATE ASSISTANT SENTINEL"));
    }

    #[test]
    fn response_path_must_be_absolute_regular_and_bounded() {
        let error = validate_response_path(Path::new("relative.json"))
            .expect_err("relative path must fail");
        assert!(error.contains("absolute"));

        let temp = TempDir::new().unwrap();
        let directory_error = validate_response_path(temp.path())
            .expect_err("a response directory must not be accepted");
        assert!(directory_error.contains("regular"));
        let oversized = temp.path().join("oversized.json");
        let file = fs::File::create(&oversized).unwrap();
        file.set_len(MAX_APP_SERVER_RESPONSE_BYTES + 1).unwrap();
        let oversized_error =
            validate_response_path(&oversized).expect_err("an oversized response must fail");
        assert!(oversized_error.contains("limit"));
    }
}
