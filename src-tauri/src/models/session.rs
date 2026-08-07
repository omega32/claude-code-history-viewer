use serde::{Deserialize, Serialize};

/// Git worktree 유형
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GitWorktreeType {
    /// 메인 레포지토리 (.git이 디렉토리)
    Main,
    /// 링크드 워크트리 (.git이 파일)
    Linked,
    /// Git 레포가 아님
    NotGit,
}

/// Git worktree 정보
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GitInfo {
    /// 워크트리 유형
    pub worktree_type: GitWorktreeType,
    /// 메인 레포의 프로젝트 경로 (링크드 워크트리인 경우)
    /// 예: "/Users/jack/my-project"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_project_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeProject {
    pub name: String,
    /// Claude session storage path (e.g., "~/.claude/projects/-Users-jack-client-my-project")
    pub path: String,
    /// Decoded actual filesystem path (e.g., "/Users/jack/client/my-project")
    pub actual_path: String,
    pub session_count: usize,
    pub message_count: usize,
    pub last_modified: String,
    /// Git worktree 정보
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_info: Option<GitInfo>,
    /// Provider identifier (claude, codex, opencode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Storage type (json, sqlite)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_type: Option<String>,
    /// Label for custom Claude directory source (e.g., "Personal")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_directory_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeSession {
    pub session_id: String,        // Unique ID based on file path
    pub actual_session_id: String, // Actual session ID from the messages
    pub file_path: String,
    pub project_name: String,
    pub message_count: usize,
    pub first_message_time: String,
    pub last_message_time: String,
    pub last_modified: String,
    pub has_tool_use: bool,
    pub has_errors: bool,
    pub summary: Option<String>,
    /// Whether this session was explicitly renamed via the /rename command
    #[serde(default)]
    pub is_renamed: bool,
    /// Provider identifier (claude, codex, opencode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Storage type (json, sqlite)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_type: Option<String>,
    /// Originating client for Claude Code sessions: "cli" / "claude-vscode" / "claude-desktop".
    /// `None` for non-Claude providers or sessions predating the entrypoint field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<String>,
    /// Parent session id for an explicitly forked session.
    ///
    /// Codex supplies this as `session_meta.payload.forked_from_id`. Other
    /// providers and native sessions leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forked_from_id: Option<String>,
    /// Provider-authenticated provenance for a spawned sub-agent session.
    ///
    /// Codex derives this only from the rollout's first
    /// `session_meta.payload.source.subagent.thread_spawn` object. The direct
    /// parent remains available separately as `forked_from_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_provenance: Option<SubagentProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentProvenance {
    /// Top-level timestamp of the child's first `session_meta` record.
    pub spawned_at: String,
    /// Stable path assigned by the spawning agent tree (for example `/root/research`).
    pub agent_path: String,
    /// Human-readable agent nickname, when the provider supplied a non-empty value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_nickname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommit {
    pub hash: String,
    pub author: String,
    pub date: String,
    pub message: String,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_session_serialization() {
        let session = ClaudeSession {
            session_id: "/path/to/file.jsonl".to_string(),
            actual_session_id: "actual-session-id".to_string(),
            file_path: "/path/to/file.jsonl".to_string(),
            project_name: "my-project".to_string(),
            message_count: 42,
            first_message_time: "2025-06-01T10:00:00Z".to_string(),
            last_message_time: "2025-06-01T12:00:00Z".to_string(),
            last_modified: "2025-06-01T12:00:00Z".to_string(),
            has_tool_use: true,
            has_errors: false,
            summary: Some("Test conversation".to_string()),
            is_renamed: false,
            provider: None,
            storage_type: None,
            entrypoint: None,
            forked_from_id: None,
            subagent_provenance: None,
        };

        let serialized = serde_json::to_string(&session).unwrap();
        assert!(!serialized.contains("forked_from_id"));
        assert!(!serialized.contains("subagent_provenance"));
        let deserialized: ClaudeSession = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.project_name, "my-project");
        assert_eq!(deserialized.message_count, 42);
        assert!(deserialized.has_tool_use);
        assert!(!deserialized.has_errors);
    }

    #[test]
    fn test_legacy_claude_session_deserialization_defaults_fork_provenance() {
        let json = r#"{
            "session_id":"/path/to/file.jsonl",
            "actual_session_id":"actual-session-id",
            "file_path":"/path/to/file.jsonl",
            "project_name":"my-project",
            "message_count":1,
            "first_message_time":"2025-06-01T10:00:00Z",
            "last_message_time":"2025-06-01T10:00:00Z",
            "last_modified":"2025-06-01T10:00:00Z",
            "has_tool_use":false,
            "has_errors":false,
            "summary":null,
            "is_renamed":false
        }"#;

        let session: ClaudeSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.forked_from_id, None);
        assert_eq!(session.subagent_provenance, None);
    }
}
