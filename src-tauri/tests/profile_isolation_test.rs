use claude_code_history_viewer_lib::commands::{mcp_presets, unified_presets};

#[tokio::test]
async fn guarded_profile_writers_stay_inside_the_owned_home() {
    let profile = claude_code_history_viewer_lib::profile_paths::TestProfile::new()
        .expect("run integration tests through scripts/test-rust.ts");
    let mcp = mcp_presets::save_mcp_preset(mcp_presets::MCPPresetInput {
        id: Some("isolated-mcp".to_string()),
        name: "isolated MCP".to_string(),
        description: None,
        servers: "{}".to_string(),
    })
    .await
    .unwrap();
    let unified = unified_presets::save_unified_preset(unified_presets::UnifiedPresetInput {
        id: None,
        name: "isolated unified".to_string(),
        description: None,
        settings: "{}".to_string(),
        mcp_servers: "{}".to_string(),
    })
    .await
    .unwrap();
    let root = profile.path().join(".claude-history-viewer");
    for (folder, id) in [("mcp-presets", mcp.id), ("unified-presets", unified.id)] {
        let path = root.join(folder).join(format!("{id}.json"));
        assert!(path.is_file(), "{}", path.display());
        assert!(path
            .canonicalize()
            .unwrap()
            .starts_with(root.canonicalize().unwrap()));
    }
}

#[test]
fn isolated_library_cannot_inherit_provider_overrides_or_unrelated_scopes() {
    let profile = claude_code_history_viewer_lib::profile_paths::TestProfile::new()
        .expect("run integration tests through scripts/test-rust.ts");
    assert!(claude_code_history_viewer_lib::profile_paths::test_profile_enabled());
    assert_eq!(
        claude_code_history_viewer_lib::profile_paths::env::var_os("CODEX_HOME"),
        None
    );
    let home = claude_code_history_viewer_lib::profile_paths::home_dir().unwrap();
    assert_eq!(home, profile.path());
    assert_eq!(
        std::thread::spawn(claude_code_history_viewer_lib::profile_paths::home_dir)
            .join()
            .unwrap(),
        None
    );
    assert!(!claude_code_history_viewer_lib::wsl::is_wsl_available());
    assert!(claude_code_history_viewer_lib::wsl::detect_distros().is_empty());
    #[cfg(target_os = "windows")]
    assert!(claude_code_history_viewer_lib::wsl::resolve_home_path("Ubuntu").is_err());
}
