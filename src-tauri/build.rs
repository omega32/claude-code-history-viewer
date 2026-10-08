#[allow(clippy::disallowed_methods)] // Build-mode selection precedes the application adapter.
fn main() {
    println!("cargo:rerun-if-env-changed=CCHV_TEST_PROFILE");
    println!("cargo:rustc-check-cfg=cfg(cc_history_test_isolation)");
    println!("cargo:rustc-check-cfg=cfg(cc_history_live_profile_tests)");
    match std::env::var("CCHV_TEST_PROFILE") {
        Ok(value) if value == "isolated-v1" => {
            println!("cargo:rustc-cfg=cc_history_test_isolation");
        }
        Ok(value) if value == "live-v1" => {
            println!("cargo:rustc-cfg=cc_history_live_profile_tests");
        }
        Err(std::env::VarError::NotPresent) => {}
        _ => panic!("CCHV_TEST_PROFILE must be absent, isolated-v1 or live-v1"),
    }
    tauri_build::build();
}
