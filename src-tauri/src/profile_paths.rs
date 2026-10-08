//! The only adapter allowed to resolve native profile directories or process environment.

#![allow(clippy::disallowed_methods)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::TempDir;

#[derive(Clone, Default)]
struct Profile {
    home: Option<PathBuf>,
    environment: HashMap<OsString, OsString>,
}

#[derive(Default)]
struct ScopedProfile {
    current: Profile,
    guards: Vec<u64>,
}

thread_local! {
    static PROFILE: RefCell<ScopedProfile> = RefCell::new(ScopedProfile::default());
}

static NEXT_GUARD: AtomicU64 = AtomicU64::new(1);

#[doc(hidden)]
pub const fn test_profile_enabled() -> bool {
    cfg!(any(test, cc_history_test_isolation))
}

#[doc(hidden)]
pub const fn live_profile_tests_enabled() -> bool {
    cfg!(all(cc_history_live_profile_tests, not(test)))
}

fn directory(native: fn() -> Option<PathBuf>, child: Option<&str>) -> Option<PathBuf> {
    directory_with_mode(test_profile_enabled(), native, child)
}

fn directory_with_mode(
    isolated: bool,
    native: fn() -> Option<PathBuf>,
    child: Option<&str>,
) -> Option<PathBuf> {
    if isolated {
        PROFILE.with(|profile| {
            profile
                .borrow()
                .current
                .home
                .as_ref()
                .map(|home| child.map_or_else(|| home.clone(), |child| home.join(child)))
        })
    } else {
        native()
    }
}

pub fn home_dir() -> Option<PathBuf> {
    directory(native_dirs::home_dir, None)
}

pub fn data_dir() -> Option<PathBuf> {
    directory(native_dirs::data_dir, Some(".test-data"))
}

pub fn data_local_dir() -> Option<PathBuf> {
    directory(native_dirs::data_local_dir, Some(".test-local-data"))
}

pub fn config_dir() -> Option<PathBuf> {
    directory(native_dirs::config_dir, Some(".test-config"))
}

pub fn cache_dir() -> Option<PathBuf> {
    directory(native_dirs::cache_dir, Some(".test-cache"))
}

pub fn download_dir() -> Option<PathBuf> {
    directory(native_dirs::download_dir, Some("Downloads"))
}

pub fn document_dir() -> Option<PathBuf> {
    directory(native_dirs::document_dir, Some("Documents"))
}

pub fn desktop_dir() -> Option<PathBuf> {
    directory(native_dirs::desktop_dir, Some("Desktop"))
}

pub fn managed_settings_path() -> Result<PathBuf, String> {
    if test_profile_enabled() {
        return home_dir()
            .map(|home| home.join(".test-managed").join("managed-settings.json"))
            .ok_or_else(|| "Could not find scoped test home directory".to_string());
    }
    if cfg!(target_os = "macos") {
        Ok(PathBuf::from(
            "/Library/Application Support/ClaudeCode/managed-settings.json",
        ))
    } else {
        Err("Managed settings are only available on macOS".to_string())
    }
}

/// Process environment stays native in ordinary builds. Test builds can read
/// only explicitly installed values, including provider-specific home overrides.
pub mod env {
    use super::{test_profile_enabled, OsStr, OsString, PROFILE};
    use std::env::VarError;

    pub fn var<K: AsRef<OsStr>>(key: K) -> Result<String, VarError> {
        if test_profile_enabled() {
            var_os(key)
                .ok_or(VarError::NotPresent)?
                .into_string()
                .map_err(VarError::NotUnicode)
        } else {
            std::env::var(key)
        }
    }

    pub fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
        if test_profile_enabled() {
            PROFILE.with(|profile| {
                profile
                    .borrow()
                    .current
                    .environment
                    .get(key.as_ref())
                    .cloned()
            })
        } else {
            std::env::var_os(key)
        }
    }

    pub fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
        if test_profile_enabled() {
            PROFILE.with(|profile| {
                profile
                    .borrow_mut()
                    .current
                    .environment
                    .insert(key.as_ref().to_os_string(), value.as_ref().to_os_string());
            });
        } else {
            std::env::set_var(key, value);
        }
    }

    pub fn remove_var<K: AsRef<OsStr>>(key: K) {
        if test_profile_enabled() {
            PROFILE.with(|profile| {
                profile
                    .borrow_mut()
                    .current
                    .environment
                    .remove(key.as_ref());
            });
        } else {
            std::env::remove_var(key);
        }
    }
}

#[doc(hidden)]
pub struct HomeGuard {
    previous: Option<Profile>,
    id: u64,
    _thread: PhantomData<Rc<()>>,
}

impl HomeGuard {
    pub fn set(home: &Path) -> Self {
        assert!(
            test_profile_enabled(),
            "test profile isolation is not enabled"
        );
        assert!(home.is_absolute(), "test home must be absolute");
        let mut current = PROFILE.with(|profile| profile.borrow().current.clone());
        current.home = Some(home.to_path_buf());
        let guard = Self::install(current);
        env::set_var("HOME", home);
        env::set_var("USERPROFILE", home);
        env::set_var("APPDATA", home.join(".test-data"));
        env::set_var("LOCALAPPDATA", home.join(".test-local-data"));
        env::set_var("XDG_DATA_HOME", home.join(".test-data"));
        env::set_var("XDG_CONFIG_HOME", home.join(".test-config"));
        env::set_var("XDG_CACHE_HOME", home.join(".test-cache"));
        guard
    }

    fn install(current: Profile) -> Self {
        let id = NEXT_GUARD.fetch_add(1, Ordering::Relaxed);
        let previous = PROFILE.with(|profile| {
            let mut profile = profile.borrow_mut();
            let previous = std::mem::replace(&mut profile.current, current);
            profile.guards.push(id);
            previous
        });
        Self {
            previous: Some(previous),
            id,
            _thread: PhantomData,
        }
    }
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        PROFILE.with(|profile| {
            let mut profile = profile.borrow_mut();
            if profile.guards.last() == Some(&self.id) {
                profile.guards.pop();
                profile.current = self.previous.take().unwrap_or_default();
            } else {
                // Out-of-order drops cannot restore a context whose owner has ended.
                *profile = ScopedProfile::default();
            }
        });
    }
}

#[doc(hidden)]
pub struct TestProfile {
    _guard: HomeGuard,
    directory: TempDir,
}

impl TestProfile {
    pub fn new() -> Result<Self, String> {
        if !test_profile_enabled() {
            return Err("run integration tests through scripts/test-rust.ts".to_string());
        }
        let directory = TempDir::new().map_err(|error| error.to_string())?;
        let guard = HomeGuard::set(directory.path());
        Ok(Self {
            _guard: guard,
            directory,
        })
    }

    pub fn path(&self) -> &Path {
        self.directory.path()
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct ProfileContext {
    captured: Option<Profile>,
}

impl ProfileContext {
    pub fn capture() -> Self {
        Self {
            captured: test_profile_enabled()
                .then(|| PROFILE.with(|profile| profile.borrow().current.clone())),
        }
    }

    pub fn run<F, R>(&self, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = self.captured.clone().map(HomeGuard::install);
        operation()
    }
}

/// Propagate a test scope only to explicitly owned work. A global override
/// would expose one test's profile to unrelated pool workers.
pub fn spawn_blocking<F, R>(operation: F) -> tauri::async_runtime::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let captured = ProfileContext::capture();
    tauri::async_runtime::spawn_blocking(move || captured.run(operation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_directory_dispatch_preserves_the_native_result() {
        let native: fn() -> Option<PathBuf> = || Some(PathBuf::from("native-directory-sentinel"));
        assert_eq!(
            directory_with_mode(false, native, Some("ignored")),
            native()
        );
        assert_eq!(directory_with_mode(true, native, None), None);
    }

    #[test]
    fn all_profile_directories_fail_closed_without_a_scope() {
        for resolve in [
            home_dir,
            data_dir,
            data_local_dir,
            config_dir,
            cache_dir,
            download_dir,
            document_dir,
            desktop_dir,
        ] {
            assert_eq!(resolve(), None);
        }
        for key in ["HOME", "USERPROFILE", "CODEX_HOME", "HERMES_HOME"] {
            assert_eq!(env::var_os(key), None);
        }
        assert!(!live_profile_tests_enabled());
    }

    #[test]
    fn managed_settings_require_and_remain_inside_the_test_home() {
        assert!(managed_settings_path().is_err());
        let profile = TestProfile::new().unwrap();
        assert_eq!(
            managed_settings_path().unwrap(),
            profile
                .path()
                .join(".test-managed")
                .join("managed-settings.json")
        );
        drop(profile);
        assert!(managed_settings_path().is_err());
    }

    #[test]
    fn unrelated_threads_cannot_observe_profile_or_environment() {
        let profile = TestProfile::new().unwrap();
        env::set_var("CODEX_HOME", profile.path().join(".codex"));
        let child = std::thread::spawn(|| (home_dir(), env::var_os("CODEX_HOME")));
        assert_eq!(child.join().unwrap(), (None, None));
        assert_eq!(home_dir().as_deref(), Some(profile.path()));
    }

    #[test]
    fn concurrent_scopes_keep_distinct_homes() {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                let profile = TestProfile::new().unwrap();
                barrier.wait();
                assert_eq!(home_dir().as_deref(), Some(profile.path()));
                profile.path().to_path_buf()
            }));
        }
        assert_ne!(
            threads.remove(0).join().unwrap(),
            threads.remove(0).join().unwrap()
        );
        assert_eq!(home_dir(), None);
    }

    #[test]
    fn owned_rayon_work_restores_scope_after_success_and_unwind() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let profile = TestProfile::new().unwrap();
        env::set_var("CODEX_HOME", profile.path().join(".codex"));
        let captured = ProfileContext::capture();
        let (home, codex) =
            pool.install(|| captured.run(|| (home_dir(), env::var_os("CODEX_HOME"))));
        assert_eq!(home.as_deref(), Some(profile.path()));
        assert_eq!(codex, env::var_os("CODEX_HOME"));
        assert_eq!(
            pool.install(|| (home_dir(), env::var_os("CODEX_HOME"))),
            (None, None)
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.install(|| captured.run(|| panic!("exercise owned worker cleanup")));
        }));
        assert!(result.is_err());
        assert_eq!(
            pool.install(|| (home_dir(), env::var_os("CODEX_HOME"))),
            (None, None)
        );
    }

    #[test]
    fn nested_scope_restores_environment_and_home_after_unwind() {
        let outer = TestProfile::new().unwrap();
        env::set_var("CODEX_HOME", "outer");
        let result = std::panic::catch_unwind(|| {
            let _inner = TestProfile::new().unwrap();
            env::set_var("CODEX_HOME", "inner");
            panic!("exercise scope cleanup");
        });
        assert!(result.is_err());
        assert_eq!(home_dir().as_deref(), Some(outer.path()));
        assert_eq!(env::var("CODEX_HOME").unwrap(), "outer");
        drop(outer);
        assert_eq!(home_dir(), None);
        assert_eq!(env::var_os("CODEX_HOME"), None);
    }

    #[test]
    fn out_of_order_scope_drop_fails_closed() {
        let outer = TestProfile::new().unwrap();
        let inner = TestProfile::new().unwrap();
        drop(outer);
        assert_eq!(home_dir(), None);
        assert_eq!(env::var_os("HOME"), None);
        drop(inner);
        assert_eq!(home_dir(), None);
    }

    #[test]
    fn environment_overrides_never_mutate_the_process_environment() {
        let key = "CCHV_PROFILE_ISOLATION_ENV_PROBE";
        let before = std::env::var_os(key);
        let _profile = TestProfile::new().unwrap();
        env::set_var(key, "isolated");
        assert_eq!(env::var(key).unwrap(), "isolated");
        assert_eq!(std::env::var_os(key), before);
        env::remove_var(key);
        assert_eq!(env::var_os(key), None);
        assert_eq!(std::env::var_os(key), before);
    }

    #[tokio::test]
    async fn owned_blocking_tasks_receive_only_the_captured_scope() {
        let profile = TestProfile::new().unwrap();
        env::set_var("CODEX_HOME", profile.path().join(".codex"));
        let (home, codex_home) = spawn_blocking(|| (home_dir(), env::var_os("CODEX_HOME")))
            .await
            .unwrap();
        assert_eq!(home.as_deref(), Some(profile.path()));
        assert_eq!(codex_home, env::var_os("CODEX_HOME"));
        drop(profile);
        assert_eq!(spawn_blocking(home_dir).await.unwrap(), None);
    }

    #[tokio::test]
    async fn profile_writers_refuse_missing_scope_before_filesystem_work() {
        use crate::commands::{mcp_presets, unified_presets};

        assert_eq!(home_dir(), None);
        let mcp = mcp_presets::save_mcp_preset(mcp_presets::MCPPresetInput {
            id: Some("isolation-refusal".to_string()),
            name: "isolation refusal".to_string(),
            description: None,
            servers: "{}".to_string(),
        })
        .await
        .unwrap_err();
        assert!(mcp.contains("Could not find home directory"), "{mcp}");
        let unified = unified_presets::save_unified_preset(unified_presets::UnifiedPresetInput {
            id: Some("isolation-refusal".to_string()),
            name: "isolation refusal".to_string(),
            description: None,
            settings: "{}".to_string(),
            mcp_servers: "{}".to_string(),
        })
        .await
        .unwrap_err();
        assert!(
            unified.contains("Could not find home directory"),
            "{unified}"
        );
    }
}
