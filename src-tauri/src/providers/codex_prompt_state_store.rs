use super::{
    decode_snapshot_cursor, dpapi, full_control, get_base_path, validate_session_path, Bounds,
    CountBytes, Engine, Retained, SNAPSHOT_CURSOR_VERSION,
};
use crate::models::ClaudeMessage;
use crate::utils::is_symlink_or_reparse;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zeroize::Zeroizing;

const SCHEMA: u32 = 1;
const MAX_ENTRIES: usize = 8;
const TTL_SECONDS: u64 = 24 * 60 * 60;
const MAX_CIPHERTEXT: u64 = 64 * 1024 * 1024;
const MAX_PLAINTEXT: usize = 48 * 1024 * 1024;
const MAX_TOTAL: u64 = 128 * 1024 * 1024;
const MAX_DIRECTORY_ENTRIES: usize = 64;
const MAGIC: &[u8] = b"CCHV-CODEX-PROMPT-STATE\0";
const OWNER: &[u8] = b"claude-code-history-viewer codex-prompt-state-v1\n";
const OWNER_FILE: &str = ".owner";
const LOCK_FILE: &str = ".lock";

#[cfg(test)]
thread_local! {
    static OBSERVATION: std::cell::Cell<(bool, &'static str)> = const { std::cell::Cell::new((false, "full")) };
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    schema: u32,
    epoch: u32,
    native_version: String,
    provider: String,
    session_id: String,
    carrier: PathBuf,
    codex_home: PathBuf,
    cache_root: PathBuf,
    executable: PathBuf,
    image_sha256: String,
}

#[derive(Serialize)]
struct Saved<'a> {
    schema: u32,
    updated_at_seconds: u64,
    binding: &'a Binding,
    state: &'a Retained,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Restored {
    schema: u32,
    updated_at_seconds: u64,
    binding: Binding,
    state: Retained,
}

struct BoundedBytes(Zeroizing<Vec<u8>>);

impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .map_or(true, |length| length > MAX_PLAINTEXT)
        {
            return Err(std::io::Error::other("state bound"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(|_| std::io::Error::other("state allocation"))?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn seconds() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|time| time.as_secs())
}

fn hash_image(path: &Path) -> Option<String> {
    let mut file = regular_open(path, false, false)?;
    let before = file.metadata().ok()?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let length = file.read(&mut buffer).ok()?;
        if length == 0 {
            break;
        }
        hash.update(&buffer[..length]);
    }
    let after = file.metadata().ok()?;
    if before.len() != after.len() || before.modified().ok()? != after.modified().ok()? {
        return None;
    }
    Some(format!("{:x}", hash.finalize()))
}

fn regular_open(path: &Path, write: bool, create: bool) -> Option<File> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(write)
        .create(create)
        .truncate(false);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path).ok()?;
    let metadata = file.metadata().ok()?;
    (metadata.is_file() && !is_symlink_or_reparse(&metadata)).then_some(file)
}

fn direct_directory(path: &Path) -> Option<()> {
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).ok()?;
        if !metadata.is_dir() || is_symlink_or_reparse(&metadata) {
            return None;
        }
    }
    Some(())
}

fn root_directory(home: &Path) -> Option<PathBuf> {
    let requested = crate::profile_paths::env::var_os("CCHV_PROMPT_STATE_CACHE_DIR").map_or_else(
        || {
            crate::profile_paths::cache_dir()
                .map(|root| root.join("claude-code-history-viewer/codex-prompt-state-v1"))
        },
        |override_path| Some(PathBuf::from(override_path)),
    )?;
    if !requested.is_absolute()
        || requested
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return None;
    }
    let existing = requested.ancestors().find(|candidate| candidate.exists())?;
    direct_directory(existing)?;
    let canonical_existing = existing.canonicalize().ok()?;
    let future = canonical_existing.join(requested.strip_prefix(existing).ok()?);
    if future.starts_with(home) || home.starts_with(&future) {
        return None;
    }
    fs::create_dir_all(&future).ok()?;
    direct_directory(&future)?;
    let root = future.canonicalize().ok()?;
    if root.starts_with(home) || home.starts_with(&root) {
        return None;
    }
    Some(root)
}

fn slot_name(name: &str) -> bool {
    name.strip_suffix(".bin").is_some_and(|stem| {
        stem.len() == 64
            && stem
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn temporary_name(name: &str) -> bool {
    name.strip_suffix(".tmp")
        .is_some_and(|stem| stem.starts_with(".pending-") && name.len() < 128)
}

struct Entry {
    path: PathBuf,
    length: u64,
    modified: SystemTime,
    temporary: bool,
}

fn inventory(root: &Path) -> Option<Vec<Entry>> {
    let mut entries = Vec::new();
    for (count, entry) in fs::read_dir(root).ok()?.enumerate() {
        if count >= MAX_DIRECTORY_ENTRIES {
            return None;
        }
        let entry = entry.ok()?;
        let name = entry.file_name().into_string().ok()?;
        let metadata = fs::symlink_metadata(entry.path()).ok()?;
        if !metadata.is_file() || is_symlink_or_reparse(&metadata) {
            return None;
        }
        if name == LOCK_FILE || name == OWNER_FILE {
            continue;
        }
        if !slot_name(&name) && !temporary_name(&name) {
            return None;
        }
        entries.push(Entry {
            path: entry.path(),
            length: metadata.len(),
            modified: metadata.modified().ok()?,
            temporary: temporary_name(&name),
        });
    }
    Some(entries)
}

fn establish_owner(root: &Path) -> Option<()> {
    let path = root.join(OWNER_FILE);
    if !path.exists() {
        if fs::read_dir(root).ok()?.next().is_some() {
            return None;
        }
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .ok()?;
        file.write_all(OWNER).ok()?;
        file.sync_all().ok()?;
    }
    let file = regular_open(&path, false, false)?;
    if file.metadata().ok()?.len() != OWNER.len() as u64 {
        return None;
    }
    let mut value = Vec::new();
    file.take(OWNER.len() as u64 + 1)
        .read_to_end(&mut value)
        .ok()?;
    (value == OWNER).then_some(())
}

struct Slot {
    binding: Binding,
    key: String,
    path: PathBuf,
    _lock: File,
}

impl Slot {
    fn open(carrier: &Path) -> Option<Self> {
        let carrier = validate_session_path(carrier, &carrier.to_string_lossy()).ok()?;
        let home = PathBuf::from(get_base_path()?).canonicalize().ok()?;
        let session_id = super::super::session_id_from_rollout_filename(&carrier)?;
        if !super::super::is_canonical_session_id(&session_id) {
            return None;
        }
        let executable = std::env::current_exe().ok()?.canonicalize().ok()?;
        let image_sha256 = hash_image(&executable)?;
        let root = root_directory(&home)?;
        establish_owner(&root)?;
        inventory(&root)?;
        let lock = regular_open(&root.join(LOCK_FILE), true, true)?;
        FileExt::try_lock_exclusive(&lock).ok()?;
        let binding = Binding {
            schema: SCHEMA,
            epoch: SNAPSHOT_CURSOR_VERSION,
            native_version: env!("CARGO_PKG_VERSION").to_string(),
            provider: "codex".to_string(),
            session_id,
            carrier,
            codex_home: home,
            cache_root: root,
            executable,
            image_sha256,
        };
        let key = format!("{:x}", Sha256::digest(serde_json::to_vec(&binding).ok()?));
        let path = binding.cache_root.join(format!("{key}.bin"));
        let slot = Self {
            binding,
            key,
            path,
            _lock: lock,
        };
        slot.prune(0)?;
        Some(slot)
    }

    fn prune(&self, incoming: u64) -> Option<()> {
        let now = SystemTime::now();
        let mut kept = Vec::new();
        for entry in inventory(&self.binding.cache_root)? {
            let stale = now
                .duration_since(entry.modified)
                .is_ok_and(|age| age.as_secs() > TTL_SECONDS);
            if entry.temporary || entry.length > MAX_CIPHERTEXT || stale {
                fs::remove_file(&entry.path).ok()?;
            } else {
                kept.push(entry);
            }
        }
        kept.sort_by_key(|entry| entry.modified);
        let replacing = kept.iter().any(|entry| entry.path == self.path);
        let mut count = kept.len() + usize::from(incoming > 0 && !replacing);
        // Old and incoming ciphertext coexist until atomic replacement completes.
        let mut bytes = kept
            .iter()
            .try_fold(incoming, |sum, entry| sum.checked_add(entry.length))?;
        for entry in &kept {
            if count <= MAX_ENTRIES && bytes <= MAX_TOTAL {
                break;
            }
            if incoming > 0 && entry.path == self.path {
                continue;
            }
            fs::remove_file(&entry.path).ok()?;
            count -= 1;
            bytes -= entry.length;
        }
        (count <= MAX_ENTRIES && bytes <= MAX_TOTAL).then_some(())
    }

    fn read(&self) -> Option<Retained> {
        let mut file = regular_open(&self.path, false, false)?;
        let length = file.metadata().ok()?.len();
        if length == 0 || length > MAX_CIPHERTEXT {
            return None;
        }
        let mut ciphertext = Vec::new();
        ciphertext
            .try_reserve_exact(usize::try_from(length).ok()?)
            .ok()?;
        Read::by_ref(&mut file)
            .take(MAX_CIPHERTEXT + 1)
            .read_to_end(&mut ciphertext)
            .ok()?;
        if ciphertext.len() as u64 != length || file.metadata().ok()?.len() != length {
            return None;
        }
        let clear = Zeroizing::new(dpapi::unprotect(
            &ciphertext,
            self.key.as_bytes(),
            MAX_PLAINTEXT + MAGIC.len() + 44,
        )?);
        let payload = unpack(&clear)?;
        let restored: Restored = serde_json::from_slice(payload).ok()?;
        if restored.schema != SCHEMA
            || restored.binding != self.binding
            || seconds()?.checked_sub(restored.updated_at_seconds)? > TTL_SECONDS
            || !valid_state(&restored.state, &self.binding)
        {
            return None;
        }
        Some(restored.state)
    }

    fn save(&self, state: &Retained) -> Option<()> {
        if !valid_state(state, &self.binding) {
            return None;
        }
        let mut sink = BoundedBytes(Zeroizing::new(Vec::new()));
        serde_json::to_writer(
            &mut sink,
            &Saved {
                schema: SCHEMA,
                updated_at_seconds: seconds()?,
                binding: &self.binding,
                state,
            },
        )
        .ok()?;
        let clear = pack(&sink.0)?;
        let cipher = dpapi::protect(&clear, self.key.as_bytes(), MAX_CIPHERTEXT as usize)?;
        self.prune(cipher.len() as u64)?;
        if hash_image(&self.binding.executable)? != self.binding.image_sha256 {
            return None;
        }
        direct_directory(&self.binding.cache_root)?;
        if self.path.exists() {
            regular_open(&self.path, false, false)?;
        }
        let mut temporary = tempfile::Builder::new()
            .prefix(".pending-")
            .suffix(".tmp")
            .tempfile_in(&self.binding.cache_root)
            .ok()?;
        temporary.write_all(&cipher).ok()?;
        temporary.as_file().sync_all().ok()?;
        temporary.persist(&self.path).ok()?;
        Some(())
    }
}

fn pack(payload: &[u8]) -> Option<Zeroizing<Vec<u8>>> {
    if payload.len() > MAX_PLAINTEXT {
        return None;
    }
    let mut clear = Zeroizing::new(Vec::new());
    clear
        .try_reserve_exact(MAGIC.len() + 44 + payload.len())
        .ok()?;
    clear.extend_from_slice(MAGIC);
    clear.extend_from_slice(&SCHEMA.to_le_bytes());
    clear.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    clear.extend_from_slice(&Sha256::digest(payload));
    clear.extend_from_slice(payload);
    Some(clear)
}

fn unpack(clear: &[u8]) -> Option<&[u8]> {
    let header = MAGIC.len() + 44;
    if clear.len() < header || &clear[..MAGIC.len()] != MAGIC {
        return None;
    }
    let fields = &clear[MAGIC.len()..];
    if u32::from_le_bytes(fields[..4].try_into().ok()?) != SCHEMA {
        return None;
    }
    let declared = usize::try_from(u64::from_le_bytes(fields[4..12].try_into().ok()?)).ok()?;
    let payload = &clear[header..];
    if declared > MAX_PLAINTEXT
        || declared != payload.len()
        || Sha256::digest(payload).as_slice() != &fields[12..44]
    {
        return None;
    }
    Some(payload)
}

fn valid_state(state: &Retained, binding: &Binding) -> bool {
    let bounds = Bounds::default();
    if state.epoch != SNAPSHOT_CURSOR_VERSION
        || state.path != binding.carrier
        || state.cursor.len() > super::super::MAX_SNAPSHOT_CURSOR_BYTES
        || state.view.check(bounds).is_err()
    {
        return false;
    }
    let Ok(cursor) = decode_snapshot_cursor(&state.cursor) else {
        return false;
    };
    if cursor.version != SNAPSHOT_CURSOR_VERSION
        || cursor.provider != "codex"
        || cursor.canonical_path != binding.carrier.to_string_lossy()
        || cursor.checkpoint.state.session_id != binding.session_id
        || cursor.checkpoint.replace_from > state.view.raw_count
    {
        return false;
    }
    if cursor.lineage.as_ref().is_some_and(|lineage| {
        lineage.is_empty()
            || lineage.len() > 128
            || lineage.iter().any(|segment| {
                let path = Path::new(&segment.canonical_path);
                !path.is_absolute()
                    || path
                        .components()
                        .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
                    || segment
                        .canonical_path
                        .split(['/', '\\'])
                        .any(|part| matches!(part, "." | ".."))
                    || (!path.starts_with(binding.codex_home.join("sessions"))
                        && !path.starts_with(binding.codex_home.join("archived_sessions")))
            })
    }) {
        return false;
    }
    let reference_bytes = state
        .view
        .refs
        .iter()
        .try_fold(0usize, |sum, (_, reference)| {
            sum.checked_add(reference.scope.len())?
                .checked_add(reference.scheme.len())?
                .checked_add(reference.id.len())
        });
    if reference_bytes != Some(state.view.ref_bytes) {
        return false;
    }
    let mut counter = CountBytes {
        bytes: 0,
        limit: bounds.candidate_bytes,
    };
    for (_, message) in &state.view.candidates {
        if message.session_id != binding.session_id
            || message.provider.as_deref() != Some("codex")
            || serde_json::to_writer(&mut counter, message).is_err()
        {
            return false;
        }
    }
    counter.bytes == state.view.candidate_bytes
}

pub(super) fn load_candidates(
    path: &Path,
    tasks: &HashSet<String>,
) -> Result<Vec<ClaudeMessage>, String> {
    #[cfg(test)]
    OBSERVATION.with(|observation| observation.set((false, "full")));
    if !cfg!(windows) {
        return full_control(path, tasks);
    }
    let Some(slot) = Slot::open(path) else {
        return full_control(path, tasks);
    };
    let mut engine = Engine {
        retained: slot.read(),
        ..Engine::default()
    };
    #[cfg(test)]
    let cache_hit = engine.retained.is_some();
    let acquired = match engine.acquire(path, tasks) {
        Ok(acquired) => acquired,
        Err(_) => return full_control(path, tasks),
    };
    if acquired.reason == "seed-unconfirmed" && engine.retained.is_some() {
        return full_control(path, tasks);
    }
    if let Some(state) = &engine.retained {
        if slot.save(state).is_none() && acquired.mode != "full" {
            return full_control(path, tasks);
        }
    }
    #[cfg(test)]
    OBSERVATION.with(|observation| observation.set((cache_hit, acquired.mode)));
    Ok(acquired.candidates)
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::super::tests::{append, paginated_fixture, rows, SESSION, TASK2};
    use super::super::tests::{fixture, ids, values, TASK1};
    use super::*;
    #[cfg(windows)]
    use serde_json::json;

    fn parity(path: &Path, task: &str) -> Vec<ClaudeMessage> {
        let acquired = load_candidates(path, &ids(task)).unwrap();
        assert_eq!(
            values(&acquired),
            values(&full_control(path, &ids(task)).unwrap())
        );
        acquired
    }

    #[test]
    fn prompt_state_cache_inner_integrity_is_domain_length_and_digest_bound() {
        let payload = b"synthetic state";
        let original = pack(payload).unwrap();
        assert_eq!(unpack(&original), Some(payload.as_slice()));
        for offset in [
            0,
            MAGIC.len(),
            MAGIC.len() + 4,
            MAGIC.len() + 12,
            original.len() - 1,
        ] {
            let mut changed = original.to_vec();
            changed[offset] ^= 1;
            assert!(unpack(&changed).is_none());
        }
        let mut trailing = original.to_vec();
        trailing.push(0);
        assert!(unpack(&trailing).is_none());
    }

    #[test]
    fn prompt_state_cache_invalid_root_is_a_non_mutating_full_fallback() {
        let (_profile, path) = fixture();
        let relative = format!("cache-relative-{}", uuid::Uuid::new_v4());
        crate::profile_paths::env::set_var("CCHV_PROMPT_STATE_CACHE_DIR", &relative);
        assert_eq!(parity(&path, TASK1).len(), 1);
        assert!(!Path::new(&relative).exists());
        let provider_root = path.parent().unwrap().join("forbidden-cache");
        crate::profile_paths::env::set_var("CCHV_PROMPT_STATE_CACHE_DIR", &provider_root);
        parity(&path, TASK1);
        assert!(!provider_root.exists());
    }

    #[test]
    fn prompt_state_cache_foreign_directory_is_preserved() {
        let (profile, path) = fixture();
        let root = profile.path().join("foreign-cache");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("foreign.txt"), b"preserve").unwrap();
        crate::profile_paths::env::set_var("CCHV_PROMPT_STATE_CACHE_DIR", &root);
        parity(&path, TASK1);
        assert_eq!(fs::read(root.join("foreign.txt")).unwrap(), b"preserve");
        assert_eq!(fs::read_dir(root).unwrap().count(), 1);
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_two_owners_acquire_a_new_prompt_from_encrypted_state() {
        let (_profile, path) = fixture();
        assert!(parity(&path, TASK2).is_empty());
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (false, "full")));
        let slot = Slot::open(&path).unwrap();
        let cipher = fs::read(&slot.path).unwrap();
        assert!(!cipher
            .windows(b"ordinary authored prompt".len())
            .any(|window| window == b"ordinary authored prompt"));
        assert!(slot.read().is_some());
        drop(slot);
        append(&path, &rows(TASK2, "new-persisted-user"));
        assert_eq!(parity(&path, TASK2).len(), 1);
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (true, "replace")));
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_source_correct_plaintext_and_corrupt_ciphertext_are_misses() {
        let (_profile, path) = fixture();
        parity(&path, TASK1);
        let slot = Slot::open(&path).unwrap();
        let state = slot.read().unwrap();
        let cipher = fs::read(&slot.path).unwrap();
        let plaintext = serde_json::to_vec(&Saved {
            schema: SCHEMA,
            updated_at_seconds: seconds().unwrap(),
            binding: &slot.binding,
            state: &state,
        })
        .unwrap();
        fs::write(&slot.path, &plaintext).unwrap();
        assert!(slot.read().is_none());
        for offset in [0, cipher.len() / 2, cipher.len() - 1] {
            let mut changed = cipher.clone();
            changed[offset] ^= 1;
            fs::write(&slot.path, changed).unwrap();
            assert!(slot.read().is_none());
        }
        drop(slot);
        parity(&path, TASK1);
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (false, "full")));
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_valid_dpapi_still_requires_inner_integrity_and_semantic_binding() {
        let (_profile, path) = fixture();
        parity(&path, TASK1);
        let slot = Slot::open(&path).unwrap();
        let state = slot.read().unwrap();
        let original = serde_json::to_value(Saved {
            schema: SCHEMA,
            updated_at_seconds: seconds().unwrap(),
            binding: &slot.binding,
            state: &state,
        })
        .unwrap();
        for mutation in [
            "image",
            "version",
            "home",
            "carrier",
            "schema",
            "expired",
            "future",
            "epoch",
            "accounting",
            "ledger",
            "unknown",
        ] {
            let mut changed = original.clone();
            match mutation {
                "image" => changed["binding"]["image_sha256"] = json!("wrong-image"),
                "version" => changed["binding"]["native_version"] = json!("stale-version"),
                "home" => changed["binding"]["codex_home"] = json!("C:/foreign"),
                "carrier" => changed["binding"]["carrier"] = json!("C:/foreign.jsonl"),
                "schema" => changed["schema"] = json!(2),
                "expired" => {
                    changed["updated_at_seconds"] = json!(seconds().unwrap() - TTL_SECONDS - 1);
                }
                "future" => changed["updated_at_seconds"] = json!(seconds().unwrap() + 3600),
                "epoch" => changed["state"]["epoch"] = json!(0),
                "accounting" => changed["state"]["view"]["candidate_bytes"] = json!(0),
                "ledger" => changed["state"]["view"]["refs"] = json!([]),
                "unknown" => changed["state"]["unexpected"] = json!(true),
                _ => unreachable!(),
            }
            let clear = pack(&serde_json::to_vec(&changed).unwrap()).unwrap();
            fs::write(
                &slot.path,
                dpapi::protect(&clear, slot.key.as_bytes(), MAX_CIPHERTEXT as usize).unwrap(),
            )
            .unwrap();
            assert!(slot.read().is_none(), "{mutation}");
        }
        let mut clear = pack(&serde_json::to_vec(&original).unwrap()).unwrap();
        let last = clear.len() - 1;
        clear[last] ^= 1;
        fs::write(
            &slot.path,
            dpapi::protect(&clear, slot.key.as_bytes(), MAX_CIPHERTEXT as usize).unwrap(),
        )
        .unwrap();
        assert!(slot.read().is_none());
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_lock_contention_never_publishes() {
        let (_profile, path) = fixture();
        parity(&path, TASK1);
        let slot = Slot::open(&path).unwrap();
        let before = fs::read(&slot.path).unwrap();
        parity(&path, TASK1);
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (false, "full")));
        assert_eq!(fs::read(&slot.path).unwrap(), before);
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_rewrite_and_collisions_match_ordinary_reads() {
        let (_profile, path) = fixture();
        let original = fs::read(&path).unwrap();
        parity(&path, TASK1);
        append(&path, &rows(TASK2, "native-1"));
        assert!(parity(&path, TASK1)[0].record_ref.is_none());
        fs::write(&path, original).unwrap();
        append(&path, &rows(TASK2, "native-2"));
        assert!(parity(&path, TASK1)[0].record_ref.is_some());
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (true, "full")));
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_lineage_twin_drift_rebuilds_after_restoration() {
        let (_profile, parent, path, bytes) = paginated_fixture();
        parity(&path, TASK2);
        fs::write(parent, bytes).unwrap();
        parity(&path, TASK2);
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (true, "full")));
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_eviction_and_maintenance_are_bounded() {
        let (_profile, path) = fixture();
        let body = fs::read_to_string(&path).unwrap();
        for _ in 0..MAX_ENTRIES + 2 {
            let session = uuid::Uuid::new_v4().to_string();
            let next = path
                .parent()
                .unwrap()
                .join(format!("rollout-2026-10-30T00-00-00-{session}.jsonl"));
            fs::write(&next, body.replace(SESSION, &session)).unwrap();
            parity(&next, TASK1);
        }
        let slot = Slot::open(&path).unwrap();
        let entries = inventory(&slot.binding.cache_root).unwrap();
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert!(entries.iter().map(|entry| entry.length).sum::<u64>() <= MAX_TOTAL);
        fs::write(
            slot.binding.cache_root.join(".pending-abandoned.tmp"),
            b"ciphertext fragment",
        )
        .unwrap();
        slot.prune(0).unwrap();
        assert!(!slot
            .binding
            .cache_root
            .join(".pending-abandoned.tmp")
            .exists());
    }

    #[cfg(windows)]
    #[test]
    fn prompt_state_cache_failed_atomic_replace_keeps_old_ciphertext_and_returns_full() {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;
        let (_profile, path) = fixture();
        parity(&path, TASK1);
        let slot = Slot::open(&path).unwrap();
        let before = fs::read(&slot.path).unwrap();
        let _deny_replace = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&slot.path)
            .unwrap();
        let target = slot.path.clone();
        drop(slot);
        append(&path, &rows(TASK2, "new-user-after-lock"));
        parity(&path, TASK2);
        OBSERVATION.with(|observation| assert_eq!(observation.get(), (false, "full")));
        assert_eq!(fs::read(target).unwrap(), before);
    }
}
