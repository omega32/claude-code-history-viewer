use super::{CodexRolloutHeader, CodexSessionListing};
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

const MAX_DECODED_HEADER: u64 = 1024 * 1024;
const MAX_ENCODED_HEADER: u64 = 4 * 1024 * 1024;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct CodexSessionSource {
    provider: &'static str,
    actual_session_id: String,
    file_path: String,
    is_hidden: bool,
    is_archived: bool,
}

impl From<CodexSessionListing> for CodexSessionSource {
    fn from(listed: CodexSessionListing) -> Self {
        Self {
            provider: "codex",
            actual_session_id: listed.session.actual_session_id,
            file_path: listed.session.file_path,
            is_hidden: false,
            is_archived: listed.is_archived,
        }
    }
}

pub(crate) fn load(session_id: &str) -> Result<Option<CodexSessionSource>, String> {
    if let Some(source) = indexed_source(session_id) {
        return Ok(Some(source));
    }
    super::load_session_metadata_by_id(session_id).map(|listed| listed.map(Into::into))
}

fn indexed_source(session_id: &str) -> Option<CodexSessionSource> {
    if !super::is_canonical_session_id(session_id) {
        return None;
    }
    let base = super::get_base_path()?;
    crate::utils::require_absolute_path(&base, "Codex base path").ok()?;
    let base = PathBuf::from(base);
    let (selection, path) = super::indexed_rollout_selection(&base, session_id)?;
    let path_text = path.to_str()?;
    let archived = super::exact_rollout_path_provenance(&base, path_text).ok()??;
    let canonical_path = path.canonicalize().ok()?;
    let before = super::session_info_fingerprint(&path)?;
    let header = bounded_header(&path).ok()??;
    if header.session_id != session_id || !supported_header(&path, &header) {
        return None;
    }
    let (confirmed_selection, confirmed_path) =
        super::indexed_rollout_selection(&base, session_id)?;
    if selection.rollout_path != confirmed_selection.rollout_path
        || selection.history_mode != confirmed_selection.history_mode
        || path != confirmed_path
        || super::exact_rollout_path_provenance(&base, path_text).ok()?? != archived
        || path.canonicalize().ok()? != canonical_path
        || super::session_info_fingerprint(&path)? != before
    {
        return None;
    }
    Some(CodexSessionSource {
        provider: "codex",
        actual_session_id: header.session_id,
        file_path: path_text.to_string(),
        is_hidden: false,
        is_archived: archived,
    })
}

fn supported_header(path: &Path, header: &CodexRolloutHeader) -> bool {
    match header.history_mode.as_deref() {
        None => true,
        Some("paginated") => {
            super::rollout_file_identity(path)
                .is_some_and(|identity| identity.thread_id == header.session_id)
                && header.history_base.as_ref().map_or(true, |base| {
                    header.first_ordinal == Some(base.end_ordinal_exclusive)
                })
        }
        Some(_) => false,
    }
}

fn bounded_header(path: &Path) -> Result<Option<CodexRolloutHeader>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    if super::is_plain_rollout(path) {
        header_from_reader(file)
    } else {
        let decoder =
            zstd::Decoder::new(file.take(MAX_ENCODED_HEADER)).map_err(|error| error.to_string())?;
        header_from_reader(decoder)
    }
}

fn header_from_reader(reader: impl Read) -> Result<Option<CodexRolloutHeader>, String> {
    let mut reader = BufReader::new(reader.take(MAX_DECODED_HEADER));
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .map_err(|error| error.to_string())?
            == 0
            || !line.ends_with(b"\n")
        {
            return Ok(None);
        }
        if let Some(header) = super::parse_rollout_header(&line)? {
            return Ok(Some(header));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_paths::TestProfile;
    use crate::providers::codex::{self, SESSION_INFO_PARSE_COUNT};
    use rusqlite::{params, Connection};
    use serde_json::{json, Value};
    use std::fmt::Write;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    const ID: &str = "01a0bb2f-24a5-7712-8e2d-dbf43a5057d6";
    const OTHER: &str = "01a0903e-6b82-7df3-8892-5e67623a4b9c";

    fn fixture(archived: bool) -> (TestProfile, PathBuf) {
        let profile = TestProfile::new().unwrap();
        let root = profile.path().join(".codex").join(if archived {
            "archived_sessions"
        } else {
            "sessions"
        });
        fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("rollout-2026-10-30T10-00-00-{ID}.jsonl"));
        write_rows(&path, &[header(ID), user()]);
        (profile, path)
    }

    fn header(id: &str) -> Value {
        json!({"type":"session_meta","payload":{"id":id,"cwd":"C:/Fixture"}})
    }

    fn user() -> Value {
        json!({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"fixture"}]}})
    }

    fn write_rows(path: &Path, rows: &[Value]) {
        let mut content = String::new();
        for row in rows {
            writeln!(content, "{row}").unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn select(profile: &TestProfile, path: &Path, mode: Option<&str>) {
        let conn = Connection::open(profile.path().join(".codex/state_5.sqlite")).unwrap();
        conn.execute_batch("CREATE TABLE IF NOT EXISTS threads(id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, history_mode TEXT);").unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO threads VALUES (?1,?2,?3)",
            params![ID, path.to_string_lossy(), mode],
        )
        .unwrap();
    }

    fn expected() -> Result<Option<CodexSessionSource>, String> {
        codex::load_session_metadata_by_id(ID).map(|listed| listed.map(Into::into))
    }

    #[test]
    fn session_source_indexed_header_avoids_summary_parsing_and_matches_five_fields() {
        let (profile, path) = fixture(false);
        select(&profile, &path, None);
        let ordinary = expected().unwrap();
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);
        fs::remove_file(
            profile
                .path()
                .join(".codex")
                .join(codex::SESSION_METADATA_CACHE_FILENAME),
        )
        .unwrap();
        let source = load(ID).unwrap();
        assert_eq!(source, ordinary);
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 0);
        let fields = serde_json::to_value(source.unwrap()).unwrap();
        assert_eq!(fields.as_object().unwrap().len(), 5);
        assert_eq!(fields["provider"], "codex");
    }

    #[test]
    fn session_source_archived_and_compressed_twin_precedence_matches_metadata() {
        let (profile, path) = fixture(true);
        let bytes = fs::read(&path).unwrap();
        let compressed = path.with_extension("jsonl.zst");
        fs::write(&compressed, zstd::encode_all(bytes.as_slice(), 1).unwrap()).unwrap();
        select(&profile, &path, None);
        assert_eq!(load(ID), expected());
        assert_eq!(load(ID).unwrap().unwrap().file_path, path.to_string_lossy());
        fs::remove_file(&path).unwrap();
        assert_eq!(load(ID), expected());
        let source = load(ID).unwrap().unwrap();
        assert_eq!(source.file_path, compressed.to_string_lossy());
        assert!(source.is_archived);
        fs::write(&path, bytes).unwrap();
        select(&profile, &compressed, None);
        assert_eq!(load(ID).unwrap().unwrap().file_path, path.to_string_lossy());
    }

    #[test]
    fn session_source_fallback_preserves_metaless_no_index_and_ambiguity() {
        let (profile, path) = fixture(false);
        assert_eq!(load(ID), expected());
        write_rows(&path, &[user()]);
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_some());
        select(&profile, &path, None);
        assert_eq!(load(ID), expected());
        fs::remove_file(profile.path().join(".codex/state_5.sqlite")).unwrap();
        let other = path.with_file_name(format!("rollout-2026-10-30T11-00-00-{ID}.jsonl"));
        fs::copy(&path, other).unwrap();
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn session_source_first_header_and_native_id_mismatch_do_not_guess() {
        let (profile, path) = fixture(false);
        select(&profile, &path, None);
        write_rows(&path, &[header(OTHER), header(ID), user()]);
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_none());
        write_rows(&path, &[header(ID), header(OTHER), user()]);
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_some());
    }

    #[test]
    fn session_source_missing_selected_paginated_carrier_does_not_resurrect() {
        let (profile, path) = fixture(false);
        let missing =
            path.with_file_name(format!("rollout-2026-10-30T11-00-00-{ID}_{OTHER}.jsonl"));
        select(&profile, &missing, Some("paginated"));
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_none());
    }

    #[test]
    fn session_source_paginated_header_with_null_sqlite_mode_avoids_ancestry_summary_work() {
        let (profile, path) = fixture(false);
        let mut meta = header(ID);
        meta["ordinal"] = json!(0);
        meta["payload"]["history_mode"] = json!("paginated");
        write_rows(&path, &[meta, user()]);
        #[cfg(windows)]
        let selected = PathBuf::from(format!(r"\\?\{}", path.display()));
        #[cfg(not(windows))]
        let selected = path.clone();
        select(&profile, &selected, None);
        SESSION_INFO_PARSE_COUNT.store(0, Ordering::SeqCst);
        let fast = load(ID).unwrap();
        assert_eq!(SESSION_INFO_PARSE_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(fast, expected().unwrap());
    }

    #[test]
    fn session_source_header_prefix_bounds_and_torn_lines_fall_back_without_changing_availability()
    {
        let (profile, path) = fixture(false);
        select(&profile, &path, None);
        let row = format!("{}\n", header(ID));
        let padding = usize::try_from(MAX_DECODED_HEADER).unwrap() - row.len();
        let at_bound = " ".repeat(padding) + &row;
        assert!(header_from_reader(at_bound.as_bytes()).unwrap().is_some());
        let oversized = " ".repeat(padding + 1) + &row;
        assert!(header_from_reader(oversized.as_bytes()).unwrap().is_none());
        for bytes in [
            oversized,
            header(ID).to_string(),
            format!("not-json\n{{\"type\":\"session_meta\"}}\n{row}"),
        ] {
            fs::write(&path, bytes).unwrap();
            assert!(indexed_source(ID).is_none());
            assert_eq!(load(ID), expected());
        }
    }

    #[test]
    fn session_source_encoded_header_limit_preserves_full_compressed_fallback() {
        let (profile, path) = fixture(false);
        let content = fs::read(&path).unwrap();
        let compressed = path.with_extension("jsonl.zst");
        let skip = u32::try_from(MAX_ENCODED_HEADER).unwrap();
        let mut bytes = vec![0x50, 0x2a, 0x4d, 0x18];
        bytes.extend_from_slice(&skip.to_le_bytes());
        bytes.resize(bytes.len() + usize::try_from(skip).unwrap(), 0);
        bytes.extend(zstd::encode_all(content.as_slice(), 1).unwrap());
        fs::write(&compressed, bytes).unwrap();
        fs::remove_file(&path).unwrap();
        select(&profile, &path, None);
        assert!(indexed_source(ID).is_none());
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_some());
    }

    #[test]
    fn session_source_unsafe_plain_twin_does_not_choose_compressed_or_escape_root() {
        let (profile, path) = fixture(false);
        let compressed = path.with_extension("jsonl.zst");
        fs::write(
            &compressed,
            zstd::encode_all(fs::read(&path).unwrap().as_slice(), 1).unwrap(),
        )
        .unwrap();
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        select(&profile, &compressed, Some("paginated"));
        assert!(indexed_source(ID).is_none());
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_none());
        let outside = profile
            .path()
            .join(format!("rollout-2026-10-30T10-00-00-{ID}.jsonl"));
        write_rows(&outside, &[header(ID), user()]);
        select(&profile, &outside, Some("paginated"));
        assert!(indexed_source(ID).is_none());
        assert_eq!(load(ID), expected());
        assert!(load(ID).unwrap().is_none());
    }

    #[test]
    fn session_source_selected_rotation_and_archive_change_are_live() {
        let (profile, path) = fixture(false);
        select(&profile, &path, None);
        let first = load(ID).unwrap().unwrap();
        let archive = profile.path().join(".codex").join("archived_sessions");
        fs::create_dir_all(&archive).unwrap();
        let next = archive.join(format!("rollout-2026-10-30T11-00-00-{ID}_{OTHER}.jsonl"));
        fs::copy(&path, &next).unwrap();
        select(&profile, &next, None);
        let second = load(ID).unwrap().unwrap();
        assert_ne!(first.file_path, second.file_path);
        assert!(!first.is_archived);
        assert!(second.is_archived);
        assert_eq!(Some(second), expected().unwrap());
    }

    #[test]
    fn session_source_does_not_claim_lineage_validation_or_mask_native_candidate_refusal() {
        let (profile, path) = fixture(false);
        let mut meta = header(ID);
        meta["ordinal"] = json!(1);
        meta["payload"]["history_mode"] = json!("paginated");
        meta["payload"]["history_base"] =
            json!({"thread_id":OTHER,"end_byte_offset":10,"end_ordinal_exclusive":1});
        write_rows(&path, &[meta, user()]);
        select(&profile, &path, Some("paginated"));
        assert!(
            load(ID).unwrap().is_some(),
            "source only observes the selected carrier"
        );
        assert!(expected().unwrap().is_none());
        assert!(codex::load_session_metadata_by_path(path.to_str().unwrap()).is_err());
        assert!(codex::load_messages(path.to_str().unwrap()).is_err());
    }
}
