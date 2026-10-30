use super::super::{
    is_canonical_session_id, is_plain_rollout, metadata_stayed_stable, parse_rollout_header,
};
use super::*;
use crate::utils::is_symlink_or_reparse;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Component;
use std::time::Instant;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProbeConfig {
    schema_version: u32,
    fixture_root: PathBuf,
    session_path: PathBuf,
    capture_path: PathBuf,
    seed_end: u64,
    steps: Vec<ProbeStep>,
    #[serde(default)]
    private_frames: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProbeStep {
    end: u64,
    task_ids: Vec<String>,
}

fn relative_path(path: &Path) -> Result<(), &'static str> {
    if path.as_os_str().is_empty()
        || path.to_str().is_none_or(|text| {
            text.split(['/', '\\']).any(|component| {
                component.is_empty()
                    || matches!(component, "." | "..")
                    || component.ends_with(['.', ' '])
                    || component
                        .chars()
                        .any(|character| character.is_control() || character == ':')
            })
        })
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("invalid-path");
    }
    Ok(())
}

fn direct_path(root: &Path, path: &Path, allow_missing: bool) -> Result<(), &'static str> {
    let mut current = root.to_path_buf();
    for component in path.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if is_symlink_or_reparse(&metadata) => return Err("invalid-path"),
            Ok(_) => {}
            Err(error) if allow_missing && error.kind() == io::ErrorKind::NotFound => {}
            Err(_) => return Err("invalid-path"),
        }
    }
    Ok(())
}

impl ProbeConfig {
    fn validate(&self) -> Result<(PathBuf, PathBuf, PathBuf, String), &'static str> {
        if self.schema_version != 1
            || self.steps.is_empty()
            || self.steps.len() > 4
            || !self.fixture_root.is_absolute()
            || self
                .fixture_root
                .components()
                .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err("invalid-config");
        }
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("invalid-path")?;
        let scratch = repository
            .join("scratch")
            .canonicalize()
            .map_err(|_| "invalid-path")?;
        let root = self
            .fixture_root
            .canonicalize()
            .map_err(|_| "invalid-path")?;
        if root == scratch || !root.starts_with(&scratch) {
            return Err("invalid-path");
        }
        for ancestor in self.fixture_root.ancestors() {
            if is_symlink_or_reparse(&fs::symlink_metadata(ancestor).map_err(|_| "invalid-path")?) {
                return Err("invalid-path");
            }
        }
        let relative_root = root.strip_prefix(&scratch).map_err(|_| "invalid-path")?;
        direct_path(&scratch, relative_root, false)?;
        relative_path(&self.session_path)?;
        relative_path(&self.capture_path)?;
        let components = self.session_path.components().collect::<Vec<_>>();
        if components.len() < 3
            || components[0].as_os_str() != ".codex"
            || !matches!(
                components[1].as_os_str().to_str(),
                Some("sessions" | "archived_sessions")
            )
            || self
                .capture_path
                .components()
                .next()
                .is_some_and(|component| component.as_os_str() == ".codex")
        {
            return Err("invalid-path");
        }
        let selected = root.join(&self.session_path);
        let capture = root.join(&self.capture_path);
        direct_path(&root, &self.session_path, true)?;
        direct_path(&root, &self.capture_path, false)?;
        if selected.exists() || !is_plain_rollout(&selected) || !capture.is_file() {
            return Err("invalid-fixture");
        }
        let source_len = fs::metadata(&capture).map_err(|_| "invalid-fixture")?.len();
        if source_len > 512 * 1024 * 1024 || self.seed_end == 0 {
            return Err("invalid-bound");
        }
        let mut source = File::open(&capture).map_err(|_| "fixture-io")?;
        let mut previous = self.seed_end;
        for end in std::iter::once(self.seed_end).chain(self.steps.iter().map(|step| step.end)) {
            if end > source_len || end == 0 {
                return Err("invalid-bound");
            }
            source
                .seek(SeekFrom::Start(end - 1))
                .map_err(|_| "fixture-io")?;
            let mut byte = [0];
            source.read_exact(&mut byte).map_err(|_| "fixture-io")?;
            if byte != *b"\n" {
                return Err("invalid-bound");
            }
        }
        for step in &self.steps {
            let tasks = step.task_ids.iter().collect::<HashSet<_>>();
            if step.end <= previous
                || tasks.is_empty()
                || tasks.len() > 8
                || tasks.len() != step.task_ids.len()
                || tasks.iter().any(|task| !is_canonical_session_id(task))
            {
                return Err("invalid-bound");
            }
            previous = step.end;
        }
        source.seek(SeekFrom::Start(0)).map_err(|_| "fixture-io")?;
        let mut header_bytes = Vec::new();
        BufReader::new(source.take(1024 * 1024))
            .read_until(b'\n', &mut header_bytes)
            .map_err(|_| "fixture-io")?;
        let header = parse_rollout_header(&header_bytes)
            .map_err(|_| "invalid-fixture")?
            .ok_or("invalid-fixture")?;
        let identity = rollout_file_identity(&selected).ok_or("invalid-fixture")?;
        if !is_canonical_session_id(&header.session_id) || identity.thread_id != header.session_id {
            return Err("invalid-fixture");
        }
        Ok((root, selected, capture, header.session_id))
    }
}

fn frame(
    config: &ProbeConfig,
    session: &str,
    kind: &str,
    index: usize,
    acquired: &Result<Acquisition, String>,
    full: &Result<Vec<ClaudeMessage>, String>,
    metrics: Value,
) -> Result<(), &'static str> {
    let tasks = &config.steps[index].task_ids;
    let mut value = json!({"schemaVersion":1,"kind":kind,"stepIndex":index,"prototypeRefused":acquired.is_err(),"fullRefused":full.is_err(),"metrics":metrics});
    if config.private_frames {
        value["selectors"] = json!(tasks
            .iter()
            .map(|task| json!({"provider":"codex","sessionId":session,"taskId":task}))
            .collect::<Vec<_>>());
        value["prototypeCandidates"] = serde_json::to_value(
            acquired
                .as_ref()
                .map(|result| result.candidates.as_slice())
                .unwrap_or_default(),
        )
        .map_err(|_| "frame-encoding")?;
        value["fullCandidates"] =
            serde_json::to_value(full.as_ref().map(Vec::as_slice).unwrap_or_default())
                .map_err(|_| "frame-encoding")?;
    }
    let encoded = serde_json::to_string(&value).map_err(|_| "frame-encoding")?;
    if encoded.len() > 16 * 1024 * 1024 {
        return Err("frame-bound");
    }
    println!("\nCCHV_PROMPT_PROTO_V1 {encoded}");
    Ok(())
}

fn timed<T>(action: impl FnOnce() -> T) -> (T, f64) {
    let started = Instant::now();
    let result = action();
    (result, started.elapsed().as_secs_f64() * 1000.0)
}

pub(super) fn run() -> Result<(), &'static str> {
    let mut input = Vec::new();
    io::stdin()
        .lock()
        .take(65_537)
        .read_to_end(&mut input)
        .map_err(|_| "invalid-config")?;
    if input.len() > 65_536 {
        return Err("invalid-config");
    }
    let config: ProbeConfig = serde_json::from_slice(&input).map_err(|_| "invalid-config")?;
    let (root, selected, capture, session) = config.validate()?;
    let _home = crate::profile_paths::HomeGuard::set(&root);
    let mut captured = File::open(&capture).map_err(|_| "fixture-io")?;
    let before = captured.metadata().map_err(|_| "fixture-io")?;
    fs::create_dir_all(selected.parent().ok_or("invalid-path")?).map_err(|_| "fixture-io")?;
    direct_path(&root, &config.session_path, true)?;
    let mut working = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&selected)
        .map_err(|_| "fixture-io")?;
    if io::copy(
        &mut Read::by_ref(&mut captured).take(config.seed_end),
        &mut working,
    )
    .map_err(|_| "fixture-io")?
        != config.seed_end
    {
        return Err("fixture-io");
    }
    working.flush().map_err(|_| "fixture-io")?;
    let mut engine = Engine::default();
    let first_tasks = config.steps[0]
        .task_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let (seed, seed_ms) = timed(|| engine.acquire(&selected, &first_tasks));
    let (seed_full, seed_control_ms) = timed(|| full_control(&selected, &first_tasks));
    if !seed
        .as_ref()
        .is_ok_and(|result| result.candidates.is_empty())
        || !seed_full.as_ref().is_ok_and(Vec::is_empty)
    {
        return Err("seed-selector-present");
    }
    let (raw_count, refs, candidates, accounted) = engine.retained_counts();
    frame(
        &config,
        &session,
        "seed",
        0,
        &seed,
        &seed_full,
        json!({"mode":seed.as_ref().map(|result| result.mode).unwrap_or("refused"),"reason":seed.as_ref().map(|result| result.reason).unwrap_or("source-refused"),"seedMs":seed_ms,"seedControlMs":seed_control_ms,"nativeParity":true,"retainedRecords":raw_count,"normalizedRecords":engine.retained.as_ref().map(|retained| retained.view.raw_count),"acceptedLogicalBytes":engine.accepted_logical_bytes(),"retainedRefs":refs,"retainedCandidates":candidates,"retainedBytes":accounted}),
    )?;
    let mut previous = config.seed_end;
    for (index, step) in config.steps.iter().enumerate() {
        let count = step.end - previous;
        if io::copy(&mut Read::by_ref(&mut captured).take(count), &mut working)
            .map_err(|_| "fixture-io")?
            != count
        {
            return Err("fixture-io");
        }
        working.flush().map_err(|_| "fixture-io")?;
        previous = step.end;
        let tasks = step.task_ids.iter().cloned().collect::<HashSet<_>>();
        let ((acquired, warm_ms), (full, full_ms)) = if index % 2 == 0 {
            let warm = timed(|| engine.acquire(&selected, &tasks));
            let full = timed(|| full_control(&selected, &tasks));
            (warm, full)
        } else {
            let full = timed(|| full_control(&selected, &tasks));
            let warm = timed(|| engine.acquire(&selected, &tasks));
            (warm, full)
        };
        let parity = match (&acquired, &full) {
            (Ok(acquired), Ok(full)) => {
                serde_json::to_value(&acquired.candidates).map_err(|_| "frame-encoding")?
                    == serde_json::to_value(full).map_err(|_| "frame-encoding")?
            }
            (Err(_), Err(_)) => true,
            _ => false,
        };
        let (raw_count, refs, candidates, accounted) = engine.retained_counts();
        frame(
            &config,
            &session,
            "step",
            index,
            &acquired,
            &full,
            json!({"mode":acquired.as_ref().map(|result| result.mode).unwrap_or("refused"),"reason":acquired.as_ref().map(|result| result.reason).unwrap_or("source-refused"),"warmMs":warm_ms,"fullMs":full_ms,"nativeParity":parity,"retainedRecords":raw_count,"normalizedRecords":engine.retained.as_ref().map(|retained| retained.view.raw_count),"acceptedLogicalBytes":engine.accepted_logical_bytes(),"retainedRefs":refs,"retainedCandidates":candidates,"retainedBytes":accounted}),
        )?;
        if !parity {
            return Err("native-parity");
        }
    }
    let after = fs::metadata(&capture).map_err(|_| "fixture-io")?;
    if !metadata_stayed_stable(&before, &after) {
        return Err("fixture-changed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> (tempfile::TempDir, Value) {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let root = tempfile::tempdir_in(repository.join("scratch")).unwrap();
        fs::create_dir_all(root.path().join("captured")).unwrap();
        let session = "019f5acd-6b03-7653-ae71-6321002ca8d4";
        let header = format!(
            "{}\n",
            json!({"type":"session_meta","payload":{"id":session}})
        );
        let source = format!(
            "{header}{}\n",
            json!({"type":"event_msg","payload":{"type":"task_started","turn_id":session}})
        );
        fs::write(
            root.path().join("captured/current.jsonl"),
            source.as_bytes(),
        )
        .unwrap();
        let value = json!({"schemaVersion":1,"fixtureRoot":root.path(),"sessionPath":format!(".codex/sessions/rollout-2026-10-19T12-30-47-{session}.jsonl"),"capturePath":"captured/current.jsonl","seedEnd":header.len(),"steps":[{"end":source.len(),"taskIds":[session]}]});
        (root, value)
    }

    #[test]
    fn codex_prompt_prototype_probe_manifest_is_strict_and_non_mutating() {
        let (_profile, original) = manifest();
        let config: ProbeConfig = serde_json::from_value(original.clone()).unwrap();
        assert!(!config.private_frames);
        let (_, selected, _, _) = config.validate().unwrap();
        assert!(!selected.exists());
        for (field, value) in [
            ("schemaVersion", json!(2)),
            ("sessionPath", json!("../live.jsonl")),
            ("sessionPath", json!(".codex/sessions/./alias.jsonl")),
            ("sessionPath", json!(".codex/sessions/rollout.jsonl.zst")),
            ("capturePath", json!(".codex/sessions/capture.jsonl")),
            ("seedEnd", json!(0)),
            ("steps", json!([])),
        ] {
            let mut changed = original.clone();
            changed[field] = value;
            assert!(serde_json::from_value::<ProbeConfig>(changed)
                .unwrap()
                .validate()
                .is_err());
        }
        let mut unknown = original;
        unknown["unsafeOption"] = json!(true);
        assert!(serde_json::from_value::<ProbeConfig>(unknown).is_err());
        assert!(!selected.exists());
    }

    #[test]
    fn codex_prompt_prototype_probe_rejects_bounds_duplicates_and_existing_selected() {
        let (_profile, original) = manifest();
        for mutation in [
            "partial",
            "past-eof",
            "duplicate-task",
            "repeated-end",
            "unknown-step",
        ] {
            let mut changed = original.clone();
            match mutation {
                "partial" => {
                    changed["steps"][0]["end"] =
                        json!(original["steps"][0]["end"].as_u64().unwrap() - 1);
                }
                "past-eof" => changed["steps"][0]["end"] = json!(u64::MAX),
                "duplicate-task" => {
                    let task = changed["steps"][0]["taskIds"][0].clone();
                    changed["steps"][0]["taskIds"]
                        .as_array_mut()
                        .unwrap()
                        .push(task);
                }
                "repeated-end" => {
                    let step = changed["steps"][0].clone();
                    changed["steps"].as_array_mut().unwrap().push(step);
                }
                "unknown-step" => changed["steps"][0]["unsafeOption"] = json!(true),
                _ => unreachable!(),
            }
            assert!(serde_json::from_value::<ProbeConfig>(changed)
                .map_or(true, |config| config.validate().is_err()));
        }
        let config: ProbeConfig = serde_json::from_value(original).unwrap();
        let (_, selected, _, _) = config.validate().unwrap();
        fs::create_dir_all(selected.parent().unwrap()).unwrap();
        fs::write(&selected, "owned preexisting file").unwrap();
        assert!(config.validate().is_err());
        assert_eq!(
            fs::read_to_string(selected).unwrap(),
            "owned preexisting file"
        );
    }
}
