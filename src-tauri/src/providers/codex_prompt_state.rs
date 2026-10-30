use super::{
    decode_snapshot_cursor, find_rollout_path_by_id, get_base_path, load_messages,
    load_session_snapshot, rollout_file_identity, snapshot_cursor_replace_from,
    validate_session_path, ClaudeMessage, RecordRef, SessionSnapshotLoad, Value,
    AUTHORED_USER_SUBTYPE, SNAPSHOT_CURSOR_VERSION,
};
use crate::commands::multi_provider::finalize_loaded_messages;
use crate::prompt_candidates::{select_candidates, valid_record_ref};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::fs;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

#[path = "codex_prompt_state_dpapi.rs"]
mod dpapi;
#[cfg(test)]
#[path = "codex_prompt_prototype_probe.rs"]
mod probe;
#[path = "codex_prompt_state_store.rs"]
mod store;

#[cfg(test)]
pub(super) fn run_private_probe() -> Result<(), &'static str> {
    probe::run()
}

pub(super) fn load_cached_candidates(
    path: &Path,
    task_ids: &HashSet<String>,
) -> Result<Vec<ClaudeMessage>, String> {
    store::load_candidates(path, task_ids)
}

#[derive(Clone, Copy)]
struct Bounds {
    records: usize,
    ref_bytes: usize,
    candidates: usize,
    candidate_bytes: usize,
}

const MAX_RETAINED_RECORDS: usize = 65_536;
const MAX_RETAINED_CANDIDATES: usize = 2_048;

fn bounded_sequence<'de, D, T, const LIMIT: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Sequence<T, const LIMIT: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const LIMIT: usize> serde::de::Visitor<'de> for Sequence<T, LIMIT> {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("bounded prompt state sequence")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while values.len() < LIMIT {
                match sequence.next_element()? {
                    Some(value) => values.push(value),
                    None => return Ok(values),
                }
            }
            if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom("prompt state sequence bound"));
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Sequence::<T, LIMIT>(PhantomData))
}

fn deserialize_refs<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<(usize, RecordRef)>, D::Error> {
    bounded_sequence::<D, _, MAX_RETAINED_RECORDS>(deserializer)
}

fn deserialize_candidates<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<(usize, ClaudeMessage)>, D::Error> {
    bounded_sequence::<D, _, MAX_RETAINED_CANDIDATES>(deserializer)
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            records: MAX_RETAINED_RECORDS,
            ref_bytes: 16 * 1024 * 1024,
            candidates: MAX_RETAINED_CANDIDATES,
            candidate_bytes: 32 * 1024 * 1024,
        }
    }
}

thread_local! {
    static CAPTURE: RefCell<Option<(Bounds, Option<CapturedView>)>> = const { RefCell::new(None) };
}

struct CaptureScope;

impl CaptureScope {
    fn start(bounds: Bounds) -> Self {
        CAPTURE.with_borrow_mut(|slot| {
            assert!(slot.is_none(), "nested prototype capture");
            *slot = Some((bounds, None));
        });
        Self
    }

    fn finish(self) -> Option<CapturedView> {
        let view = CAPTURE.with_borrow_mut(|slot| slot.as_mut().and_then(|(_, view)| view.take()));
        drop(self);
        view
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        CAPTURE.with_borrow_mut(|slot| *slot = None);
    }
}

struct CountBytes {
    bytes: usize,
    limit: usize,
}

impl Write for CountBytes {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buffer.len())
            .filter(|length| *length <= self.limit)
            .ok_or_else(|| io::Error::other("prototype payload bound"))?;
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedView {
    raw_count: usize,
    #[serde(deserialize_with = "deserialize_refs")]
    refs: Vec<(usize, RecordRef)>,
    #[serde(deserialize_with = "deserialize_candidates")]
    candidates: Vec<(usize, ClaudeMessage)>,
    ref_bytes: usize,
    candidate_bytes: usize,
    #[serde(skip)]
    ineligible: Option<&'static str>,
}

fn is_candidate(message: &ClaudeMessage) -> bool {
    message.message_type == "user"
        && message.subtype.as_deref() == Some(AUTHORED_USER_SUBTYPE)
        && message
            .data
            .as_ref()
            .and_then(|data| data.get("providerTurnId"))
            .and_then(Value::as_str)
            .is_some()
}

fn requires_merge(message: &ClaudeMessage) -> bool {
    message.message_type == "user"
        && message
            .content
            .as_ref()
            .and_then(Value::as_array)
            .is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            })
}

impl CapturedView {
    fn from_messages(messages: &[ClaudeMessage], bounds: Bounds) -> Self {
        let mut view = Self {
            raw_count: messages.len(),
            ..Self::default()
        };
        if messages.iter().any(requires_merge) {
            view.ineligible = Some("merge-required");
            return view;
        }
        if messages.len() > bounds.records {
            view.ineligible = Some("state-budget");
            return view;
        }
        let mut counter = CountBytes {
            bytes: 0,
            limit: bounds.candidate_bytes,
        };
        for (index, message) in messages.iter().enumerate() {
            if let Some(record) = &message.record_ref {
                let bytes = record
                    .scope
                    .len()
                    .saturating_add(record.scheme.len())
                    .saturating_add(record.id.len());
                view.ref_bytes = view.ref_bytes.saturating_add(bytes);
                if view.ref_bytes > bounds.ref_bytes {
                    view.ineligible = Some("state-budget");
                    break;
                }
                view.refs.push((index, record.clone()));
            }
            if is_candidate(message) {
                if view.candidates.len() >= bounds.candidates
                    || serde_json::to_writer(&mut counter, message).is_err()
                {
                    view.ineligible = Some("state-budget");
                    break;
                }
                view.candidates.push((index, message.clone()));
            }
        }
        view.candidate_bytes = counter.bytes;
        if view.ineligible.is_some() {
            view.refs.clear();
            view.candidates.clear();
        }
        view
    }

    fn check(&self, bounds: Bounds) -> Result<(), &'static str> {
        if let Some(reason) = self.ineligible {
            return Err(reason);
        }
        if self.raw_count > bounds.records
            || self.ref_bytes > bounds.ref_bytes
            || self.candidates.len() > bounds.candidates
            || self.candidate_bytes > bounds.candidate_bytes
        {
            return Err("state-budget");
        }
        let mut previous = None;
        for (position, _) in &self.refs {
            if *position >= self.raw_count || previous.is_some_and(|old| old >= *position) {
                return Err("coordinate-mismatch");
            }
            previous = Some(*position);
        }
        previous = None;
        for (position, message) in &self.candidates {
            if *position >= self.raw_count
                || previous.is_some_and(|old| old >= *position)
                || !is_candidate(message)
                || self
                    .refs
                    .iter()
                    .find(|(index, _)| index == position)
                    .map(|(_, record)| record)
                    != message.record_ref.as_ref()
            {
                return Err("coordinate-mismatch");
            }
            previous = Some(*position);
        }
        Ok(())
    }

    fn replace(
        &mut self,
        replace_from: usize,
        mut suffix: Self,
        bounds: Bounds,
    ) -> Result<(), &'static str> {
        self.check(bounds)?;
        suffix.check(bounds)?;
        if replace_from > self.raw_count {
            return Err("coordinate-mismatch");
        }
        self.raw_count = replace_from
            .checked_add(suffix.raw_count)
            .ok_or("state-budget")?;
        self.refs.retain(|(index, _)| *index < replace_from);
        self.candidates.retain(|(index, _)| *index < replace_from);
        for (index, _) in &mut suffix.refs {
            *index += replace_from;
        }
        for (index, _) in &mut suffix.candidates {
            *index += replace_from;
        }
        self.refs.extend(suffix.refs);
        self.candidates.extend(suffix.candidates);
        self.ref_bytes = self
            .refs
            .iter()
            .map(|(_, record)| record.scope.len() + record.scheme.len() + record.id.len())
            .sum();
        let mut counter = CountBytes {
            bytes: 0,
            limit: bounds.candidate_bytes,
        };
        for (_, message) in &self.candidates {
            serde_json::to_writer(&mut counter, message).map_err(|_| "state-budget")?;
        }
        self.candidate_bytes = counter.bytes;
        self.check(bounds)
    }

    fn select(&self, task_ids: &HashSet<String>) -> Vec<ClaudeMessage> {
        let mut exact = HashMap::<&RecordRef, usize>::new();
        for (_, record) in &self.refs {
            *exact.entry(record).or_default() += 1;
        }
        let mut valid = HashMap::<(&str, &str), usize>::new();
        for (_, record) in &self.refs {
            if exact.get(record) == Some(&1) && valid_record_ref(record) {
                *valid.entry((&record.scheme, &record.id)).or_default() += 1;
            }
        }
        self.candidates
            .iter()
            .filter(|(_, message)| {
                message
                    .data
                    .as_ref()
                    .and_then(|data| data.get("providerTurnId"))
                    .and_then(Value::as_str)
                    .is_some_and(|task| task_ids.contains(task))
            })
            .map(|(_, original)| {
                let mut message = original.clone();
                if message.record_ref.as_ref().is_some_and(|record| {
                    exact.get(record).is_some_and(|count| *count > 1)
                        || (valid_record_ref(record)
                            && valid
                                .get(&(record.scheme.as_str(), record.id.as_str()))
                                .is_some_and(|count| *count > 1))
                }) {
                    message.record_ref = None;
                }
                message
            })
            .collect()
    }
}

pub(super) fn capture_before_finalization(messages: &[ClaudeMessage]) {
    CAPTURE.with_borrow_mut(|slot| {
        if let Some((bounds, active)) = slot {
            *active = Some(CapturedView::from_messages(messages, *bounds));
        }
    });
}

fn load_captured(
    path: &Path,
    cursor: Option<&str>,
    bounds: Bounds,
) -> Result<(SessionSnapshotLoad, Option<CapturedView>), String> {
    let capture = CaptureScope::start(bounds);
    let result = load_session_snapshot(&path.to_string_lossy(), cursor)?;
    Ok((result, capture.finish()))
}

fn verify_selection(path: &Path, encoded: &str) -> Result<(), &'static str> {
    let cursor = decode_snapshot_cursor(encoded).map_err(|_| "missing-cursor")?;
    let base = PathBuf::from(get_base_path().ok_or("selection-drift")?);
    if cursor.canonical_path != path.to_string_lossy() {
        return Err("path-rotation");
    }
    let paths = cursor
        .lineage
        .as_ref()
        .map(|segments| {
            segments
                .iter()
                .map(|segment| Path::new(&segment.canonical_path))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![path]);
    for selected in paths {
        let identity = rollout_file_identity(selected).ok_or("selection-drift")?;
        let fresh =
            find_rollout_path_by_id(&base, &identity.rollout_id).map_err(|_| "selection-drift")?;
        if fresh != selected {
            return Err("selection-drift");
        }
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Retained {
    path: PathBuf,
    cursor: String,
    epoch: u32,
    view: CapturedView,
}

struct Engine {
    retained: Option<Retained>,
    epoch: u32,
    bounds: Bounds,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            retained: None,
            epoch: SNAPSHOT_CURSOR_VERSION,
            bounds: Bounds::default(),
        }
    }
}

struct Acquisition {
    candidates: Vec<ClaudeMessage>,
    mode: &'static str,
    reason: &'static str,
}

impl Engine {
    #[cfg(test)]
    fn accepted_logical_bytes(&self) -> Option<u64> {
        let cursor = decode_snapshot_cursor(&self.retained.as_ref()?.cursor).ok()?;
        cursor.lineage.map_or(Some(cursor.accepted_len), |lineage| {
            lineage.iter().try_fold(0u64, |total, segment| {
                total.checked_add(segment.accepted_len)
            })
        })
    }

    #[cfg(test)]
    fn retained_counts(&self) -> (usize, usize, usize, usize) {
        self.retained.as_ref().map_or((0, 0, 0, 0), |retained| {
            (
                retained.view.raw_count,
                retained.view.refs.len(),
                retained.view.candidates.len(),
                retained.cursor.len()
                    + retained.view.ref_bytes
                    + retained.view.candidate_bytes
                    + retained.view.refs.len() * std::mem::size_of::<(usize, RecordRef)>(),
            )
        })
    }

    fn view_for(
        capture: Option<CapturedView>,
        returned_count: usize,
        cursor: &str,
        bounds: Bounds,
    ) -> Result<CapturedView, &'static str> {
        let view = capture.ok_or("capture-missing")?;
        view.check(bounds)?;
        if view.raw_count != returned_count
            || snapshot_cursor_replace_from(cursor).map_err(|_| "missing-cursor")? > view.raw_count
        {
            return Err("coordinate-mismatch");
        }
        Ok(view)
    }

    fn apply_replace(
        retained: &mut Retained,
        replace_from: usize,
        messages: Vec<ClaudeMessage>,
        capture: Option<CapturedView>,
        cursor: String,
        bounds: Bounds,
    ) -> Result<(), &'static str> {
        if snapshot_cursor_replace_from(&retained.cursor).map_err(|_| "missing-cursor")?
            != replace_from
        {
            return Err("coordinate-mismatch");
        }
        let suffix = capture.ok_or("capture-missing")?;
        if suffix.raw_count != messages.len() {
            return Err("coordinate-mismatch");
        }
        drop(messages);
        retained.view.replace(replace_from, suffix, bounds)?;
        if snapshot_cursor_replace_from(&cursor).map_err(|_| "missing-cursor")?
            > retained.view.raw_count
        {
            return Err("coordinate-mismatch");
        }
        retained.cursor = cursor;
        Ok(())
    }

    fn full(
        &mut self,
        path: &Path,
        task_ids: &HashSet<String>,
        reason: &'static str,
    ) -> Result<Acquisition, String> {
        self.retained = None;
        let (result, capture) = load_captured(path, None, self.bounds)?;
        self.finish_full(path, task_ids, reason, result, capture)
    }

    fn finish_full(
        &mut self,
        path: &Path,
        task_ids: &HashSet<String>,
        reason: &'static str,
        loaded: SessionSnapshotLoad,
        capture: Option<CapturedView>,
    ) -> Result<Acquisition, String> {
        let result = loaded;
        let SessionSnapshotLoad::Full {
            messages, cursor, ..
        } = result
        else {
            return Err("unexpected prototype full result".to_string());
        };
        let returned_count = messages.len();
        let candidates = select_candidates(messages, "codex", task_ids);
        let mut result = Acquisition {
            candidates,
            mode: "full",
            reason,
        };
        let Some(cursor) = cursor else {
            result.reason = "missing-cursor";
            return Ok(result);
        };
        let view = match Self::view_for(capture, returned_count, &cursor, self.bounds) {
            Ok(view) => view,
            Err(reason) => {
                result.reason = reason;
                return Ok(result);
            }
        };
        let mut retained = Retained {
            path: path.to_path_buf(),
            cursor,
            epoch: self.epoch,
            view,
        };
        if let Err(reason) = verify_selection(path, &retained.cursor) {
            return self.terminal_full(path, task_ids, reason);
        }
        let (confirmation, capture) = load_captured(path, Some(&retained.cursor), self.bounds)?;
        match confirmation {
            SessionSnapshotLoad::Unchanged { cursor } => retained.cursor = cursor,
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor,
                ..
            } => {
                if let Err(reason) = Self::apply_replace(
                    &mut retained,
                    replace_from,
                    messages,
                    capture,
                    cursor,
                    self.bounds,
                ) {
                    return self.terminal_full(path, task_ids, reason);
                }
                result.candidates = retained.view.select(task_ids);
            }
            SessionSnapshotLoad::Full { messages, .. } => {
                result.candidates = select_candidates(messages, "codex", task_ids);
                result.reason = "seed-unconfirmed";
                return Ok(result);
            }
        }
        if let Err(reason) = verify_selection(path, &retained.cursor) {
            return self.terminal_full(path, task_ids, reason);
        }
        self.retained = Some(retained);
        Ok(result)
    }

    fn terminal_full(
        &mut self,
        path: &Path,
        task_ids: &HashSet<String>,
        reason: &'static str,
    ) -> Result<Acquisition, String> {
        self.retained = None;
        let SessionSnapshotLoad::Full { messages, .. } =
            load_session_snapshot(&path.to_string_lossy(), None)?
        else {
            return Err("unexpected prototype terminal full result".to_string());
        };
        Ok(Acquisition {
            candidates: select_candidates(messages, "codex", task_ids),
            mode: "full",
            reason,
        })
    }

    fn acquire(&mut self, path: &Path, task_ids: &HashSet<String>) -> Result<Acquisition, String> {
        let path = validate_session_path(path, &path.to_string_lossy())?;
        let Some(mut retained) = self.retained.take() else {
            return self.full(&path, task_ids, "initial");
        };
        if retained.epoch != self.epoch {
            return self.full(&path, task_ids, "parser-drift");
        }
        if retained.path != path {
            return self.full(&path, task_ids, "path-rotation");
        }
        if let Err(reason) = retained
            .view
            .check(self.bounds)
            .and_then(|()| verify_selection(&path, &retained.cursor))
        {
            return self.full(&path, task_ids, reason);
        }
        let (loaded, capture) = load_captured(&path, Some(&retained.cursor), self.bounds)?;
        let mode = match loaded {
            SessionSnapshotLoad::Unchanged { cursor } => {
                retained.cursor = cursor;
                "unchanged"
            }
            SessionSnapshotLoad::Replace {
                replace_from,
                messages,
                cursor,
                ..
            } => {
                if let Err(reason) = Self::apply_replace(
                    &mut retained,
                    replace_from,
                    messages,
                    capture,
                    cursor,
                    self.bounds,
                ) {
                    return self.full(&path, task_ids, reason);
                }
                "replace"
            }
            full @ SessionSnapshotLoad::Full { .. } => {
                return self.finish_full(&path, task_ids, "provider-full", full, capture);
            }
        };
        if let Err(reason) = verify_selection(&path, &retained.cursor) {
            return self.full(&path, task_ids, reason);
        }
        let candidates = retained.view.select(task_ids);
        self.retained = Some(retained);
        Ok(Acquisition {
            candidates,
            mode,
            reason: if mode == "replace" {
                "append"
            } else {
                "unchanged"
            },
        })
    }
}

fn select_codex_candidates(
    messages: Vec<ClaudeMessage>,
    task_ids: &HashSet<String>,
) -> Vec<ClaudeMessage> {
    select_candidates(messages, "codex", task_ids)
}

fn full_control(path: &Path, task_ids: &HashSet<String>) -> Result<Vec<ClaudeMessage>, String> {
    Ok(select_codex_candidates(
        finalize_loaded_messages(load_messages(&path.to_string_lossy())?),
        task_ids,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::parse_live_rollout;
    use super::*;
    use serde_json::json;
    use std::io::Write;

    pub(super) const SESSION: &str = "019f5acd-6b03-7653-ae71-6321002ca8d4";
    pub(super) const TASK1: &str = "01a18a3e-1c77-7423-9393-9e7424dbf380";
    pub(super) const TASK2: &str = "01a18a3e-1c77-7423-9393-9e7424dbf381";

    #[test]
    fn prompt_state_cache_sequence_decode_stops_at_the_bound() {
        let mut accepted = serde_json::Deserializer::from_str("[1,2]");
        assert_eq!(
            bounded_sequence::<_, u8, 2>(&mut accepted).unwrap(),
            vec![1, 2]
        );
        let mut excessive = serde_json::Deserializer::from_str("[1,2,3]");
        assert!(bounded_sequence::<_, u8, 2>(&mut excessive).is_err());
    }

    pub(super) fn rows(task: &str, id: &str) -> Vec<Value> {
        vec![
            json!({"timestamp":"2026-10-19T12:30:48Z","type":"event_msg","payload":{"type":"task_started","turn_id":task}}),
            json!({"timestamp":"2026-10-19T12:30:49Z","type":"event_msg","payload":{"type":"item_completed","turn_id":task,"item":{"type":"UserMessage","id":id,"content":[{"type":"text","text":"ordinary authored prompt"}]}}}),
        ]
    }

    pub(super) fn append(path: &Path, rows: &[Value]) {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        for row in rows {
            writeln!(file, "{row}").unwrap();
        }
    }

    pub(super) fn fixture() -> (crate::profile_paths::TestProfile, PathBuf) {
        let profile = crate::profile_paths::TestProfile::new().unwrap();
        let root = profile.path().join(".codex/sessions");
        fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("rollout-2026-10-19T12-30-47-{SESSION}.jsonl"));
        append(
            &path,
            &[
                json!({"timestamp":"2026-10-19T12:30:47Z","type":"session_meta","payload":{"id":SESSION,"cwd":"C:/repo","source":"cli"}}),
            ],
        );
        append(&path, &rows(TASK1, "native-1"));
        append(
            &path,
            &[
                json!({"timestamp":"2026-10-19T12:30:50Z","type":"event_msg","payload":{"type":"task_complete","turn_id":TASK1}}),
            ],
        );
        (profile, path)
    }

    pub(super) fn ids(task: &str) -> HashSet<String> {
        HashSet::from([task.to_string()])
    }

    pub(super) fn values(messages: &[ClaudeMessage]) -> Value {
        serde_json::to_value(messages).unwrap()
    }

    #[test]
    fn codex_prompt_prototype_first_new_prompt_replays_with_full_parity() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        assert!(engine
            .acquire(&path, &ids(TASK2))
            .unwrap()
            .candidates
            .is_empty());
        append(&path, &rows(TASK2, "native-2"));
        let acquired = engine.acquire(&path, &ids(TASK2)).unwrap();
        assert_eq!(acquired.mode, "replace");
        assert_eq!(acquired.candidates.len(), 1);
        assert_eq!(
            values(&acquired.candidates),
            values(&full_control(&path, &ids(TASK2)).unwrap())
        );
    }

    #[test]
    fn codex_prompt_prototype_ledger_captures_original_refs_and_omitted_records() {
        let (_profile, path) = fixture();
        let messages = parse_live_rollout(&path).unwrap();
        let scope = CaptureScope::start(Bounds::default());
        capture_before_finalization(&messages);
        let captured = scope.finish().unwrap();
        assert_eq!(captured.raw_count, messages.len());
        assert_eq!(
            captured.refs.len(),
            messages
                .iter()
                .filter(|message| message.record_ref.is_some())
                .count()
        );
        assert_eq!(captured.candidates.len(), 1);
        assert!(captured.ineligible.is_none());
        assert!(captured
            .refs
            .iter()
            .all(|(_, record)| valid_record_ref(record)));
    }

    fn assert_full_parity(
        engine: &mut Engine,
        path: &Path,
        tasks: &HashSet<String>,
    ) -> Acquisition {
        let acquired = engine.acquire(path, tasks).unwrap();
        assert_eq!(
            values(&acquired.candidates),
            values(&full_control(path, tasks).unwrap())
        );
        acquired
    }

    #[test]
    fn codex_prompt_prototype_all_record_collisions_are_reversible() {
        let (_profile, path) = fixture();
        let original = parse_live_rollout(&path).unwrap();
        let mut omitted = original
            .iter()
            .find(|message| is_candidate(message))
            .unwrap()
            .clone();
        omitted.message_type = "assistant".to_string();
        omitted.subtype = None;
        let mut colliding = original.clone();
        colliding.push(omitted);
        let mut view = CapturedView::from_messages(&colliding, Bounds::default());
        assert!(view.select(&ids(TASK1))[0].record_ref.is_none());
        assert_eq!(
            values(&view.select(&ids(TASK1))),
            values(&select_codex_candidates(
                finalize_loaded_messages(colliding),
                &ids(TASK1)
            ))
        );
        view.replace(original.len(), CapturedView::default(), Bounds::default())
            .unwrap();
        assert!(view.select(&ids(TASK1))[0].record_ref.is_some());
        assert_eq!(
            values(&view.select(&ids(TASK1))),
            values(&select_codex_candidates(
                finalize_loaded_messages(original),
                &ids(TASK1)
            ))
        );
    }

    #[test]
    fn codex_prompt_prototype_keeps_unref_and_duplicate_task_candidates() {
        let (_profile, path) = fixture();
        let mut messages = parse_live_rollout(&path).unwrap();
        let mut duplicate = messages
            .iter()
            .find(|message| is_candidate(message))
            .unwrap()
            .clone();
        duplicate.uuid = "unreferenced".to_string();
        duplicate.record_ref = None;
        messages.push(duplicate);
        let view = CapturedView::from_messages(&messages, Bounds::default());
        assert_eq!(view.select(&ids(TASK1)).len(), 2);
        assert_eq!(
            values(&view.select(&ids(TASK1))),
            values(&select_codex_candidates(
                finalize_loaded_messages(messages),
                &ids(TASK1)
            ))
        );
    }

    #[test]
    fn codex_prompt_prototype_exact_and_consumer_ref_validation_match_full() {
        let (_profile, path) = fixture();
        let original = parse_live_rollout(&path).unwrap();
        for reference in [
            RecordRef::session("codex-response-item", "😀".repeat(256)),
            RecordRef::session("codex-response-item", "\u{feff}"),
            RecordRef::session("codex-response-item", "\u{0085}"),
            RecordRef {
                version: 2,
                ..RecordRef::session("codex-response-item", "native-1")
            },
        ] {
            let mut messages = original.clone();
            let candidate = messages
                .iter_mut()
                .find(|message| is_candidate(message))
                .unwrap();
            candidate.record_ref = Some(reference);
            let mut other = candidate.clone();
            other.message_type = "assistant".to_string();
            messages.push(other);
            let view = CapturedView::from_messages(&messages, Bounds::default());
            assert_eq!(
                values(&view.select(&ids(TASK1))),
                values(&select_codex_candidates(
                    finalize_loaded_messages(messages),
                    &ids(TASK1)
                ))
            );
        }
    }

    #[test]
    fn codex_prompt_prototype_rewrite_shrink_epoch_and_rotation_rebuild() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK1));
        assert!(engine.retained.is_some());
        let original = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            original.replace("ordinary authored prompt", "rewritten authored input"),
        )
        .unwrap();
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK1)).mode,
            "full"
        );
        fs::write(&path, &original).unwrap();
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK1)).mode,
            "full"
        );
        engine.epoch += 1;
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK1)).reason,
            "parser-drift"
        );
        let rotated = path.parent().unwrap().join(format!(
            "rollout-2026-10-19T12-30-47-{SESSION}_{TASK2}.jsonl"
        ));
        fs::rename(&path, &rotated).unwrap();
        assert_eq!(
            assert_full_parity(&mut engine, &rotated, &ids(TASK1)).reason,
            "path-rotation"
        );
    }

    #[test]
    fn codex_prompt_prototype_bounds_fallback_preserves_available_content() {
        let (_profile, path) = fixture();
        for bounds in [
            Bounds {
                records: 0,
                ..Bounds::default()
            },
            Bounds {
                ref_bytes: 0,
                ..Bounds::default()
            },
            Bounds {
                candidates: 0,
                ..Bounds::default()
            },
            Bounds {
                candidate_bytes: 1,
                ..Bounds::default()
            },
        ] {
            let mut engine = Engine {
                bounds,
                ..Engine::default()
            };
            let result = assert_full_parity(&mut engine, &path, &ids(TASK1));
            assert_eq!(result.reason, "state-budget");
            assert_eq!(result.candidates.len(), 1);
            assert!(engine.retained.is_none());
        }
    }

    #[test]
    fn codex_prompt_prototype_missing_capture_and_coordinates_reject_retention() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK1));
        let state = engine.retained.as_mut().unwrap();
        assert!(matches!(
            Engine::view_for(None, 1, &state.cursor, Bounds::default()),
            Err("capture-missing")
        ));
        state.view.raw_count = 0;
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK1)).reason,
            "coordinate-mismatch"
        );
    }

    #[test]
    fn codex_prompt_prototype_merge_gate_rejects_even_count_preserving_mutation() {
        let (_profile, path) = fixture();
        let mut messages = parse_live_rollout(&path).unwrap();
        let index = messages.iter().position(is_candidate).unwrap();
        let mut tool = messages[index].clone();
        tool.message_type = "assistant".to_string();
        tool.content =
            Some(json!([{"type":"tool_use","id":"matching-tool","name":"shell","input":{}}]));
        messages.insert(index, tool);
        messages[index + 1].content = Some(
            json!([{"type":"text","text":"kept"},{"type":"tool_result","tool_use_id":"matching-tool","content":"result"}]),
        );
        let finalized = finalize_loaded_messages(messages.clone());
        assert_eq!(finalized.len(), messages.len());
        assert_ne!(values(&finalized), values(&messages));
        assert_eq!(
            CapturedView::from_messages(&messages, Bounds::default()).ineligible,
            Some("merge-required")
        );
    }

    #[test]
    fn codex_prompt_prototype_final_capture_replaces_failed_delta_attempt() {
        let (_profile, path) = fixture();
        let messages = parse_live_rollout(&path).unwrap();
        let scope = CaptureScope::start(Bounds::default());
        capture_before_finalization(&[]);
        capture_before_finalization(&messages);
        assert_eq!(scope.finish().unwrap().raw_count, messages.len());
    }

    #[test]
    fn codex_prompt_prototype_new_async_reply_uses_native_full_fallback() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK2));
        let mut reply = rows(TASK2, "async-reply");
        reply[1]["payload"]["item"]["content"][0]["text"] =
            json!("<send_user_message_question_reply>\n[]\n</send_user_message_question_reply>");
        append(&path, &reply);
        let result = assert_full_parity(&mut engine, &path, &ids(TASK2));
        assert_eq!(result.mode, "full");
        assert_eq!(result.reason, "provider-full");
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK2)).mode,
            "unchanged"
        );
    }

    #[test]
    fn codex_prompt_prototype_selected_duplicate_carrier_drops_retention() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK1));
        let duplicate = path
            .parent()
            .unwrap()
            .join(format!("rollout-2026-10-20T12-30-47-{SESSION}.jsonl"));
        fs::copy(&path, duplicate).unwrap();
        let result = assert_full_parity(&mut engine, &path, &ids(TASK1));
        assert_eq!(result.reason, "selection-drift");
        assert!(engine.retained.is_none());
    }

    #[test]
    fn codex_prompt_prototype_multiple_new_selectors_and_live_suffix_parity() {
        let (_profile, path) = fixture();
        let mut engine = Engine::default();
        let tasks = HashSet::from([TASK1.to_string(), TASK2.to_string()]);
        assert_full_parity(&mut engine, &path, &tasks);
        append(&path, &rows(TASK2, "native-2"));
        assert_eq!(
            assert_full_parity(&mut engine, &path, &tasks).mode,
            "replace"
        );
        assert_eq!(
            assert_full_parity(&mut engine, &path, &tasks).mode,
            "unchanged"
        );
        append(
            &path,
            &[
                json!({"type":"response_item","timestamp":"2026-10-19T12:30:51Z","payload":{"id":"later-assistant","type":"message","role":"assistant","content":[{"type":"output_text","text":"later output"}]}}),
            ],
        );
        assert_eq!(
            assert_full_parity(&mut engine, &path, &tasks).mode,
            "replace"
        );
    }
    #[test]
    fn codex_prompt_prototype_seed_is_provisional_until_prefix_confirmation() {
        let (_profile, path) = fixture();
        let path = path.canonicalize().unwrap();
        let (loaded, capture) = load_captured(&path, None, Bounds::default()).unwrap();
        let original = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            original.replace("ordinary authored prompt", "changed during seed read"),
        )
        .unwrap();
        let mut engine = Engine::default();
        let result = engine
            .finish_full(&path, &ids(TASK1), "initial", loaded, capture)
            .unwrap();
        assert_eq!(result.reason, "seed-unconfirmed");
        assert!(engine.retained.is_none());
        assert_eq!(
            values(&result.candidates),
            values(&full_control(&path, &ids(TASK1)).unwrap())
        );
    }

    #[test]
    fn codex_prompt_prototype_collision_rewrite_restores_prefix_identity() {
        let (_profile, path) = fixture();
        let original = fs::read(&path).unwrap();
        let mut engine = Engine::default();
        let tasks = HashSet::from([TASK1.to_string(), TASK2.to_string()]);
        assert_full_parity(&mut engine, &path, &tasks);
        append(&path, &rows(TASK2, "native-1"));
        let collision = assert_full_parity(&mut engine, &path, &tasks);
        assert_eq!(collision.mode, "replace");
        assert_eq!(collision.candidates.len(), 2);
        assert!(collision
            .candidates
            .iter()
            .all(|message| message.record_ref.is_none()));
        fs::write(&path, original).unwrap();
        append(&path, &rows(TASK2, "native-2"));
        let restored = assert_full_parity(&mut engine, &path, &tasks);
        assert_eq!(restored.mode, "full");
        assert!(restored
            .candidates
            .iter()
            .all(|message| message.record_ref.is_some()));
    }

    #[test]
    fn codex_prompt_prototype_unsafe_tool_backreference_rebuilds() {
        let (_profile, path) = fixture();
        let body = fs::read_to_string(&path).unwrap();
        let mut lines = body
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        lines.insert(lines.len() - 1, json!({"type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"old-call","arguments":"{}"}}));
        fs::write(&path, "").unwrap();
        append(&path, &lines);
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK1));
        append(
            &path,
            &[
                json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"old-call","output":"late result"}}),
            ],
        );
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(TASK1)).mode,
            "full"
        );
    }

    pub(super) fn paginated_fixture(
    ) -> (crate::profile_paths::TestProfile, PathBuf, PathBuf, Vec<u8>) {
        let (profile, parent) = fixture();
        let body = fs::read_to_string(&parent).unwrap();
        let lines = body
            .lines()
            .enumerate()
            .map(|(index, line)| {
                let mut value: Value = serde_json::from_str(line).unwrap();
                value["ordinal"] = json!(index);
                if index == 0 {
                    value["payload"]["history_mode"] = json!("paginated");
                }
                value
            })
            .collect::<Vec<_>>();
        fs::write(&parent, "").unwrap();
        append(&parent, &lines);
        let parent_bytes = fs::read(&parent).unwrap();
        let child = parent.parent().unwrap().join(format!(
            "rollout-2026-10-20T12-30-47-{SESSION}_{TASK2}.jsonl"
        ));
        append(
            &child,
            &[
                json!({"timestamp":"2026-10-19T12:30:47Z","ordinal":4,"type":"session_meta","payload":{"id":SESSION,"cwd":"C:/repo","source":"cli","history_mode":"paginated","history_base":{"thread_id":SESSION,"end_ordinal_exclusive":4,"end_byte_offset":parent_bytes.len()}}}),
            ],
        );
        let mut current = rows(TASK2, "current-page");
        current
            .push(json!({"type":"event_msg","payload":{"type":"task_complete","turn_id":TASK2}}));
        for (index, row) in current.iter_mut().enumerate() {
            row["ordinal"] = json!(index + 5);
        }
        append(&child, &current);
        let compressed = parent.with_extension("jsonl.zst");
        fs::write(
            &compressed,
            zstd::stream::encode_all(parent_bytes.as_slice(), 1).unwrap(),
        )
        .unwrap();
        fs::remove_file(&parent).unwrap();
        (profile, parent, child, parent_bytes)
    }

    #[test]
    fn codex_prompt_prototype_paginated_compressed_ancestor_append_and_twin_drift() {
        let (_profile, parent, path, parent_bytes) = paginated_fixture();
        let mut engine = Engine::default();
        let future = "01a18a3e-1c77-7423-9393-9e7424dbf382";
        assert_full_parity(&mut engine, &path, &ids(future));
        assert!(engine.retained.is_some());
        let mut new = rows(future, "new-page-user");
        for (index, row) in new.iter_mut().enumerate() {
            row["ordinal"] = json!(index + 8);
        }
        append(&path, &new);
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(future)).mode,
            "replace"
        );
        fs::write(&parent, parent_bytes).unwrap();
        assert_eq!(
            assert_full_parity(&mut engine, &path, &ids(future)).reason,
            "selection-drift"
        );
    }

    #[test]
    fn codex_prompt_prototype_new_ancestor_ambiguity_matches_full_refusal() {
        let (_profile, parent, path, _) = paginated_fixture();
        let mut engine = Engine::default();
        assert_full_parity(&mut engine, &path, &ids(TASK2));
        let duplicate = parent
            .parent()
            .unwrap()
            .join(format!("rollout-2026-10-21T12-30-47-{SESSION}.jsonl.zst"));
        fs::copy(parent.with_extension("jsonl.zst"), duplicate).unwrap();
        assert!(engine.acquire(&path, &ids(TASK2)).is_err());
        assert!(full_control(&path, &ids(TASK2)).is_err());
        assert!(engine.retained.is_none());
    }
}
