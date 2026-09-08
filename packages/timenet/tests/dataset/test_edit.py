"""Copy-on-write edits: remove_records validate-and-cascade, and the edit_version round trip."""

import pyarrow.parquet as pq
import pytest

from timenet.dataset.edit import edit_version, remove_records
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def test_remove_records_preserves_registered_annotations_and_their_refs():
    # A registered annotation no record carries, and the task refs to it, must survive a record removal.
    dataset = make_dataset()
    dataset.register_annotations([Annotation(key="answer_options", value=["yes", "no"], id="opts-shared")])
    answer = next(task for task in dataset.tasks if task.id == "task-answer-0")
    answer.input_annotation_ids = (*answer.input_annotation_ids, "opts-shared")

    edited = remove_records(dataset, ["record-2"], cascade=True)  # record-2's own task cascades out

    assert [ann.id for ann in edited.registered_annotations] == ["opts-shared"]  # not dropped by from_parts
    kept = next(task for task in edited.tasks if task.id == "task-answer-0")
    assert "opts-shared" in kept.input_annotation_ids  # registered = always reachable, so the ref is not stripped


def _base(tmp_path, dataset=None):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version)


def _answer_reads_context_via_record1():
    """make_dataset() rewired so task-answer-0 spans record-0/1 but reaches cohort-shared only via record-1.

    Still a valid dataset — the annotation sits on a record the task is attached to — so it writes and
    reads back fine. Removing record-1 is what strands the reference.
    """
    dataset = make_dataset()
    answer = next(t for t in dataset.tasks if t.id == "task-answer-0")
    answer.record_ids = ("record-0", "record-1")
    dataset.records[1].task_ids = (*dataset.records[1].task_ids, "task-answer-0")
    record0 = dataset.records[0]
    record0.annotations = tuple(a for a in record0.annotations if a.id != "cohort-shared")
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, ["record-1"])
    assert {s.record_id for s in edited.records} == {"record-0", "record-2"}
    # tasks untouched (record-1 had none)
    assert {t.id for t in edited.tasks} == {
        "task-cls-0",
        "task-answer-0",
        "task-scalar-0",
        "task-localize-0",
        "task-cls-2",
    }


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, ["nope"])


def test_remove_record_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        remove_records(dataset, ["record-0"])  # every task on record-0 would dangle


def test_remove_record_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, ["record-0"], cascade=True)
    assert {s.record_id for s in edited.records} == {"record-1", "record-2"}
    # every task on record-0 goes, and task-answer-0 (which derives from task-cls-0) cascades out
    assert {t.id for t in edited.tasks} == {"task-cls-2"}


def test_shared_annotation_survives_on_remaining_record():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, ["record-0"], cascade=True)
    record1 = next(s for s in edited.records if s.record_id == "record-1")
    assert any(a.id == "cohort-shared" for a in record1.annotations)


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping record-1 strands cohort-shared for task-answer-0. It is context rather than the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_record1()
    dataset.derive_schema()

    edited = remove_records(dataset, ["record-1"])
    rebuilt = next(t for t in edited.tasks if t.id == "task-answer-0")
    assert rebuilt.record_ids == ("record-0",)
    assert rebuilt.input_annotation_ids == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The mirror case: cohort-shared stays on record-0, so removing record-1 leaves the ref intact.
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, ["record-1"])
    rebuilt = next(t for t in edited.tasks if t.id == "task-answer-0")
    assert rebuilt.input_annotation_ids == ("cohort-shared",)


def test_target_annotation_refs_are_required():
    # task-localize-0 stores its answer as target_annotation_ids pointing at stim-0/art-0 on record-0.
    # Widening it to record-2 means dropping record-0 no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(t for t in dataset.tasks if t.id == "task-localize-0")
    localize.target = None
    localize.target_annotation_ids = ("stim-0", "art-0")
    localize.record_ids = ("record-0", "record-2")
    dataset.records[2].task_ids = (*dataset.records[2].task_ids, "task-localize-0")
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match="task-localize-0"):
        remove_records(dataset, ["record-0"])

    edited = remove_records(dataset, ["record-0"], cascade=True)
    assert "task-localize-0" not in {t.id for t in edited.tasks}


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the records that task is attached to. Removing record-1 strands cohort-shared for
    # task-answer-0, which previously wrote the dangling reference straight into the new version.
    base = _base(tmp_path, _answer_reads_context_via_record1())
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=["record-1"])
    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()

    by_record = {s.record_id: {a.id for a in s.annotations} for s in restored.records}
    for task in restored.tasks:
        reachable = set().union(*(by_record[sid] for sid in task.record_ids))
        named = (*task.input_annotation_ids, *task.target_annotation_ids)
        assert not [aid for aid in named if aid not in reachable], f"{task.id} dangles"


def test_payload_record_refs_are_required_whatever_the_task_type():
    # The editor reads TaskRefs rather than special-casing forecasting, so an edit task's source and a
    # correspondence task's candidate pool are protected the same way.
    dataset = make_dataset()
    edited_record = dataset.records[1]
    dataset.add_task(
        dataset.records[0],
        TSEditingTask(
            prompt="Denoise it.",
            source_record_id="record-0",
            target_record_id=edited_record.record_id,
            id="task-edit-0",
        ),
    )
    dataset.add_task(
        dataset.records[0],
        TSCorrespondenceTask(
            prompt="Which trace matches?",
            candidate_record_ids=("record-1", "record-2"),
            target=("record-2",),
            id="task-corr-0",
        ),
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="task-edit-0"):
        remove_records(dataset, ["record-1"])  # the edit's produced record
    with pytest.raises(TimeFEditError, match="task-corr-0"):
        remove_records(dataset, ["record-2"])  # a candidate the correspondence answer names


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    base = _base(tmp_path)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=["record-1"])

    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()
    assert {s.record_id for s in restored.records} == {"record-0", "record-2"}

    manifest = Manifest.from_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    index = pq.read_table(out / "time_series_index/part-00000000.parquet").to_pylist()
    assert all(row["record_id"] != "record-1" for row in index)  # no orphaned index rows


def test_edit_version_same_version_rejected(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_record_ids=["record-1"])
