"""Copy-on-write edits: remove_records validate-and-cascade, and the edit_version round trip."""

import duckdb
import pytest

from timenet.dataset.edit import edit_version, remove_records
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, InputModality, TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def test_remove_records_preserves_dataset_annotations_and_their_refs():
    # A dataset annotation and a task reference to its occurrence survive an unrelated record removal.
    dataset = make_dataset()
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"], id="opts-shared"))
    answer = next(task for task in dataset.tasks if task.id == "task-answer-0")
    answer.input_annotations = (*answer.input_annotations, options)

    edited = remove_records(dataset, ["record-2"], cascade=True)  # record-2's own task cascades out

    assert [ann.id for ann in edited.annotations] == ["opts-shared"]
    kept = next(task for task in edited.tasks if task.id == "task-answer-0")
    assert options in kept.input_annotations


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
    answer.inputs = (dataset.records[0], dataset.records[1])
    dataset.records[1].task_ids = (*dataset.records[1].task_ids, "task-answer-0")
    record0 = dataset.records[0]
    record0.annotations = tuple(a for a in record0.annotations if a.id != "cohort-shared")
    answer.input_annotations = tuple(a for a in dataset.records[1].annotations if a.id == "cohort-shared")
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
    assert rebuilt.inputs == (dataset.records[0],)
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The mirror case: cohort-shared stays on record-0, so removing record-1 leaves the ref intact.
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, ["record-1"])
    rebuilt = next(t for t in edited.tasks if t.id == "task-answer-0")
    assert tuple(annotation.id for annotation in rebuilt.input_annotations) == ("cohort-shared",)


def test_target_annotation_refs_are_required():
    # task-localize-0 stores its answer as annotation occurrences on record-0.
    # Widening it to record-2 means dropping record-0 no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(t for t in dataset.tasks if t.id == "task-localize-0")
    localize.inputs = (dataset.records[0], dataset.records[2])
    localize.targets = None
    localize.target_annotations = tuple(
        annotation for annotation in dataset.records[0].annotations if annotation.id in {"stim-0", "art-0"}
    )
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

    by_record = {
        record.record_id: {annotation.occurrence_id for annotation in record.annotations} for record in restored.records
    }
    for task in restored.tasks:
        reachable = set().union(*(by_record[record.id] for record in task.inputs))
        named = (*task.input_annotations, *task.target_annotations)
        assert not [annotation for annotation in named if annotation.occurrence_id not in reachable]


def test_target_and_candidate_records_are_required_whatever_the_task_type():
    dataset = make_dataset()
    edited_record = dataset.records[1]
    dataset.add_task(
        task=TSEditingTask(
            input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
            inputs=(dataset.records[0],),
            prompt="Denoise it.",
            targets=(edited_record,),
            id="task-edit-0",
        )
    )
    dataset.add_task(
        task=TSCorrespondenceTask(
            input_modalities=frozenset({InputModality.TEXT, InputModality.TIME_SERIES}),
            inputs=(dataset.records[0],),
            prompt="Which trace matches?",
            candidate_records=(dataset.records[1], dataset.records[2]),
            targets=(dataset.records[2],),
            id="task-corr-0",
        )
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

    with duckdb.connect(str(out / "control.duckdb"), read_only=True) as connection:
        row = connection.execute("SELECT count(*) FROM records WHERE record_id = 'record-1'").fetchone()
        assert row is not None
        assert row[0] == 0


def test_edit_version_same_version_rejected(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_record_ids=["record-1"])
