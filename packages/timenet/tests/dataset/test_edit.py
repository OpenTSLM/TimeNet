"""Copy-on-write edits: remove_records validate-and-cascade, and the edit_version round trip."""

import duckdb
import pytest

from timenet.dataset.edit import edit_version, remove_records
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    TemporalLocalizationTask,
    TSCorrespondenceTask,
    TSEditingTask,
    Version,
)
from timenet.writer import TimeFWriter


def test_remove_records_preserves_dataset_annotations_and_their_refs():
    # A dataset annotation and a task reference to its occurrence survive an unrelated record removal.
    dataset = make_dataset()
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"], id=401))
    answer = _answer(dataset)
    answer.input_annotations = (*answer.input_annotations, options)

    # record2's own task cascades out
    edited = remove_records(dataset, [dataset.records[2].id], cascade=True)

    assert [ann.id for ann in edited.annotations] == [401]
    kept = next(task for task in edited.tasks if task.id == answer.id)
    assert options in kept.input_annotations


def _answer(dataset):
    return next(task for task in dataset.tasks if isinstance(task, AnswerTask))


def _cohort_id(dataset):
    return next(a.id for a in dataset.records[0].annotations if a.key == "cohort")


def _ids(objects):
    return {obj.id for obj in objects}


def _base_ids(tmp_path, dataset=None):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        ids = writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version), ids


def _base(tmp_path, dataset=None):
    return _base_ids(tmp_path, dataset)[0]


def _answer_reads_context_via_record1():
    """make_dataset() rewired so the answer task spans records 0/1 but reaches cohort only via record 1.

    Still a valid dataset — the annotation sits on a record the task is attached to — so it writes and
    reads back fine. Removing record 1 is what strands the reference.
    """
    dataset = make_dataset()
    answer = _answer(dataset)
    cohort_id = _cohort_id(dataset)
    answer.inputs = (dataset.records[0], dataset.records[1])
    dataset.records[1].task_ids = (*dataset.records[1].task_ids, answer.id)
    record0 = dataset.records[0]
    record0.annotations = tuple(a for a in record0.annotations if a.id != cohort_id)
    answer.input_annotations = tuple(a for a in dataset.records[1].annotations if a.id == cohort_id)
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    record0, record1, record2 = dataset.records
    edited = remove_records(dataset, [record1.id])
    assert _ids(edited.records) == {record0.id, record2.id}
    # tasks untouched (record1 had none)
    assert len(dataset.tasks) == 5
    assert _ids(edited.tasks) == _ids(dataset.tasks)


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, [999])


def test_remove_record_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        # every task on record0 would dangle
        remove_records(dataset, [dataset.records[0].id])


def test_remove_record_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    record0, record1, record2 = dataset.records
    edited = remove_records(dataset, [record0.id], cascade=True)
    assert _ids(edited.records) == {record1.id, record2.id}
    # every task on record0 goes, and the answer task (which derives from record0's classification)
    # cascades out
    assert _ids(edited.tasks) == set(record2.task_ids)
    assert len(record2.task_ids) == 1


def test_shared_annotation_survives_on_remaining_record():
    dataset = make_dataset()
    dataset.derive_schema()
    record0, record1, _ = dataset.records
    cohort_id = _cohort_id(dataset)
    edited = remove_records(dataset, [record0.id], cascade=True)
    kept = next(s for s in edited.records if s.id == record1.id)
    assert any(a.id == cohort_id for a in kept.annotations)


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping record1 strands the cohort annotation for the answer task. It is context rather than the
    # answer, so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_record1()
    dataset.derive_schema()

    edited = remove_records(dataset, [dataset.records[1].id])
    rebuilt = next(t for t in edited.tasks if t.id == _answer(dataset).id)
    assert rebuilt.inputs == (dataset.records[0],)
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The mirror case: the cohort annotation stays on record0, so removing record1 leaves the ref intact.
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, [dataset.records[1].id])
    rebuilt = next(t for t in edited.tasks if t.id == _answer(dataset).id)
    assert tuple(annotation.id for annotation in rebuilt.input_annotations) == (_cohort_id(dataset),)


def test_target_annotation_refs_are_required():
    # The localization task stores its answer as annotation occurrences on record0.
    # Widening it to record2 means dropping record0 no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(t for t in dataset.tasks if isinstance(t, TemporalLocalizationTask))
    localize.inputs = (dataset.records[0], dataset.records[2])
    localize.targets = None
    localize.target_annotations = tuple(
        annotation for annotation in dataset.records[0].annotations if annotation.key in {"stimulus", "artifact"}
    )
    dataset.records[2].task_ids = (*dataset.records[2].task_ids, localize.id)
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match=str(localize.id)):
        remove_records(dataset, [dataset.records[0].id])

    edited = remove_records(dataset, [dataset.records[0].id], cascade=True)
    assert localize.id not in _ids(edited.tasks)


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the records that task is attached to. Removing record1 strands the cohort annotation for
    # the answer task, which previously wrote the dangling reference straight into the new version.
    dataset = _answer_reads_context_via_record1()
    base, ids = _base_ids(tmp_path, dataset)
    out = edit_version(
        base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[ids.records[dataset.records[1].id]]
    )
    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()

    by_record = {
        record.id: {annotation.occurrence_id for annotation in record.annotations} for record in restored.records
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
            inputs=(dataset.records[0],),
            prompt="Denoise it.",
            targets=(edited_record,),
            id=306,
        )
    )
    dataset.add_task(
        task=TSCorrespondenceTask(
            inputs=(dataset.records[0],),
            prompt="Which trace matches?",
            candidate_records=(dataset.records[1], dataset.records[2]),
            targets=(dataset.records[2],),
            id=307,
        )
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match=r"dangle tasks \[[^\]]*\b306\b"):
        remove_records(dataset, [dataset.records[1].id])  # the edit's produced record
    with pytest.raises(TimeFEditError, match=r"dangle tasks \[[^\]]*\b307\b"):
        remove_records(dataset, [dataset.records[2].id])  # a candidate the correspondence answer names


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    dataset = make_dataset()
    base, ids = _base_ids(tmp_path, dataset)
    record0, record1, record2 = (ids.records[record.id] for record in dataset.records)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[record1])

    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()
    assert _ids(restored.records) == {record0, record2}

    manifest = Manifest.model_validate_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    with duckdb.connect(str(out / "control.duckdb"), read_only=True) as connection:
        row = connection.execute("SELECT count(*) FROM records WHERE record_id = ?", [record1]).fetchone()
        assert row is not None
        assert row[0] == 0


def test_edit_version_same_version_rejected(tmp_path):
    dataset = make_dataset()
    base = _base(tmp_path, dataset)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(
            base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_record_ids=[dataset.records[1].id]
        )
