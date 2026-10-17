"""Copy-on-write edits: remove_records validate-and-cascade, and the edit_version round trip."""

import duckdb
import pytest

from timenet.dataset.edit import edit_version, remove_records
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def test_remove_records_preserves_dataset_annotations_and_their_refs():
    # A dataset annotation and a task reference to its occurrence survive an unrelated record removal.
    dataset = make_dataset()
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"], id=401))
    answer = next(task for task in dataset.tasks if task.id == 302)
    answer.input_annotations = (*answer.input_annotations, options)

    edited = remove_records(dataset, [12], cascade=True)  # record 12's own task cascades out

    assert [ann.id for ann in edited.annotations] == [401]
    kept = next(task for task in edited.tasks if task.id == 302)
    assert options in kept.input_annotations


def _base(tmp_path, dataset=None):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version)


def _answer_reads_context_via_record1():
    """make_dataset() rewired so answer task 302 spans records 10/11 but reaches cohort 201 only via record 11.

    Still a valid dataset — the annotation sits on a record the task is attached to — so it writes and
    reads back fine. Removing record 11 is what strands the reference.
    """
    dataset = make_dataset()
    answer = next(t for t in dataset.tasks if t.id == 302)
    answer.inputs = (dataset.records[0], dataset.records[1])
    dataset.records[1].task_ids = (*dataset.records[1].task_ids, 302)
    record0 = dataset.records[0]
    record0.annotations = tuple(a for a in record0.annotations if a.id != 201)
    answer.input_annotations = tuple(a for a in dataset.records[1].annotations if a.id == 201)
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, [11])
    assert {s.record_id for s in edited.records} == {10, 12}
    # tasks untouched (record 11 had none)
    assert {t.id for t in edited.tasks} == {
        301,
        302,
        303,
        304,
        305,
    }


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, [999])


def test_remove_record_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        remove_records(dataset, [10])  # every task on record 10 would dangle


def test_remove_record_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, [10], cascade=True)
    assert {s.record_id for s in edited.records} == {11, 12}
    # every task on record 10 goes, and answer task 302 (which derives from task 301) cascades out
    assert {t.id for t in edited.tasks} == {305}


def test_shared_annotation_survives_on_remaining_record():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, [10], cascade=True)
    record1 = next(s for s in edited.records if s.record_id == 11)
    assert any(a.id == 201 for a in record1.annotations)


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping record 11 strands cohort annotation 201 for answer task 302. It is context rather than the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_record1()
    dataset.derive_schema()

    edited = remove_records(dataset, [11])
    rebuilt = next(t for t in edited.tasks if t.id == 302)
    assert rebuilt.inputs == (dataset.records[0],)
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The mirror case: cohort annotation 201 stays on record 10, so removing record 11 leaves the ref intact.
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_records(dataset, [11])
    rebuilt = next(t for t in edited.tasks if t.id == 302)
    assert tuple(annotation.id for annotation in rebuilt.input_annotations) == (201,)


def test_target_annotation_refs_are_required():
    # task 304 stores its answer as annotation occurrences on record 10.
    # Widening it to record 12 means dropping record 10 no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(t for t in dataset.tasks if t.id == 304)
    localize.inputs = (dataset.records[0], dataset.records[2])
    localize.targets = None
    localize.target_annotations = tuple(
        annotation for annotation in dataset.records[0].annotations if annotation.id in {203, 204}
    )
    dataset.records[2].task_ids = (*dataset.records[2].task_ids, 304)
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match="304"):
        remove_records(dataset, [10])

    edited = remove_records(dataset, [10], cascade=True)
    assert 304 not in {t.id for t in edited.tasks}


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the records that task is attached to. Removing record 11 strands cohort annotation 201 for
    # answer task 302, which previously wrote the dangling reference straight into the new version.
    base = _base(tmp_path, _answer_reads_context_via_record1())
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[11])
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
    with pytest.raises(TimeFEditError, match="306"):
        remove_records(dataset, [11])  # the edit's produced record
    with pytest.raises(TimeFEditError, match="307"):
        remove_records(dataset, [12])  # a candidate the correspondence answer names


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    base = _base(tmp_path)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[11])

    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()
    assert {s.record_id for s in restored.records} == {10, 12}

    manifest = Manifest.model_validate_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    with duckdb.connect(str(out / "control.duckdb"), read_only=True) as connection:
        row = connection.execute("SELECT count(*) FROM records WHERE record_id = 11").fetchone()
        assert row is not None
        assert row[0] == 0


def test_edit_version_same_version_rejected(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_record_ids=[11])
