"""Copy-on-write edits: remove_records validate-and-cascade, and the edit_version round trip."""

import re

import duckdb
import pytest

from timenet.dataset.edit import edit_version, remove_records
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, TemporalLocalizationTask, TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def test_remove_records_preserves_dataset_annotations_and_their_refs():
    # A dataset annotation and a task reference to its occurrence survive an unrelated record removal.
    dataset = make_dataset()
    options = dataset.annotate(Annotation(key="answer_options", value=["yes", "no"]))
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.input_annotations = (*answer.input_annotations, options)

    edited = remove_records(dataset, [dataset.records[2].id], cascade=True)

    assert list(edited.annotations) == [options]
    kept = next(task for task in edited.tasks if task.id == answer.id)
    assert options in kept.input_annotations


def _base(tmp_path, dataset=None):
    dataset = make_dataset() if dataset is None else dataset
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version)


def _answer_reads_context_via_additional_input():
    """Rewire the answer task to reach its context annotation only through an additional input record.

    Still a valid dataset — the annotation sits on a record the task is attached to — so it writes and
    reads back fine. Removing the additional input record is what strands the reference.
    """
    dataset = make_dataset()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (record,) = answer.inputs
    (shared_annotation,) = answer.input_annotations
    context_record = dataset.records[1]
    context_annotation = next(
        annotation for annotation in context_record.annotations if annotation == shared_annotation
    )
    answer.inputs = (record, context_record)
    record.annotations = tuple(annotation for annotation in record.annotations if annotation != shared_annotation)
    answer.input_annotations = (context_annotation,)
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_record():
    dataset = make_dataset()
    dataset.derive_schema()
    removed_record = dataset.records[1]
    retained_records = (dataset.records[0], dataset.records[2])
    expected_task_ids = {task.id for task in dataset.tasks}
    assert not dataset.tasks_for(removed_record), "the removed record must have no tasks"
    edited = remove_records(dataset, [removed_record.id])
    assert {record.id for record in edited.records} == {record.id for record in retained_records}
    assert {task.id for task in edited.tasks} == expected_task_ids


def test_remove_unknown_record_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown record ids"):
        remove_records(dataset, ["nope"])


def test_remove_record_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        remove_records(dataset, [dataset.records[0].id])


def test_remove_record_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    removed_record = dataset.records[0]
    retained_records = dataset.records[1:]
    independent_record = dataset.records[2]
    surviving_task = next(task for task in dataset.tasks if task.inputs == (independent_record,))
    edited = remove_records(dataset, [removed_record.id], cascade=True)
    assert {record.id for record in edited.records} == {record.id for record in retained_records}
    assert {task.id for task in edited.tasks} == {surviving_task.id}


def test_shared_annotation_survives_on_remaining_record():
    dataset = make_dataset()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (removed_record,) = answer.inputs
    (shared_annotation,) = answer.input_annotations
    retained_record = dataset.records[1]
    edited = remove_records(dataset, [removed_record.id], cascade=True)
    restored_record = next(record for record in edited.records if record.id == retained_record.id)
    assert shared_annotation in restored_record.annotations


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping the additional input record strands the task's context annotation. It is not the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_additional_input()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    kept_record, removed_record = answer.inputs
    edited = remove_records(dataset, [removed_record.id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert rebuilt.inputs == (kept_record,)
    assert rebuilt.input_annotations == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The context annotation stays on the task's input, so removing another record leaves it intact.
    dataset = make_dataset()
    dataset.derive_schema()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    (context_annotation,) = answer.input_annotations
    edited = remove_records(dataset, [dataset.records[1].id])
    rebuilt = next(task for task in edited.tasks if task.id == answer.id)
    assert rebuilt.input_annotations == (context_annotation,)


def test_target_annotation_refs_are_required():
    # The localization task stores its answer as an annotation occurrence on its input record.
    # Adding another input means dropping the original no longer costs it every record, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(task for task in dataset.tasks if isinstance(task, TemporalLocalizationTask))
    (removed_record,) = localize.inputs
    retained_record = dataset.records[2]
    target_annotation = next(annotation for annotation in removed_record.annotations if annotation.span is not None)
    localize.inputs = (removed_record, retained_record)
    localize.targets = None
    localize.target_annotations = (target_annotation,)
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match=re.escape(localize.id)):
        remove_records(dataset, [removed_record.id])

    edited = remove_records(dataset, [removed_record.id], cascade=True)
    assert localize.id not in {task.id for task in edited.tasks}


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the records that task is attached to. Removing the context record previously
    # wrote the dangling reference straight into the new version.
    dataset = _answer_reads_context_via_additional_input()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    context_record = answer.inputs[1]
    base = _base(tmp_path, dataset)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[context_record.id])
    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()

    by_record = {record.id: {id(annotation) for annotation in record.annotations} for record in restored.records}
    for task in restored.tasks:
        reachable = set().union(*(by_record[record.id] for record in task.inputs))
        named = (*task.input_annotations, *task.target_annotations)
        assert not [annotation for annotation in named if id(annotation) not in reachable]


def test_target_and_candidate_records_are_required_whatever_the_task_type():
    dataset = make_dataset()
    edited_record = dataset.records[1]
    candidate_record = dataset.records[2]
    edit_task = dataset.add_task(
        task=TSEditingTask(
            inputs=(dataset.records[0],),
            prompt="Denoise it.",
            targets=(edited_record,),
            id="task-edit-0",
        )
    )
    correspondence_task = dataset.add_task(
        task=TSCorrespondenceTask(
            inputs=(dataset.records[0],),
            prompt="Which trace matches?",
            candidate_records=(edited_record, candidate_record),
            targets=(candidate_record,),
            id="task-corr-0",
        )
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match=re.escape(edit_task.id)):
        remove_records(dataset, [edited_record.id])
    with pytest.raises(TimeFEditError, match=re.escape(correspondence_task.id)):
        remove_records(dataset, [candidate_record.id])


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    dataset = make_dataset()
    removed_record = dataset.records[1]
    retained_records = (dataset.records[0], dataset.records[2])
    base = _base(tmp_path, dataset)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_record_ids=[removed_record.id])

    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()
    assert {record.id for record in restored.records} == {record.id for record in retained_records}

    manifest = Manifest.model_validate_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    with duckdb.connect(str(out / "control.duckdb"), read_only=True) as connection:
        row = connection.execute("SELECT count(*) FROM records WHERE record_id = ?", [removed_record.id]).fetchone()
        assert row is not None
        assert row[0] == 0


def test_edit_version_same_version_rejected(tmp_path):
    dataset = make_dataset()
    base = _base(tmp_path, dataset)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(
            base,
            tmp_path / "out",
            dataset_version=dataset.metadata.dataset_version,
            remove_record_ids=[dataset.records[1].id],
        )
