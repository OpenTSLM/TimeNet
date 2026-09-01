"""Copy-on-write edits: remove_samples validate-and-cascade, and the edit_version round trip."""

import pyarrow.parquet as pq
import pytest

from timenet.dataset.edit import edit_version, remove_samples
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def test_remove_samples_preserves_registered_annotations_and_their_refs():
    # A registered annotation no sample carries, and the task refs to it, must survive a sample removal.
    dataset = make_dataset()
    dataset.register_annotations([Annotation(key="answer_options", value=["yes", "no"], id="opts-shared")])
    answer = next(task for task in dataset.tasks if task.id == "task-answer-0")
    answer.input_annotation_ids = (*answer.input_annotation_ids, "opts-shared")

    edited = remove_samples(dataset, ["sample-2"], cascade=True)  # sample-2's own task cascades out

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


def _answer_reads_context_via_sample1():
    """make_dataset() rewired so task-answer-0 spans sample-0/1 but reaches cohort-shared only via sample-1.

    Still a valid dataset — the annotation sits on a sample the task is attached to — so it writes and
    reads back fine. Removing sample-1 is what strands the reference.
    """
    dataset = make_dataset()
    answer = next(t for t in dataset.tasks if t.id == "task-answer-0")
    answer.sample_ids = ("sample-0", "sample-1")
    dataset.samples[1].task_ids = (*dataset.samples[1].task_ids, "task-answer-0")
    sample0 = dataset.samples[0]
    sample0.annotations = tuple(a for a in sample0.annotations if a.id != "cohort-shared")
    return dataset


# ---- in-memory transform ----------------------------------------------------------------------


def test_remove_leaf_sample():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_samples(dataset, ["sample-1"])
    assert {s.sample_id for s in edited.samples} == {"sample-0", "sample-2"}
    # tasks untouched (sample-1 had none)
    assert {t.id for t in edited.tasks} == {
        "task-cls-0",
        "task-answer-0",
        "task-scalar-0",
        "task-localize-0",
        "task-cls-2",
    }


def test_remove_unknown_sample_raises():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="unknown sample ids"):
        remove_samples(dataset, ["nope"])


def test_remove_sample_with_task_rejects_without_cascade():
    dataset = make_dataset()
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="cascade=True"):
        remove_samples(dataset, ["sample-0"])  # every task on sample-0 would dangle


def test_remove_sample_cascades_dependent_tasks():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_samples(dataset, ["sample-0"], cascade=True)
    assert {s.sample_id for s in edited.samples} == {"sample-1", "sample-2"}
    # every task on sample-0 goes, and task-answer-0 (which derives from task-cls-0) cascades out
    assert {t.id for t in edited.tasks} == {"task-cls-2"}


def test_shared_annotation_survives_on_remaining_sample():
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_samples(dataset, ["sample-0"], cascade=True)
    sample1 = next(s for s in edited.samples if s.sample_id == "sample-1")
    assert any(a.id == "cohort-shared" for a in sample1.annotations)


def test_input_annotation_refs_are_stripped_when_unreachable():
    # Dropping sample-1 strands cohort-shared for task-answer-0. It is context rather than the answer,
    # so the task survives with the reference pruned instead of being rejected.
    dataset = _answer_reads_context_via_sample1()
    dataset.derive_schema()

    edited = remove_samples(dataset, ["sample-1"])
    rebuilt = next(t for t in edited.tasks if t.id == "task-answer-0")
    assert rebuilt.sample_ids == ("sample-0",)
    assert rebuilt.input_annotation_ids == ()


def test_input_annotation_refs_survive_a_reachable_removal():
    # The mirror case: cohort-shared stays on sample-0, so removing sample-1 leaves the ref intact.
    dataset = make_dataset()
    dataset.derive_schema()
    edited = remove_samples(dataset, ["sample-1"])
    rebuilt = next(t for t in edited.tasks if t.id == "task-answer-0")
    assert rebuilt.input_annotation_ids == ("cohort-shared",)


def test_target_annotation_refs_are_required():
    # task-localize-0 stores its answer as target_annotation_ids pointing at stim-0/art-0 on sample-0.
    # Widening it to sample-2 means dropping sample-0 no longer costs it every sample, but it does cost
    # it the answer, so the edit is rejected rather than silently rewriting the ground truth.
    dataset = make_dataset()
    localize = next(t for t in dataset.tasks if t.id == "task-localize-0")
    localize.target = None
    localize.target_annotation_ids = ("stim-0", "art-0")
    localize.sample_ids = ("sample-0", "sample-2")
    dataset.samples[2].task_ids = (*dataset.samples[2].task_ids, "task-localize-0")
    dataset.derive_schema()

    with pytest.raises(TimeFEditError, match="task-localize-0"):
        remove_samples(dataset, ["sample-0"])

    edited = remove_samples(dataset, ["sample-0"], cascade=True)
    assert "task-localize-0" not in {t.id for t in edited.tasks}


def test_edited_version_has_no_dangling_annotation_refs(tmp_path):
    # The end-to-end guarantee: whatever a committed edit contains, every annotation a task names is
    # resolvable from the samples that task is attached to. Removing sample-1 strands cohort-shared for
    # task-answer-0, which previously wrote the dangling reference straight into the new version.
    base = _base(tmp_path, _answer_reads_context_via_sample1())
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_sample_ids=["sample-1"])
    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()

    by_sample = {s.sample_id: {a.id for a in s.annotations} for s in restored.samples}
    for task in restored.tasks:
        reachable = set().union(*(by_sample[sid] for sid in task.sample_ids))
        named = (*task.input_annotation_ids, *task.target_annotation_ids)
        assert not [aid for aid in named if aid not in reachable], f"{task.id} dangles"


def test_payload_sample_refs_are_required_whatever_the_task_type():
    # The editor reads TaskRefs rather than special-casing forecasting, so an edit task's source and a
    # correspondence task's candidate pool are protected the same way.
    dataset = make_dataset()
    edited_sample = dataset.samples[1]
    dataset.add_task(
        dataset.samples[0],
        TSEditingTask(
            prompt="Denoise it.",
            source_sample_id="sample-0",
            target_sample_id=edited_sample.sample_id,
            id="task-edit-0",
        ),
    )
    dataset.add_task(
        dataset.samples[0],
        TSCorrespondenceTask(
            prompt="Which trace matches?",
            candidate_sample_ids=("sample-1", "sample-2"),
            target=("sample-2",),
            id="task-corr-0",
        ),
    )
    dataset.derive_schema()
    with pytest.raises(TimeFEditError, match="task-edit-0"):
        remove_samples(dataset, ["sample-1"])  # the edit's produced sample
    with pytest.raises(TimeFEditError, match="task-corr-0"):
        remove_samples(dataset, ["sample-2"])  # a candidate the correspondence answer names


# ---- copy-on-write version --------------------------------------------------------------------


def test_edit_version_round_trip(tmp_path):
    base = _base(tmp_path)
    out = edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 1), remove_sample_ids=["sample-1"])

    with TimeFReader(DatasetVersion.open_local(out)) as reader:
        restored = reader.read()
    assert {s.sample_id for s in restored.samples} == {"sample-0", "sample-2"}

    manifest = Manifest.from_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"

    index = pq.read_table(out / "time_series_index/part-00000000.parquet").to_pylist()
    assert all(row["sample_id"] != "sample-1" for row in index)  # no orphaned index rows


def test_edit_version_same_version_rejected(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_sample_ids=["sample-1"])
