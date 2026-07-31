"""Copy-on-write edits: remove_samples validate-and-cascade, and the edit_version round trip."""

import pyarrow.parquet as pq
import pytest

from timenet.dataset.edit import edit_version, remove_samples
from timenet.errors import TimeFEditError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.testing import make_dataset
from timenet.types import TSCorrespondenceTask, TSEditingTask, Version
from timenet.writer import TimeFWriter


def _base(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "base", dataset) as writer:
        writer.write()
    meta = dataset.metadata
    return tmp_path / "base" / meta.dataset_id / str(meta.dataset_version)


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

    with TimeFReader(out) as reader:
        restored = reader.read()
    assert {s.sample_id for s in restored.samples} == {"sample-0", "sample-2"}

    manifest = Manifest.from_json((out / "manifest.json").read_text())
    assert str(manifest.metadata.dataset_version) == "1.0.1"
    assert manifest.derived_from == {"dataset_version": "1.0.0", "op": "remove_samples"}

    index = pq.read_table(out / "time_series_index.parquet").to_pylist()
    assert all(row["sample_id"] != "sample-1" for row in index)  # no orphaned index rows


def test_edit_version_same_version_rejected(tmp_path):
    base = _base(tmp_path)
    with pytest.raises(TimeFEditError, match="must differ"):
        edit_version(base, tmp_path / "out", dataset_version=Version(1, 0, 0), remove_sample_ids=["sample-1"])
