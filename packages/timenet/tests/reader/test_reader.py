import json
from pathlib import Path
import pickle

import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal, make_dataset
from timenet.types import (
    ClassificationTask,
    IntervalAnnotation,
    PointAnnotation,
    QATask,
    ReasoningTask,
    StaticAnnotation,
    View,
)
from timenet.writer import TimeFWriter


def _write(tmp_path, dataset=None, **kwargs) -> Path:
    dataset = dataset if dataset is not None else make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


# ---- round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_full_round_trip(tmp_path, backend):
    original = make_dataset()
    version_dir = _write(tmp_path, dataset=make_dataset(), values_backend=backend)
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert isinstance(restored, TimeFDataset)
    assert_datasets_equal(original, restored)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_round_trip_with_chunk_splitting(tmp_path, backend):
    original = make_dataset()
    version_dir = _write(
        tmp_path, dataset=make_dataset(), values_backend=backend, chunk_max_bytes=64, row_group_target_bytes=64
    )
    with TimeFReader(version_dir) as reader:
        restored = reader.read()
    assert_datasets_equal(original, restored)


def test_metadata_and_schema(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        assert reader.metadata.dataset_id == "timenet/hello-world"
        assert {s.spec_type for s in reader.schema.time_series_specs} == {"sine", "cosine"}
        assert ClassificationTask in reader.schema.tasks


def test_shared_series_distinct_objects_same_id(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        samples = {s.sample_id: s for s in reader.read().samples}
    a = next(ts for ts in samples["sample-0"].time_series if ts.time_series_id == "ts-shared")
    b = next(ts for ts in samples["sample-1"].time_series if ts.time_series_id == "ts-shared")
    assert a is not b
    assert a.to_arrow().equals(b.to_arrow())


def test_annotation_value_types_round_trip(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        samples = {s.sample_id: s for s in reader.read().samples}
    anns = {a.key: a for a in samples["sample-0"].annotations}
    assert isinstance(anns["age"], StaticAnnotation)
    assert anns["age"].value == 64 and isinstance(anns["age"].value, int)
    assert isinstance(anns["stimulus"], PointAnnotation)
    assert isinstance(anns["artifact"], IntervalAnnotation)
    assert anns["artifact"].time_series_ids == ("ts-shared",)


def test_task_chain_round_trips(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        tasks = {t.id: t for t in reader.tasks}
    qa = tasks["task-qa-0"]
    assert isinstance(qa, QATask)
    assert qa.from_tasks and qa.from_tasks[0].id == "task-cls-0"


def test_reasoning_task_rationale_round_trips(tmp_path):
    dataset = make_dataset()
    sample = dataset.samples[0]
    dataset.add_task(
        sample,
        ReasoningTask(
            question="Is the rhythm normal?",
            rationale="Regular R-R intervals with a P wave before each QRS.",
            target="Yes.",
            id="task-reason-0",
        ),
    )
    dataset.add_task(sample, ReasoningTask(question="Any ectopy?", target="No.", id="task-reason-1"))  # rationale=None
    version_dir = _write(tmp_path, dataset=dataset)
    with TimeFReader(version_dir) as reader:
        tasks = {t.id: t for t in reader.tasks}
    with_rationale = tasks["task-reason-0"]
    assert isinstance(with_rationale, ReasoningTask)
    assert with_rationale.question == "Is the rhythm normal?"
    assert with_rationale.rationale == "Regular R-R intervals with a P wave before each QRS."
    assert with_rationale.target == "Yes."
    without_rationale = tasks["task-reason-1"]
    assert isinstance(without_rationale, ReasoningTask)
    assert without_rationale.rationale is None  # optional field round-trips as None


def test_iter_samples_matches_read(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        streamed = {s.sample_id for s in reader.iter_samples()}
    with TimeFReader(version_dir) as reader:
        read_ids = {s.sample_id for s in reader.read().samples}
    assert streamed == read_ids


def test_window_sample_preserved(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        views = {s.sample_id: s.view for s in reader.read().samples}
    assert views["sample-2"] == View.WINDOW


# ---- laziness ---------------------------------------------------------------------------------


def test_values_are_lazy(tmp_path, monkeypatch):
    version_dir = _write(tmp_path)
    original_open = pq.ParquetFile
    opens = {"n": 0}

    def counting_open(*args, **kwargs):
        opens["n"] += 1
        return original_open(*args, **kwargs)

    with TimeFReader(version_dir) as reader:
        dataset = reader.read()
        monkeypatch.setattr(pq, "ParquetFile", counting_open)
        ts = dataset.samples[0].time_series[0]
        assert opens["n"] == 0  # building samples opened no shards
        ts.to_arrow()
        assert opens["n"] >= 1  # reading values opened a shard


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_read_back_dataset_is_picklable(tmp_path, backend):
    # A multi-worker torch DataLoader pickles the dataset to each worker, so lazy loaders must pickle
    # even after values (and thus the backend's handles/caches) have been touched.
    version_dir = _write(tmp_path, values_backend=backend)
    dataset = TimeFReader(version_dir).read()
    first = dataset.samples[0].time_series[0]
    original = first.to_arrow()  # populates the values backend's caches

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.samples[0].time_series[0].to_arrow().equals(original)


# ---- validation -------------------------------------------------------------------------------


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TimeFReader(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        TimeFReader(tmp_path / "empty")


@pytest.mark.parametrize("format_version", [2, 99])
def test_unsupported_format_version_raises(tmp_path, format_version):
    version_dir = _write(tmp_path)
    manifest_path = version_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["timef_format_version"] = format_version
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(TimeFFormatError):
        TimeFReader(version_dir)


def test_corrupt_index_locator_has_series_context(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        key = next(iter(reader._index))
        del reader._index[key][0]["chunk_file"]
        with pytest.raises(ValueError, match=f"failed to read series {key[1]!r} for sample {key[0]!r}"):
            reader._load_values(*key)


def test_missing_listed_file_raises(tmp_path):
    version_dir = _write(tmp_path)
    (version_dir / "samples.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        TimeFReader(version_dir)


def test_verify_passes_on_an_intact_dataset(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        reader.verify()  # must not raise


def test_verify_detects_a_corrupted_shard(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        # a shard: read lazily, so __init__ still succeeds and verify() is what catches it
        rel = next(r for r in reader._manifest.checksums if r.startswith("time_series/shard-"))
    target = version_dir / rel
    target.write_bytes(target.read_bytes() + b"corruption")
    with TimeFReader(version_dir) as reader, pytest.raises(TimeFFormatError, match="checksum mismatch"):
        reader.verify()


def test_verify_detects_a_deleted_file(tmp_path):
    version_dir = _write(tmp_path)
    with TimeFReader(version_dir) as reader:
        rel = next(r for r in reader._manifest.checksums if r.startswith("time_series/shard-"))
    (version_dir / rel).unlink()
    # __init__ already refuses a manifest-listed file that is gone
    with pytest.raises(FileNotFoundError):
        TimeFReader(version_dir)


def test_corrupt_task_partition_raises_format_error(tmp_path):
    # An unknown task partition name is corrupt on-disk data, so it must surface as TimeFFormatError
    # rather than the bare ValueError that TaskType() happens to raise.
    version_dir = _write(tmp_path)
    tasks_dir = next((version_dir / "tasks").iterdir())
    tasks_dir.rename(tasks_dir.parent / "task=not_a_real_task_type")
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tasks"] = [
        p.replace(tasks_dir.name, "task=not_a_real_task_type") for p in manifest["files"]["tasks"]
    ]
    manifest["checksums"] = {
        k.replace(tasks_dir.name, "task=not_a_real_task_type"): v for k, v in manifest["checksums"].items()
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TimeFFormatError):
        TimeFReader(version_dir)
