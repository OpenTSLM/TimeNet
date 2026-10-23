from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.storage import StorageSources
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import PackedLocation, PackedValueKind, PackedValueReader, StoredOffsets, StoredText
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, ureg
from timenet.types.reference_keys import OwnerKey
from timenet.types.references import DatasetRef
from timenet.values_backends.types import ChunkDataIndex, StoredChunk
from timenet.writer import TimeFWriter


def store(root, dataset, **kwargs):
    dataset.derive_schema()
    with TimeFWriter(root, dataset, **kwargs) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_prepared_shared_payloads_do_not_open_parent_catalogs(tmp_path, monkeypatch, backend):
    original = make_dataset()
    text = "A detailed clinical report. " * 300
    original.tasks[0].prompt = text
    report = original.records[0].annotate(Annotation(key="report", value=text))
    original.records[0].signals[0].annotate(report)
    expected = original.records[0].signals[0].to_arrow()
    parent_dir = store(tmp_path, original, values_backend=backend)
    reader = TimeFReader(open_local_version(parent_dir))
    parent = reader.get_handle().reader._read_catalog()
    child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/child"}))
    child.add_record(record=deepcopy(parent.records[0]))
    prompt = next(task.prompt for task in parent.tasks if task.id == original.tasks[0].id)
    assert prompt is not None
    child.add_task(task=AnswerTask(id="question", inputs=(child.records[0],), prompt=prompt, targets=(prompt,)))

    def unexpected_read(*args):
        pytest.fail("inheriting stored values must not read or copy them")

    with monkeypatch.context() as patch:
        # Packed text and time offsets are read through read_scalar, and signal arrays through
        # the reader's array reads.
        patch.setattr(PackedValueReader, "read_scalar", unexpected_read)
        patch.setattr(TimeFReader, "_read_array", unexpected_read)
        patch.setattr(TimeFReader, "_read_array_steps", unexpected_read)
        child_dir = store(tmp_path, child)
    reader.close()
    manifest = Manifest.model_validate_json((child_dir / "manifest.json").read_text())
    assert not manifest.files.time_series
    assert not manifest.files.payloads
    assert len(manifest.payload_sources) == 1
    # ID/version is the pin; descriptive metadata does not change the reference.
    source = Manifest.model_validate_json((parent_dir / "manifest.json").read_text())
    metadata = source.metadata.model_copy(update={"description": "Updated description"})
    source = source.model_copy(update={"metadata": metadata})
    (parent_dir / "manifest.json").write_text(source.to_json())
    prepared = open_local_version(child_dir)
    # Preparation requires the full release, but processing never opens parent catalogs.
    (parent_dir / "control.duckdb").rename(parent_dir / "unavailable.duckdb")
    (parent_dir / "manifest.json").unlink()
    with TimeFReader(prepared) as child_reader:
        assert child_reader.read().read().records[0].signals[0].to_arrow().equals(expected)
        assert child_reader.read_task_table()["prompt"].to_pylist() == [text]
        assert child_reader.read_target_table()["text_value"].to_pylist() == [text]
        assert text in child_reader.read_annotation_table()["text_value"].to_pylist()


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_a_child_reads_inherited_chunks_through_the_release_that_stores_them(tmp_path, backend):
    parent_dir = store(tmp_path, make_dataset(), values_backend=backend)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader._read_catalog()
        parent_values = parent.records[0].signals[0]._values
        assert isinstance(parent_values, StoredArray)
        child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
        child.add_record(record=deepcopy(parent.records[0]))
        child_dir = store(tmp_path, child)
        expected = parent_values.read()
    owner = OwnerKey(f"{parent.metadata.dataset_id}@{parent.metadata.dataset_version}")

    with TimeFReader(open_local_version(child_dir)) as reader:
        values = reader._read_catalog().records[0].signals[0]._values
        assert isinstance(values, StoredArray)
        assert values._location == ChunkLocation(owner=owner, chunks=parent_values._location.chunks)
        assert values.read().equals(expected)
        assert values.read_steps(3, 10).equals(expected.slice(3, 7))


def test_shared_payloads_cannot_reference_the_output_release(tmp_path):
    path = store(tmp_path, make_dataset())
    original_manifest = (path / "manifest.json").read_bytes()
    with TimeFReader(open_local_version(path)) as reader:
        loaded = reader._read_catalog()
        with pytest.raises(TimeFValidationError, match="also the output release"):
            store(tmp_path / "copy", loaded)
    assert (path / "manifest.json").read_bytes() == original_manifest
    with TimeFReader(open_local_version(path)) as reader:
        assert reader.read().read().records


_SPEC = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless)


def _release(dataset_id, records):
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0),
            name=dataset_id,
            description="d",
            license=License.MIT,
        )
    )
    for record in records:
        dataset.add_record(record=record)
    return dataset


def _record_in_memory(record_id, text, offsets):
    """Return a Record whose Signal values, time offsets, and report text are held in memory."""
    signal = Signal.from_irregular(
        np.arange(len(offsets), dtype=np.float32), time_offsets_us=offsets, spec=_SPEC, name=record_id
    )
    record = Record(record_id=record_id, sources=(Source(name="env", signals=(signal,)),))
    record.annotate(Annotation(key="report", value=text))
    return record


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_a_grandchild_names_the_release_that_stores_each_payload(tmp_path, backend):
    text = {"a": "Stored by release A. " * 300, "b": "Stored by release B. " * 300}
    offsets = {"a": [0, 5_000, 9_000], "b": [0, 1_000, 2_000, 7_000]}
    a_dir = store(
        tmp_path, _release("test/a", [_record_in_memory("a", text["a"], offsets["a"])]), values_backend=backend
    )
    with TimeFReader(open_local_version(a_dir)) as reader:
        (a,) = reader._read_catalog().records
        b = _release("test/b", [deepcopy(a), _record_in_memory("b", text["b"], offsets["b"])])
        b_dir = store(tmp_path, b, values_backend=backend)
    with TimeFReader(open_local_version(b_dir)) as reader:
        # C writes with the default backend; reused Zarr chunks keep their own encoding.
        c_dir = store(tmp_path, _release("test/c", [deepcopy(record) for record in reader._read_catalog().records]))

    manifest = Manifest.model_validate_json((c_dir / "manifest.json").read_text())
    refs = {name: DatasetRef(dataset_id=f"test/{name}", version=Version(1, 0, 0)) for name in "ab"}
    assert manifest.payload_sources == {OwnerKey(str(ref)): ref for ref in refs.values()}
    assert not manifest.files.time_series
    assert not manifest.files.payloads
    with TimeFReader(open_local_version(c_dir)) as reader:
        records = reader._read_catalog().records
        assert sorted(record.id for record in records) == ["a", "b"]
        for record in records:
            (signal,) = record.signals
            (report,) = record.annotations
            assert isinstance(signal._values, StoredArray)
            assert isinstance(signal._time_offsets, StoredOffsets)
            assert isinstance(report.value, StoredText)
            # A payload that B reused from A keeps naming A, so C's reader resolves it in one step.
            owner = str(refs[record.id])
            assert signal._values._location.owner == owner
            assert signal._time_offsets._location.owner == owner
            assert report.value._location.owner == owner
            assert signal.to_numpy().tolist() == list(range(len(offsets[record.id])))
            assert signal.time_offsets_us().tolist() == offsets[record.id]
            assert report.value.read() == text[record.id]


def test_a_registry_refuses_a_release_whose_payload_sources_it_does_not_hold(tmp_path):
    parent_dir = store(tmp_path / "parents", make_dataset())
    elsewhere = LocalRegistry(tmp_path / "children")
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader._read_catalog()
        child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "test/child"}))
        child.add_record(record=deepcopy(parent.records[0]))
        with pytest.raises(
            TimeFValidationError,
            match=r"test/child@1\.0\.0 would reuse payloads stored by timenet/hello-world@1\.0\.0, which is not",
        ):
            elsewhere.store(child)
        assert not elsewhere.exists("test/child", "1.0.0")
        assert not list((tmp_path / "children").rglob("*.tmp-*"))

        # Next to the release that stores its payloads, the same dataset is stored and opens.
        LocalRegistry(tmp_path / "parents").store(child)
        expected = parent.records[0].signals[0].to_arrow()
    with TimeFReader(open_local_version(tmp_path / "parents/test/child/1.0.0")) as reader:
        assert reader.read().read().records[0].signals[0].to_arrow().equals(expected)


def test_a_reused_payload_must_be_listed_by_the_release_that_stores_it(tmp_path):
    version = open_local_version(store(tmp_path, make_dataset()))
    sources = StorageSources()
    chunks = ChunkLocation(chunks=(StoredChunk(0, "time_series/part-99999999.parquet", ChunkDataIndex(0, 0), 1),))
    text = PackedLocation(path="payloads/part-99999999.parquet", row=0, kind=PackedValueKind.TEXT, length=1)
    for location, path in ((chunks, "time_series"), (text, "payloads")):
        with pytest.raises(
            TimeFValidationError, match=rf"payload '{path}/part-99999999\.parquet' is absent from timenet/hello-world"
        ):
            sources.add(version, location)
    assert not sources.sources


def test_a_reused_nullable_zarr_signal_keeps_its_validity_arrays(tmp_path):
    spec = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless, nullable=True)
    signal = Signal.from_values([1.0, None, 3.0], spec=spec, name="x", time_axis=RegularAxis.from_rate_hz(1))
    parent = _release("test/a", [Record(record_id="a", sources=(Source(name="env", signals=(signal,)),))])
    version = open_local_version(store(tmp_path, parent, values_backend="zarr"))
    with TimeFReader(version) as reader:
        (record,) = reader._read_catalog().records
        values = record.signals[0]._values
        assert isinstance(values, StoredArray)
        child_dir = store(tmp_path, _release("test/child", [deepcopy(record)]))

        # A release that lacks the validity arrays cannot lend a nullable Signal its chunks.
        files = version.manifest.files
        kept = tuple(part for part in files.time_series if "/_validity/" not in part.path)
        assert len(kept) < len(files.time_series)
        without_validity = replace(
            version,
            manifest=version.manifest.model_copy(update={"files": files.model_copy(update={"time_series": kept})}),
        )
        StorageSources().add(without_validity, values._location)
        with pytest.raises(TimeFValidationError, match=r"payload 'time_series\.zarr/_validity/.+' is absent"):
            StorageSources().add(without_validity, values._location, nullable=True)
    with TimeFReader(open_local_version(child_dir)) as reader:
        assert reader.read().read().records[0].signals[0].to_arrow().to_pylist() == [1.0, None, 3.0]


def test_a_reused_signal_must_keep_its_stored_shape(tmp_path):
    with TimeFReader(open_local_version(store(tmp_path, make_dataset()))) as reader:
        stored = reader._read_catalog().records[0].signals[0]
        shorter = Signal.from_loader(
            name=stored.name,
            spec=stored.spec,
            time_axis=stored.time_axis,
            n_values=stored.n_values - 1,
            loader=stored._values,
        )
        child = _release("test/child", [Record(record_id="r", sources=(Source(name="s", signals=(shorter,)),))])
        with pytest.raises(TimeFValidationError, match=rf"shared signal '{shorter.id}' changed its storage shape"):
            store(tmp_path, child)


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_eagerly_read_signals_reuse_stored_chunks(tmp_path, backend):
    original = make_dataset()
    parent_dir = store(tmp_path, original, values_backend=backend)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader._read_dataset()
        child = TimeFDataset(metadata=original.metadata.model_copy(update={"dataset_id": "test/child"}))
        child.add_record(record=deepcopy(parent.records[0]))
        child_dir = store(tmp_path, child)
    manifest = Manifest.model_validate_json((child_dir / "manifest.json").read_text())
    # The in-memory values still point at their stored chunks, so the child writes no value files.
    assert manifest.files.time_series == ()
    assert [str(ref) for ref in manifest.payload_sources.values()] == [
        f"{original.metadata.dataset_id}@{original.metadata.dataset_version}"
    ]
