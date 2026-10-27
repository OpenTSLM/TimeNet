from dataclasses import replace
import shutil

import numpy as np
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.format.storage import StorageSources
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import PackedLocation, PackedValueKind, PackedValueReader, StoredOffsets, StoredText
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    DatasetSchema,
    License,
    TimeSeriesSpec,
    Version,
    ureg,
)
from timenet.types.reference_keys import SourceKey
from timenet.types.references import DatasetRef
from timenet.types.text import TextHandle
from timenet.values_backends.types import ChunkDataIndex, StoredChunk
from timenet.writer import TimeFWriter


def _derive(parent, dataset_id, records):
    """Return a release under ``dataset_id`` that keeps read Records and their ids, as an edit does.

    The Records keep their stored handles, which name the release they were read from.
    """
    return TimeFDataset.from_parts(
        metadata=parent.metadata.model_copy(update={"dataset_id": dataset_id}),
        records=records,
        tasks=(),
        schema=DatasetSchema(),
        last_ids=parent.last_ids,
    )


def _name(record):
    return record.metadata["name"]


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
    parent = reader.read()
    child = _derive(parent, "test/child", [parent.records[0]])
    prompt = next(task.prompt for task in parent.tasks if task.id == original.tasks[0].id)
    assert prompt is not None
    child.add_task(task=AnswerTask(inputs=(child.records[0],), prompt=prompt, targets=(prompt,)))

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
        assert child_reader.read().records[0].signals[0].to_arrow().equals(expected)
        assert child_reader.read_task_table()["prompt"].to_pylist() == [text]
        assert child_reader.read_target_table()["text_value"].to_pylist() == [text]
        assert text in child_reader.read_annotation_table()["text_value"].to_pylist()


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_a_child_reads_inherited_chunks_through_the_release_that_stores_them(tmp_path, backend):
    parent_dir = store(tmp_path, make_dataset(), values_backend=backend)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader.read()
        parent_values = parent.records[0].signals[0]._values
        assert isinstance(parent_values, StoredArray)
        child_dir = store(tmp_path, _derive(parent, "test/child", [parent.records[0]]))
        expected = parent_values.read()
    source = SourceKey(f"{parent.metadata.dataset_id}@{parent.metadata.dataset_version}")

    with TimeFReader(open_local_version(child_dir)) as reader:
        values = reader.read().records[0].signals[0]._values
        assert isinstance(values, StoredArray)
        assert values._location == ChunkLocation(source=source, chunks=parent_values._location.chunks)
        assert values.read().equals(expected)
        assert values.read_steps(3, 10).equals(expected.slice(3, 7))


def test_shared_payloads_cannot_reference_the_output_release(tmp_path):
    path = store(tmp_path, make_dataset())
    original_manifest = (path / "manifest.json").read_bytes()
    with TimeFReader(open_local_version(path)) as reader:
        loaded = reader.read()
        with pytest.raises(TimeFValidationError, match="also the output release"):
            store(tmp_path / "copy", loaded)
    assert (path / "manifest.json").read_bytes() == original_manifest
    with TimeFReader(open_local_version(path)) as reader:
        assert reader.read().records


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


def _record_in_memory(name, text, offsets):
    """Return a named Record whose Signal values, time offsets, and report text are held in memory."""
    signal = Signal.from_irregular(
        np.arange(len(offsets), dtype=np.float32), time_offsets_us=offsets, spec=_SPEC, name=name
    )
    record = Record(sources=(Source(name="env", signals=(signal,)),), metadata={"name": name})
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
        a = reader.read()
        b = _derive(a, "test/b", a.records)
        b.add_record(record=_record_in_memory("b", text["b"], offsets["b"]))
        b_dir = store(tmp_path, b, values_backend=backend)
    with TimeFReader(open_local_version(b_dir)) as reader:
        b = reader.read()
        # C writes with the default backend; reused Zarr chunks keep their own encoding.
        c_dir = store(tmp_path, _derive(b, "test/c", b.records))

    manifest = Manifest.model_validate_json((c_dir / "manifest.json").read_text())
    refs = {name: DatasetRef(dataset_id=f"test/{name}", version=Version(1, 0, 0)) for name in "ab"}
    assert {key: source.release for key, source in manifest.payload_sources.items()} == {
        SourceKey(str(ref)): ref for ref in refs.values()
    }
    # Each source is pinned to the files C reads from it, as that source records them.
    for name, directory in (("a", a_dir), ("b", b_dir)):
        recorded = Manifest.model_validate_json((directory / "manifest.json").read_text()).files.all_files()
        pinned = manifest.payload_sources[SourceKey(str(refs[name]))].files
        assert pinned
        assert set(pinned) <= set(recorded)
    assert not manifest.files.time_series
    assert not manifest.files.payloads
    with TimeFReader(open_local_version(c_dir)) as reader:
        records = reader.read().records
        assert sorted(_name(record) for record in records) == ["a", "b"]
        for record in records:
            (signal,) = record.signals
            (report,) = record.annotations
            assert isinstance(signal._values, StoredArray)
            assert isinstance(signal._time_offsets, StoredOffsets)
            assert isinstance(report.value, StoredText)
            # A payload that B reused from A keeps naming A, so C's reader resolves it in one step.
            source = str(refs[_name(record)])
            assert signal._values._location.source == source
            assert signal._time_offsets._location.source == source
            assert report.value._location.source == source
            assert signal.to_numpy().tolist() == list(range(len(offsets[_name(record)])))
            assert signal.time_offsets_us().tolist() == offsets[_name(record)]
            assert report.value.read() == text[_name(record)]


def test_a_registry_refuses_a_release_whose_payload_sources_it_does_not_hold(tmp_path):
    parent_dir = store(tmp_path / "parents", make_dataset())
    elsewhere = LocalRegistry(tmp_path / "children")
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader.read()
        child = _derive(parent, "test/child", [parent.records[0]])
        with pytest.raises(
            TimeNetDatasetNotFoundError,
            match=rf"payload source timenet/hello-world@1\.0\.0 of test/child@1\.0\.0 is not in local registry "
            rf"{tmp_path / 'children'}; store test/child@1\.0\.0 in the registry that holds",
        ):
            elsewhere.store(child)
        assert not elsewhere.exists("test/child", "1.0.0")
        assert not list((tmp_path / "children").rglob("*.tmp-*"))

        # Next to the release that stores its payloads, the same dataset is stored and opens.
        LocalRegistry(tmp_path / "parents").store(child)
        expected = parent.records[0].signals[0].to_arrow()
    with TimeFReader(open_local_version(tmp_path / "parents/test/child/1.0.0")) as reader:
        assert reader.read().records[0].signals[0].to_arrow().equals(expected)


def test_a_refused_forced_store_keeps_the_committed_release(tmp_path):
    parent_dir = store(tmp_path / "parents", make_dataset())
    children = LocalRegistry(tmp_path / "children")
    with TimeFReader(open_local_version(parent_dir)) as reader:
        parent = reader.read()
        # The committed release owns its values, so this registry holds everything it needs.
        children.store(_release("test/child", [_record_in_memory("old", "Old report.", [0, 1_000])]))
        child_dir = tmp_path / "children/test/child/1.0.0"
        before = (child_dir / "manifest.json").read_bytes()
        child = _derive(parent, "test/child", [parent.records[0]])
        with pytest.raises(TimeNetDatasetNotFoundError, match=r"payload source .* of test/child@1\.0\.0 is not in"):
            children.store(child, force=True)
    assert (child_dir / "manifest.json").read_bytes() == before
    with TimeFReader(open_local_version(child_dir)) as reader:
        assert [_name(record) for record in reader.read().records] == ["old"]
    assert [path.name for path in child_dir.parent.iterdir()] == ["1.0.0"]


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
    parent = _release("test/a", [Record(sources=(Source(name="env", signals=(signal,)),))])
    version = open_local_version(store(tmp_path, parent, values_backend="zarr"))
    with TimeFReader(version) as reader:
        loaded = reader.read()
        values = loaded.records[0].signals[0]._values
        assert isinstance(values, StoredArray)
        child_dir = store(tmp_path, _derive(loaded, "test/child", loaded.records))

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
    # The child pins the validity arrays it reads, with the values array.
    pinned = Manifest.model_validate_json((child_dir / "manifest.json").read_text()).payload_sources
    assert any("/_validity/" in part.path for part in pinned[SourceKey("test/a@1.0.0")].files)
    with TimeFReader(open_local_version(child_dir)) as reader:
        assert reader.read().records[0].signals[0].to_arrow().to_pylist() == [1.0, None, 3.0]


@pytest.mark.parametrize("change", ["length", "spec"])
def test_a_reused_signal_must_keep_its_stored_shape(tmp_path, change):
    with TimeFReader(open_local_version(store(tmp_path, make_dataset()))) as reader:
        stored = reader.read().records[0].signals[0]
        changed = Signal.from_loader(
            name=stored.name,
            spec=stored.spec.model_copy(update={"name": "Renamed"}) if change == "spec" else stored.spec,
            time_axis=stored.time_axis,
            n_values=stored.n_values - 1 if change == "length" else stored.n_values,
            loader=stored._values,
        )
        child = _release("test/child", [Record(sources=(Source(name="s", signals=(changed,)),))])
        with pytest.raises(TimeFValidationError, match=r"shared signal \d+ changed its storage shape"):
            store(tmp_path, child)


def _reports(directory, **kwargs):
    """Read each Record's name, values, time offsets, and report text from a release."""
    with TimeFReader(open_local_version(directory, **kwargs)) as reader:
        return {
            _name(record): (
                record.signals[0].to_numpy().tolist(),
                record.signals[0].time_offsets_us().tolist(),
                _text(record.annotations[0].value),
            )
            for record in reader.read().records
        }


def _text(value):
    assert isinstance(value, TextHandle)
    return value.read()


def _parent_and_child(root, backend="parquet", text="Parent report. "):
    """Store a parent release and a child that reuses all its payloads, under one registry root."""
    parent = _release("test/p", [_record_in_memory("p", text * 300, [0, 1_000, 5_000])])
    parent_dir = store(root, parent, values_backend=backend)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        loaded = reader.read()
        child_dir = store(root, _derive(loaded, "test/c", loaded.records))
    return parent_dir, child_dir


def _pins(directory):
    return Manifest.model_validate_json((directory / "manifest.json").read_text()).payload_sources


@pytest.mark.parametrize("backend", ["parquet", "zarr"])
def test_a_child_pins_exactly_the_source_files_it_reads(tmp_path, backend):
    parent_dir, child_dir = _parent_and_child(tmp_path, backend)
    recorded = {
        part.path: part
        for part in Manifest.model_validate_json((parent_dir / "manifest.json").read_text()).files.all_files()
    }
    with TimeFReader(open_local_version(child_dir)) as reader:
        (record,) = reader.read().records
        signal, report = record.signals[0], record.annotations[0].value
        assert isinstance(signal._values, StoredArray)
        assert isinstance(signal._time_offsets, StoredOffsets)
        assert isinstance(report, StoredText)
        read = {chunk.chunk_file for chunk in signal._values._location.chunks}
        read |= {signal._time_offsets._location.path, report._location.path}
    # A Zarr array is a directory of files, so the child pins every file under the arrays it reads.
    expected = {
        path for path in recorded if any(path == read_path or path.startswith(f"{read_path}/") for read_path in read)
    }
    (pinned,) = _pins(child_dir).values()
    assert pinned.release == DatasetRef.parse("test/p@1.0.0")
    assert [part.path for part in pinned.files] == sorted(expected)
    assert all(part == recorded[part.path] for part in pinned.files)
    assert "control.duckdb" not in expected


def test_opening_a_child_detects_a_forced_replace_of_its_source(tmp_path):
    _, child_dir = _parent_and_child(tmp_path)
    LocalRegistry(tmp_path).store(
        _release("test/p", [_record_in_memory("p", "Replaced. " * 300, [0, 2_000])]), force=True
    )
    first = min(part.path for part in _pins(child_dir)["test/p@1.0.0"].files)
    message = rf"payload source test/p@1\.0\.0 differs from the release test/c@1\.0\.0 was written against: test/c@1\.0\.0 pins '{first}' as \d+ bytes"
    with pytest.raises(TimeFFormatError, match=message):
        open_local_version(child_dir)
    with pytest.raises(TimeFFormatError, match=message):
        LocalRegistry(tmp_path).open_version("test/c", "1.0.0")


def test_storing_a_child_detects_a_forced_replace_of_its_source(tmp_path):
    parent_dir = store(tmp_path, _release("test/p", [_record_in_memory("p", "Parent. " * 300, [0, 1_000])]))
    registry = LocalRegistry(tmp_path)
    with TimeFReader(open_local_version(parent_dir)) as reader:
        loaded = reader.read()
        registry.store(_release("test/p", [_record_in_memory("p", "Replaced. " * 300, [0, 2_000])]), force=True)
        # Both releases store the same values, so their shards match; the time offsets differ, and the
        # error names that payload file.
        with pytest.raises(
            TimeFFormatError,
            match=rf"payload source test/p@1\.0\.0 in local registry {tmp_path} differs from the release the reused "
            r"values were read from: they read 'payloads/part-00000000\.parquet' as \d+ bytes with checksum "
            r"sha256:[0-9a-f]{64}, but test/p@1\.0\.0 there records \d+ bytes with checksum sha256:[0-9a-f]{64}$",
        ):
            registry.store(_derive(loaded, "test/c", loaded.records))
    assert not (tmp_path / "test/c").exists() or not list((tmp_path / "test/c").iterdir())


def test_a_child_is_refused_by_a_registry_that_holds_another_release_of_its_source(tmp_path):
    ours, foreign = LocalRegistry(tmp_path / "ours"), LocalRegistry(tmp_path / "foreign")
    ours.store(_release("test/p", [_record_in_memory("p", "Ours. " * 300, [0, 1_000])]))
    foreign.store(_release("test/p", [_record_in_memory("p", "Foreign. " * 300, [0, 3_000])]))
    with TimeFReader(ours.open_version("test/p", "1.0.0")) as reader:
        loaded = reader.read()
        with pytest.raises(
            TimeFFormatError,
            match=r"payload source test/p@1\.0\.0 in local registry .*foreign differs from the release",
        ):
            foreign.store(_derive(loaded, "test/c", loaded.records))
        assert not foreign.exists("test/c", "1.0.0")
        ours.store(_derive(loaded, "test/c", loaded.records))
    assert _reports(tmp_path / "ours/test/c/1.0.0") == {"p": ([0.0, 1.0], [0, 1_000], "Ours. " * 300)}


def test_a_write_fails_at_the_first_reused_source_and_publishes_nothing(tmp_path):
    parent_dir = store(tmp_path / "parents", _release("test/p", [_record_in_memory("p", "Parent. " * 300, [0, 1_000])]))
    elsewhere = tmp_path / "elsewhere"

    def unexpected_read():
        pytest.fail("a write that fails at its first reused source must not write other values")

    own = Signal.from_loader(
        name="own", spec=_SPEC, time_axis=RegularAxis.from_rate_hz(1), n_values=2, loader=unexpected_read
    )
    with TimeFReader(open_local_version(parent_dir)) as reader:
        loaded = reader.read()
        child = _derive(loaded, "test/c", loaded.records)
        child.add_record(record=Record(sources=(Source(name="own", signals=(own,)),)))
        child.derive_schema()
        with (
            pytest.raises(
                TimeNetDatasetNotFoundError, match=r"payload source test/p@1\.0\.0 of test/c@1\.0\.0 is not in"
            ),
            TimeFWriter(elsewhere, child) as writer,
        ):
            writer.write()
    # The write failed before it wrote any values, and the aborted writer left nothing behind.
    assert not [path for path in elsewhere.rglob("*") if path.is_file()]
    assert not (elsewhere / "test/c/1.0.0").exists()


def test_a_release_cannot_reuse_payloads_of_a_source_that_reads_it(tmp_path):
    registry = LocalRegistry(tmp_path)
    parent_dir = store(tmp_path, _release("test/p", [_record_in_memory("p", "Parent. " * 300, [0, 1_000])]))
    with TimeFReader(open_local_version(parent_dir)) as reader:
        loaded = reader.read()
        child = _derive(loaded, "test/c", loaded.records)
        child.add_record(record=_record_in_memory("c", "Child. " * 300, [0, 7_000]))
        child_dir = store(tmp_path, child)
    before = {name: _reports(directory) for name, directory in (("p", parent_dir), ("c", child_dir))}
    with TimeFReader(open_local_version(child_dir)) as reader:
        loaded = reader.read()
        own = [record for record in loaded.records if _name(record) == "c"]
        # A forced P@1 built from C's own payloads would make C, which reads P@1, a source of P@1.
        with pytest.raises(
            TimeFValidationError, match=r"test/c@1\.0\.0, which reads payloads of test/p@1\.0\.0 itself"
        ):
            registry.store(_derive(loaded, "test/p", own), force=True)
    assert {name: _reports(directory) for name, directory in (("p", parent_dir), ("c", child_dir))} == before


def test_opening_names_a_missing_source_and_the_root_it_searched(tmp_path):
    parent_dir, child_dir = _parent_and_child(tmp_path)
    shutil.rmtree(parent_dir)
    with pytest.raises(
        TimeNetDatasetNotFoundError, match=rf"payload source test/p@1\.0\.0 of test/c@1\.0\.0 is not in {tmp_path}$"
    ):
        open_local_version(child_dir)


@pytest.mark.parametrize("damage", ["missing", "truncated"])
@pytest.mark.parametrize("release", ["child", "source"])
def test_opening_names_a_missing_or_truncated_file_with_its_sizes(tmp_path, release, damage):
    parent_dir, child_dir = _parent_and_child(tmp_path)
    if release == "child":
        owner, part = (
            r"test/c@1\.0\.0",
            Manifest.model_validate_json((child_dir / "manifest.json").read_text()).files.control[0],
        )
        path = child_dir / part.path
    else:
        owner, part = r"payload source test/p@1\.0\.0 of test/c@1\.0\.0", _pins(child_dir)["test/p@1.0.0"].files[0]
        path = parent_dir / part.path
    if damage == "missing":
        path.unlink()
        message = rf"{owner}: file '{part.path}' is missing; expected {part.size} bytes"
    else:
        path.write_bytes(path.read_bytes()[:-1])
        message = rf"{owner}: file '{part.path}' has {part.size - 1} bytes; expected {part.size}"
    with pytest.raises(TimeFFormatError, match=message):
        open_local_version(child_dir)


def test_a_grandchild_opens_without_the_sources_of_its_sources(tmp_path):
    a_dir = store(tmp_path, _release("test/a", [_record_in_memory("a", "Stored by A. " * 300, [0, 1_000])]))
    with TimeFReader(open_local_version(a_dir)) as reader:
        a = reader.read()
        b = _derive(a, "test/b", a.records)
        b.add_record(record=_record_in_memory("b", "Stored by B. " * 300, [0, 2_000, 3_000]))
        b_dir = store(tmp_path, b)
    with TimeFReader(open_local_version(b_dir)) as reader:
        b = reader.read()
        c_dir = store(tmp_path, _derive(b, "test/c", [record for record in b.records if _name(record) == "b"]))
    # C reads only B's own payloads, so A is not one of its sources, though B reads A.
    assert list(_pins(c_dir)) == ["test/b@1.0.0"]
    shutil.rmtree(a_dir)
    assert _reports(c_dir) == {"b": ([0.0, 1.0, 2.0], [0, 2_000, 3_000], "Stored by B. " * 300)}
    with pytest.raises(TimeNetDatasetNotFoundError, match=r"payload source test/a@1\.0\.0 of test/b@1\.0\.0"):
        open_local_version(b_dir)


def test_a_child_moved_with_its_sources_to_another_root_keeps_working(tmp_path):
    _, child_dir = _parent_and_child(tmp_path / "before")
    expected = _reports(child_dir)
    shutil.copytree(tmp_path / "before", tmp_path / "after")
    shutil.rmtree(tmp_path / "before")
    assert _reports(tmp_path / "after/test/c/1.0.0") == expected


def test_sources_root_opens_a_child_outside_the_registry_layout(tmp_path):
    _, child_dir = _parent_and_child(tmp_path / "registry")
    loose = tmp_path / "loose"
    shutil.copytree(child_dir, loose)
    with pytest.raises(TimeFValidationError, match=r"open it from <root>/test/c/1\.0\.0 or pass sources_root="):
        open_local_version(loose)
    assert _reports(loose, sources_root=tmp_path / "registry") == _reports(child_dir)


def test_a_release_without_sources_opens_from_any_directory(tmp_path):
    parent_dir, _ = _parent_and_child(tmp_path / "registry")
    shutil.copytree(parent_dir, tmp_path / "anywhere")
    assert _reports(tmp_path / "anywhere") == _reports(parent_dir)


def test_verify_hashes_the_files_a_child_reuses(tmp_path):
    parent_dir, child_dir = _parent_and_child(tmp_path)
    with TimeFReader(open_local_version(child_dir)) as reader:
        reader.verify()
    part = next(part for part in _pins(child_dir)["test/p@1.0.0"].files if part.path.startswith("payloads/"))
    payload = parent_dir / part.path
    data = payload.read_bytes()
    payload.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    # The size still matches, so opening, which reads only metadata, accepts the corrupted file.
    with (
        TimeFReader(open_local_version(child_dir)) as reader,
        pytest.raises(
            TimeFFormatError, match=rf"checksum mismatch for {part.path} of payload source test/p@1\.0\.0: pinned"
        ),
    ):
        reader.verify()


def test_a_download_fetches_only_the_direct_payload_sources(tmp_path):
    registry_root = tmp_path / "registry"
    a_dir = store(registry_root, _release("test/a", [_record_in_memory("a", "Stored by A. " * 300, [0, 1_000])]))
    with TimeFReader(open_local_version(a_dir)) as reader:
        a = reader.read()
        b = _derive(a, "test/b", a.records)
        b.add_record(record=_record_in_memory("b", "Stored by B. " * 300, [0, 2_000]))
        b_dir = store(registry_root, b)
    with TimeFReader(open_local_version(b_dir)) as reader:
        b = reader.read()
        store(registry_root, _derive(b, "test/c", [record for record in b.records if _name(record) == "b"]))
    cache = tmp_path / "cache"
    LocalRegistry(registry_root).download_version("test/c", "1.0.0", cache / "test/c/1.0.0")
    assert sorted(path.parent.relative_to(cache).as_posix() for path in cache.rglob("manifest.json")) == [
        "test/b/1.0.0",
        "test/c/1.0.0",
    ]
    assert _reports(cache / "test/c/1.0.0") == {"b": ([0.0, 1.0], [0, 2_000], "Stored by B. " * 300)}
