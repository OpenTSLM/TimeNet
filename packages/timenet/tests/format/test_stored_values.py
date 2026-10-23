import copy
import dataclasses
import inspect
import pickle
from typing import get_type_hints

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import IrregularAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.duckdb import connect_control
from timenet.format.storage import StorageSources
from timenet.format.stored_array import ChunkLocation, StoredArray
from timenet.format.stored_values import (
    CatalogValue,
    ExternalValue,
    PackedField,
    PackedLocation,
    PackedValueKind,
    PackedValueLocator,
    PackedValueReader,
    PackedValueWriter,
    StoredOffsets,
    StoredText,
    StoredTextList,
    offsets_digest,
)
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ArrayHandle,
    ObjectKind,
    TextHandle,
    TextListHandle,
    TimeSeriesSpec,
    ValueHandle,
    Version,
    ureg,
)
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray
from timenet.types.reference_keys import OwnerKey
from timenet.types.text import InlineText, InlineTextList
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report with café notes. " * 300
_TEXTS = [_TEXT, "short report"]


def _write(root, prompt=_TEXT):
    dataset = make_dataset()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.prompt = prompt
    answer.inputs[0].annotate(Annotation(id="reports", key="reports", value=_TEXTS))
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _stored(reader):
    dataset = reader.read()
    (prompt,) = [task.prompt for task in dataset.tasks if isinstance(task.prompt, StoredText)]
    (reports,) = [
        annotation.value
        for record in dataset.records
        for annotation in record.annotations
        if annotation.key == "reports"
    ]
    return prompt, reports


@pytest.fixture
def reader(tmp_path):
    with TimeFReader(open_local_version(_write(tmp_path))) as reader:
        yield reader


@pytest.fixture
def payload_reads(monkeypatch):
    reads = []
    original = PackedValueReader.read_scalar

    def read_scalar(self, location):
        reads.append(location)
        return original(self, location)

    monkeypatch.setattr(PackedValueReader, "read_scalar", read_scalar)
    return reads


def test_the_stored_kind_picks_the_handle_class(reader):
    prompt, reports = _stored(reader)
    assert isinstance(prompt, StoredText)
    assert isinstance(prompt, TextHandle)
    assert not isinstance(prompt, TextListHandle)
    assert isinstance(reports, StoredTextList)
    assert isinstance(reports, TextListHandle)
    assert prompt.read() == _TEXT
    assert reports.read() == _TEXTS


def test_a_stored_handle_carries_its_packed_location(reader):
    prompt, reports = _stored(reader)
    assert isinstance(prompt._location, PackedLocation)
    assert prompt._location.kind is PackedValueKind.TEXT
    assert prompt._location.length == len(_TEXT)
    assert reports._location.kind is PackedValueKind.TEXT_LIST
    assert reports._location.length == len(_TEXTS)
    assert prompt._location.path == reports._location.path
    assert prompt._location.row != reports._location.row
    assert not hasattr(prompt, "__dict__")
    assert "PackedValueReader" not in repr(prompt)


@pytest.mark.parametrize(
    ("handle_type", "location_type"),
    [
        (ValueHandle, None),
        (NumpyArray, None),
        (ArrowArray, None),
        (CallableArray, None),
        (InlineText, None),
        (InlineTextList, None),
        (StoredText, PackedLocation),
        (StoredTextList, PackedLocation),
        (StoredOffsets, PackedLocation),
        (StoredArray, ChunkLocation),
    ],
)
def test_type_hints_resolve_on_every_handle_class(handle_type, location_type):
    # The base declares _location for type checkers only, so timenet.types needs no format names.
    hints = get_type_hints(handle_type)
    assert "_location" not in hints
    location = inspect.getattr_static(handle_type, "_location")
    if location_type is None:
        assert location is None
    else:
        # A stored handle gives its location through a property, which type checkers accept as
        # an override of the base's read-only one.
        assert get_type_hints(location.fget)["return"] is location_type


def test_the_annotation_table_returns_a_packed_string_list(reader):
    rows = reader.read_annotation_table(ObjectKind.RECORD).to_pylist()
    (reports,) = [row for row in rows if row["key"] == "reports"]
    assert reports["text_list_value"] == _TEXTS


def test_stored_text_is_never_loaded_and_reads_through_every_time(reader, payload_reads):
    prompt, reports = _stored(reader)
    assert not prompt.is_loaded
    assert not reports.is_loaded
    assert prompt.read() == prompt.read() == _TEXT
    assert len(payload_reads) == 2
    assert not prompt.is_loaded


def test_length_and_truth_read_no_payload(reader, payload_reads):
    prompt, reports = _stored(reader)
    assert prompt
    assert reports
    assert (prompt.length, reports.length) == (len(_TEXT), len(_TEXTS))
    assert not payload_reads


def test_handles_from_one_reader_at_one_location_are_equal_without_a_read(reader, payload_reads):
    prompt, reports = _stored(reader)
    # Closing drops the reader's cached objects, so the next read builds new handles.
    reader.close()
    again, reports_again = _stored(reader)
    assert again is not prompt
    assert again == prompt
    assert reports_again == reports
    assert copy.copy(prompt) == prompt
    assert prompt != reports
    assert not payload_reads


def test_stored_text_compares_by_content_with_inline_text_and_native_values(reader, payload_reads):
    prompt, reports = _stored(reader)
    assert prompt == InlineText(_TEXT)
    assert prompt == _TEXT
    assert _TEXT == prompt  # noqa: SIM300 - the reflected comparison is the point
    assert reports == InlineTextList(_TEXTS)
    assert reports == _TEXTS
    assert prompt != _TEXT[:-1] + "!"
    assert prompt != [_TEXT]
    assert len(payload_reads) == 6


def test_handles_at_one_location_in_different_releases_compare_by_content(tmp_path):
    # Both prompts land in the first row of the first payload file and have the same length.
    with (
        TimeFReader(open_local_version(_write(tmp_path / "a"))) as first,
        TimeFReader(open_local_version(_write(tmp_path / "b", prompt=_TEXT.upper()))) as second,
    ):
        (prompt, _), (other, _) = _stored(first), _stored(second)
        assert prompt._location == other._location
        assert prompt != other
        assert other == _TEXT.upper()


def test_stored_handles_are_not_hashable(reader):
    for handle in _stored(reader):
        with pytest.raises(TypeError, match="unhashable"):
            hash(handle)


def test_a_stored_handle_pickles_its_location_without_its_text(reader):
    prompt, reports = _stored(reader)
    for handle, value in ((prompt, _TEXT), (reports, _TEXTS)):
        pickled = pickle.dumps(handle)
        assert b"clinical report" not in pickled
        back = pickle.loads(pickled)
        assert type(back) is type(handle)
        assert back._location == handle._location
        assert not back.is_loaded
        assert back.read() == value


def test_a_deep_copy_shares_the_stored_handle(reader):
    prompt, reports = _stored(reader)
    assert copy.deepcopy(prompt) is prompt
    assert copy.deepcopy(reports) is reports


def test_writing_stored_text_reuses_its_payload_row_without_reading_it(tmp_path, payload_reads):
    source = _write(tmp_path)
    with TimeFReader(open_local_version(source)) as reader:
        dataset = reader.read()
        prompt, reports = _stored(reader)
        (record,) = [record for record in dataset.records if any(a.key == "reports" for a in record.annotations)]
        child = TimeFDataset(metadata=dataset.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)}))
        child.add_record(record=copy.deepcopy(record))
        child.add_task(task=AnswerTask(id="question", inputs=child.records, prompt=prompt, targets=(prompt,)))
        child.derive_schema()
        with TimeFWriter(tmp_path, child) as writer:
            writer.write()
    assert not payload_reads

    target = tmp_path / child.metadata.dataset_id / str(child.metadata.dataset_version)
    owner = OwnerKey(f"{dataset.metadata.dataset_id}@{dataset.metadata.dataset_version}")
    manifest = Manifest.model_validate_json((target / "manifest.json").read_text())
    assert not manifest.files.payloads
    assert list(manifest.payload_sources) == [owner]
    with TimeFReader(open_local_version(target)) as reader:
        child_prompt, child_reports = _stored(reader)
        # The child's locations name the parent payload rows, which the child reads through its owner.
        assert child_prompt._location == dataclasses.replace(prompt._location, owner=owner)
        assert child_reports._location == dataclasses.replace(reports._location, owner=owner)
        assert child_prompt.read() == _TEXT
        assert child_reports.read() == _TEXTS


@pytest.mark.parametrize(
    ("assignment", "condition", "stored", "message"),
    [
        ("row_index = 99", "field = 'prompt'", 0, "row out of bounds"),
        ("value_kind = 'text'", "value_kind = 'texts'", 1, "missing external text"),
        ("n_values = n_values + 1", "field = 'prompt'", 0, "has length"),
        ("n_values = n_values + 1", "value_kind = 'texts'", 1, "has length"),
    ],
)
def test_a_location_that_disagrees_with_its_payload_fails_the_read(tmp_path, assignment, condition, stored, message):
    version_dir = _write(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute(f"UPDATE external_values SET {assignment} WHERE {condition}")  # noqa: S608 - fixed test SQL

    with TimeFReader(version) as reader:
        handle = _stored(reader)[stored]
        with pytest.raises(TimeFFormatError, match=message):
            handle.read()


def test_a_payload_without_the_value_kind_column_fails_the_read(tmp_path):
    version_dir = _write(tmp_path)
    version = open_local_version(version_dir)
    (payload,) = (version_dir / "payloads").iterdir()
    pq.write_table(pq.read_table(payload).select(["text"]), payload)

    with TimeFReader(version) as reader:
        _, reports = _stored(reader)
        with pytest.raises(TimeFFormatError, match="unknown external value kind"):
            reports.read()


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("short", "short"),
        (InlineText("short"), "short"),
        (InlineTextList(["a", "b"]), ["a", "b"]),
        (3, 3),
        (None, None),
    ],
)
def test_values_that_fit_their_catalog_row_stay_there(tmp_path, value, kept):
    packed = PackedValueWriter(tmp_path, StorageSources()).pack(
        ObjectKind.ANNOTATION, "content", PackedField.VALUE, value
    )
    assert packed == CatalogValue(kept)


@pytest.mark.parametrize(("value", "kind"), [(_TEXT, PackedValueKind.TEXT), (_TEXTS, PackedValueKind.TEXT_LIST)])
def test_large_text_is_packed(tmp_path, value, kind):
    packed = PackedValueWriter(tmp_path, StorageSources()).pack(
        ObjectKind.ANNOTATION, "content", PackedField.VALUE, value
    )
    assert isinstance(packed, ExternalValue)
    assert packed.locator.value_kind is kind
    assert packed.locator.location == PackedLocation(
        path=packed.locator.value_path, row=0, kind=kind, length=len(value)
    )


_OFFSETS_US = [0, 5_000, 9_000]


def _write_irregular(root):
    dataset = make_dataset()
    axis = IrregularAxis.spanning(_OFFSETS_US)
    spec = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless)
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=spec,
            time_axis=axis,
            loader=lambda: pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            n_values=len(_OFFSETS_US),
            time_offsets_loader=lambda: pa.array(_OFFSETS_US, type=pa.int64()),
        )
        for name in ("temp", "humidity")
    )
    dataset.add_record(record=Record(record_id="irregular", sources=(Source(name="env", signals=signals),)))
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _stored_offsets(reader):
    (record,) = reader.iter_records(["irregular"])
    temp, humidity = record.signals
    assert isinstance(temp._time_offsets, StoredOffsets)
    assert temp._time_offsets is humidity._time_offsets
    return temp._time_offsets


@pytest.fixture
def irregular_reader(tmp_path):
    with TimeFReader(open_local_version(_write_irregular(tmp_path))) as reader:
        yield reader


def test_stored_offsets_carry_their_packed_location(irregular_reader, payload_reads):
    handle = _stored_offsets(irregular_reader)
    assert isinstance(handle, ArrayHandle)
    assert handle._location.kind is PackedValueKind.TIME_OFFSETS
    assert handle._location.digest == offsets_digest(np.asarray(_OFFSETS_US, dtype=np.int64))
    assert handle.length == handle._location.length == len(_OFFSETS_US)
    assert (handle.first, handle.last) == (_OFFSETS_US[0], _OFFSETS_US[-1])
    assert not hasattr(handle, "__dict__")
    assert "PackedValueReader" not in repr(handle)
    assert not payload_reads


def test_stored_offsets_give_the_digest_their_location_records(irregular_reader, payload_reads):
    handle = _stored_offsets(irregular_reader)
    assert handle.digest == handle._location.digest == offsets_digest(np.asarray(_OFFSETS_US, dtype=np.int64))
    assert not payload_reads

    digestless = dataclasses.replace(handle, _packed_location=dataclasses.replace(handle._location, digest=None))
    with pytest.raises(TimeFFormatError, match="has time offsets without a digest"):
        _ = digestless.digest


def test_stored_offsets_are_loaded_by_their_first_read_and_keep_the_checked_time_offsets(
    irregular_reader, payload_reads, monkeypatch
):
    checks = []
    check = StoredOffsets._check

    def counting_check(self, offsets):
        checks.append(self.axis_id)
        check(self, offsets)

    monkeypatch.setattr(StoredOffsets, "_check", counting_check)
    handle = _stored_offsets(irregular_reader)
    assert not handle.is_loaded

    time_offsets = handle.read()
    assert handle.is_loaded
    assert time_offsets.to_pylist() == _OFFSETS_US
    assert handle.read() is time_offsets
    assert handle.read_numpy().tolist() == _OFFSETS_US
    assert handle.read_steps(1, 3).to_pylist() == _OFFSETS_US[1:]
    assert len(payload_reads) == 1
    assert checks == [handle.axis_id]


def test_stored_offsets_pickle_their_location_and_only_kept_time_offsets(irregular_reader, payload_reads):
    handle = _stored_offsets(irregular_reader)
    unloaded = pickle.loads(pickle.dumps(handle))
    assert type(unloaded) is StoredOffsets
    assert unloaded._location == handle._location
    assert not unloaded.is_loaded

    handle.read()
    loaded = pickle.loads(pickle.dumps(handle))
    assert loaded.is_loaded
    assert loaded.read().to_pylist() == _OFFSETS_US
    assert len(payload_reads) == 1
    assert unloaded.read().to_pylist() == _OFFSETS_US
    assert len(payload_reads) == 2


def test_a_deep_copy_shares_the_stored_offsets(irregular_reader):
    handle = _stored_offsets(irregular_reader)
    assert copy.deepcopy(handle) is handle


def test_text_handles_are_not_built_for_stored_time_offsets(irregular_reader):
    handle = _stored_offsets(irregular_reader)
    with pytest.raises(TimeFValidationError, match="read through StoredOffsets"):
        handle.store.handle(handle._location)


def test_time_offsets_that_disagree_with_their_location_fail_the_read(tmp_path):
    version_dir = _write_irregular(tmp_path)
    version = open_local_version(version_dir)
    for payload in (version_dir / "payloads").iterdir():
        table = pq.read_table(payload)
        rows = table.to_pylist()
        for row in rows:
            if row["time_offsets"] is not None:
                del row["time_offsets"][1]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), payload)

    with TimeFReader(version) as reader:
        handle = _stored_offsets(reader)
        with pytest.raises(TimeFFormatError, match="stores 2 offsets but its location records 3"):
            handle.read()
        assert not handle.is_loaded


def test_kept_time_offsets_hold_only_their_axis(tmp_path):
    # Every axis lands in one payload file, whose time offset column the kept array must not pin.
    n_steps = 10_000
    dataset = make_dataset()
    signals = tuple(
        Signal.from_irregular(
            np.zeros(n_steps, dtype=np.float32),
            time_offsets_us=np.arange(n_steps, dtype=np.int64) * 1_000 + axis,
            spec=TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless),
            name=f"axis-{axis}",
        )
        for axis in range(5)
    )
    dataset.add_record(record=Record(record_id="irregular", sources=(Source(name="env", signals=signals),)))
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    (payload,) = (version_dir / "payloads").iterdir()
    assert pq.read_table(payload)["time_offsets"].combine_chunks().flatten().nbytes == 5 * n_steps * 8

    with TimeFReader(open_local_version(version_dir)) as reader:
        (record,) = reader.iter_records(["irregular"])
        handle = record.signals[2]._time_offsets
        assert isinstance(handle, StoredOffsets)
        kept = handle.read()
        assert kept.to_numpy()[0] == 2
        assert kept.offset == 0
        assert kept.get_total_buffer_size() == kept.nbytes == n_steps * 8
        assert len(pickle.dumps(handle)) < 2 * n_steps * 8


@pytest.mark.parametrize(
    ("object_kind", "field", "value_kind", "digest"),
    [
        (ObjectKind.AXIS, PackedField.TIME_OFFSETS, PackedValueKind.TIME_OFFSETS, None),
        (ObjectKind.TASK, PackedField.PROMPT, PackedValueKind.TEXT, "sha256:" + "0" * 64),
    ],
)
def test_only_time_offset_locators_carry_a_digest(object_kind, field, value_kind, digest):
    with pytest.raises(ValueError, match="time offsets require a digest; text values must not store one"):
        PackedValueLocator(
            object_kind=object_kind,
            object_id="object",
            field=field,
            position=0,
            value_path="payloads/part-00000000.parquet",
            row_index=0,
            value_kind=value_kind,
            n_values=1,
            value_digest=digest,
        )


@pytest.mark.parametrize(
    "offsets",
    [pa.array([0.0, 1.0]), pa.array([0, 1], type=pa.int32()), pa.array([0, None], type=pa.int64())],
)
def test_only_non_null_int64_time_offsets_are_packed(tmp_path, offsets):
    with pytest.raises(TimeFValidationError, match="time offsets must be non-null int64 values"):
        PackedValueWriter(tmp_path, StorageSources()).pack(ObjectKind.AXIS, "axis", PackedField.TIME_OFFSETS, offsets)
