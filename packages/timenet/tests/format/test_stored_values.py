import copy
import pickle
import shutil
from typing import get_type_hints

import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeFFormatError
from timenet.format.duckdb import connect_control
from timenet.format.stored_values import (
    CatalogValue,
    ExternalValue,
    PackedField,
    PackedLocation,
    PackedValueKind,
    PackedValueReader,
    PackedValueWriter,
    StoredText,
    StoredTextList,
)
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, ObjectKind, TextHandle, TextListHandle, ValueHandle
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray
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
    with TimeFReader(DatasetVersion.open_local(_write(tmp_path))) as reader:
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
    ],
)
def test_type_hints_resolve_on_every_handle_class(handle_type, location_type):
    # The base declares _location for type checkers only, so timenet.types needs no format names.
    assert get_type_hints(handle_type).get("_location") is location_type


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
        TimeFReader(DatasetVersion.open_local(_write(tmp_path / "a"))) as first,
        TimeFReader(DatasetVersion.open_local(_write(tmp_path / "b", prompt=_TEXT.upper()))) as second,
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


def test_writing_a_read_release_packs_its_stored_text_into_the_new_release(tmp_path):
    source = _write(tmp_path / "a")
    with TimeFReader(DatasetVersion.open_local(source)) as reader:
        dataset = reader.read()
        dataset.derive_schema()
        with TimeFWriter(tmp_path / "b", dataset) as writer:
            writer.write()
    shutil.rmtree(source)

    target = tmp_path / "b" / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(target)) as reader:
        prompt, reports = _stored(reader)
        assert prompt == _TEXT
        assert reports == _TEXTS


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
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute(f"UPDATE external_values SET {assignment} WHERE {condition}")  # noqa: S608 - fixed test SQL

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        handle = _stored(reader)[stored]
        with pytest.raises(TimeFFormatError, match=message):
            handle.read()


def test_a_payload_without_the_value_kind_column_fails_the_read(tmp_path):
    version_dir = _write(tmp_path)
    (payload,) = (version_dir / "payloads").iterdir()
    pq.write_table(pq.read_table(payload).select(["text"]), payload)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    packed = PackedValueWriter(tmp_path).pack(ObjectKind.ANNOTATION, "content", PackedField.VALUE, value)
    assert packed == CatalogValue(kept)


@pytest.mark.parametrize(("value", "kind"), [(_TEXT, PackedValueKind.TEXT), (_TEXTS, PackedValueKind.TEXT_LIST)])
def test_large_text_is_packed(tmp_path, value, kind):
    packed = PackedValueWriter(tmp_path).pack(ObjectKind.ANNOTATION, "content", PackedField.VALUE, value)
    assert isinstance(packed, ExternalValue)
    assert packed.locator.value_kind is kind
    assert packed.locator.location == PackedLocation(
        path=packed.locator.value_path, row=0, kind=kind, length=len(value)
    )
