import copy
import dataclasses
import inspect
import json
from pathlib import Path
import pickle
import re
import shutil
from typing import get_type_hints

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import IrregularAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.control_audit import audit_control_database
import timenet.format.control_writer as control_writer_module
from timenet.format.duckdb import connect_control
from timenet.format.storage import StorageSources
from timenet.format.stored_array import ChunkLocation, StoredArray
import timenet.format.stored_values as stored_values_module
from timenet.format.stored_values import (
    CatalogValue,
    ExternalValue,
    PackedContentKey,
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
from timenet.storage.parquet import PayloadTableCache
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ArrayHandle,
    DatasetSchema,
    ObjectKind,
    TextHandle,
    TextListHandle,
    TimeSeriesSpec,
    ValueHandle,
    Version,
    ureg,
)
from timenet.types.annotations import value_type_of
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray
from timenet.types.reference_keys import SourceKey
from timenet.types.text import InlineText, InlineTextList, read_value
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report with café notes. " * 300
_TEXTS = [_TEXT, "short report"]
# JSON-encoded with its quotes, the first takes exactly 4,096 bytes and the second one more.
_AT_LIMIT = "a" * 4094
_PAST_LIMIT = "a" * 4095
# Spelled out rather than imported, so a change to the format's payload schema fails a test.
_PAYLOAD_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("texts", pa.list_(pa.field("element", pa.string(), nullable=False))),
        ("time_offsets", pa.list_(pa.field("element", pa.int64(), nullable=False))),
    ]
)


def _write_dataset(root, dataset):
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _packed_dataset(prompt=_TEXT):
    dataset = make_dataset()
    answer = next(task for task in dataset.tasks if isinstance(task, AnswerTask))
    answer.prompt = prompt
    record = answer.inputs[0]
    # Two occurrences of one content put the occurrence ids ahead of the content ids, so a locator
    # keyed by the wrong id fails.
    marker = Annotation(key="marker", value=1)
    record.annotate(marker)
    record.annotate(marker)
    record.annotate(Annotation(key="reports", value=_TEXTS))
    return dataset, answer


def _write(root, prompt=_TEXT):
    return _write_dataset(root, _packed_dataset(prompt)[0])


def _external_rows(version_dir):
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        return connection.execute(
            "SELECT object_kind, object_id, field, position FROM external_values ORDER BY ALL"
        ).fetchall()


def _content_ids(version_dir, key):
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        rows = connection.execute("SELECT content_id FROM annotation_contents WHERE name = ?", [key]).fetchall()
    return [content_id for (content_id,) in rows]


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


def _child_of(dataset, records):
    """Return a derived version of a read dataset that keeps the given Records and their ids."""
    return TimeFDataset.from_parts(
        metadata=dataset.metadata.model_copy(update={"dataset_version": Version(1, 1, 0)}),
        records=records,
        tasks=(),
        schema=DatasetSchema(),
        last_ids=dataset.last_ids,
    )


def test_writing_stored_text_reuses_its_payload_row_without_reading_it(tmp_path, payload_reads):
    source = _write(tmp_path)
    with TimeFReader(open_local_version(source)) as reader:
        dataset = reader.read()
        prompt, reports = _stored(reader)
        (record,) = [record for record in dataset.records if any(a.key == "reports" for a in record.annotations)]
        child = _child_of(dataset, [record])
        child.add_task(task=AnswerTask(inputs=(record,), prompt=prompt, targets=(prompt,)))
        child.derive_schema()
        with TimeFWriter(tmp_path, child) as writer:
            writer.write()
    assert not payload_reads

    target = tmp_path / child.metadata.dataset_id / str(child.metadata.dataset_version)
    source = SourceKey(f"{dataset.metadata.dataset_id}@{dataset.metadata.dataset_version}")
    manifest = Manifest.model_validate_json((target / "manifest.json").read_text())
    assert not manifest.files.payloads
    assert list(manifest.payload_sources) == [source]
    with TimeFReader(open_local_version(target)) as reader:
        child_prompt, child_reports = _stored(reader)
        # The child's locations name the parent payload rows, which the child reads through its source.
        assert child_prompt._location == dataclasses.replace(prompt._location, source=source)
        assert child_reports._location == dataclasses.replace(reports._location, source=source)
        # The source's manifest gives the checksum, so the child's content keys are the parent's.
        assert child_prompt._content_key() == prompt._content_key()
        assert child_reports._content_key() == reports._content_key()
        assert child_prompt.read() == _TEXT
        assert child_reports.read() == _TEXTS


@pytest.mark.parametrize("second_reader", ["opened", "pickled"])
def test_child_tasks_find_parent_owned_annotations_on_records_from_another_reader(tmp_path, second_reader):
    dataset, answer = _packed_dataset(prompt="short")
    answer.inputs[0].annotate(Annotation(key="report", value=_TEXT))
    with TimeFReader(open_local_version(_write_dataset(tmp_path, dataset))) as reader:
        loaded = reader.read()
        (record,) = [record for record in loaded.records if record.id == answer.inputs[0].id]
        (report,) = [annotation for annotation in record.annotations if annotation.key == "report"]
        child = _child_of(loaded, [record])
        task = child.add_task(task=AnswerTask(inputs=(record,), prompt="Which report?", targets=("this one",)))
        task.input_annotations = (report,)
        target = _write_dataset(tmp_path, child)
    version = open_local_version(target)

    with TimeFReader(version) as first:
        records = list(first.iter_records())
        second = TimeFReader(version) if second_reader == "opened" else pickle.loads(pickle.dumps(first))
        with second:
            tasks = {task.id: task for task in second.iter_tasks(records)}

    (child_record,) = records
    (stored,) = [annotation for annotation in child_record.annotations if annotation.key == "report"]
    assert isinstance(stored.value, StoredText)
    assert isinstance(report.value, StoredText)
    assert stored.value._location.source is not None
    assert stored.value._content_key() == report.value._content_key()
    assert tasks[task.id].input_annotations[0] is stored


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


def test_written_payloads_have_the_payload_schema_and_read_back(tmp_path):
    version_dir = _write(tmp_path)
    for payload in (version_dir / "payloads").iterdir():
        assert pq.ParquetFile(payload).schema_arrow.equals(_PAYLOAD_SCHEMA)

    with TimeFReader(open_local_version(version_dir)) as reader:
        prompt, reports = _stored(reader)
        assert prompt.read() == _TEXT
        assert reports.read() == _TEXTS


def test_a_payload_rebuilt_with_the_payload_schema_still_reads(tmp_path):
    version_dir = _write(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    (payload,) = (version_dir / "payloads").iterdir()
    rows = pq.read_table(payload).to_pylist()
    # Another compression and one row per group: only the Arrow schema has to match.
    pq.write_table(pa.Table.from_pylist(rows, schema=_PAYLOAD_SCHEMA), payload, compression="snappy", row_group_size=1)

    with TimeFReader(version) as reader:
        prompt, reports = _stored(reader)
        assert prompt.read() == _TEXT
        assert reports.read() == _TEXTS


@pytest.mark.parametrize(
    ("rewrite", "found"),
    [
        pytest.param(
            lambda table: table.cast(table.schema.set(0, pa.field("text", pa.binary()))),
            "text: binary, texts: list<element: string not null>, time_offsets: list<element: int64 not null>",
            id="binary-text",
        ),
        pytest.param(
            lambda table: table.cast(table.schema.set(1, pa.field("texts", pa.list_(pa.string())))),
            "text: string, texts: list<element: string>, time_offsets: list<element: int64 not null>",
            id="nullable-list-elements",
        ),
        pytest.param(lambda table: table.select(["text"]), "text: string", id="missing-texts"),
    ],
)
def test_a_payload_with_another_schema_fails_every_read(tmp_path, rewrite, found):
    version_dir = _write(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    (payload,) = (version_dir / "payloads").iterdir()
    pq.write_table(rewrite(pq.read_table(payload)), payload)

    with TimeFReader(version) as reader:
        # A rejected file stays out of the cache, so each read checks it again.
        for handle in _stored(reader):
            with pytest.raises(TimeFFormatError, match=re.escape(f"payloads/{payload.name} has schema ({found}),")):
                handle.read()


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("short", "short"),
        (InlineText("short"), "short"),
        (InlineTextList(["a", "b"]), ["a", "b"]),
        (3, 3),
        (None, None),
        pytest.param(_AT_LIMIT, _AT_LIMIT, id="4096-bytes"),
    ],
)
def test_values_that_fit_their_catalog_row_stay_there(tmp_path, value, kept):
    packed = PackedValueWriter(tmp_path, StorageSources()).pack(
        ObjectKind.ANNOTATION_CONTENT, 1, PackedField.VALUE, value
    )
    assert packed == CatalogValue(kept)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (_TEXT, PackedValueKind.TEXT),
        (_TEXTS, PackedValueKind.TEXT_LIST),
        pytest.param(_PAST_LIMIT, PackedValueKind.TEXT, id="4097-bytes"),
    ],
)
def test_large_text_is_packed(tmp_path, value, kind):
    packed = PackedValueWriter(tmp_path, StorageSources()).pack(
        ObjectKind.ANNOTATION_CONTENT, 1, PackedField.VALUE, value
    )
    assert isinstance(packed, ExternalValue)
    assert packed.locator.value_kind is kind
    assert packed.locator.location == PackedLocation(
        path=packed.locator.value_path, row=0, kind=kind, length=len(value)
    )


def test_locators_name_their_owner_by_kind_and_int_id(tmp_path):
    dataset, answer = _packed_dataset()
    version_dir = _write_dataset(tmp_path, dataset)
    (content_id,) = _content_ids(version_dir, "reports")
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        (occurrence_id,) = connection.execute(
            "SELECT occurrence_id FROM annotation_occurrences WHERE content_id = ?", [content_id]
        ).fetchone() or (None,)
    assert occurrence_id != content_id

    assert _external_rows(version_dir) == sorted(
        [
            (ObjectKind.ANNOTATION_CONTENT.value, content_id, PackedField.VALUE.value, 0),
            (ObjectKind.TASK.value, answer.id, PackedField.PROMPT.value, 0),
        ]
    )
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        (object_id_type,) = connection.execute(
            "SELECT data_type FROM duckdb_columns() WHERE table_name = 'external_values' AND column_name = 'object_id'"
        ).fetchone() or (None,)
        assert object_id_type == "BIGINT"
        audit_control_database(connection)


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("object_id = 999", "external value 'prompt' refers to missing Task 999"),
        ("object_kind = 'Annotation'", r"invalid external_values\.object_kind value"),
    ],
)
def test_the_audit_rejects_a_locator_without_its_owner(tmp_path, assignment, message):
    version_dir = _write(tmp_path)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute(f"UPDATE external_values SET {assignment} WHERE field = 'prompt'")  # noqa: S608 - fixed test SQL
        with pytest.raises(TimeFFormatError, match=message):
            audit_control_database(connection)


def test_a_content_shared_by_occurrences_is_packed_once(tmp_path):
    dataset, _ = _packed_dataset(prompt="short")
    report = Annotation(key="report", value=_TEXT)
    for record in dataset.records:
        record.annotate(report)
    version_dir = _write_dataset(tmp_path, dataset)

    (content_id,) = _content_ids(version_dir, "report")
    assert (ObjectKind.ANNOTATION_CONTENT.value, content_id, PackedField.VALUE.value, 0) in _external_rows(version_dir)
    assert len(_external_rows(version_dir)) == 2


def test_a_stored_content_key_names_the_payload_file_row_and_kind_without_a_read(reader, payload_reads):
    prompt, reports = _stored(reader)
    version_dir = Path(prompt.store.version.root)
    for handle in (prompt, reports):
        key = handle._content_key()
        location = handle._location
        assert key == PackedContentKey(
            checksum=file_checksum(version_dir / location.path), row=location.row, kind=location.kind
        )
        hash(key)
    # The keys of fresh handles at the same location still agree.
    reader.close()
    again, reports_again = _stored(reader)
    assert again._content_key() == prompt._content_key()
    assert reports_again._content_key() == reports._content_key()
    assert prompt._content_key() != reports._content_key()
    assert not payload_reads


def test_a_stored_content_key_never_equals_an_inline_key(reader, payload_reads):
    prompt, reports = _stored(reader)
    assert prompt._content_key() != InlineText(_TEXT)._content_key()
    assert reports._content_key() != InlineTextList(_TEXTS)._content_key()
    stored = Annotation(key="reports", value=reports)
    inline = Annotation(key="reports", value=_TEXTS)
    assert value_type_of(stored.value) == value_type_of(inline.value)
    assert stored._content_key != inline._content_key
    assert not payload_reads


def test_one_location_in_two_payload_files_gives_two_content_keys(tmp_path):
    # Both prompts land in the first row of the first payload file, but their text differs.
    with (
        TimeFReader(open_local_version(_write(tmp_path / "a"))) as first,
        TimeFReader(open_local_version(_write(tmp_path / "b", prompt=_TEXT.upper()))) as second,
    ):
        (prompt, _), (other, _) = _stored(first), _stored(second)
        assert prompt._location == other._location
        assert prompt._content_key() != other._content_key()


def test_one_payload_file_in_two_releases_gives_one_content_key(tmp_path, payload_reads):
    source = _write(tmp_path / "a")
    # The same release in a second registry.
    twin = tmp_path / "b" / source.relative_to(tmp_path / "a")
    shutil.copytree(source, twin)
    with (
        TimeFReader(open_local_version(source)) as first,
        TimeFReader(open_local_version(twin)) as second,
    ):
        (prompt, reports), (other, other_reports) = _stored(first), _stored(second)
        assert prompt.store is not other.store
        assert prompt._content_key() == other._content_key()
        assert reports._content_key() == other_reports._content_key()
        # Equal keys mean equal text, so the handles are equal without a read.
        assert prompt == other
        assert reports == other_reports
    assert not payload_reads


@pytest.mark.parametrize("second_reader", ["opened", "pickled"])
def test_tasks_find_packed_annotations_on_records_from_another_reader(tmp_path, second_reader):
    dataset, answer = _packed_dataset(prompt="short")
    report = answer.inputs[0].annotate(Annotation(key="report", value=_TEXT))
    answer.input_annotations = (report,)
    version = open_local_version(_write_dataset(tmp_path, dataset))

    with TimeFReader(version) as first:
        records = list(first.iter_records())
        second = TimeFReader(version) if second_reader == "opened" else pickle.loads(pickle.dumps(first))
        with second:
            tasks = {task.id: task for task in second.iter_tasks(records)}

    (record,) = [record for record in records if record.id == answer.inputs[0].id]
    (stored,) = [annotation for annotation in record.annotations if annotation.key == "report"]
    assert isinstance(stored.value, StoredText)
    assert tasks[answer.id].input_annotations[0] is stored


def test_a_payload_file_outside_the_manifest_is_rejected(tmp_path):
    version_dir = _write(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("UPDATE external_values SET value_path = 'payloads/part-00000099.parquet'")

    with (
        TimeFReader(version) as reader,
        pytest.raises(TimeFFormatError, match=r"payloads/part-00000099\.parquet.* not listed in the manifest"),
    ):
        reader.read()


def test_rewriting_stored_text_at_one_location_keeps_one_content_row(tmp_path):
    dataset, _ = _packed_dataset(prompt="short")
    report = Annotation(key="report", value=_TEXT)
    for record in dataset.records:
        record.annotate(report)
    source = _write_dataset(tmp_path / "a", dataset)
    with TimeFReader(open_local_version(source)) as reader:
        loaded = reader.read()
        values = [
            annotation.value
            for record in loaded.records
            for annotation in record.annotations
            if annotation.key == "report"
        ]
        reports = [value for value in values if isinstance(value, StoredText)]
        assert len(reports) == len(values) == len(dataset.records) > 1
        assert len({value._content_key() for value in reports}) == 1
        # A new version beside the source, which stores the payloads it reuses.
        target = _write_dataset(tmp_path / "a", _child_of(loaded, loaded.records))

    # Each release keeps one content row for the shared report; the child names the parent's payload row.
    assert [len(_content_ids(release, "report")) for release in (source, target)] == [1, 1]
    assert len(_external_rows(target)) == 2
    with TimeFReader(open_local_version(target)) as reader:
        back = [
            annotation.value
            for record in reader.read().records
            for annotation in record.annotations
            if annotation.key == "report"
        ]
    assert back == [_TEXT] * len(dataset.records)


def _with_packed_second_target(tmp_path):
    dataset = make_dataset()
    task = dataset.add_task(task=AnswerTask(inputs=(dataset.records[0],), prompt="Report?", targets=("short", _TEXT)))
    return _write_dataset(tmp_path, dataset), task.id


def _targets(version, task_id):
    with TimeFReader(version) as reader:
        (task,) = [task for task in reader.read_tasks() if task.id == task_id]
        rows = [
            (row["position"], row["text_value"])
            for row in reader.read_target_table().to_pylist()
            if row["task_id"] == task_id
        ]
    assert task.targets is not None
    return [read_value(target) for target in task.targets], rows


def test_a_packed_target_after_the_first_keeps_its_position(tmp_path):
    version_dir, task_id = _with_packed_second_target(tmp_path)
    assert [row for row in _external_rows(version_dir) if row[1] == task_id and row[2] == "targets"] == [
        (ObjectKind.TASK.value, task_id, PackedField.TARGETS.value, 1)
    ]
    assert _targets(open_local_version(version_dir), task_id) == (["short", _TEXT], [(0, "short"), (1, _TEXT)])


def test_targets_are_matched_to_their_locators_by_stored_position(tmp_path):
    version_dir, task_id = _with_packed_second_target(tmp_path)
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("UPDATE task_targets SET position = position * 2 WHERE task_id = ?", [task_id])
        connection.execute(
            "UPDATE external_values SET position = position * 2 WHERE field = 'targets' AND object_id = ?", [task_id]
        )
    assert _targets(version, task_id) == (["short", _TEXT], [(0, "short"), (2, _TEXT)])


def test_two_stored_texts_of_one_kind_in_one_payload_compare_by_content(tmp_path, payload_reads):
    dataset = make_dataset()
    other = _TEXT.upper()
    assert len(other) == len(_TEXT)
    task = dataset.add_task(
        task=AnswerTask(inputs=(dataset.records[0],), prompt=_TEXT, rationale=other, targets=("yes",))
    )
    version_dir = _write_dataset(tmp_path, dataset)
    with TimeFReader(open_local_version(version_dir)) as reader:
        (stored,) = [stored for stored in reader.read_tasks() if stored.id == task.id]
        prompt, rationale = stored.prompt, stored.rationale
        assert isinstance(prompt, StoredText)
        assert isinstance(rationale, StoredText)
        assert prompt._location.path == rationale._location.path
        assert prompt._content_key() != rationale._content_key()
        assert prompt != rationale
    assert len(payload_reads) == 2


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        # Occurrences outnumber contents here, so the highest occurrence id names no content.
        (
            "UPDATE external_values SET object_id = (SELECT max(occurrence_id) FROM annotation_occurrences) "
            "WHERE object_kind = 'Content'",
            r"external value 'value' refers to missing Content \d+",
        ),
        (
            "UPDATE external_values SET position = 7 WHERE field = 'targets'",
            r"external target 7 of Task \d+ has no text target row",
        ),
        (
            "UPDATE task_targets SET target_kind = 'integer', text_value = NULL, integer_value = 5 WHERE position = 1",
            r"external target 1 of Task \d+ has no text target row",
        ),
        (
            "UPDATE annotation_contents SET value_kind = 'text' WHERE name = 'reports'",
            r"external value of Content \d+ is texts, but the content holds text",
        ),
        (
            "UPDATE tasks SET input_modalities = ['time_series'] "
            "WHERE task_id IN (SELECT object_id FROM external_values WHERE field = 'prompt')",
            r"task \d+ has invalid input_modalities",
        ),
    ],
)
def test_the_audit_checks_locators_against_their_rows(tmp_path, statement, message):
    dataset, answer = _packed_dataset()
    answer.targets = ("short", _TEXT)
    version_dir = _write_dataset(tmp_path, dataset)
    with connect_control(version_dir / "control.duckdb") as connection:
        audit_control_database(connection)
        connection.execute(statement)
        with pytest.raises(TimeFFormatError, match=message):
            audit_control_database(connection)


def test_a_release_without_payloads_reads_no_locators(tmp_path, monkeypatch):
    version_dir = _write_dataset(tmp_path, make_dataset())
    assert not (version_dir / "payloads").exists()

    def no_locators(*_args, **_kwargs):
        raise AssertionError("looked up locators")

    monkeypatch.setattr(PackedValueReader, "handles", no_locators)
    monkeypatch.setattr(PackedValueReader, "locations", no_locators)
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.read()
        list(reader.iter_tasks())
        reader.read_task_table()
        reader.read_target_table()
        reader.read_annotation_table()


def test_a_catalog_without_its_locator_table_fails_as_a_format_error(tmp_path):
    version_dir = _write(tmp_path)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("DROP TABLE external_values")

    with (
        TimeFReader(open_local_version(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="cannot read external values") as raised,
    ):
        reader.read()
    assert isinstance(raised.value.__cause__, duckdb.Error)


def test_the_writer_stores_locators_one_task_batch_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer_module, "_TASK_BATCH_SIZE", 2)
    held = []
    pack = PackedValueWriter.pack

    def counting_pack(self, *args, **kwargs):
        packed = pack(self, *args, **kwargs)
        held.append(len(self.locators))
        return packed

    monkeypatch.setattr(PackedValueWriter, "pack", counting_pack)
    dataset = make_dataset()
    template = dataset.tasks[0]
    record = template.inputs[0]
    tasks = [AnswerTask(inputs=(record,), prompt=f"{_TEXT}{index}", targets=("yes",)) for index in range(7)]
    streamed = TimeFDataset.from_parts(
        metadata=dataset.metadata,
        records=dataset.records,
        tasks=(),
        schema=dataset.derive_schema(),
        last_ids=dataset.last_ids,
    )
    streamed.set_task_stream([AnswerTask], lambda: iter(tasks))

    version_dir = _write_dataset(tmp_path, streamed)
    assert max(held) <= 2
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM external_values WHERE field = 'prompt'").fetchone() == (7,)
    with TimeFReader(open_local_version(version_dir)) as reader:
        assert {read_value(task.prompt) for task in reader.read_tasks()} == {f"{_TEXT}{index}" for index in range(7)}


def test_the_writer_stores_annotation_locators_in_bounded_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer_module, "_ROW_BATCH_SIZE", 2)
    held = []
    pack = PackedValueWriter.pack

    def counting_pack(self, *args, **kwargs):
        packed = pack(self, *args, **kwargs)
        held.append(len(self.locators))
        return packed

    monkeypatch.setattr(PackedValueWriter, "pack", counting_pack)
    dataset = make_dataset()
    for index in range(5):
        dataset.records[0].annotate(Annotation(key=f"report-{index}", value=f"{_TEXT}{index}"))

    version_dir = _write_dataset(tmp_path, dataset)
    assert max(held) <= 2
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM external_values WHERE object_kind = 'Content'").fetchone() == (
            5,
        )
        audit_control_database(connection)


def _release_with_payload_files(tmp_path, monkeypatch, count):
    # Each packed value starts a new payload file.
    monkeypatch.setattr(stored_values_module, "_PACK_BYTES", 1)
    dataset = make_dataset()
    for index in range(count):
        dataset.records[0].annotate(Annotation(key=f"report-{index}", value=f"{_TEXT}{index}"))
    version = open_local_version(_write_dataset(tmp_path, dataset))
    paths = [part.path for part in version.manifest.files.payloads]
    assert len(paths) == count
    return version, paths


def test_the_payload_cache_keeps_decoded_files_within_its_byte_budget(tmp_path, monkeypatch):
    version, paths = _release_with_payload_files(tmp_path, monkeypatch, 4)
    sizes = [pq.read_table(Path(version.root) / path).nbytes for path in paths]
    cache = PayloadTableCache(sizes[-1] + sizes[-2])
    for path in paths:
        cache.read_table(version, path, _PAYLOAD_SCHEMA)
        assert cache.nbytes <= sizes[-1] + sizes[-2]
    assert list(cache._tables) == [(version.root, path) for path in paths[-2:]]
    assert cache.nbytes == sizes[-1] + sizes[-2]
    cache.close()
    assert cache.nbytes == 0


def test_the_payload_cache_keeps_the_newest_file_past_its_budget(tmp_path, monkeypatch):
    version, paths = _release_with_payload_files(tmp_path, monkeypatch, 3)
    cache = PayloadTableCache(1)
    for path in paths:
        table = cache.read_table(version, path, _PAYLOAD_SCHEMA)
        assert cache.read_table(version, path, _PAYLOAD_SCHEMA) is table
        assert list(cache._tables) == [(version.root, path)]


def test_a_reader_keeps_decoded_payloads_within_its_byte_budget(tmp_path, monkeypatch):
    version, paths = _release_with_payload_files(tmp_path, monkeypatch, 6)
    budget = 2 * pq.read_table(Path(version.root) / paths[0]).nbytes
    monkeypatch.setattr(stored_values_module, "_CACHE_BYTES", budget)
    with TimeFReader(version) as reader:
        reports = [
            annotation.value
            for record in reader.read().records
            for annotation in record.annotations
            if annotation.key.startswith("report-")
        ]
        assert len(reports) == len(paths)
        packed = reader._packed_value_reader
        assert packed is not None
        cache = packed._parquet
        for index, report in enumerate(reports):
            assert read_value(report) == f"{_TEXT}{index}"
            assert cache.nbytes <= budget


def test_the_payload_cache_evicts_the_least_recently_read_file(tmp_path, monkeypatch):
    version, (first, second, third) = _release_with_payload_files(tmp_path, monkeypatch, 3)
    sizes = [pq.read_table(Path(version.root) / path).nbytes for path in (first, second, third)]
    cache = PayloadTableCache(max(sizes) * 2)
    cache.read_table(version, first, _PAYLOAD_SCHEMA)
    cache.read_table(version, second, _PAYLOAD_SCHEMA)
    # Reading the first file again makes the second the least recently read.
    cache.read_table(version, first, _PAYLOAD_SCHEMA)
    cache.read_table(version, third, _PAYLOAD_SCHEMA)
    assert list(cache._tables) == [(version.root, first), (version.root, third)]


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
    dataset.add_record(record=Record(sources=(Source(name="env", signals=signals),)))
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_time_offsets_are_keyed_by_their_axis_row_and_packed_once_for_a_shared_axis(tmp_path):
    version_dir = _write_irregular(tmp_path)
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        (axis_id,) = connection.execute(
            "SELECT DISTINCT axis_id FROM signals WHERE name IN ('temp', 'humidity')"
        ).fetchone() or (None,)
        assert connection.execute(
            "SELECT count(DISTINCT axis_id) FROM signals WHERE name IN ('temp', 'humidity')"
        ).fetchone() == (1,)
        audit_control_database(connection)
    assert [row for row in _external_rows(version_dir) if row[2] == PackedField.TIME_OFFSETS.value] == [
        (ObjectKind.AXIS.value, axis_id, PackedField.TIME_OFFSETS.value, 0)
    ]
    (payload,) = (version_dir / "payloads").iterdir()
    assert pq.read_table(payload)["time_offsets"].drop_null().to_pylist() == [_OFFSETS_US]


def _irregular_record(reader):
    (record,) = [record for record in reader.iter_records() if record.sources[0].name == "env"]
    return record


def _stored_offsets(reader):
    record = _irregular_record(reader)
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
    # Opening checks the release's files, so open it before changing them.
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


@pytest.mark.parametrize(
    ("element", "found"),
    [
        pytest.param(pa.field("element", pa.int32(), nullable=False), "list<element: int32 not null>", id="int32"),
        pytest.param(pa.field("element", pa.int64()), "list<element: int64>", id="nullable"),
    ],
)
def test_time_offsets_of_another_element_type_fail_the_read(tmp_path, element, found):
    version_dir = _write_irregular(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    (payload,) = (version_dir / "payloads").iterdir()
    table = pq.read_table(payload)
    index = table.schema.get_field_index("time_offsets")
    pq.write_table(table.cast(table.schema.set(index, pa.field("time_offsets", pa.list_(element)))), payload)

    with TimeFReader(version) as reader:
        handle = _stored_offsets(reader)
        message = re.escape(f"payloads/{payload.name} has schema (") + ".*" + re.escape(f"time_offsets: {found}), ")
        with pytest.raises(TimeFFormatError, match=message):
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
    dataset.add_record(record=Record(sources=(Source(name="env", signals=signals),)))
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    (payload,) = (version_dir / "payloads").iterdir()
    assert pq.read_table(payload)["time_offsets"].combine_chunks().flatten().nbytes == 5 * n_steps * 8

    with TimeFReader(open_local_version(version_dir)) as reader:
        record = _irregular_record(reader)
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
            object_id=1,
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
        PackedValueWriter(tmp_path, StorageSources()).pack(ObjectKind.AXIS, 1, PackedField.TIME_OFFSETS, offsets)


_LARGE_OFFSETS_US = [2**53 + 1, 2**62, 2**62 + 1]


def test_a_payload_of_interleaved_text_lists_and_time_offsets_round_trips_exactly(tmp_path):
    values = [
        (ObjectKind.TASK, PackedField.PROMPT, _TEXT),
        (ObjectKind.AXIS, PackedField.TIME_OFFSETS, pa.array(_OFFSETS_US, type=pa.int64())),
        (ObjectKind.ANNOTATION_CONTENT, PackedField.VALUE, _TEXTS),
        (ObjectKind.AXIS, PackedField.TIME_OFFSETS, pa.array(_LARGE_OFFSETS_US, type=pa.int64())),
        (ObjectKind.TASK, PackedField.RATIONALE, _TEXT.upper()),
    ]
    writer = PackedValueWriter(tmp_path, StorageSources())
    packed = [writer.pack(kind, index, field, value) for index, (kind, field, value) in enumerate(values, start=1)]
    writer.flush()

    locators = [item.locator for item in packed if isinstance(item, ExternalValue)]
    assert [locator.row_index for locator in locators] == list(range(len(values)))
    (path,) = writer.files
    table = pq.read_table(tmp_path / path)
    assert table.schema.equals(_PAYLOAD_SCHEMA)
    assert table.to_pylist() == [
        {"text": _TEXT, "texts": None, "time_offsets": None},
        {"text": None, "texts": None, "time_offsets": _OFFSETS_US},
        {"text": None, "texts": _TEXTS, "time_offsets": None},
        {"text": None, "texts": None, "time_offsets": _LARGE_OFFSETS_US},
        {"text": _TEXT.upper(), "texts": None, "time_offsets": None},
    ]


def test_large_time_offsets_round_trip_as_non_null_int64(tmp_path):
    dataset = make_dataset()
    signal = Signal.from_irregular(
        np.zeros(len(_LARGE_OFFSETS_US), dtype=np.float32),
        time_offsets_us=np.array(_LARGE_OFFSETS_US, dtype=np.int64),
        spec=TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless),
        name="large",
    )
    dataset.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    version_dir = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    (payload,) = (version_dir / "payloads").iterdir()
    assert str(pq.read_schema(payload).field("time_offsets").type) == "list<element: int64 not null>"

    with TimeFReader(open_local_version(version_dir)) as reader:
        record = _irregular_record(reader)
        assert record.signals[0].time_offsets_us().tolist() == _LARGE_OFFSETS_US


def test_packed_time_offsets_reach_the_payload_as_the_arrays_packed(tmp_path, monkeypatch):
    # Arrow's array types can't be patched to spy on to_pylist, so follow the packed arrays instead.
    concatenated = []
    concat_arrays = pa.concat_arrays

    def spy(arrays, *args, **kwargs):
        concatenated.append(list(arrays))
        return concat_arrays(concatenated[-1], *args, **kwargs)

    monkeypatch.setattr(pa, "concat_arrays", spy)
    first, second = pa.array(_OFFSETS_US, type=pa.int64()), pa.array(_LARGE_OFFSETS_US, type=pa.int64())
    writer = PackedValueWriter(tmp_path, StorageSources())
    writer.pack(ObjectKind.AXIS, 1, PackedField.TIME_OFFSETS, first)
    writer.pack(ObjectKind.ANNOTATION_CONTENT, 1, PackedField.VALUE, _TEXTS)
    writer.pack(ObjectKind.AXIS, 2, PackedField.TIME_OFFSETS, second)
    writer.flush()

    (arrays,) = concatenated
    assert len(arrays) == 2
    assert arrays[0] is first
    assert arrays[1] is second


def test_the_writer_stores_axis_locators_in_bounded_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(control_writer_module, "_ROW_BATCH_SIZE", 2)
    held = []
    pack = PackedValueWriter.pack

    def counting_pack(self, *args, **kwargs):
        packed = pack(self, *args, **kwargs)
        held.append(len(self.locators))
        return packed

    monkeypatch.setattr(PackedValueWriter, "pack", counting_pack)
    dataset = make_dataset()
    spec = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless)
    for index in range(5):
        signal = Signal.from_irregular(
            np.zeros(3, dtype=np.float32), time_offsets_us=[0, 5_000 + index, 9_000], spec=spec, name="env"
        )
        dataset.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))

    version_dir = _write_dataset(tmp_path, dataset)
    assert max(held) <= 2
    with connect_control(version_dir / "control.duckdb", read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM external_values WHERE object_kind = 'Axis'").fetchone() == (5,)
        audit_control_database(connection)


def _rewrite_time_offsets(version_dir, change):
    for payload in (version_dir / "payloads").iterdir():
        table = pq.read_table(payload)
        rows = table.to_pylist()
        for row in rows:
            if row["time_offsets"] is not None:
                change(row["time_offsets"])
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), payload)


def test_time_offsets_whose_digest_differs_from_their_locator_fail_the_read(tmp_path):
    version_dir = _write_irregular(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    # Same length, endpoints and order: only the digest tells the change apart.
    _rewrite_time_offsets(version_dir, lambda offsets: offsets.__setitem__(1, offsets[1] - 1_000))

    with TimeFReader(version) as reader:
        handle = _stored_offsets(reader)
        with pytest.raises(TimeFFormatError, match="time offsets whose digest differs from their locator's"):
            handle.read()
        assert not handle.is_loaded


def test_repeated_time_offsets_round_trip(tmp_path):
    dataset = make_dataset()
    offsets = [0, 5_000, 5_000, 9_000]
    signal = Signal.from_irregular(
        np.zeros(len(offsets), dtype=np.float32),
        time_offsets_us=offsets,
        spec=TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless),
        name="env",
    )
    dataset.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))
    version_dir = _write_dataset(tmp_path, dataset)

    with TimeFReader(open_local_version(version_dir)) as reader:
        assert _irregular_record(reader).signals[0].time_offsets_us().tolist() == offsets


def test_time_offsets_in_a_payload_file_outside_the_manifest_are_rejected(tmp_path):
    version_dir = _write_irregular(tmp_path)
    # Opening checks the release's files, so open it before changing them.
    version = open_local_version(version_dir)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute(
            "UPDATE external_values SET value_path = 'payloads/part-00000099.parquet' WHERE object_kind = 'Axis'"
        )

    with (
        TimeFReader(version) as reader,
        pytest.raises(TimeFFormatError, match=r"payloads/part-00000099\.parquet.* not listed in the manifest"),
    ):
        _irregular_record(reader)


def test_a_catalog_written_before_packed_values_fails_as_a_format_error(tmp_path):
    # A catalog with irregular axes but no locator table and no payload files.
    version_dir = _write_irregular(tmp_path)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("DROP TABLE external_values")
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["payloads"] = []
    manifest_path.write_text(json.dumps(manifest))

    with (
        TimeFReader(open_local_version(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="cannot read external values") as raised,
    ):
        list(reader.iter_records())
    assert isinstance(raised.value.__cause__, duckdb.Error)


def test_packed_values_in_a_release_that_lists_no_payload_files_are_refused(tmp_path):
    version_dir = _write_irregular(tmp_path)
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["payloads"] = []
    manifest_path.write_text(json.dumps(manifest))

    with (
        TimeFReader(open_local_version(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match="stores packed values, but the release lists no payload files"),
    ):
        list(reader.iter_records())


def test_a_reader_keeps_decoded_time_offsets_within_its_byte_budget(tmp_path, monkeypatch):
    # Each axis lands in a payload file of its own, and each Record is read on its own.
    monkeypatch.setattr(stored_values_module, "_PACK_BYTES", 1)
    n_steps = 5_000
    dataset = make_dataset()
    spec = TimeSeriesSpec(spec_type="env", name="env", unit_value=ureg.dimensionless)
    for index in range(6):
        signal = Signal.from_irregular(
            np.zeros(n_steps, dtype=np.float32),
            time_offsets_us=np.arange(n_steps, dtype=np.int64) * (index + 1),
            spec=spec,
            name="env",
        )
        dataset.add_record(record=Record(sources=(Source(name="env", signals=(signal,)),)))
    version_dir = _write_dataset(tmp_path, dataset)
    budget = 2 * n_steps * 8
    monkeypatch.setattr(stored_values_module, "_CACHE_BYTES", budget)

    with TimeFReader(open_local_version(version_dir)) as reader:
        packed = reader._packed_value_reader
        assert packed is not None
        record_ids = [record.id for record in reader.iter_records(with_annotations=False)]
        read = 0
        for record_id in record_ids:
            (record,) = reader.iter_records([record_id], with_annotations=False)
            for signal in record.signals:
                if signal.name == "env":
                    signal.time_offsets_us()
                    read += 1
            assert packed._parquet.nbytes <= budget
        assert read == 6
