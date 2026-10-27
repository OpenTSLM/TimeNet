import copy
from pathlib import Path
import pickle
import re
import shutil
from typing import get_type_hints

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.format.control_audit import audit_control_database
import timenet.format.control_writer as control_writer_module
from timenet.format.duckdb import connect_control
import timenet.format.stored_values as stored_values_module
from timenet.format.stored_values import (
    CatalogValue,
    ExternalValue,
    PackedContentKey,
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
from timenet.storage.parquet import PayloadTableCache
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, ObjectKind, TextHandle, TextListHandle, ValueHandle
from timenet.types.annotations import value_type_of
from timenet.types.handles import ArrowArray, CallableArray, NumpyArray
from timenet.types.text import InlineText, InlineTextList, read_value
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report with café notes. " * 300
_TEXTS = [_TEXT, "short report"]
# JSON-encoded with its quotes, the first takes exactly 4,096 bytes and the second one more.
_AT_LIMIT = "a" * 4094
_PAST_LIMIT = "a" * 4095
# Spelled out rather than imported, so a change to the format's payload schema fails a test.
_PAYLOAD_SCHEMA = pa.schema(
    [("text", pa.string()), ("texts", pa.list_(pa.field("element", pa.string(), nullable=False)))]
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


def test_written_payloads_have_the_payload_schema_and_read_back(tmp_path):
    version_dir = _write(tmp_path)
    for payload in (version_dir / "payloads").iterdir():
        assert pq.ParquetFile(payload).schema_arrow.equals(_PAYLOAD_SCHEMA)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        prompt, reports = _stored(reader)
        assert prompt.read() == _TEXT
        assert reports.read() == _TEXTS


def test_a_payload_rebuilt_with_the_payload_schema_still_reads(tmp_path):
    version_dir = _write(tmp_path)
    (payload,) = (version_dir / "payloads").iterdir()
    rows = pq.read_table(payload).to_pylist()
    # Another compression and one row per group: only the Arrow schema has to match.
    pq.write_table(pa.Table.from_pylist(rows, schema=_PAYLOAD_SCHEMA), payload, compression="snappy", row_group_size=1)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        prompt, reports = _stored(reader)
        assert prompt.read() == _TEXT
        assert reports.read() == _TEXTS


@pytest.mark.parametrize(
    ("rewrite", "found"),
    [
        pytest.param(
            lambda table: table.cast(table.schema.set(0, pa.field("text", pa.binary()))),
            "text: binary, texts: list<element: string not null>",
            id="binary-text",
        ),
        pytest.param(
            lambda table: table.cast(table.schema.set(1, pa.field("texts", pa.list_(pa.string())))),
            "text: string, texts: list<element: string>",
            id="nullable-list-elements",
        ),
        pytest.param(lambda table: table.select(["text"]), "text: string", id="missing-texts"),
    ],
)
def test_a_payload_with_another_schema_fails_every_read(tmp_path, rewrite, found):
    version_dir = _write(tmp_path)
    (payload,) = (version_dir / "payloads").iterdir()
    pq.write_table(rewrite(pq.read_table(payload)), payload)

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    packed = PackedValueWriter(tmp_path).pack(ObjectKind.ANNOTATION_CONTENT, 1, PackedField.VALUE, value)
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
    packed = PackedValueWriter(tmp_path).pack(ObjectKind.ANNOTATION_CONTENT, 1, PackedField.VALUE, value)
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
        TimeFReader(DatasetVersion.open_local(_write(tmp_path / "a"))) as first,
        TimeFReader(DatasetVersion.open_local(_write(tmp_path / "b", prompt=_TEXT.upper()))) as second,
    ):
        (prompt, _), (other, _) = _stored(first), _stored(second)
        assert prompt._location == other._location
        assert prompt._content_key() != other._content_key()


def test_one_payload_file_in_two_releases_gives_one_content_key(tmp_path, payload_reads):
    source = _write(tmp_path / "a")
    twin = tmp_path / "b" / "twin"
    shutil.copytree(source, twin)
    with (
        TimeFReader(DatasetVersion.open_local(source)) as first,
        TimeFReader(DatasetVersion.open_local(twin)) as second,
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
    version = DatasetVersion.open_local(_write_dataset(tmp_path, dataset))

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
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("UPDATE external_values SET value_path = 'payloads/part-00000099.parquet'")

    with (
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
        pytest.raises(TimeFFormatError, match=r"payloads/part-00000099\.parquet.* not listed in the manifest"),
    ):
        reader.read()


def test_rewriting_stored_text_at_one_location_keeps_one_content_row(tmp_path):
    dataset, _ = _packed_dataset(prompt="short")
    report = Annotation(key="report", value=_TEXT)
    for record in dataset.records:
        record.annotate(report)
    with TimeFReader(DatasetVersion.open_local(_write_dataset(tmp_path / "a", dataset))) as reader:
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
        target = _write_dataset(tmp_path / "b", loaded)

    assert len(_content_ids(target, "report")) == 1
    assert len(_external_rows(target)) == 2
    with TimeFReader(DatasetVersion.open_local(target)) as reader:
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


def _targets(version_dir, task_id):
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    assert _targets(version_dir, task_id) == (["short", _TEXT], [(0, "short"), (1, _TEXT)])


def test_targets_are_matched_to_their_locators_by_stored_position(tmp_path):
    version_dir, task_id = _with_packed_second_target(tmp_path)
    with connect_control(version_dir / "control.duckdb") as connection:
        connection.execute("UPDATE task_targets SET position = position * 2 WHERE task_id = ?", [task_id])
        connection.execute(
            "UPDATE external_values SET position = position * 2 WHERE field = 'targets' AND object_id = ?", [task_id]
        )
    assert _targets(version_dir, task_id) == (["short", _TEXT], [(0, "short"), (2, _TEXT)])


def test_two_stored_texts_of_one_kind_in_one_payload_compare_by_content(tmp_path, payload_reads):
    dataset = make_dataset()
    other = _TEXT.upper()
    assert len(other) == len(_TEXT)
    task = dataset.add_task(
        task=AnswerTask(inputs=(dataset.records[0],), prompt=_TEXT, rationale=other, targets=("yes",))
    )
    version_dir = _write_dataset(tmp_path, dataset)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
        TimeFReader(DatasetVersion.open_local(version_dir)) as reader,
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
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
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
    version = DatasetVersion.open_local(_write_dataset(tmp_path, dataset))
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
