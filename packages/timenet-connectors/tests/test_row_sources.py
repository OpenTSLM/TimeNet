import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataRow, NonBlankText
from timenet_connectors.sources.readers.csv import CsvRowSource
from timenet_connectors.sources.readers.json_object import JsonObjectRowSource
from timenet_connectors.sources.readers.jsonl import JsonlRowSource
from timenet_connectors.sources.readers.parquet import ParquetRowSource


class Row(DataRow):
    identifier: str = Field(alias="id")
    text: NonBlankText


@pytest.fixture(params=[CsvRowSource, ParquetRowSource, JsonObjectRowSource, JsonlRowSource])
def reader_class(request):
    return request.param


@pytest.fixture
def source_factory(reader_class, tmp_path):
    def create(values, **kwargs):
        if reader_class is CsvRowSource:
            path = tmp_path / "rows.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        elif reader_class is ParquetRowSource:
            path = tmp_path / "rows.parquet"
            pq.write_table(pa.Table.from_pylist(values), path, row_group_size=1)
        elif reader_class is JsonObjectRowSource:
            path = tmp_path / "rows.json"
            path.write_text(json.dumps({str(index): value for index, value in enumerate(values)}))
        else:
            path = tmp_path / "rows.jsonl"
            path.write_text("".join(json.dumps(value) + "\n" for value in values))
        return reader_class(path, Row, **kwargs)

    return create


@pytest.fixture
def opened_files(monkeypatch):
    opened = []
    original = Path.open
    parquet_open = pq.ParquetFile

    def track(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        opened.append(handle)
        return handle

    def track_parquet(*args, **kwargs):
        handle = parquet_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(pq, "ParquetFile", track_parquet)
    return opened


@pytest.mark.parametrize("field", ["id", "missing"])
def test_invalid_unique_fails_before_file_access(reader_class, tmp_path, monkeypatch, field):
    def unexpected_open(*args, **kwargs):
        pytest.fail("configuration validation must not open a file")

    monkeypatch.setattr(Path, "open", unexpected_open)
    monkeypatch.setattr(pq, "ParquetFile", unexpected_open)
    with pytest.raises(TimeNetConnectorError, match="is not a field of Row"):
        reader_class(tmp_path / "missing", Row, unique=field)


def test_unread_context_closes_its_source(source_factory, opened_files):
    source = source_factory([{"id": "001", "text": "first"}])
    opened_files.clear()
    with source.open_rows():
        pass
    assert len(opened_files) == 1 and opened_files[0].closed


def test_early_exit_closes_its_source(source_factory, opened_files):
    source = source_factory([{"id": "001", "text": "first"}, {"id": "002", "text": "second"}])
    opened_files.clear()
    with source.open_rows() as rows:
        assert next(rows).row.identifier == "001"
    assert len(opened_files) == 1 and opened_files[0].closed


def test_caller_failure_closes_its_source(source_factory, opened_files):
    source = source_factory([{"id": "001", "text": "first"}])
    opened_files.clear()
    caller_error = OSError("consumer storage failed")
    with pytest.raises(OSError) as error, source.open_rows() as rows:
        next(rows)
        raise caller_error
    assert error.value is caller_error
    assert len(opened_files) == 1 and opened_files[0].closed


def test_row_validation_failure_closes_its_source(source_factory, opened_files):
    source = source_factory([{"id": "001", "text": "first"}, {"id": "002", "text": " "}])
    opened_files.clear()
    with source.open_rows() as rows:
        assert next(rows).row.identifier == "001"
        with pytest.raises(TimeNetConnectorError, match="text:"):
            next(rows)
    assert len(opened_files) == 1 and opened_files[0].closed


def test_reads_have_independent_positions_and_uniqueness(source_factory):
    source = source_factory([{"id": "001", "text": "first"}, {"id": "002", "text": "second"}], unique="identifier")
    with source.open_rows() as first, source.open_rows() as second:
        assert next(first) == next(second)
        assert next(first) == next(second)
    with source.open_rows() as again:
        assert [entry.row.identifier for entry in again] == ["001", "002"]


def test_uniqueness_reports_both_original_locations(source_factory):
    source = source_factory(
        [{"id": "001", "text": "first"}, {"id": "002", "text": "second"}, {"id": "001", "text": "third"}],
        unique="identifier",
    )
    with source.open_rows() as rows:
        first = next(rows)
        next(rows)
        with pytest.raises(TimeNetConnectorError, match="duplicate '001'") as error:
            next(rows)
    assert first.location.context in str(error.value)
    assert "first at" in str(error.value)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_file_errors_use_connector_error_with_path_and_cause(source_factory, failure):
    source = source_factory([{"id": "001", "text": "first"}])
    if failure == "missing":
        source.path.unlink()
    else:
        source.path.write_bytes(b"\xff\xfe")
    with pytest.raises(TimeNetConnectorError) as error, source.open_rows() as rows:
        list(rows)
    assert str(source.path) in str(error.value)
    assert error.value.__cause__ is not None
