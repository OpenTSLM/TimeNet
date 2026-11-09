from contextlib import closing
from pathlib import Path

from pydantic import ValidationError
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers import CsvReader, DataRow


class Row(DataRow):
    name: str
    count: int


def test_data_row_rejects_unexpected_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Row.model_validate({"name": "sensor", "count": 1, "extra": "value"})


def test_csv_reads_rows_in_order(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nfirst,1\nsecond,2\n")
    reader = CsvReader(path, Row, strict=True)
    assert list(reader.iter_rows()) == [Row(name="first", count=1), Row(name="second", count=2)]


def test_csv_invalid_row_reports_location(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nx,bad\n")
    with pytest.raises(TimeNetConnectorError, match="line 2") as exc:
        list(CsvReader(path, Row).iter_rows())
    assert str(path) in str(exc.value)
    assert exc.value.__cause__ is not None


@pytest.mark.parametrize("content", ["name,name\nx,y\n", "name,count\nx,1,extra\n", "name,count\nx\n"])
def test_csv_rejects_ambiguous_rows(tmp_path, content):
    path = tmp_path / "source.csv"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError):
        list(CsvReader(path, Row).iter_rows())


def test_csv_checks_expected_header_even_without_rows(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("count,name\n")
    with pytest.raises(TimeNetConnectorError, match="expected columns"):
        list(CsvReader(path, Row, columns=("name", "count")).iter_rows())


def test_closing_a_partially_consumed_csv_reader_closes_its_file(tmp_path, monkeypatch):
    path = tmp_path / "source.csv"
    path.write_text("name,count\nfirst,1\nsecond,2\n")
    handles = []
    open_file = Path.open

    def track_open(file_path, *args, **kwargs):
        handle = open_file(file_path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", track_open)
    with closing(CsvReader(path, Row).iter_rows()) as rows:
        assert next(rows) == Row(name="first", count=1)
        assert not handles[0].closed
    assert handles[0].closed
