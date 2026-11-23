from datetime import datetime, timedelta
import gzip
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ureg
from timenet_connectors.datasets.icsdataset.hai.connector import (
    FLAGS_SOURCE,
    POINTS_SOURCE,
    DataFile,
    HaiConnector,
    HaiRelease,
    LabelFile,
)
from timenet_connectors.datasets.icsdataset.hai.files import DATA_FILES, LABEL_FILES, SourceFile
from timenet_connectors.datasets.icsdataset.hai.points import MANUAL_POINTS


_N_ROWS = 400
_START = {
    "20.07": datetime(2019, 10, 29, 11, 0, 0),
    "22.04": datetime(2021, 7, 10, 0, 0, 1),
    "23.05": datetime(2022, 8, 12, 16, 0, 1),
}
# Synthetic files with the real column names of each version, a few columns each. 20.07 writes ";" and
# "time" with four flags; 22.04 writes "," and "timestamp" with CRLF line ends and one flag; 23.05 has
# no flag in its data files, and a separate label file for its test files.
_FILES = (
    SourceFile("20.07", "test1.csv.gz", "0" * 64),
    SourceFile("22.04", "test1.csv", "0" * 64),
    SourceFile("23.05", "hai-test1.csv", "0" * 64),
    SourceFile("23.05", "hai-train3.csv", "0" * 64),
)
_COLUMNS = {
    "20.07": ("P1_PIT01", "P3_LT01", "P3_LH", "P3_LCP01D", "P2_On", "attack", "attack_P1", "attack_P2", "attack_P3"),
    "22.04": ("P1_PIT01", "P3_LIT01", "P3_LH01", "P3_LCP01D", "P2_OnOff", "Attack"),
    "23.05": ("P1_PIT01", "P3_LIT01", "x1001_15_ASSIGN_OUT", "x1002_07_SETPOINT_OUT", "P1_PP04D"),
}
_FORMAT = {"20.07": (";", "time", "\n"), "22.04": (",", "timestamp", "\r\n"), "23.05": (",", "timestamp", "\r\n")}
_ATTACK = range(100, 160)


def _value(column: str, row: int) -> float:
    # Each column and row gets its own value, with a few decimals like HAI prints them. Flags are 0 or 1.
    if column.lower().startswith("attack"):
        return float(row in _ATTACK)
    return round(len(column) * 10 + row * 0.25 + 0.003, 4)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8", newline="")
    return path


def _write_data_file(directory: Path, source: SourceFile, *, skip_row: int | None = None) -> Path:
    delimiter, time_column, line_end = _FORMAT[source.version]
    columns = _COLUMNS[source.version]
    lines = [delimiter.join((time_column, *columns))]
    for row in range(_N_ROWS):
        if row == skip_row:
            continue
        time = _START[source.version] + timedelta(seconds=row)
        cells = [str(int(_value(c, row))) if c.lower().startswith("attack") else repr(_value(c, row)) for c in columns]
        lines.append(delimiter.join([f"{time:%Y-%m-%d %H:%M:%S}", *cells]))
    return _write(directory / source.path, line_end.join(lines) + line_end)


def _write_label_file(directory: Path, *, to_the_minute: bool = True, shift: int = 0) -> Path:
    # label-test2.csv prints its times to the minute, without a leading zero on the hour.
    lines = ["timestamp,label"]
    for row in range(_N_ROWS):
        time = _START["23.05"] + timedelta(seconds=row + shift)
        stamp = f"{time:%Y-%m-%d} {time.hour}:{time:%M}" if to_the_minute else f"{time:%Y-%m-%d %H:%M:%S}"
        lines.append(f"{stamp},{int(row in _ATTACK)}")
    return _write(directory / "hai-23.05" / "label-test1.csv", "\r\n".join(lines) + "\r\n")


def _release(directory: Path, *, skip_row: int | None = None, **label_options) -> HaiRelease:
    files = tuple(DataFile(source, _write_data_file(directory, source, skip_row=skip_row)) for source in _FILES)
    label = LabelFile(SourceFile("23.05", "label-test1.csv", "0" * 64), _write_label_file(directory, **label_options))
    return HaiRelease(files=files, labels={"hai-test1": label})


def _signals(record) -> dict:
    return {signal.name: signal for signal in record.signals}


def test_every_column_of_every_file_becomes_a_signal(tmp_path):
    dataset = HaiConnector().convert(_release(tmp_path))
    assert [record.metadata["path"] for record in dataset.records] == [source.path for source in _FILES]
    for record, source in zip(dataset.records, _FILES, strict=True):
        assert record.metadata["version"] == source.version
        assert record.metadata["file"] == source.stem
        assert record.metadata["start_time"] == f"{_START[source.version]:%Y-%m-%d %H:%M:%S}"
        assert record.metadata["n_seconds"] == _N_ROWS
        sources = {one.name: [signal.name for signal in one.signals] for one in record.sources}
        flags = source.format.flag_columns + (("label",) if source.stem == "hai-test1" else ())
        assert sources[POINTS_SOURCE] == [c for c in _COLUMNS[source.version] if c not in flags]
        assert sources.get(FLAGS_SOURCE, []) == list(flags)
        for name, signal in _signals(record).items():
            column = "attack" if name == "label" else name
            # Exact equality: float64 keeps every digit of the source text.
            np.testing.assert_array_equal(signal.to_numpy(), [_value(column, row) for row in range(_N_ROWS)])
    # HAI publishes no labels for the 23.05 training files.
    assert FLAGS_SOURCE not in {one.name for one in dataset.records[3].sources}


def test_each_signal_has_the_quantity_and_unit_of_the_manual(tmp_path):
    old, new, end = (_signals(record) for record in HaiConnector().convert(_release(tmp_path)).records[:3])
    assert old["P1_PIT01"].spec.unit_value == ureg.bar
    assert old["P1_PIT01"].spec.name == "Heat-exchanger outlet pressure"
    # The manual names P3_LT01 and P2_On in brackets in the rows of P3_LIT01 and P2_OnOff.
    assert old["P3_LT01"].spec == new["P3_LIT01"].spec
    assert old["P3_LT01"].spec.unit_value == ureg.percent
    assert old["P3_LT01"].metadata == {"manual_point": "P3_LIT01"}
    assert old["P2_On"].spec == new["P2_OnOff"].spec
    assert new["P2_OnOff"].spec.unit_value == ureg.dimensionless
    # The manual prints "-" as the unit of P3_LCP01D, and does not list P3_LH: no unit.
    assert old["P3_LCP01D"].spec.unit_value is None
    assert old["P3_LH"].spec.unit_value is None
    assert old["P3_LH"].metadata == {"manual_point": None}
    # 23.05: the manual ties DCS block 1001.15 to P1_B2016 and gives the other blocks no unit.
    assert end["x1001_15_ASSIGN_OUT"].spec.unit_value == ureg.bar
    assert end["x1002_07_SETPOINT_OUT"].spec.name == "Output of DCS block 1002.07"
    assert end["x1002_07_SETPOINT_OUT"].spec.unit_value is None
    assert end["label"].spec.spec_type == "hai_attack_flag"


def test_the_manual_table_has_its_87_points():
    assert len(MANUAL_POINTS) == len({point.name for point in MANUAL_POINTS}) == 87


def test_the_pins_cover_every_file_of_the_four_versions():
    assert len(DATA_FILES) == 28
    assert len({(one.version, one.stem) for one in DATA_FILES}) == 28
    assert set(LABEL_FILES) == {"hai-test1", "hai-test2"}
    assert all(len(one.sha256) == 64 for one in (*DATA_FILES, *LABEL_FILES.values()))


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    dataset = HaiConnector().convert(_release(source))
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())


def test_each_file_is_parsed_once(tmp_path, monkeypatch):
    calls = []
    read = DataFile.read_columns

    def counted(self, points):
        calls.append(self.source.path)
        return read(self, points)

    monkeypatch.setattr(DataFile, "read_columns", counted)
    dataset = HaiConnector().convert(_release(tmp_path / "source"))
    dataset.write(path=tmp_path / "registry")
    assert sorted(calls) == sorted(source.path for source in _FILES)


def test_a_missing_second_raises(tmp_path):
    with pytest.raises(TimeNetConnectorError, match=r"hai-20\.07/test1\.csv\.gz: data rows 200 and 201 are 2 s apart"):
        HaiConnector().convert(_release(tmp_path, skip_row=200))


def test_a_label_file_that_does_not_line_up_raises(tmp_path):
    with pytest.raises(TimeNetConnectorError, match=r"label-test1\.csv: data row 1 names another time"):
        HaiConnector().convert(_release(tmp_path, to_the_minute=False, shift=1))
