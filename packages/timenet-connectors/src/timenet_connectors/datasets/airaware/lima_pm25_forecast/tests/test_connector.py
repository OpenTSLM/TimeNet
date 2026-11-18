from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from timenet.reader import TimeFReader
from timenet_connectors.datasets.airaware.lima_pm25_forecast.connector import (
    LimaPM25Connector,
    _parse,
    _split,
)


def test_chronological_assignment_purges_crossing_history_or_target():
    cut = datetime(2022, 3, 16, 8, tzinfo=UTC)
    assert _split(cut - timedelta(hours=30), cut - timedelta(hours=1)) == "train"
    assert _split(cut - timedelta(hours=29), cut) is None
    assert _split(cut, cut + timedelta(hours=29)) == "validation"


def test_duplicate_zero_conflict_and_missing_are_distinct():
    header = "ESTACION,LONGITUD,LATITUD,ALTITUD,DEPARTAMENTO,PROVINCIA,DISTRITO,FECHA,HORA,PM2_5"
    prefix = "FICTICIA,-77,-12,10,LIMA,LIMA,TEST,20200101"
    lines = [
        header,
        f"{prefix},0,0",
        f"{prefix},0,0.0",
        f"{prefix},0,",
        f"{prefix},10000,3",
        f"{prefix},10000,4",
        f"{prefix},20000,",
    ]
    (readings,) = _parse("\n".join(lines).encode()).values()
    assert list(readings.values()) == [(0.0, 2), None]


def test_build_from_verified_fixture_and_read_timef(tmp_path, monkeypatch):
    csv_path = tmp_path / "input.csv"
    rows = ["ESTACION,LONGITUD,LATITUD,ALTITUD,DEPARTAMENTO,PROVINCIA,DISTRITO,FECHA,HORA,PM2_5"]
    for start in ("20200101", "20220501", "20230501"):
        day = datetime.strptime(start, "%Y%m%d").replace(tzinfo=UTC)
        for hour in range(30):
            tick = day + timedelta(hours=hour)
            rows.append(f"FICTICIA,-77,-12,10,LIMA,LIMA,TEST,{tick:%Y%m%d},{tick:%H%M%S},{hour}")
    body = ("\n".join(rows) + "\n").encode()
    csv_path.write_bytes(body)
    monkeypatch.setattr(LimaPM25Connector, "source_sha256", sha256(body).hexdigest())
    dataset = LimaPM25Connector().convert([csv_path])
    assert [str(task.split) for task in dataset.tasks] == ["train", "validation", "test"]
    with TimeFReader.open(dataset.write(path=tmp_path / "registry")) as reader:
        reader.verify()
        task = next(reader.iter_tasks(batch_size=1))
        assert task.inputs[0].signals[0].to_arrow().to_pylist() == list(range(24))
        assert task.targets == tuple(range(24, 30))
    csv_path.write_bytes(body + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        LimaPM25Connector().convert([csv_path])
