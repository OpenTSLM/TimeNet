"""Tests for the MIMIC-IV vital signs and notes connector.

All rows are synthetic. The tests do not contain protected MIMIC-IV data.
"""

import csv
import gzip
from pathlib import Path

import numpy as np
import pytest

from timenet.connectors import BaseConnector
from timenet.engine import store_dataset
from timenet.errors import TimeFValidationError, TimeNetBuildError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask, License, ureg
from timenet_connectors.datasets.physionet.mimic_iv_ext_vitals_notes.connector import (
    MIMIC_IV_NOTE_ROOT_ENV,
    MIMIC_IV_ROOT_ENV,
    MimicIvExtVitalsNotesConnector,
)


_ADMISSIONS_COLUMNS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "admit_provider_id",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "edregtime",
    "edouttime",
    "hospital_expire_flag",
]
_ICUSTAYS_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "first_careunit",
    "last_careunit",
    "intime",
    "outtime",
    "los",
]
_CHARTEVENTS_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "caregiver_id",
    "charttime",
    "storetime",
    "itemid",
    "value",
    "valuenum",
    "valueuom",
    "warning",
]
_DISCHARGE_COLUMNS = [
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
]


def _write_gzip_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _chart_event(hadm_id: int, subject_id: int, charttime: str, itemid: int, value: float) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": hadm_id + 10_000,
        "charttime": charttime,
        "storetime": charttime,
        "itemid": itemid,
        "value": str(value),
        "valuenum": value,
        "warning": 0,
    }


def _source_roots(tmp_path: Path) -> tuple[Path, Path]:
    mimic_root = tmp_path / "mimiciv"
    note_root = tmp_path / "mimic-iv-note"
    _write_gzip_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        _ADMISSIONS_COLUMNS,
        [
            {
                "subject_id": 1001,
                "hadm_id": 2001,
                "admittime": "2020-01-01 00:00:00",
                "dischtime": "2020-01-01 04:00:00",
                "admission_type": "URGENT",
                "hospital_expire_flag": 0,
            },
            {
                "subject_id": 1002,
                "hadm_id": 2002,
                "admittime": "2020-02-01 00:00:00",
                "dischtime": "2020-02-01 04:00:00",
                "admission_type": "ELECTIVE",
                "hospital_expire_flag": 0,
            },
        ],
    )
    _write_gzip_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        _ICUSTAYS_COLUMNS,
        [
            {
                "subject_id": 1001,
                "hadm_id": 2001,
                "stay_id": 12001,
                "first_careunit": "Medical ICU",
                "last_careunit": "Medical ICU",
                "intime": "2020-01-01 00:00:00",
                "outtime": "2020-01-01 03:00:00",
                "los": 0.125,
            },
            {
                "subject_id": 1002,
                "hadm_id": 2002,
                "stay_id": 12002,
                "first_careunit": "Surgical ICU",
                "last_careunit": "Surgical ICU",
                "intime": "2020-02-01 00:00:00",
                "outtime": "2020-02-01 03:00:00",
                "los": 0.125,
            },
        ],
    )
    _write_gzip_csv(
        mimic_root / "icu" / "chartevents.csv.gz",
        _CHARTEVENTS_COLUMNS,
        [
            _chart_event(2001, 1001, "2020-01-01 00:10:00", 220045, 80),
            _chart_event(2001, 1001, "2020-01-01 00:40:00", 220045, 100),
            _chart_event(2001, 1001, "2020-01-01 01:10:00", 220045, 400),
            _chart_event(2001, 1001, "2020-01-01 02:05:00", 220045, 70),
            _chart_event(2001, 1001, "2020-01-01 04:00:00", 220045, 65),
            _chart_event(2001, 1001, "2020-01-01 00:15:00", 220179, 120),
            _chart_event(2001, 1001, "2020-01-01 00:15:00", 220180, 70),
            _chart_event(2001, 1001, "2020-01-01 00:15:00", 220052, 85),
            _chart_event(2001, 1001, "2020-01-01 00:20:00", 220277, 98),
            _chart_event(2001, 1001, "2020-01-01 00:25:00", 220210, 18),
            _chart_event(2001, 1001, "2020-01-01 00:30:00", 223761, 98.6),
            _chart_event(2001, 1001, "2020-01-01 02:30:00", 223762, 38),
            _chart_event(2002, 1002, "2020-02-01 00:10:00", 220045, 75),
        ],
    )
    _write_gzip_csv(
        note_root / "note" / "discharge.csv.gz",
        _DISCHARGE_COLUMNS,
        [
            {
                "note_id": "synthetic-ds-1",
                "subject_id": 1001,
                "hadm_id": 2001,
                "note_type": "DS",
                "note_seq": 1,
                "charttime": "2020-01-01 04:00:00",
                "storetime": "2020-01-01 05:00:00",
                "text": "Synthetic discharge summary one.",
            },
            {
                "note_id": "synthetic-ds-2",
                "subject_id": 1001,
                "hadm_id": 2001,
                "note_type": "DS",
                "note_seq": 2,
                "charttime": "2020-01-01 04:00:00",
                "storetime": "2020-01-01 05:30:00",
                "text": "Synthetic discharge summary two.",
            },
            {
                "note_id": "synthetic-ad-1",
                "subject_id": 1002,
                "hadm_id": 2002,
                "note_type": "AD",
                "note_seq": 1,
                "charttime": "2020-02-01 00:00:00",
                "storetime": "2020-02-01 00:00:00",
                "text": "Synthetic admission note.",
            },
        ],
    )
    return mimic_root, note_root


def _connector_and_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mimic_root, note_root = _source_roots(tmp_path)
    monkeypatch.setenv(MIMIC_IV_ROOT_ENV, str(mimic_root))
    monkeypatch.setenv(MIMIC_IV_NOTE_ROOT_ENV, str(note_root))
    connector = MimicIvExtVitalsNotesConnector()
    return connector, connector.download(tmp_path / "cache")


def test_is_a_connector_and_has_credentialed_license():
    connector = MimicIvExtVitalsNotesConnector()
    assert isinstance(connector, BaseConnector)
    assert connector.metadata().dataset_id == "physionet/mimic-iv-ext-vitals-notes"
    assert connector.metadata().license is License.PHYSIONET_CREDENTIALED_HEALTH_DATA_1_5_0


def test_download_requires_authorized_local_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MIMIC_IV_ROOT_ENV, raising=False)
    monkeypatch.delenv(MIMIC_IV_NOTE_ROOT_ENV, raising=False)
    with pytest.raises(TimeNetBuildError, match=MIMIC_IV_ROOT_ENV):
        MimicIvExtVitalsNotesConnector().download(tmp_path / "cache")


def test_download_only_discovers_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    connector, sources = _connector_and_source(tmp_path, monkeypatch)
    assert isinstance(connector, MimicIvExtVitalsNotesConnector)
    assert sources[0].raw_files.chartevents.is_file()
    assert not sources[0].prepared_dir.exists()


def test_convert_builds_hourly_sparse_vitals_and_discharge_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    connector, sources = _connector_and_source(tmp_path, monkeypatch)
    dataset = connector.convert(sources)

    assert [sample.sample_id for sample in dataset.samples] == ["mimic-hadm-2001"]
    sample = dataset.samples[0]
    assert sample.subject_ids == ("mimic-subject-1001",)
    assert sample.start_time is None
    assert {annotation.key: annotation.value for annotation in sample.annotations} == {
        "admission_type": "URGENT",
        "first_careunit": "Medical ICU",
    }

    series = {item.channel: item for item in sample.time_series}
    assert set(series) == {
        "heart-rate",
        "systolic",
        "diastolic",
        "mean",
        "spo2",
        "respiratory-rate",
        "temperature",
    }
    np.testing.assert_allclose(series["heart-rate"].to_numpy(), [90, 70])
    np.testing.assert_array_equal(series["heart-rate"].time_offsets_us(), [0, 7_200_000_000])
    np.testing.assert_allclose(series["temperature"].to_numpy(), [37, 38], rtol=1e-6)
    assert series["heart-rate"].spec.unit_value == ureg.bpm
    assert series["systolic"].spec.unit_value == ureg.mmHg
    assert series["respiratory-rate"].spec.unit_value == ureg.brpm
    assert series["temperature"].spec.unit_value == ureg.degree_Celsius

    tasks = list(dataset.iter_tasks())
    assert len(tasks) == 2
    assert all(isinstance(task, AnswerTask) for task in tasks)
    assert {task.target for task in tasks} == {
        "Synthetic discharge summary one.",
        "Synthetic discharge summary two.",
    }
    assert all(task.sample_ids == ("mimic-hadm-2001",) for task in tasks)
    assert all(task.prompt is None and task.rationale is None for task in tasks)


def test_prepared_cache_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    connector, sources = _connector_and_source(tmp_path, monkeypatch)
    connector.convert(sources)
    prepared_files = tuple(sources[0].prepared_dir.glob("*.parquet"))
    assert {path.name for path in prepared_files} == {"admissions.parquet", "vitals.parquet", "notes.parquet"}
    first_mtimes = {path.name: path.stat().st_mtime_ns for path in prepared_files}

    connector.convert(sources)

    assert {path.name: path.stat().st_mtime_ns for path in prepared_files} == first_mtimes


def test_convert_rejects_an_invalid_source_count():
    with pytest.raises(TimeFValidationError, match="needs one source"):
        MimicIvExtVitalsNotesConnector().convert([])


def test_convert_round_trips_through_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    connector, sources = _connector_and_source(tmp_path / "source", monkeypatch)
    dataset = connector.convert(sources)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "output")

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()

    assert [sample.sample_id for sample in restored.samples] == ["mimic-hadm-2001"]
    assert len(restored.samples[0].time_series) == 7
    assert len(restored.tasks) == 2
