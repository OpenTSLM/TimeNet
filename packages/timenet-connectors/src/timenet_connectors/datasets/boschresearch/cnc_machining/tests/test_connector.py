"""Focused tests for the Bosch CNC Machining connector."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from timenet.connectors import BaseConnector
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, ureg
from timenet_connectors.datasets.boschresearch.cnc_machining.connector import (
    _LOCAL_SOURCE_ENV,
    _SOURCE_COMMIT,
    _SOURCE_SHA256,
    BoschCncConnector,
    BoschCncSource,
    _discover,
)


def _write_record(  # noqa: PLR0913
    root: Path,
    *,
    machine: str = "M01",
    process: str = "OP00",
    health: str = "good",
    timeframe: str = "Aug_2019",
    example: str = "000",
    values: np.ndarray | None = None,
    filename_machine: str | None = None,
    filename_process: str | None = None,
) -> Path:
    """Write one deterministic source-shaped HDF5 fixture."""
    if values is None:
        values = np.array([[1.25, 2.5, 3.75], [-4.0, -5.0, -6.0]], dtype=np.float64)
    directory = root / "data" / machine / process / health
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{filename_machine or machine}_{timeframe}_{filename_process or process}_{example}.h5"
    path = directory / filename
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vibration_data", data=values)
    return path


def _sources(root: Path) -> list[BoschCncSource]:
    return _discover(root / "data")


def _convert(root: Path):
    return BoschCncConnector().convert(_sources(root))


def _annotations(record) -> dict[str, object]:
    return {annotation.key: annotation.value for annotation in record.annotations}


def test_connector_identity_and_metadata():
    connector = BoschCncConnector()
    assert isinstance(connector, BaseConnector)
    assert connector.metadata().dataset_id == "boschresearch/cnc-machining"
    assert str(connector.metadata().license) == "CC-BY-4.0"


def test_download_discovers_local_source_deterministically(tmp_path, monkeypatch):
    second = _write_record(tmp_path, example="002")
    first = _write_record(tmp_path, example="001")
    monkeypatch.setenv(_LOCAL_SOURCE_ENV, str(tmp_path))

    one = BoschCncConnector().download(tmp_path / "cache-one")
    two = BoschCncConnector().download(tmp_path / "cache-two")

    assert one == two
    assert [source.path for source in one] == [first, second]
    assert [source.record_id for source in one] == [
        "M01/OP00/good/M01_Aug_2019_OP00_001",
        "M01/OP00/good/M01_Aug_2019_OP00_002",
    ]


def test_default_download_uses_commit_pinned_archive(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    _write_record(source_root)
    (source_root / "README.md").write_text("fixture", encoding="utf-8")
    seen = {}

    async def fake_ensure_archive(url, cache_dir, *, filename, sha256):
        seen.update(url=url, cache_dir=cache_dir, filename=filename, sha256=sha256)
        return source_root

    monkeypatch.delenv(_LOCAL_SOURCE_ENV, raising=False)
    monkeypatch.setattr(
        "timenet_connectors.datasets.boschresearch.cnc_machining.connector.ensure_archive",
        fake_ensure_archive,
    )

    sources = BoschCncConnector().download(tmp_path / "cache")

    assert len(sources) == 1
    assert _SOURCE_COMMIT in seen["url"]
    assert _SOURCE_COMMIT in seen["filename"]
    assert seen["sha256"] == _SOURCE_SHA256
    assert seen["cache_dir"] == tmp_path / "cache"


def test_discovery_parses_metadata_and_builds_one_record_per_file(tmp_path):
    _write_record(tmp_path, machine="M02", process="OP03", health="bad", example="007")
    source = _sources(tmp_path)[0]
    assert source.machine_number == "M02"
    assert source.process_number == "OP03"
    assert source.process_health == "bad"
    assert source.timeframe == "Aug_2019"
    assert source.example_number == "007"

    dataset = _convert(tmp_path)
    assert len(dataset.records) == 1
    assert _annotations(dataset.records[0]) == {
        "source_file": "M02/OP03/bad/M02_Aug_2019_OP03_007.h5",
        "machine_number": "M02",
        "process_number": "OP03",
        "process_health": "bad",
        "timeframe": "Aug_2019",
        "example_number": "007",
    }


def test_directory_filename_disagreement_is_rejected(tmp_path):
    _write_record(tmp_path, filename_machine="M02")
    with pytest.raises(TimeFValidationError, match="metadata disagreement"):
        _sources(tmp_path)


@pytest.mark.parametrize("health", ["unknown", "BAD"])
def test_unknown_health_label_is_rejected(tmp_path, health):
    _write_record(tmp_path, health=health)
    with pytest.raises(TimeFValidationError, match="'good' or 'bad'"):
        _sources(tmp_path)


def test_signal_mapping_values_time_and_unit(tmp_path):
    values = np.array([[1.25, 20.5, -3.0], [4.5, -50.25, 6.75]], dtype=np.float64)
    _write_record(tmp_path, values=values)
    record = _convert(tmp_path).records[0]

    assert [series.signal for series in record.time_series] == ["x", "y", "z"]
    for channel, series in enumerate(record.time_series):
        np.testing.assert_array_equal(series.to_numpy(), values[:, channel])
        assert series.n_values == values.shape[0]
        assert series.spec.dtype == "float64"
        assert series.to_numpy().dtype == np.dtype("float64")
        assert series.spec.unit_value == ureg.Unit("milligravity")
        assert series.spec.unit_value != ureg.Unit("mg")
        assert isinstance(series.time_axis, RegularAxis)
        assert series.time_axis == RegularAxis.from_rate_hz(2000)
        assert series.time_axis.start_index == 0
        assert series.time_offsets_loader is None
    assert record.start_time is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_supported_source_dtypes_become_exact_float64(tmp_path, dtype):
    values = np.array([[1, 2, 3], [-4, -5, -6]], dtype=dtype)
    _write_record(tmp_path, values=values)
    record = _convert(tmp_path).records[0]
    for channel, series in enumerate(record.time_series):
        actual = series.to_numpy()
        assert actual.dtype == np.dtype("float64")
        np.testing.assert_array_equal(actual, values[:, channel].astype(np.float64))


def test_unsafe_int64_to_float64_conversion_is_rejected(tmp_path):
    values = np.array([[2**53 + 1, 0, 0]], dtype=np.int64)
    _write_record(tmp_path, values=values)
    series = _convert(tmp_path).records[0].time_series[0]
    with pytest.raises(TimeFValidationError, match="unsafe int64 to float64"):
        series.to_numpy()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda values: values[:, :2], r"shape \(N, 3\)"),
        (lambda values: values.astype(np.int32), "unsupported.*dtype"),
    ],
)
def test_invalid_hdf5_schema_is_rejected(tmp_path, mutator, message):
    values = mutator(np.arange(12, dtype=np.float64).reshape(4, 3))
    _write_record(tmp_path, values=values)
    with pytest.raises(TimeFValidationError, match=message):
        _convert(tmp_path)


def test_missing_vibration_data_is_rejected(tmp_path):
    path = _write_record(tmp_path)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("other", data=[1.0])
    with pytest.raises(TimeFValidationError, match="missing 'vibration_data'"):
        _convert(tmp_path)


def test_unexpected_hdf5_root_schema_is_rejected(tmp_path):
    path = _write_record(tmp_path)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("unexpected", data=[1.0])
    with pytest.raises(TimeFValidationError, match="unexpected HDF5 root schema"):
        _convert(tmp_path)


@pytest.mark.parametrize("health", ["good", "bad"])
def test_process_health_is_a_whole_record_classification_without_events(tmp_path, health):
    _write_record(tmp_path, health=health)
    dataset = _convert(tmp_path)
    record = dataset.records[0]
    task = dataset.tasks[0]

    assert isinstance(task, ClassificationTask)
    assert task.target == health
    assert task.target_schema == "process_health"
    assert task.record_ids == (record.record_id,)
    assert task.scope is None
    assert all(annotation.span is None for annotation in record.annotations)
    assert not hasattr(dataset, "events")
