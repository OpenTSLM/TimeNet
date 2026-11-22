"""Synthetic NASA FD001 fixtures; no licensed source-data rows or network calls."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timenet.types import ScalarPredictionTask, Split, read_value
from timenet_connectors.datasets.musitu.cmapss_fd001.connector import (
    COLS,
    WINDOW,
    CmapssFd001Connector,
    _read,
)


def _write_engine_rows(path: Path, counts: dict[int, int]) -> None:
    rows = []
    for unit, length in counts.items():
        for cycle in range(1, length + 1):
            row = [unit, cycle] + [float(unit * 100 + cycle + i / 100) for i in range(len(COLS) - 2)]
            rows.append(row)
    np.savetxt(path, np.array(rows), fmt="%.6f")


def _release(root: Path) -> Path:
    # Invented telemetry with multiple engine lengths to exercise overlapping windows.
    # Development engine #1 is validation; #2 and #3 remain training.
    _write_engine_rows(root / "train_FD001.txt", {1: 40, 2: 47, 3: 30})
    _write_engine_rows(root / "test_FD001.txt", {1: 32, 2: 35})
    (root / "RUL_FD001.txt").write_text("15\n50\n", encoding="utf-8")
    return root


def _targets(dataset) -> list[tuple[Split, float]]:
    tasks = dataset.tasks_of(ScalarPredictionTask)
    return [(task.split, float(read_value(task.targets[0]))) for task in tasks]


def test_fixture_conversion_and_engine_grouping(tmp_path):
    dataset = CmapssFd001Connector().convert(_release(tmp_path))
    assert len(dataset.records) == 6
    assert len(dataset.tasks_of(ScalarPredictionTask)) == 6
    labels = _targets(dataset)
    assert labels.count((Split.VALIDATION, 10.0)) == 1
    assert labels.count((Split.TRAIN, 17.0)) == 1  # engine 2 @ cycle 30, final cycle 47
    assert labels.count((Split.TRAIN, 2.0)) == 1  # engine 2 @ cycle 45
    assert labels.count((Split.TRAIN, 0.0)) == 1  # engine 3 @ cycle 30
    assert labels.count((Split.TEST, 15.0)) == 1
    assert labels.count((Split.TEST, 50.0)) == 1
    seen = {}
    for record, task in zip(dataset.records, dataset.tasks, strict=True):
        meta = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
        key = (meta["cohort"], meta["engine_id"])
        seen.setdefault(key, set()).add(task.split)
        assert len(record.signals) == 24
        assert len({signal.name for signal in record.signals}) == 24
        for signal in record.signals:
            values = signal.to_arrow().to_numpy(zero_copy_only=False)
            assert len(values) == WINDOW
            assert np.isfinite(values).all()
    assert all(len(split_set) == 1 for split_set in seen.values())
    assert seen[("official_test", 1)] == {Split.TEST}
    assert seen[("official_test", 2)] == {Split.TEST}
    assert seen[("development", 1)] == {Split.VALIDATION}
    assert set(seen[("development", 2)]) == {Split.TRAIN}


def test_exact_sensor_values_are_past_only(tmp_path):
    dataset = CmapssFd001Connector().convert(_release(tmp_path))
    record = dataset.records[0]
    sensors = {signal.name: signal.to_arrow().to_numpy(zero_copy_only=False) for signal in record.signals}
    assert sensors["setting_1"].tolist() == pytest.approx([101 + i for i in range(WINDOW)])
    assert sensors["sensor_21"][-1] == pytest.approx(130.23)
    assert "unit" not in sensors
    assert "cycle" not in sensors


@pytest.mark.parametrize("field,first,second,expected", [
    ("cycle", 2, 1, "Duplicate"),
    ("cycle", 2, 9, "Noncontiguous"),
    ("unit", 1, 1.5, "positive integers"),
    ("sensor_1", 2, np.nan, "Non-finite"),
])
def test_corrupt_source_rows_fail_closed(tmp_path, field, first, second, expected):
    path = _release(tmp_path) / "train_FD001.txt"
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    frame.columns = COLS
    frame.loc[first, field] = second
    frame.to_csv(path, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match=expected):
        _read(path)


@pytest.mark.parametrize("offset,match", [
    ("-2\n50\n", "nonnegative integers"),
    ("1.5\n50\n", "nonnegative integers"),
    ("NaN\n50\n", "nonnegative integers"),
    ("15\n", "do not match engine count"),
])
def test_invalid_official_labels_are_rejected(tmp_path, offset, match):
    _release(tmp_path)
    (tmp_path / "RUL_FD001.txt").write_text(offset, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        CmapssFd001Connector().convert(tmp_path)


def test_download_uses_explicit_local_files_without_network(tmp_path, monkeypatch):
    root = _release(tmp_path)
    monkeypatch.setenv("CMAPSS_DATA_DIR", str(root))
    def block(*_args, **_kwargs):
        raise AssertionError("Local data should not trigger a remote download")
    monkeypatch.setattr("timenet_connectors.datasets.musitu.cmapss_fd001.connector.urlopen", block)
    assert CmapssFd001Connector().download(tmp_path / "unused") == root
