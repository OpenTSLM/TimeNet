import pickle

import numpy as np
import pandas as pd
import pytest

from timenet.composition import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet.types import TimeSeriesSpec
from timenet_connectors.datasets.oca_john.harespod.connector import HarespodConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.connector import HeartsHarespodConnector, _window
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.release import TASKS


def signal(values, name="original", rate=100):
    return Signal(
        id=name,
        name="rsp",
        data=np.asarray(values, dtype=np.float64),
        time_axis=RegularAxis.from_rate_hz(rate),
        spec=TimeSeriesSpec(spec_type="rsp", name="Respiration", dtype="float64", unit_value=None),
    )


def test_timestamp_windows_roundtrip(tmp_path, monkeypatch):
    directory = "harespod/altitude_ranking_respiration"
    monkeypatch.setattr(
        "timenet_connectors.datasets.yang_ai_lab.hearts_harespod.connector.TASKS", {directory: TASKS[directory]}
    )
    parent = TimeFDataset(metadata=HarespodConnector().metadata())
    parent.add_record(
        record=Record(
            record_id="harespod-219a",
            sources=(Source(id="original-source", name="Respiration", signals=(signal(np.arange(15) / 15),)),),
            metadata={"recording_start_local": "2023-02-19T08:56:34"},
        )
    )
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(parent)
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2023-02-19 08:56:34", periods=15, freq="10ms"), "rsp": np.arange(15) / 15}
    )
    folder = tmp_path / "cases/harespod/altitude_ranking_respiration"
    folder.mkdir(parents=True)
    payload = {
        "subject_id": "219a",
        "segment_dfs": {"A": frame.iloc[:5], "B": frame.iloc[5:10], "C": frame.iloc[10:]},
        "GT": ["C", "A", "B"],
    }
    (folder / "0.pkl").write_bytes(pickle.dumps(payload))
    connector = HeartsHarespodConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([tmp_path / "cases"], context)
        child.set_dependencies(context.dependency_lock())
        assert child.tasks[0].targets == ('["C", "A", "B"]',)
        np.testing.assert_allclose(child.tasks[0].inputs[1].signals[0].to_numpy(), np.arange(5, 10) / 15)
        registry.store(child)
    with registry.open_reader("yang-ai-lab/hearts-harespod", "1.0.0") as reader:
        assert len(tuple(reader.iter_tasks())) == 1


def test_reset_clock_requires_unique_matching_window():
    original = signal([0.1, 0.3, 0.4, 0.2, 0.8])
    expected = signal([0.3, 0.4, 0.2], "child")
    child = _window(expected, original, 0, 0, reset_clock=True)
    np.testing.assert_array_equal(child.to_numpy(), [0.3, 0.4, 0.2])
    assert child.metadata["parent_signal"] == original.id


def test_ambiguous_matching_window_is_rejected():
    child = _window(signal([0, 0], "child"), signal([0, 0, 0, 0]), 0, 0, reset_clock=True)
    with pytest.raises(TimeFFormatError, match="found 3"):
        child.to_arrow()


def test_reset_window_checks_sample_clock():
    child = _window(signal([0.3, 0.4], "child", rate=1), signal([0.1, 0.3, 0.4]), 0, 0, reset_clock=True)
    with pytest.raises(TimeFFormatError, match="different sample clock"):
        child.to_arrow()
