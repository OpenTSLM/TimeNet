import numpy as np
import pandas as pd
import pytest

from timenet.dataset import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.oca_john.harespod.connector import CHANNELS, HarespodConnector


@pytest.fixture
def subject(tmp_path):
    subject = tmp_path / "219a"
    subject.mkdir()
    for channel in CHANNELS:
        frame = pd.DataFrame(
            {"timestamp": pd.date_range("2023-02-19 08:56:34", periods=5, freq="10ms"), "value": np.linspace(0, 1, 5)}
        )
        frame.to_csv(subject / f"{channel}_5cut.csv", header=False, index=False)
    (subject / "key_timestamp.txt").write_text("t1 = '2023-02-19 08:56:34.254' # 1500\n")
    return tmp_path


def test_preserves_all_channels_and_original_clock(subject, tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    dataset = HarespodConnector().convert([subject])
    registry.store(dataset)
    with registry.open_reader("oca-john/harespod", "1.0.0") as reader:
        (record,) = reader.iter_records()
        assert len(record.signals) == 5
        assert record.metadata["recording_start_local"] == "2023-02-19T08:56:34.000000"
        assert "1500" in str(record.metadata["key_timestamps"])
        signal = record.signals[0]
        assert isinstance(signal.time_axis, RegularAxis)
        assert signal.time_axis.period_us == 10000
        assert signal.to_numpy().tolist() == [0, 0.25, 0.5, 0.75, 1]


def test_nonfinite_values_are_rejected_lazily(subject):
    (subject / "219a/rsp_5cut.csv").write_text("2023-02-19 08:56:34,inf\n")
    dataset = HarespodConnector().convert([subject])
    with pytest.raises(TimeFFormatError, match="non-finite"):
        next(signal for signal in dataset.records[0].signals if signal.name == "rsp").to_arrow()


def test_unordered_time_is_rejected(subject):
    (subject / "219a/rsp_5cut.csv").write_text("2023-02-19 08:56:34,0\n2023-02-19 08:56:34,1\n")
    with pytest.raises(TimeFFormatError, match="unordered"):
        HarespodConnector().convert([subject])
