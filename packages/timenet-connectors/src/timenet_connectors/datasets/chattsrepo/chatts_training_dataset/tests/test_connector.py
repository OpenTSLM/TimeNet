"""Check that the ChatTS source shape survives conversion and TimeF storage."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from timenet.dataset import OrdinalAxis, TimeFDataset
from timenet.types import InputModality
from timenet_connectors.datasets.chattsrepo.chatts_training_dataset.connector import (
    CONFIGS,
    ChatTSConnector,
    ChatTSFile,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _files() -> list[ChatTSFile]:
    return [ChatTSFile(config, FIXTURES / config / "train.jsonl") for config in CONFIGS]


def test_convert_keeps_ordered_values_source_text_and_validation_split():
    dataset = ChatTSConnector().convert(_files())
    records = dataset.records
    expected_ids = [
        "chatts-align_256-000000",
        "chatts-align_256-000001",
        *(f"chatts-{config}-000000" for config in CONFIGS[1:]),
    ]
    assert [record.id for record in records] == expected_ids
    first = records[0]
    assert [signal.name for signal in first.signals] == ["s00", "s01"]
    assert [signal.n_values for signal in first.signals] == [3, 3]
    assert all(isinstance(signal.time_axis, OrdinalAxis) for record in records for signal in record.signals)
    assert first.signals[0].spec.unit_value is None
    assert first.signals[0].spec.dtype == "float64"
    assert first.signals[0].to_arrow().to_pylist() == [1.234567890123, 2.5, 3.75]
    assert first.signals[1].to_arrow().to_pylist() == [0.9, 0.8, 0.7]
    assert records[1].signals[0].to_arrow().to_pylist() == [10.0, 12.0, 11.0]
    assert [record.annotations[0].value for record in records] == ["align_256", *CONFIGS]

    train = dataset.get_train()
    validation = dataset.get_validation()
    assert len(train) == 5
    assert len(validation) == 1
    assert train[0].prompt == ("There are two metrics: Requests: <ts><ts/>; Success Rate: <ts><ts/>. Which one rises?")
    assert train[0].targets == ("Requests rises while Success Rate falls.",)
    assert train[0].inputs == (first,)
    assert train[0].resolved_input_modalities == frozenset({InputModality.TIME_SERIES, InputModality.TEXT})
    # The IFT source prompt says length 6, but its actual array has 3 values.
    assert records[4].signals[0].n_values == 3
    assert validation[0].targets == ('{"trend": "steady"}',)
    assert [task.id for task in dataset.get_all()] == [f"{record_id}-answer" for record_id in expected_ids]


def test_fixture_round_trips_records_values_and_task_splits(tmp_path):
    dataset = ChatTSConnector().convert(_files())
    version_dir = dataset.write(path=tmp_path)
    loaded = TimeFDataset.open(path=version_dir)

    assert len(loaded.records) == 6
    assert [signal.to_arrow().to_pylist() for signal in loaded.records[0].signals] == [
        [1.234567890123, 2.5, 3.75],
        [0.9, 0.8, 0.7],
    ]
    assert len(loaded.get_train()) == 5
    assert len(loaded.get_validation()) == 1
    assert loaded.get_validation()[0].prompt == (
        "There is a time series of length 3: <ts><ts/>. Return a JSON summary."
    )


def test_marker_mismatch_names_the_source_row(tmp_path):
    bad = tmp_path / "train.jsonl"
    bad.write_text(
        '{"input":"A: <ts><ts/>","output":"One answer.","timeseries":[[1.0],[2.0]]}\n',
        encoding="utf-8",
    )
    files = _files()
    files[0] = ChatTSFile("align_256", bad)
    with pytest.raises(ValidationError, match=r"input has 1 time-series markers for 2 series"):
        ChatTSConnector().convert(files)
