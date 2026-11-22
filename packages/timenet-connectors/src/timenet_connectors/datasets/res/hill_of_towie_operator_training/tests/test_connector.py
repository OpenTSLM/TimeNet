"""Check that the model input and supervision survive a TimeF round trip."""

from datetime import UTC, datetime
import gzip
import json

import pytest

from timenet import TimeNet
from timenet.engine import store_dataset
from timenet.errors import TimeNetConnectorError
from timenet.types import AnswerTask, ClassificationTask, Split, read_value, unix_us, ureg
from timenet_connectors.datasets.res.hill_of_towie_operator_training import connector


def _row(*, steps=36, target="Evidence.\nAnswer: generating", cutoff="2024-09-30T23:00:00+00:00"):
    return {
        "stage": "domain_qa",
        "cutoff": cutoff,
        "turbine": "T07",
        "steps": steps,
        "label": "generating",
        "example": {
            "series": [
                {"name": "P: active power (kW)", "values": [0, None, *([500] * (steps - 2))]},
                {"name": "OIL: gear oil temperature (°C)", "values": [41] * steps},
            ],
            "prompt": "What is the turbine doing?",
            "options": ["idle", "generating"],
            "answer": target,
            "category": "operating_state",
        },
    }


def _archive(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(connector, "_EXPECTED_COUNTS", {Split.TRAIN: 2, Split.VALIDATION: 1, Split.TEST: 1})
    paths = {
        Split.TRAIN: _archive(tmp_path / "train.gz", [_row(), _row(target="generating")]),
        Split.VALIDATION: _archive(tmp_path / "validation.gz", [_row(steps=1008, cutoff="2024-10-31T12:00:00+00:00")]),
        Split.TEST: _archive(tmp_path / "test.gz", [_row(cutoff="2025-02-01T12:00:00+00:00")]),
    }
    dataset = connector.HillOfTowieOperatorTrainingConnector().convert(paths)
    registry = tmp_path / "registry"
    store_dataset(dataset, registry)
    with TimeNet(sources=[registry]).open("res/hill-of-towie-operator-training@1.0.0") as reader:
        response, classification = reader.iter_tasks(split="train")
        assert isinstance(response, AnswerTask)
        assert response.prompt is not None
        assert response.targets is not None
        assert response.prompt.read() == "What is the turbine doing?"
        assert read_value(response.targets[0]) == "Evidence.\nAnswer: generating"
        assert response.metadata["options"] == ["idle", "generating"]
        assert isinstance(classification, ClassificationTask)
        assert classification.label_set.ordered
        assert [label.name for label in classification.label_set.labels] == ["idle", "generating"]
        assert classification.targets is not None
        assert read_value(classification.targets[0]) == "generating"
        record = response.inputs[0]
        assert record.metadata["turbine_id"] == "T07"
        assert record.start_time.timestamp == unix_us(datetime(2024, 9, 30, 17, 10, tzinfo=UTC))
        expected_order = ["P: active power (kW)", "OIL: gear oil temperature (°C)"]
        assert record.metadata["channel_order"] == expected_order
        by_name = {signal.name: signal for signal in record.signals}
        power, oil = (by_name[name] for name in expected_order)
        assert power.spec.unit_value == ureg.kilowatt
        assert oil.spec.unit_value == ureg.degC
        assert power.read_steps(0, 3).to_pylist() == [0, None, 500]
        (weekly,) = reader.iter_tasks(split="validation")
        assert weekly.inputs[0].signals[0].n_values == 1008
        (held_out,) = reader.iter_tasks(split="test")
        assert held_out.inputs[0].metadata["cutoff_utc"] == "2025-02-01T12:00:00+00:00"


def test_rejects_incomplete_split(tmp_path):
    raw = {Split.TRAIN: _archive(tmp_path / "train.gz", [])}
    with pytest.raises(TimeNetConnectorError, match="has 0 rows, expected 9709"):
        connector.HillOfTowieOperatorTrainingConnector().convert(raw)
