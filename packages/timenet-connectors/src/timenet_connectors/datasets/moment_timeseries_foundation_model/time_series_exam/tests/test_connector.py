import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import Split, TextHandle
from timenet_connectors.datasets.moment_timeseries_foundation_model.time_series_exam.connector import (
    TimeSeriesExamConnector,
)


_EXCERPT = Path(__file__).with_name("round3-excerpt.json")


class _ExcerptConnector(TimeSeriesExamConnector):
    """The first seven released cases, including paired signals and numeric options."""

    CARD = _EXCERPT.parent.parent / "dataset.yaml"
    SOURCE_SHA256 = hashlib.sha256(_EXCERPT.read_bytes()).hexdigest()
    QUESTION_COUNT = 7


def test_source_values_order_precision_and_test_assignments():
    rows = json.loads(_EXCERPT.read_text())
    dataset = _ExcerptConnector().convert([_EXCERPT])
    tasks = tuple(dataset.iter_tasks())
    assert len(tasks) == 7
    for row, task in zip(rows, tasks, strict=True):
        assert task.split == Split.TEST
        assert task.metadata["id"] == row["id"]
        keys = ("ts",) if "ts" in row else ("ts1", "ts2")
        signals = task.inputs[0].signals
        for signal, key in zip(signals, keys, strict=True):
            original = np.asarray(row[key], dtype=np.float64)
            restored = signal.to_numpy()
            assert restored.dtype == original.dtype
            assert restored.tobytes() == original.tobytes()
        expected_names = ("Time Series",) if "ts" in row else ("Time Series 1", "Time Series 2")
        assert tuple(signal.name for signal in signals) == expected_names
    numeric = tasks[6]
    assert numeric.metadata["options"] == [17.92, 5.37, 1.63]
    assert numeric.targets is not None
    assert isinstance(numeric.targets[0], TextHandle)
    assert numeric.prompt is not None
    assert numeric.targets[0].read() == "B) 5.37"
    assert "B) 5.37" in numeric.prompt.read()
    assert "EXAMPLE RESPONSE:" in numeric.prompt.read()
    hint = numeric.metadata["question_hint"]
    assert isinstance(hint, str)
    assert hint not in numeric.prompt.read()


def test_modified_frozen_source_is_rejected(tmp_path):
    modified = tmp_path / "qa_dataset.json"
    modified.write_bytes(_EXCERPT.read_bytes() + b"\n")
    with pytest.raises(TimeFValidationError, match="checksum"):
        _ExcerptConnector().convert([modified])
