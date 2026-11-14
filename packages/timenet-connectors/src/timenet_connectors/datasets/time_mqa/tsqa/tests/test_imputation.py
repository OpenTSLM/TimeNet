import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import TSEditingTask
from timenet.types.text import read_value
from timenet_connectors.datasets.time_mqa.tsqa.connector import TimeMqaConnector
from timenet_connectors.datasets.time_mqa.tsqa.release import SERIES_WITH_GAPS, ReleasePart


_COMPLETE = [19.47, 18.94, 19.31, 19.75, 19.44, 21.31, 21.03, 21.28]
_GAPS = (2, 5)
_INSTRUCTION = "Please give full time series with missing value imputed. The missing values are represented by 'X'."


def _row(complete: list[float], gaps: tuple[int, ...], *, answer: list[float] | None = None) -> dict[str, str]:
    # Synthetic rows in the shape of the release: the question prints 'X' in place of each gap.
    printed = ", ".join("'X'" if position in gaps else str(value) for position, value in enumerate(complete))
    imputed = ", ".join(str(value) for value in (complete if answer is None else answer))
    return {
        "application_domain": "Health",
        "task_type": "imputation",
        "QA_list": (
            f'"question": "The EEG dataset is sampled at 256 Hz. The input Time Series are [{printed}]. {_INSTRUCTION}", '
            f'"answer": "Based on the given information, the full time series with missing value imputed are [{imputed}]."'
        ),
    }


_ROWS = [_row(_COMPLETE, _GAPS)]


def _convert(write_part, rows=_ROWS):
    return TimeMqaConnector().convert([write_part(ReleasePart.FORECASTING_IMPUTATION_2, rows)])


def test_a_row_becomes_an_editing_task_from_the_record_with_gaps_to_the_complete_record(write_part):
    (task,) = _convert(write_part).tasks_of(TSEditingTask)

    (with_gaps,) = task.inputs
    assert task.targets is not None
    (complete,) = task.targets
    (signal,) = with_gaps.signals
    assert signal.spec == SERIES_WITH_GAPS
    # A missing value is an Arrow null, not a number, so the input keeps the gaps of the question.
    assert signal.to_arrow().to_pylist() == [
        None if position in _GAPS else value for position, value in enumerate(_COMPLETE)
    ]
    np.testing.assert_array_equal(complete.signals[0].to_numpy(), np.array(_COMPLETE))
    assert with_gaps.metadata == {"file": "forecasting_imputation_2", "row": 1, "role": "with_gaps"}
    assert complete.metadata == {"file": "forecasting_imputation_2", "row": 1, "role": "complete"}
    assert (
        read_value(task.prompt)
        == f"The EEG dataset is sampled at 256 Hz. The input Time Series are <series 1>. {_INSTRUCTION}"
    )
    assert task.metadata == {
        "file": "forecasting_imputation_2",
        "row": 1,
        "application_domain": "Health",
        "task_type": "imputation",
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row(_COMPLETE, ()), "the question prints a series without gaps"),
        (
            _row(_COMPLETE, _GAPS, answer=_COMPLETE[:-1]),
            "the answer does not print the series of the question with its gaps filled",
        ),
        (
            _row(_COMPLETE, _GAPS, answer=[*_COMPLETE[:-1], 0.0]),
            "the answer does not print the series of the question with its gaps filled",
        ),
    ],
    ids=["a question without gaps", "an answer one value short", "an answer that changes a known value"],
)
def test_a_malformed_row_raises_with_its_location(write_part, row, message):
    with pytest.raises(TimeNetConnectorError, match=rf"forecasting_imputation2\.csv data row 2: {message}"):
        _convert(write_part, [*_ROWS, row])


def test_the_dataset_round_trips_through_storage_with_its_gaps(write_part, tmp_path):
    dataset = _convert(write_part)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        restored = reader.read()
    assert_datasets_equal(dataset, restored)
    (task,) = restored.tasks_of(TSEditingTask)
    (signal,) = task.inputs[0].signals
    assert signal.to_arrow().null_count == len(_GAPS)
