import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import AnswerTask, ForecastingTask, InputModality
from timenet.types.text import read_value
from timenet_connectors.datasets.time_mqa.tsqa.connector import TimeMqaConnector
from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


_HISTORY = [33.0, 19.0, 14.0, 14.0, 5.0, 4.0, 44.0, 63.0, 127.0, 120.0, 187.0, 237.0]
_FUTURE = [81.0, 123.0, 232.0, 293.0]
_EARNINGS_CALL = (
    'The CFO said: "\nThanks, Mike. Results were driven by lower shipments."\nAdditionally, the historical return'
)
_RETURNS = [0.0309, -0.0183, -0.0044]


def _cell(question: str, answer: str) -> dict[str, str]:
    return {
        "application_domain": "Transport",
        "task_type": "forecasting",
        "QA_list": f'"question": "{question}", "answer": "{answer}"',
    }


def _series_row(history: list[float], future: list[float], horizon: int = len(_FUTURE)) -> dict[str, str]:
    # Synthetic rows in the shape of the release: the series is printed inside the question.
    printed = ", ".join(str(value) for value in history)
    question = (
        f"The MelbournePedestrian dataset contains pedestrian counts. The input Time Series are [{printed}]. "
        f"Please predict the next {horizon} time series point given information above."
    )
    predictions = ", ".join(str(value) for value in future)
    return _cell(question, f"Based on the given information, the predictions are [{predictions}].")


_EARNINGS_ROW = _cell(
    f"{_EARNINGS_CALL} two weeks prior to the earnings call was [{', '.join(str(value) for value in _RETURNS)}].\n"
    "Your goal is to predict whether PII's stock price will increase or decrease.",
    "[0.008520109637778939, 'up']",
)
_PROSE_ROW = _cell(
    "From August 12, 2003 the weighted ILI rate was 7.38735. Please analyze the data and provide a target result.",
    "Based on the provided textual information, I predict a moderate risk of severe influenza infections.",
)
_ROWS = [_series_row(_HISTORY, _FUTURE), _EARNINGS_ROW, _PROSE_ROW]


def _convert(write_part, rows=_ROWS):
    return TimeMqaConnector().convert([write_part(ReleasePart.FORECASTING_IMPUTATION_1, rows)])


def test_a_series_question_becomes_a_forecasting_task_from_the_history_record_to_the_future_record(write_part):
    (task,) = _convert(write_part).tasks_of(ForecastingTask)

    (history,) = task.inputs
    assert task.targets is not None
    (future,) = task.targets
    # Exact equality: the values keep every digit of the release.
    np.testing.assert_array_equal(history.signals[0].to_numpy(), np.array(_HISTORY))
    np.testing.assert_array_equal(future.signals[0].to_numpy(), np.array(_FUTURE))
    assert history.metadata == {"file": "forecasting_imputation_1", "row": 1, "role": "history"}
    assert future.metadata == {"file": "forecasting_imputation_1", "row": 1, "role": "future"}
    assert read_value(task.prompt) == (
        "The MelbournePedestrian dataset contains pedestrian counts. The input Time Series are <series 1>. "
        "Please predict the next 4 time series point given information above."
    )
    assert task.metadata == {
        "file": "forecasting_imputation_1",
        "row": 1,
        "application_domain": "Transport",
        "task_type": "forecasting",
    }


def test_a_time_mmd_question_becomes_an_answer_task_with_a_record_for_each_printed_list(write_part):
    earnings, prose = _convert(write_part).tasks_of(AnswerTask)

    (returns,) = earnings.inputs
    np.testing.assert_array_equal(returns.signals[0].to_numpy(), np.array(_RETURNS))
    assert returns.metadata == {"file": "forecasting_imputation_1", "row": 2, "role": "series_1"}
    assert read_value(earnings.prompt) == (
        f"{_EARNINGS_CALL} two weeks prior to the earnings call was <series 1>.\n"
        "Your goal is to predict whether PII's stock price will increase or decrease."
    )
    assert earnings.targets is not None
    assert read_value(earnings.targets[0]) == "[0.008520109637778939, 'up']"

    assert prose.inputs == ()
    assert prose.resolved_input_modalities == {InputModality.NO_INPUT, InputModality.TEXT}
    assert read_value(prose.prompt) == _PROSE_ROW["QA_list"].split('"question": "')[1].split('", "answer"')[0]
    assert prose.targets is not None
    assert read_value(prose.targets[0]).startswith("Based on the provided textual information")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_series_row(_HISTORY, _FUTURE[:-1]), "the question asks for 4 values, but the answer prints 3"),
        (
            _series_row(_HISTORY, _FUTURE)
            | {"QA_list": _series_row(_HISTORY, _FUTURE)["QA_list"].replace("the predictions are", "the values are")},
            "answer .* does not follow the template",
        ),
        (_series_row([], _FUTURE), "the question prints 0 series, not one"),
    ],
    ids=["one prediction too few", "an answer off the template", "a question without a series"],
)
def test_a_malformed_row_raises_with_its_location(write_part, row, message):
    with pytest.raises(TimeNetConnectorError, match=rf"forecasting_imputation1\.csv data row 4: {message}"):
        _convert(write_part, [*_ROWS, row])


def test_the_dataset_round_trips_through_storage_with_its_record_targets(write_part, tmp_path):
    dataset = _convert(write_part)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        restored = reader.read()
    assert_datasets_equal(dataset, restored)
    (task,) = restored.tasks_of(ForecastingTask)
    history, future = task.named_records()
    assert (history.metadata["role"], future.metadata["role"]) == ("history", "future")
