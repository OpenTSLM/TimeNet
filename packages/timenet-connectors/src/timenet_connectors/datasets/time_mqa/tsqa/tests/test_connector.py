import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.time_mqa.tsqa.anomaly_detection import ANOMALY_POINT
from timenet_connectors.datasets.time_mqa.tsqa.connector import TimeMqaConnector
from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


_DESCRIPTION = "This dataset includes data from Yahoo's servers and various service metrics."
_INSTRUCTION = "Please determine whether there are anomalies in this time series given information above."


def _anomaly_row(values: list[float], label: str, *, domain: str = "The Web") -> dict[str, str]:
    # Synthetic rows in the shape of the release: the series is printed inside the question.
    printed = ", ".join(repr(value) for value in values)
    question = f"{_DESCRIPTION} The input Time Serie is [{printed}]. {_INSTRUCTION}"
    answer = f"Based on the given information, this time series includes {label}."
    return {
        "application_domain": domain,
        "task_type": "anomaly detection",
        "QA_list": f'"question": "{question}", "answer": "{answer}"',
    }


_CASES = [
    ([2865.41985245168, 2917.96257617132, 2764.0783057705603, 7.84723445597e-05], "Normal Point", "The Web"),
    ([-386.0, -394.0, -403.0, -414.0, -425.0, -435.0, -444.0, -452.0], "Anomaly Point", "Machine sensors"),
]
_ROWS = [_anomaly_row(values, label, domain=domain) for values, label, domain in _CASES]


def _convert(write_part, rows: list[dict[str, str]] = _ROWS) -> TimeFDataset:
    return TimeMqaConnector().convert([write_part(ReleasePart.ANOMALY_DETECTION, rows)])


def test_each_row_becomes_a_record_and_a_classification_task(write_part):
    tasks = _convert(write_part).tasks_of(ClassificationTask)

    assert len(tasks) == len(_CASES)
    for number, (task, (values, label, domain)) in enumerate(zip(tasks, _CASES, strict=True), start=1):
        (record,) = task.inputs
        (signal,) = record.signals
        # Exact equality: the values keep every digit of the release.
        np.testing.assert_array_equal(signal.to_numpy(), np.array(values, dtype=np.float64))
        assert record.metadata == {"file": "anomaly_detection", "row": number, "role": "series_1"}
        assert read_value(task.prompt) == f"{_DESCRIPTION} The input Time Serie is <series 1>. {_INSTRUCTION}"
        assert task.label_set == ANOMALY_POINT
        assert task.targets is not None
        assert read_value(task.targets[0]) == label
        assert task.split is None
        assert task.metadata == {
            "file": "anomaly_detection",
            "row": number,
            "application_domain": domain,
            "task_type": "anomaly detection",
        }


def _with_cell(row: dict[str, str], question: str, answer: str) -> dict[str, str]:
    return row | {"QA_list": f'"question": "{question}", "answer": "{answer}"'}


_ROW = _anomaly_row([1.0, 2.0], "Normal Point")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_anomaly_row([1.0, 2.0], "Odd Point"), "label 'Odd Point' is not one of"),
        (_with_cell(_ROW, "No series here.", "Normal Point"), "the question prints 0 series, not one"),
        (_with_cell(_ROW, "Two: [1.0] and [2.0].", "Normal Point"), "the question prints 2 series, not one"),
        (_with_cell(_ROW, "One: [1.0, 2.0].", "Normal Point"), "answer 'Normal Point' does not follow the template"),
    ],
    ids=["an unknown label", "a question without a series", "a question with two series", "an answer off the template"],
)
def test_a_malformed_row_raises_with_its_location(write_part, row, message):
    with pytest.raises(TimeNetConnectorError, match=rf"anomaly_detection\.csv data row 3: {message}"):
        _convert(write_part, [*_ROWS, row])


def test_the_dataset_round_trips_through_storage(write_part, tmp_path):
    dataset = _convert(write_part)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())


def test_source_is_pinned():
    # Literal pins, so moving to another repo, commit, or file is a deliberate test edit.
    assert TimeMqaConnector.HF_REPO == "Time-MQA/TSQA"
    assert TimeMqaConnector.SOURCE_REVISION == "a7f9e95667b695d2647b9eff92bb6a38b5183a34"
    assert dict(TimeMqaConnector.SOURCE_FILES) == {
        "anomaly_detection": "Anomaly_Detection/anomaly_detection.csv",
        "classification": "Classification/classification.csv",
        "forecasting_imputation_1": "Forecasting+Imputation/Forecasting+Imputation_1/forecasting_imputation1.csv",
        "forecasting_imputation_2": "Forecasting+Imputation/Forecasting+Imputation_2/forecasting_imputation2.csv",
    }
