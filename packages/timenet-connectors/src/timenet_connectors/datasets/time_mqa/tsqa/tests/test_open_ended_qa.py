import numpy as np

from timenet.types import AnswerTask
from timenet.types.text import read_value
from timenet_connectors.datasets.time_mqa.tsqa.connector import TimeMqaConnector
from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


def _row(task_type: str, question_format: str, question: str, answer: str, *, domain: str = "Nature") -> dict[str, str]:
    return {
        "application_domain": domain,
        "task_type": task_type,
        "question_format": question_format,
        "QA_list": "" if not question else f'"question": "{question}", "answer": "{answer}"',
    }


_SERIES = [8.0, 8.0, 5.0, 13.0, 0.0, 0.0, 0.0, 10.0]
_CHOICES = ([-20.34, -20.72, -20.25], [-11.06, -8.91, -7.66], [-27.34, -28.16, -29.41])
_ROWS = [
    _row(
        "anomaly_detection",
        "open_ended_question",
        f"Identify any anomalies within the data set [{', '.join(str(value) for value in _SERIES)}].",
        "The sequence '0.0, 0.0, 0.0' is an anomaly considering the surrounding higher values.",
    ),
    _row("statistical property", "multiple_choice", "", ""),
    _row(
        "volatility analysis",
        "multiple_choice",
        "Which section shows the highest volatility? "
        + " ".join(
            f"{letter}) [{', '.join(str(value) for value in choice)}]"
            for letter, choice in zip("ABC", _CHOICES, strict=True)
        ),
        "C) [-27.34, -28.16, -29.41]. This section shows a wide range of high variance.",
        domain="Traffic",
    ),
    _row(
        "structural_break_detection",
        "true/false",
        "Identify the point that might indicate a structural break: A. 18.09, B. 10.84, C. 5.31",
        "False",
    ),
]


def test_each_question_becomes_an_answer_task_with_a_record_per_printed_series(write_part):
    tasks = TimeMqaConnector().convert([write_part(ReleasePart.OPEN_ENDED_QA, _ROWS)]).tasks_of(AnswerTask)

    anomaly, choices, no_series = tasks
    (record,) = anomaly.inputs
    # Exact equality: the values keep every digit of the release.
    np.testing.assert_array_equal(record.signals[0].to_numpy(), np.array(_SERIES))
    assert read_value(anomaly.prompt) == "Identify any anomalies within the data set <series 1>."
    assert anomaly.targets is not None
    assert read_value(anomaly.targets[0]) == (
        "The sequence '0.0, 0.0, 0.0' is an anomaly considering the surrounding higher values."
    )
    assert anomaly.metadata == {
        "file": "open_ended_qa",
        "row": 1,
        "application_domain": "Nature",
        "task_type": "anomaly_detection",
        "question_format": "open_ended_question",
    }

    # The empty cell of data row 2 holds no question, so the next task comes from data row 3.
    assert [input.signals[0].to_numpy().tolist() for input in choices.inputs] == [list(choice) for choice in _CHOICES]
    assert [input.metadata["role"] for input in choices.inputs] == ["series_1", "series_2", "series_3"]
    assert (
        read_value(choices.prompt)
        == "Which section shows the highest volatility? A) <series 1> B) <series 2> C) <series 3>"
    )
    assert choices.targets is not None
    # The answer keeps the series it prints as text.
    assert (
        read_value(choices.targets[0])
        == "C) [-27.34, -28.16, -29.41]. This section shows a wide range of high variance."
    )
    assert choices.metadata["row"] == 3

    assert no_series.inputs == ()
    assert read_value(no_series.prompt) == (
        "Identify the point that might indicate a structural break: A. 18.09, B. 10.84, C. 5.31"
    )
    assert no_series.targets is not None
    assert read_value(no_series.targets[0]) == "False"
    assert (no_series.metadata["row"], no_series.metadata["question_format"]) == (4, "true/false")
