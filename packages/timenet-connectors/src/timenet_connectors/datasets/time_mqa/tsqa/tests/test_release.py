import csv
from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.time_mqa.tsqa.release import (
    ReleaseFile,
    ReleasePart,
    find_printed_series,
    replace_printed_series_with_placeholders,
)


def _write_file(
    directory: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = ReleasePart.ANOMALY_DETECTION.columns
) -> ReleaseFile:
    path = directory / "anomaly_detection.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return ReleaseFile(part=ReleasePart.ANOMALY_DETECTION, path=path)


def _cell(question: str, answer: str) -> str:
    return f'"question": "{question}", "answer": "{answer}"'


def _row(cell: str) -> dict[str, str]:
    return {"application_domain": "AIOps", "task_type": "anomaly detection", "QA_list": cell}


def test_a_cell_keeps_raw_quotes_and_line_breaks_in_its_texts(tmp_path):
    # Like the earnings-call questions of the release, the texts hold quotes and line breaks as they are.
    question = 'The CFO said: "\nThanks, Mike." The returns were [0.01, -0.02].'
    answer = "[0.0085, 'up']"
    file = _write_file(tmp_path, [_row(_cell(question, answer)), {**_row(""), "application_domain": "Web"}])

    (row,) = file.read_rows()

    assert (row.number, row.question, row.answer) == (1, question, answer)
    assert (row.application_domain, row.task_type) == ("AIOps", "anomaly detection")


def test_only_bracketed_lists_of_decimal_numbers_are_printed_series():
    text = "Fill [?, ?, ?] in [note 1] of [7.84e-05, -386, 2.5]. Compare [] with [0.0, 1.0]."

    series = find_printed_series(text)

    assert [printed.values for printed in series] == [(7.84e-05, -386.0, 2.5), (0.0, 1.0)]
    assert replace_printed_series_with_placeholders(text, series) == (
        "Fill [?, ?, ?] in [note 1] of <series 1>. Compare [] with <series 2>."
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("question: 1, answer: 2")], r"anomaly_detection\.csv data row 1: QA_list does not have the form"),
        (
            [_row(_cell("q", "a")), {**_row(_cell("q", "a")), "task_type": "forecasting"}],
            r"data row 2: task_type 'forecasting' is not one of",
        ),
    ],
    ids=["a cell without the question-and-answer form", "a task type of another file"],
)
def test_a_malformed_row_raises_with_its_location(tmp_path, rows, message):
    file = _write_file(tmp_path, rows)
    with pytest.raises(TimeNetConnectorError, match=message):
        for row in file.read_rows():
            _ = row.family


def test_a_file_with_another_header_raises(tmp_path):
    file = _write_file(tmp_path, [], columns=("domain", "task_type", "QA_list"))
    with pytest.raises(TimeNetConnectorError, match=r"anomaly_detection\.csv: expected columns"):
        list(file.read_rows())
