import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.chengsenwang.tsqa.sample import TSQASample


@pytest.mark.parametrize("series", ["[1, true]", '[1, "2"]', "[1, NaN]", "[1, Infinity]", "[9007199254740993]"])
def test_series_rejects_coercion_and_nonfinite_values(series):
    with pytest.raises(TimeNetConnectorError, match=r"Series\[0\]\["):
        TSQASample.from_row({"Series": series, "Task": "task", "Question": "question", "Answer": "answer"})


def test_numeric_forms_preserve_grouping_keys():
    row = {"Task": "task", "Question": "question", "Answer": "answer"}
    assert (
        TSQASample.from_row(row | {"Series": "[1, 2.5]"}).signals
        == TSQASample.from_row(row | {"Series": "[[1.0, 2.5]]"}).signals
    )
