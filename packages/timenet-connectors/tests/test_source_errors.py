import pytest

from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet_connectors.sources.errors import error_context


def test_error_contexts_preserve_the_exception_and_nest_locations():
    error = TimeNetDownloadError("missing file")
    with (
        pytest.raises(TimeNetConnectorError, match=r"^exams\.csv: row 17: missing file$") as info,
        error_context("exams.csv"),
        error_context("row 17"),
    ):
        raise error
    assert info.value is error


def test_error_context_leaves_other_errors_alone():
    with pytest.raises(KeyError) as info, error_context("row 1"):
        raise KeyError("x")
    assert info.value.args == ("x",)
