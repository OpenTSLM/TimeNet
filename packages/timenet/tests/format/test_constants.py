import pytest

from timenet.errors import TimeFValidationError
from timenet.format.constants import (
    MAX_PART_INDEX,
    SHARD_TEMPLATE,
    TASK_PART_TEMPLATE,
    check_relative_path,
    part_path,
)


def test_part_path_zero_pads_the_index():
    assert part_path(SHARD_TEMPLATE, 42) == "time_series/shard-00000042.parquet"


def test_part_path_fills_named_template_fields():
    assert part_path(TASK_PART_TEMPLATE, 0, task_type="forecast") == "tasks/task=forecast/part-00000000.parquet"


def test_part_path_allows_the_largest_eight_digit_index():
    assert part_path(SHARD_TEMPLATE, MAX_PART_INDEX) == "time_series/shard-99999999.parquet"


def test_part_path_rejects_an_index_that_needs_a_ninth_digit():
    # A ninth digit widens the name and breaks the lexical ordering the zero-pad guarantees.
    with pytest.raises(TimeFValidationError):
        part_path(SHARD_TEMPLATE, MAX_PART_INDEX + 1)


def test_part_path_rejects_a_negative_index():
    with pytest.raises(TimeFValidationError):
        part_path(SHARD_TEMPLATE, -1)


def test_check_relative_path_accepts_a_path_within_the_root():
    check_relative_path("path", "samples/part-00000000.parquet")  # must not raise


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",  # absolute path
        "../../etc/passwd",  # traversal above root
        "samples/../../etc/passwd",  # traversal mid-path
    ],
)
def test_check_relative_path_rejects_a_path_that_escapes_the_root(path):
    with pytest.raises(TimeFValidationError, match="dataset root"):
        check_relative_path("path", path)
