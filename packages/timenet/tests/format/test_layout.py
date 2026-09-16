import pytest

from timenet.errors import TimeFValidationError
from timenet.format.constants import DEFAULT_CHUNK_MAX_BYTES, DEFAULT_ROW_GROUP_TARGET_BYTES
from timenet.format.layout import DEFAULT_VALUES_LAYOUT, WINDOWED_VALUES_LAYOUT, ValuesLayout


def test_the_default_layout_carries_the_writer_defaults():
    # A corpus that declares nothing must keep today's targets byte for byte.
    assert DEFAULT_VALUES_LAYOUT.chunk_max_bytes == DEFAULT_CHUNK_MAX_BYTES
    assert DEFAULT_VALUES_LAYOUT.row_group_target_bytes == DEFAULT_ROW_GROUP_TARGET_BYTES


def test_the_windowed_layout_is_the_measured_fine_one():
    assert WINDOWED_VALUES_LAYOUT.chunk_max_bytes == 64 * 2**10
    assert WINDOWED_VALUES_LAYOUT.row_group_target_bytes == 512 * 2**10


@pytest.mark.parametrize(("chunk", "row_group"), [(0, 1024), (1024, 0), (-1, 1024), (1024, -1)])
def test_a_target_that_is_not_positive_is_rejected(chunk, row_group):
    with pytest.raises(TimeFValidationError, match="must be positive"):
        ValuesLayout(chunk_max_bytes=chunk, row_group_target_bytes=row_group)
