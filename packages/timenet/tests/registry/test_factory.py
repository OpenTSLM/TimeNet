import pytest

from timenet.errors import TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry


def test_unknown_dataset_error_is_actionable(tmp_path):
    with pytest.raises(TimeNetDatasetNotFoundError, match="Build one with"):
        LocalRegistry(tmp_path).get_manifest("nope/missing")
