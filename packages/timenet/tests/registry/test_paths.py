import pytest

from timenet.errors import TimeFFormatError
from timenet.registry._paths import safe_version_path


def test_allows_a_normal_relpath(tmp_path):
    assert safe_version_path(tmp_path, "time_series/shard-0.parquet") == tmp_path / "time_series/shard-0.parquet"


def test_rejects_parent_escape(tmp_path):
    with pytest.raises(TimeFFormatError, match="escapes"):
        safe_version_path(tmp_path, "../../../../etc/passwd")


def test_rejects_absolute_path(tmp_path):
    with pytest.raises(TimeFFormatError, match="escapes"):
        safe_version_path(tmp_path, "/etc/passwd")
