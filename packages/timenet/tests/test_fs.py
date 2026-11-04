from pathlib import Path

import pytest

from timenet._fs import publish_directory


def _directories(tmp_path: Path) -> tuple[Path, Path]:
    """Make an old ``final`` and a new ``staging`` directory, each with one marker file."""
    final = tmp_path / "1.0.0"
    final.mkdir()
    (final / "marker").write_text("old")
    staging = tmp_path / "1.0.0.tmp-new"
    staging.mkdir()
    (staging / "marker").write_text("new")
    return staging, final


def test_a_stopped_rename_moves_the_old_copy_back(tmp_path, monkeypatch):
    staging, final = _directories(tmp_path)
    rename = Path.replace

    def replace(self, target):
        if self == staging:
            raise KeyboardInterrupt
        return rename(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        publish_directory(staging, final)
    assert (final / "marker").read_text() == "old"
    assert not list(tmp_path.glob("1.0.0.old-*"))


def test_a_failed_restore_names_where_the_old_copy_is(tmp_path, monkeypatch):
    staging, final = _directories(tmp_path)
    rename = Path.replace

    def replace(self, target):
        if self == staging or ".old-" in self.name:
            raise OSError("rename failed")
        return rename(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match=r"it is in .*/1\.0\.0\.old-[0-9a-f]+, so rename that to") as raised:
        publish_directory(staging, final)
    (old,) = tmp_path.glob("1.0.0.old-*")
    assert str(old) in str(raised.value)
    assert (old / "marker").read_text() == "old"
