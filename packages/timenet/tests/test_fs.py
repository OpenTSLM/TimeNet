from pathlib import Path

import pytest

from timenet._fs import lies_in, publish_directory, same_directory


def test_a_symlink_and_its_target_are_the_same_directory(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "store", target_is_directory=True)
    assert same_directory(tmp_path / "link", tmp_path / "store")


def test_a_path_lies_in_the_folder_that_a_symlink_names(tmp_path):
    (tmp_path / "store" / "acme").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "store" / "acme", target_is_directory=True)
    assert lies_in(tmp_path / "link" / "downloads", tmp_path / "store")
    assert not lies_in(tmp_path / "store", tmp_path / "link")


def test_two_missing_paths_that_differ_only_in_case_are_different_directories(tmp_path):
    assert not same_directory(tmp_path / "store", tmp_path / "Store")


def test_a_missing_path_is_never_the_same_directory_as_an_existing_one(tmp_path):
    (tmp_path / "store").mkdir()
    if (tmp_path / "Store").exists():
        pytest.skip("the temporary folder ignores case, so the case variant exists too")
    assert not same_directory(tmp_path / "Store", tmp_path / "store")


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
