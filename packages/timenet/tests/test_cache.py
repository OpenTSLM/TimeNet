from pathlib import Path

import pytest

from timenet.cache import (
    ClearedCache,
    StoredRelease,
    cache_size,
    clear_cache,
    clear_releases,
    human_bytes,
    stored_releases,
)
from timenet.config import CACHE_TAG, is_cache_folder, make_cache_folder
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import Version
from timenet.types.references import DatasetRef


def _store_with_two_releases(root):
    """Return a store that holds timenet/hello-world at 1.0.0 and 1.1.0."""
    store = LocalRegistry(root)
    for version in (Version(1, 0, 0), Version(1, 1, 0)):
        dataset = make_dataset()
        dataset._metadata = dataset.metadata.model_copy(update={"dataset_version": version})
        store.put_dataset(dataset)
    return store


def _other_dataset():
    dataset = make_dataset()
    dataset._metadata = dataset.metadata.model_copy(update={"dataset_id": "demo/other"})
    return dataset


def _size(directory):
    return sum(path.lstat().st_size for path in directory.rglob("*") if not path.is_dir())


def _raw(cache, size=10):
    """Write ``size`` bytes of raw downloads into ``<cache>/downloads``, a folder that TimeNet made."""
    make_cache_folder(cache / "downloads")
    raw = cache / "downloads/timenet/hello-world/raw.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"x" * size)
    return raw


def test_make_cache_folder_marks_the_folder_that_it_creates(tmp_path):
    make_cache_folder(tmp_path / "cache" / "downloads")
    lines = (tmp_path / "cache/downloads" / CACHE_TAG).read_text().splitlines()
    assert lines[0] == "Signature: 8a477f597d28d172789f06886806bc55"
    assert "TimeNet" in lines[1]
    assert is_cache_folder(tmp_path / "cache" / "downloads")


def test_make_cache_folder_leaves_an_existing_folder_unmarked(tmp_path):
    (tmp_path / "downloads").mkdir()
    make_cache_folder(tmp_path / "downloads")
    assert list((tmp_path / "downloads").iterdir()) == []
    assert not is_cache_folder(tmp_path / "downloads")


def test_a_marker_of_another_tool_does_not_count(tmp_path):
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / CACHE_TAG).write_text(
        "Signature: 8a477f597d28d172789f06886806bc55\n# This file is a cache directory tag created by another tool.\n"
    )
    assert not is_cache_folder(tmp_path / "downloads")


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
        (2 * 1024**4, "2.0 TB"),
        (5 * 1024**5, "5120.0 TB"),
    ],
)
def test_human_bytes(n, expected):
    assert human_bytes(n) == expected


def test_stored_releases_lists_each_release_with_its_size(tmp_path):
    store = _store_with_two_releases(tmp_path / "store")
    first, second = (tmp_path / "store/timenet/hello-world" / version for version in ("1.0.0", "1.1.0"))
    assert stored_releases(store) == [
        StoredRelease(DatasetRef.parse("timenet/hello-world@1.0.0"), first, _size(first)),
        StoredRelease(DatasetRef.parse("timenet/hello-world@1.1.0"), second, _size(second)),
    ]
    assert _size(first) > 0


def test_stored_releases_skips_what_is_not_a_release(tmp_path):
    store = _store_with_two_releases(tmp_path / "store")
    manifest = (tmp_path / "store/timenet/hello-world/1.0.0/manifest.json").read_bytes()
    # A directory whose name is not a version, a staging directory, a hidden one, a path that is not
    # a dataset ID, and a manifest at the root and one level below.
    for stray in (
        "timenet/hello-world/draft",
        "timenet/hello-world/2.0.0.tmp-1",
        ".trash/x/1.0.0",
        "a/b/c/1.0.0",
        ".",
        "1.0.0",
    ):
        (tmp_path / "store" / stray).mkdir(parents=True, exist_ok=True)
        (tmp_path / "store" / stray / "manifest.json").write_bytes(manifest)
    assert [str(item.release) for item in stored_releases(store)] == [
        "timenet/hello-world@1.0.0",
        "timenet/hello-world@1.1.0",
    ]


def test_clear_cache_keeps_a_store_inside_the_cache(tmp_path):
    cache = tmp_path / "cache"
    store = LocalRegistry(cache / "registry")
    store.put_dataset(make_dataset())
    raw = _raw(cache)
    other = cache / "other/keep.txt"
    other.parent.mkdir()
    other.write_text("kept")
    size = _size(cache / "downloads")
    assert cache_size(cache) == size
    assert clear_cache(cache, local_registry=store) == ClearedCache(size, [cache / "downloads"], [])
    assert store.exists("timenet/hello-world", "1.0.0")
    assert other.exists()
    assert not raw.exists()


def test_clear_cache_deletes_the_control_files(tmp_path):
    cache = tmp_path / "cache"
    make_cache_folder(cache / "control")
    (cache / "control/abc.duckdb").write_bytes(b"x" * 7)
    size = _size(cache / "control")
    assert cache_size(cache) == size
    cleared = clear_cache(cache, local_registry=LocalRegistry(tmp_path / "store"))
    assert cleared == ClearedCache(size, [cache / "control"], [])
    assert not (cache / "control").exists()
    assert cache.is_dir()


def test_clear_cache_keeps_and_reports_the_folders_without_the_marker(tmp_path):
    cache = tmp_path / "cache"
    (cache / "downloads").mkdir(parents=True)
    (cache / "downloads/report.pdf").write_bytes(b"x" * 10)
    (cache / "control").write_text("a file, not a folder")
    assert cache_size(cache) == 0
    cleared = clear_cache(cache, local_registry=LocalRegistry(tmp_path / "store"))
    assert cleared == ClearedCache(0, [], [cache / "downloads", cache / "control"])
    assert (cache / "downloads/report.pdf").exists()
    assert (cache / "control").read_text() == "a file, not a folder"


def test_clear_releases_deletes_every_release_but_no_other_file(tmp_path):
    store = _store_with_two_releases(tmp_path / "store")
    store.put_dataset(_other_dataset())
    notes = tmp_path / "store/notes.txt"
    notes.write_text("kept")
    staging = tmp_path / "store/timenet/hello-world/2.0.0.tmp-1/part.bin"
    staging.parent.mkdir()
    staging.write_bytes(b"kept")
    releases = stored_releases(store)

    cleared = clear_releases(store)

    assert cleared.freed == sum(item.size_bytes for item in releases)
    assert cleared.removed == [item.path for item in releases]
    assert stored_releases(store) == []
    # The dataset directories that the call emptied go, and the files that are not releases stay.
    assert not (tmp_path / "store/demo").exists()
    assert sorted(path.relative_to(tmp_path / "store").as_posix() for path in (tmp_path / "store").rglob("*")) == [
        "notes.txt",
        "timenet",
        "timenet/hello-world",
        "timenet/hello-world/2.0.0.tmp-1",
        "timenet/hello-world/2.0.0.tmp-1/part.bin",
    ]


def test_clear_sizes_a_symlink_as_the_link_not_its_target(tmp_path):
    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * 10_000)
    downloads = tmp_path / "cache/downloads"
    make_cache_folder(downloads)
    (downloads / "link.bin").symlink_to(target)

    cleared = clear_cache(tmp_path / "cache", local_registry=LocalRegistry(tmp_path / "store"))

    assert cleared.freed < 10_000  # counted the link itself, never followed it to the 10 KB target
    assert target.exists()  # the external target is left untouched
    assert not downloads.exists()


@pytest.mark.parametrize("marked", [True, False])
def test_clear_cache_unlinks_a_symlinked_downloads_folder_only_if_marked_and_keeps_its_target(tmp_path, marked):
    # The user moved the downloads folder, and left a symlink in its place.
    target = tmp_path / "elsewhere"
    if marked:
        make_cache_folder(target)
    else:
        target.mkdir()
    (target / "raw.bin").write_bytes(b"x" * 10)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "downloads").symlink_to(target, target_is_directory=True)
    link_size = (cache / "downloads").lstat().st_size
    assert cache_size(cache) == (link_size if marked else 0)  # the link, not its target

    cleared = clear_cache(cache, local_registry=LocalRegistry(tmp_path / "store"))

    if marked:
        assert cleared == ClearedCache(link_size, [cache / "downloads"], [])
        assert not (cache / "downloads").exists() and not (cache / "downloads").is_symlink()
    else:
        assert cleared == ClearedCache(0, [], [cache / "downloads"])
        assert (cache / "downloads").is_symlink()
    assert (target / "raw.bin").exists()


def test_clear_cache_refuses_a_store_in_a_folder_that_it_deletes(tmp_path):
    make_cache_folder(tmp_path / "cache" / "downloads")
    store = LocalRegistry(tmp_path / "cache" / "downloads" / "store")
    store.put_dataset(make_dataset())
    with pytest.raises(TimeFValidationError, match="overlaps"):
        clear_cache(tmp_path / "cache", local_registry=store)
    assert store.exists("timenet/hello-world", "1.0.0")


def test_clear_cache_refuses_a_folder_that_it_deletes_in_the_store(tmp_path):
    # With the cache in the store, the downloads folder can be the directory of a dataset.
    store = LocalRegistry(tmp_path / "store")
    make_cache_folder(tmp_path / "store" / "acme" / "downloads")
    (tmp_path / "store/acme/downloads/1.0.0").mkdir()
    with pytest.raises(TimeFValidationError, match="overlaps"):
        clear_cache(tmp_path / "store" / "acme", local_registry=store)
    assert (tmp_path / "store/acme/downloads/1.0.0").is_dir()


@pytest.mark.parametrize("cache", ["~", "parent"])
def test_clear_cache_refuses_the_home_folder_or_a_folder_above_it(tmp_path, home, cache):
    # With the cache at ~, a clear would delete ~/downloads, which is ~/Downloads on a volume that ignores case.
    cache_dir = home.parent if cache == "parent" else home
    raw = _raw(cache_dir)
    with pytest.raises(TimeFValidationError, match=r"^The cache .* is the home folder or a folder above it"):
        clear_cache(cache_dir, local_registry=LocalRegistry(tmp_path / "store"))
    assert raw.exists()


def test_clear_cache_works_without_a_home_folder(tmp_path, monkeypatch):
    # For example a container whose user has no home folder.
    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    raw = _raw(tmp_path / "cache")
    cleared = clear_cache(tmp_path / "cache", local_registry=LocalRegistry(tmp_path / "store"))
    assert cleared.removed == [tmp_path / "cache" / "downloads"]
    assert not raw.exists()


def test_clear_cache_of_a_missing_cache_frees_nothing(tmp_path):
    assert cache_size(tmp_path / "cache") == 0
    cleared = clear_cache(tmp_path / "cache", local_registry=LocalRegistry(tmp_path / "store"))
    assert cleared == ClearedCache(0, [], [])
