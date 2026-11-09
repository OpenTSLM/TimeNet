from pathlib import Path

from _demo_child import _child_of_the_demo
from _origin_record import origin_file
import pytest

from timenet.cache import (
    ClearedCache,
    KeptRelease,
    StoredRelease,
    cache_size,
    clear_cache,
    clear_releases,
    human_bytes,
    releases_to_clear,
    stored_releases,
)
from timenet.client import TimeNet
from timenet.config import CACHE_TAG, is_cache_folder, make_cache_folder
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetRegistryError
from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.registry.origin import OriginKind
from timenet.testing import make_dataset
from timenet.types import Version
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


PARENT = DatasetRef.parse("test/child@1.0.0")
GRAND = DatasetRef.parse("timenet/hello-world@1.0.0")
BUILT = DatasetRef.parse("test/built@1.0.0")
OTHER = DatasetRef.parse("demo/other@1.0.0")
HAND = DatasetRef.parse("demo/hand@1.0.0")


def _store_with_two_releases(root):
    """Return a store that holds timenet/hello-world at 1.0.0 and 1.1.0."""
    store = LocalRegistry(root)
    for version in (Version(1, 0, 0), Version(1, 1, 0)):
        dataset = make_dataset()
        dataset._metadata = dataset.metadata.model_copy(update={"dataset_version": version})
        store.put_dataset(dataset)
    return store


def test_clear_releases_refuses_a_read_only_local_registry(tmp_path):
    store = LocalRegistry(_store_with_two_releases(tmp_path / "store").root, read_only=True)
    before = stored_releases(store)
    with pytest.raises(TimeNetRegistryError, match=r"read_only=True, so it refuses a clear of its releases\."):
        clear_releases(store, kinds=[OriginKind.DOWNLOAD, OriginKind.BUILD])
    assert stored_releases(store) == before


def _layout(store):
    """Store the five releases of the layout.

    PARENT reads GRAND as a payload source, BUILT declares PARENT as a parent, and OTHER and HAND
    stand alone.
    """
    _child_of_the_demo(store.root)
    store.put_dataset(make_dataset(dataset_id=BUILT.dataset_id, parents={ParentAlias("parent"): PARENT}))
    store.put_dataset(make_dataset(dataset_id=OTHER.dataset_id))
    store.put_dataset(make_dataset(dataset_id=HAND.dataset_id))


def _record(store):
    """Record PARENT, GRAND and OTHER as copies and BUILT as a build. HAND gets no record."""
    for release in (PARENT, GRAND, OTHER):
        store.record_origin(release, OriginKind.DOWNLOAD, "https://registry.example.org")
    store.record_origin(BUILT, OriginKind.BUILD, "timenet-connectors 0.9.0")


def _held(store):
    """Return the releases of the layout that the store holds."""
    return {
        release
        for release in (BUILT, PARENT, GRAND, OTHER, HAND)
        if store.exists(release.dataset_id, str(release.version))
    }


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
    releases = stored_releases(store)
    assert releases == [
        StoredRelease(DatasetRef.parse("timenet/hello-world@1.0.0"), first, None),
        StoredRelease(DatasetRef.parse("timenet/hello-world@1.1.0"), second, None),
    ]
    assert [item.size() for item in releases] == [_size(first), _size(second)]
    assert _size(first) > 0


def test_stored_releases_carry_their_origin(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    releases = stored_releases(store)
    assert {item.release: item.origin for item in releases} == {
        release: store.read_origin(release) for release in (BUILT, PARENT, GRAND, OTHER, HAND)
    }
    # A release without a record counts as a build.
    assert {item.release: item.kind for item in releases} == {
        BUILT: OriginKind.BUILD,
        HAND: OriginKind.BUILD,
        PARENT: OriginKind.DOWNLOAD,
        GRAND: OriginKind.DOWNLOAD,
        OTHER: OriginKind.DOWNLOAD,
    }


def test_stored_releases_skips_what_is_not_a_release(tmp_path):
    store = _store_with_two_releases(tmp_path / "store")
    manifest = (tmp_path / "store/timenet/hello-world/1.0.0/manifest.json").read_bytes()
    # A directory whose name is not a version, a version in another spelling, a staging directory, a
    # hidden one, a path that is not a dataset ID, and a manifest at the root and one level below.
    strays = (
        "timenet/hello-world/draft",
        "timenet/hello-world/01.0.0",
        "timenet/hello-world/2.0.0.tmp-1",
        ".trash/x/1.0.0",
        "a/b/c/1.0.0",
        ".",
        "1.0.0",
    )
    for stray in strays:
        (tmp_path / "store" / stray).mkdir(parents=True, exist_ok=True)
        (tmp_path / "store" / stray / "manifest.json").write_bytes(manifest)
    assert [str(item.release) for item in stored_releases(store)] == [
        "timenet/hello-world@1.0.0",
        "timenet/hello-world@1.1.0",
    ]


def test_releases_skips_an_unfinished_write_of_a_dataset_folder(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    # The staging folder of a dataset has a name that is also a valid dataset ID.
    staging = store.root / "timenet/hello-world.tmp-1/1.0.0"
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_bytes((release_dir(store.root, GRAND) / "manifest.json").read_bytes())
    assert [str(release) for release in store.releases()] == ["timenet/hello-world@1.0.0"]


def test_clear_cache_keeps_a_local_registry_inside_the_cache(tmp_path):
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
    store.put_dataset(make_dataset(dataset_id=OTHER.dataset_id))
    # A download goes as well as the builds.
    store.record_origin(OTHER, OriginKind.DOWNLOAD, "https://registry.example.org")
    notes = tmp_path / "store/notes.txt"
    notes.write_text("kept")
    staging = tmp_path / "store/timenet/hello-world/2.0.0.tmp-1/part.bin"
    staging.parent.mkdir()
    staging.write_bytes(b"kept")
    releases = stored_releases(store)
    size = sum(item.size() for item in releases)

    cleared = clear_releases(store, kinds=[OriginKind.DOWNLOAD, OriginKind.BUILD])

    assert cleared.freed == size
    assert cleared.removed == [item.path for item in releases]
    assert cleared.kept == []
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


def test_clear_cache_refuses_a_local_registry_in_a_folder_that_it_deletes(tmp_path):
    make_cache_folder(tmp_path / "cache" / "downloads")
    store = LocalRegistry(tmp_path / "cache" / "downloads" / "store")
    store.put_dataset(make_dataset())
    with pytest.raises(TimeFValidationError, match="overlaps"):
        clear_cache(tmp_path / "cache", local_registry=store)
    assert store.exists("timenet/hello-world", "1.0.0")


def test_clear_cache_refuses_a_folder_that_it_deletes_in_the_local_registry(tmp_path):
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


def test_clearing_the_downloads_keeps_each_download_that_a_build_needs(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    cleared = clear_releases(store, kinds=[OriginKind.DOWNLOAD])
    assert _held(store) == {BUILT, PARENT, GRAND, HAND}
    # BUILT declares PARENT as a parent, and PARENT reads files of GRAND.
    assert cleared.kept == [KeptRelease(PARENT, BUILT), KeptRelease(GRAND, PARENT)]


def test_clearing_the_downloads_deletes_a_parent_that_only_a_download_declares(tmp_path):
    # A download needs its payload sources to be read, but not its parents.
    store = LocalRegistry(tmp_path / "store")
    copy = DatasetRef.parse("test/copy@1.0.0")
    store.put_dataset(make_dataset(dataset_id=OTHER.dataset_id))
    store.put_dataset(make_dataset(dataset_id=copy.dataset_id, parents={ParentAlias("parent"): OTHER}))
    store.put_dataset(make_dataset(dataset_id=BUILT.dataset_id, parents={ParentAlias("parent"): copy}))
    for release in (OTHER, copy):
        store.record_origin(release, OriginKind.DOWNLOAD, "https://registry.example.org")
    clear_releases(store, kinds=[OriginKind.DOWNLOAD])
    assert [item.release for item in stored_releases(store)] == [BUILT, copy]


def test_clearing_the_builds_keeps_each_build_that_a_download_needs(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    # PARENT, a download, reads files of GRAND, a build.
    store.record_origin(GRAND, OriginKind.BUILD, "timenet-connectors 0.9.0")
    cleared = clear_releases(store, kinds=[OriginKind.BUILD])
    assert _held(store) == {PARENT, GRAND, OTHER}
    assert cleared.kept == [KeptRelease(GRAND, PARENT)]


def test_clearing_every_kind_deletes_every_release_without_reading_a_manifest(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    # Nothing stays, so no release needs another.
    (release_dir(store.root, BUILT) / "manifest.json").write_text("{}")
    releases = stored_releases(store)

    cleared = clear_releases(store, kinds=[OriginKind.DOWNLOAD, OriginKind.BUILD])

    assert cleared.removed == [item.path for item in releases]
    assert cleared.kept == []
    assert list((tmp_path / "store").iterdir()) == []


def test_a_release_clear_with_a_damaged_record_deletes_nothing(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    origin_file(store, OTHER).write_text("{}")
    with pytest.raises(TimeFFormatError, match="is not a valid provenance record"):
        clear_releases(store, kinds=[OriginKind.DOWNLOAD])
    assert _held(store) == {BUILT, PARENT, GRAND, OTHER, HAND}


def test_clearing_the_downloads_with_a_damaged_manifest_of_a_build_deletes_nothing(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    (release_dir(store.root, BUILT) / "manifest.json").write_text("{}")
    with pytest.raises(TimeFFormatError, match="is not a valid manifest"):
        clear_releases(store, kinds=[OriginKind.DOWNLOAD])
    assert {item.release for item in stored_releases(store)} == {BUILT, PARENT, GRAND, OTHER, HAND}


def test_clearing_the_downloads_refuses_a_local_registry_with_a_symlinked_dataset_folder(tmp_path):
    # The listing skips a symlinked folder, so it would miss BUILT and delete the downloads that it needs.
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    moved = tmp_path / "elsewhere/built"
    moved.parent.mkdir()
    (store.root / "test/built").rename(moved)
    (store.root / "test/built").symlink_to(moved, target_is_directory=True)
    with pytest.raises(TimeFValidationError, match="test/built is a symlink"):
        clear_releases(store, kinds=[OriginKind.DOWNLOAD])
    assert _held(store) == {BUILT, PARENT, GRAND, OTHER, HAND}


def test_clearing_every_kind_refuses_a_local_registry_with_a_symlinked_release_folder(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _layout(store)
    _record(store)
    moved = tmp_path / "elsewhere/1.0.0"
    moved.parent.mkdir()
    release_dir(store.root, OTHER).rename(moved)
    release_dir(store.root, OTHER).symlink_to(moved, target_is_directory=True)
    with pytest.raises(TimeFValidationError, match=r"demo/other/1\.0\.0 is a symlink"):
        clear_releases(store, kinds=[OriginKind.DOWNLOAD, OriginKind.BUILD])
    assert _held(store) == {BUILT, PARENT, GRAND, OTHER, HAND}
    # The link hides the release from the listing, not from a read of its exact version.
    with TimeNet(sources=[store]).open(str(OTHER)) as reader:
        assert reader.metadata.dataset_id == OTHER.dataset_id


@pytest.mark.parametrize("clear", [releases_to_clear, clear_releases])
def test_a_release_clear_refuses_an_empty_list_of_kinds_first(tmp_path, clear):
    # A read-only local registry refuses a clear too, so the kinds must come first.
    store = LocalRegistry(tmp_path / "store", read_only=True)
    with pytest.raises(TimeFValidationError, match=r"^kinds names no kind of release$"):
        clear(store, kinds=[])


def test_the_symlink_check_skips_a_hidden_folder(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    (tmp_path / "elsewhere").mkdir()
    (store.root / ".trash").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    store.check_no_linked_folders()


def test_clear_cache_of_a_missing_cache_frees_nothing(tmp_path):
    assert cache_size(tmp_path / "cache") == 0
    cleared = clear_cache(tmp_path / "cache", local_registry=LocalRegistry(tmp_path / "store"))
    assert cleared == ClearedCache(0, [], [])
