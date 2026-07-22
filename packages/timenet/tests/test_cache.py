import pytest

from timenet.cache import cached_datasets, clear_cache, human_bytes
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _write(root):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()


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


def test_cached_datasets_lists_registry_and_storage(_home):
    _write(_home / "registry")
    _write(_home / "storage")
    cached = cached_datasets()
    locations = {(c.location, c.dataset_id, c.version) for c in cached}
    assert ("registry", "hello_world", "1.0.0") in locations
    assert ("storage", "hello_world", "1.0.0") in locations
    assert all(c.size_bytes > 0 for c in cached)


def test_cached_datasets_empty_when_nothing():
    assert cached_datasets() == []


def test_clear_removes_storage_and_cache_but_keeps_registry(_home):
    _write(_home / "registry")
    _write(_home / "storage")
    (_home / "cache" / "raw").mkdir(parents=True)
    (_home / "cache" / "raw" / "f.bin").write_bytes(b"x" * 100)

    freed, _ = clear_cache(include_registry=False)
    assert freed > 0
    assert not (_home / "storage").exists()
    assert not (_home / "cache").exists()
    assert (_home / "registry").exists()  # curated outputs kept


def test_clear_all_removes_registry_too(_home):
    _write(_home / "registry")
    clear_cache(include_registry=True)
    assert not (_home / "registry").exists()


def test_clear_sizes_a_symlink_as_the_link_not_its_target(_home, tmp_path):
    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * 10_000)
    cache = _home / "cache"
    cache.mkdir(parents=True)
    (cache / "link.bin").symlink_to(target)

    freed, _ = clear_cache(include_registry=False)

    assert freed < 10_000  # counted the link itself, never followed it to the 10 KB target
    assert target.exists()  # the external target is left untouched
    assert not cache.exists()
