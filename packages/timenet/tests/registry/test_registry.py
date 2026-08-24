from pathlib import Path

import pytest

from timenet.errors import DatasetNotFoundError, RegistryError, TimeFFormatError, TimeFValidationError
from timenet.registry import (
    TIMENET_REGISTRY_URL,
    BaseRegistry,
    DatasetVersion,
    LocalRegistry,
    RemoteRegistry,
    S3Registry,
    WritableRegistry,
    local_registry_path,
    open_registry,
    open_writable_registry,
)
from timenet.testing import make_dataset
from timenet.types import AnswerTask, Domain, License


# ---- factory ----------------------------------------------------------------------------------


def test_open_registry_local_path(registry_root):
    registry = open_registry(registry_root)
    assert isinstance(registry, LocalRegistry)
    assert isinstance(registry, BaseRegistry)


def test_open_registry_file_uri(registry_root):
    registry = open_registry(f"file://{registry_root}")
    assert isinstance(registry, LocalRegistry)


def test_open_registry_http_is_remote():
    assert isinstance(open_registry("https://registry.timenet.io"), RemoteRegistry)


def test_open_registry_s3_is_s3():
    assert isinstance(open_registry("s3://bucket/registry"), S3Registry)


def test_open_registry_timenet_scheme_aliases_hosted_remote():
    registry = open_registry("timenet://")
    assert isinstance(registry, RemoteRegistry)
    assert registry._base_url == TIMENET_REGISTRY_URL


def test_open_registry_timenet_scheme_rejects_a_path():
    # A path after timenet:// used to be dropped silently; now it is rejected, so a mistyped registry
    # URI fails loudly instead of hitting the default host.
    with pytest.raises(ValueError, match="timenet:// takes no path"):
        open_registry("timenet://chengsenwang/tsqa")


def test_open_registry_bare_timenet_scheme():
    registry = open_registry("timenet://")
    assert isinstance(registry, RemoteRegistry)
    assert registry._base_url == TIMENET_REGISTRY_URL


def test_open_writable_registry_returns_writable(registry_root):
    assert isinstance(open_writable_registry(registry_root), WritableRegistry)


def test_open_registry_unknown_scheme_rejected():
    # must not fall through to a LocalRegistry rooted at the literal "gs://bucket" string
    with pytest.raises(ValueError, match="unsupported registry scheme"):
        open_registry("gs://bucket/registry")


def test_open_registry_file_uri_with_host_rejected():
    with pytest.raises(ValueError, match="absolute"):
        open_registry("file://home/timo/registry")


def test_open_registry_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    registry = open_registry("~/registry")
    assert isinstance(registry, LocalRegistry)
    assert registry._root == tmp_path / "registry"


def test_local_registry_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert LocalRegistry(Path("~/reg"))._root == tmp_path / "reg"


# ---- local_registry_path ----------------------------------------------------------------------


def test_local_registry_path_plain_path(tmp_path):
    assert local_registry_path(tmp_path) == tmp_path


def test_local_registry_path_file_uri(tmp_path):
    assert local_registry_path(f"file://{tmp_path}") == tmp_path


def test_local_registry_path_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert local_registry_path("~/reg") == tmp_path / "reg"


@pytest.mark.parametrize(
    "uri", ["timenet://", "timenet://hello/world", "http://reg.example", "https://reg.example", "s3://bucket/reg"]
)
def test_local_registry_path_rejects_remote(uri):
    with pytest.raises(TimeNetRegistryError, match="remote"):
        local_registry_path(uri)


@pytest.mark.parametrize(("uri", "match"), [("file://host/reg", "three slashes"), ("file://", "absolute path")])
def test_local_registry_path_rejects_malformed_file_uri(uri, match):
    with pytest.raises(TimeNetRegistryError, match=match):
        local_registry_path(uri)


# ---- list / get / open ------------------------------------------------------------------------


def test_get_manifest_rejects_traversal_id(registry_root):
    # every registry path-join validates the id, so a ``..`` id can't escape the root
    with pytest.raises(TimeFValidationError, match="dataset_id"):
        LocalRegistry(registry_root).get_manifest("../evil")


def test_get_manifest_rejects_mismatched_id(registry_root):
    # a manifest whose stored id disagrees with its directory is a misplaced/corrupt artifact
    registry = LocalRegistry(registry_root)
    real_id = registry.list_datasets()[0].dataset_id
    version = str(registry.get_manifest(real_id).metadata.dataset_version)
    misplaced = registry_root / "wrong/place" / version
    misplaced.mkdir(parents=True)
    (misplaced / "manifest.json").write_text((registry_root / real_id / version / "manifest.json").read_text())
    with pytest.raises(TimeFFormatError, match="inconsistent"):
        registry.get_manifest("wrong/place")


def test_latest_version_ignores_staging_dirs(registry_root):
    # A crashed build can leave a `<version>.tmp-<uuid>` sibling (briefly holding a manifest.json).
    stale = registry_root / "demo/ecg" / "2.0.0.tmp-deadbeef"
    stale.mkdir()
    (stale / "manifest.json").write_text("{}")
    manifest = LocalRegistry(registry_root).get_manifest("demo/ecg")  # must not choke on the tmp dir
    assert manifest.metadata.dataset_version.major == 2


# ---- open_version -----------------------------------------------------------------------------


def test_open_version_returns_handle(registry_root):
    version = LocalRegistry(registry_root).open_version("demo/ecg")
    assert isinstance(version, DatasetVersion)
    assert version.manifest.metadata.dataset_id == "demo/ecg"
    assert version.root == str(registry_root / "demo/ecg" / "2.0.0")


def test_open_version_resolves_latest_to_a_concrete_directory(registry_root):
    # ``None`` means latest, but the handle roots at that resolved version, never the string "latest"
    version = LocalRegistry(registry_root).open_version("demo/ecg", version=None)
    assert version.root.endswith("/2.0.0")


def test_open_version_filesystem_reads_a_file(registry_root):
    version = LocalRegistry(registry_root).open_version("demo/ecg")
    with version.filesystem.open_input_file(version.path("manifest.json")) as handle:
        assert b"demo/ecg" in handle.read()


def test_open_version_unknown_raises(registry_root):
    with pytest.raises(TimeNetDatasetNotFoundError):
        LocalRegistry(registry_root).open_version("does/not-exist")


def test_open_version_unknown_version_raises(registry_root):
    with pytest.raises(TimeNetDatasetNotFoundError):
        LocalRegistry(registry_root).open_version("demo/ecg", version="9.9.9")


# ---- search -----------------------------------------------------------------------------------


def test_search_no_filters_returns_all(registry_root):
    assert len({m.dataset_id for m in LocalRegistry(registry_root).search()}) == 2


def test_search_by_domain(registry_root):
    results = LocalRegistry(registry_root).search(domain=Domain.CARDIOLOGY)
    assert {m.dataset_id for m in results} == {"demo/ecg"}


def test_search_by_license(registry_root):
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(license=License.MIT)} == {"demo/ecg"}


def test_search_by_task_type_filter(registry_root):
    # only hello_world has a QA task
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(task=AnswerTask)} == {"timenet/hello-world"}


def test_search_by_time_series_spec(registry_root):
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(time_series_spec="ecg_lead")} == {"demo/ecg"}


def test_search_by_tag(registry_root):
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(tag="clinical")} == {"demo/ecg"}


def test_search_by_dataset_id(registry_root):
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(dataset_id="timenet/hello-world")} == {
        "timenet/hello-world"
    }


def test_search_by_query_substring(registry_root):
    assert {m.dataset_id for m in LocalRegistry(registry_root).search(query="hello")} == {"timenet/hello-world"}


def test_search_accepts_scalar_or_list(registry_root):
    both = LocalRegistry(registry_root).search(domain=[Domain.CARDIOLOGY, Domain.GENERAL])
    assert {m.dataset_id for m in both} == {"timenet/hello-world", "demo/ecg"}


def test_search_filters_are_anded(registry_root):
    # cardiology AND MIT => ecg; cardiology AND CC-BY-4.0 => none
    assert {
        m.dataset_id for m in LocalRegistry(registry_root).search(domain=Domain.CARDIOLOGY, license=License.MIT)
    } == {"demo/ecg"}
    assert LocalRegistry(registry_root).search(domain=Domain.CARDIOLOGY, license=License.CC_BY_4_0) == []


def test_search_limit(registry_root):
    assert len(LocalRegistry(registry_root).search(limit=1)) == 1


def test_search_limit_zero_returns_empty(registry_root):
    # the limit check must run before the append, else a zero limit still yields one row
    assert LocalRegistry(registry_root).search(limit=0) == []


def test_search_negative_limit_rejected(registry_root):
    with pytest.raises(ValueError, match="non-negative"):
        LocalRegistry(registry_root).search(limit=-1)


# ---- store (write side) -----------------------------------------------------------------------


def test_store_derives_schema_if_absent(tmp_path):
    dataset = make_dataset()
    assert dataset.schema is None
    LocalRegistry(tmp_path).store(dataset)
    assert dataset.schema is not None


# ---- s3 catalog -------------------------------------------------------------------------------


def test_s3_registry_has_no_catalog():
    # The S3 backend does store + download by explicit id, but has no catalog: list/search raise.
    s3 = S3Registry("s3://bucket/registry")
    with pytest.raises(NotImplementedError):
        s3.list_datasets()
    with pytest.raises(NotImplementedError):
        s3.search()
