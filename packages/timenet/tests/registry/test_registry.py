from pathlib import Path

import pytest

from timenet.errors import DatasetNotFoundError, RegistryError, TimeFFormatError, TimeFValidationError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import (
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
from timenet.testing import assert_datasets_equal, make_dataset
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
    registry = open_registry("timenet://hello/world")
    assert isinstance(registry, RemoteRegistry)
    assert registry._base_url.startswith("https://registry.timenet.ai")


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
    with pytest.raises(RegistryError, match="remote"):
        local_registry_path(uri)


@pytest.mark.parametrize(("uri", "match"), [("file://host/reg", "three slashes"), ("file://", "absolute path")])
def test_local_registry_path_rejects_malformed_file_uri(uri, match):
    with pytest.raises(RegistryError, match=match):
        local_registry_path(uri)


# ---- list / get / open ------------------------------------------------------------------------


def test_list_datasets(registry_root):
    ids = {m.dataset_id for m in LocalRegistry(registry_root).list_datasets()}
    assert ids == {"timenet/hello-world", "demo/ecg"}


def test_get_manifest_latest(registry_root):
    manifest = LocalRegistry(registry_root).get_manifest("demo/ecg")
    assert isinstance(manifest, Manifest)
    assert manifest.metadata.dataset_version.major == 2


def test_get_manifest_unknown_raises(registry_root):
    with pytest.raises(DatasetNotFoundError):
        LocalRegistry(registry_root).get_manifest("does/not-exist")


def test_get_manifest_unknown_version_raises(registry_root):
    with pytest.raises(DatasetNotFoundError):
        LocalRegistry(registry_root).get_manifest("demo/ecg", version="9.9.9")


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


def test_open_file(registry_root):
    registry = LocalRegistry(registry_root)
    with registry.open_file("demo/ecg", "2.0.0", "manifest.json") as handle:
        assert b"demo/ecg" in handle.read()


def test_open_file_rejects_path_traversal(registry_root):
    registry = LocalRegistry(registry_root)
    with pytest.raises(ValueError, match="escapes"):
        registry.open_file("demo/ecg", "2.0.0", "../../../../etc/passwd")


# ---- open_version (storage seam) --------------------------------------------------------------


def test_open_version_returns_a_handle(registry_root):
    version = LocalRegistry(registry_root).open_version("demo/ecg")
    assert isinstance(version, DatasetVersion)
    assert version.manifest.metadata.dataset_version.major == 2  # latest resolved
    assert version.root.endswith("demo/ecg/2.0.0")
    assert version.path("manifest.json").endswith("demo/ecg/2.0.0/manifest.json")


def test_open_version_round_trips_through_the_reader(registry_root):
    with TimeFReader(LocalRegistry(registry_root).open_version("timenet/hello-world")) as reader:
        assert_datasets_equal(make_dataset(), reader.read())


def test_open_version_unknown_raises(registry_root):
    with pytest.raises(DatasetNotFoundError):
        LocalRegistry(registry_root).open_version("does/not-exist")


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


def test_store_round_trip(tmp_path):
    registry = LocalRegistry(tmp_path)
    dataset = make_dataset()
    version = registry.store(dataset)
    assert version == str(dataset.metadata.dataset_version)
    manifest = registry.get_manifest(dataset.metadata.dataset_id, version)
    assert manifest.metadata.dataset_id == dataset.metadata.dataset_id


def test_store_derives_schema_if_absent(tmp_path):
    dataset = make_dataset()
    assert dataset.schema is None
    LocalRegistry(tmp_path).store(dataset)
    assert dataset.schema is not None


def test_exists_reflects_store(tmp_path):
    registry = LocalRegistry(tmp_path)
    dataset = make_dataset()
    version = str(dataset.metadata.dataset_version)
    assert not registry.exists(dataset.metadata.dataset_id, version)
    registry.store(dataset)
    assert registry.exists(dataset.metadata.dataset_id, version)


def test_store_is_idempotent_without_force(tmp_path):
    registry = LocalRegistry(tmp_path)
    registry.store(make_dataset())
    registry.store(make_dataset())  # committed version already exists; must skip, not raise


def test_store_force_overwrites(tmp_path):
    registry = LocalRegistry(tmp_path)
    registry.store(make_dataset())
    registry.store(make_dataset(), force=True)  # must not raise


# ---- remote / s3 stubs ------------------------------------------------------------------------


def test_remote_registry_is_deferred():
    remote = RemoteRegistry("https://registry.timenet.io")
    with pytest.raises(NotImplementedError):
        remote.list_datasets()
    with pytest.raises(NotImplementedError):
        remote.store(make_dataset())
    with pytest.raises(NotImplementedError):
        remote.open_version("demo/ecg")


def test_s3_registry_is_deferred():
    s3 = S3Registry("s3://bucket/registry")
    with pytest.raises(NotImplementedError):
        s3.list_datasets()
    with pytest.raises(NotImplementedError):
        s3.store(make_dataset())
    with pytest.raises(NotImplementedError, match="later change"):
        s3.open_version("demo/ecg")
