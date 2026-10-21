import json
from pathlib import Path
import pickle
import shutil

import pyarrow.fs as pafs
import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry.dataset_graph_dependency_tools import prepare_dependency_graph, resolve_read_dependency_graph
from timenet.types.reference_keys import OwnerKey
from timenet.types.references import DatasetRef


def _version_dir(registry_root: Path) -> Path:
    return registry_root / "demo/ecg" / "2.0.0"


def test_open_local_reads_manifest_and_roots(registry_root):
    version = open_local_version(_version_dir(registry_root))
    assert version.manifest.metadata.dataset_id == "demo/ecg"
    assert version.root == str(_version_dir(registry_root))
    assert isinstance(version.filesystem, pafs.LocalFileSystem)


def test_open_local_without_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        open_local_version(tmp_path)


def test_resolve_file_path_joins_root_and_relative_path(registry_root):
    version = open_local_version(_version_dir(registry_root))
    assert version.resolve_file_path("time_series/part-000.parquet") == f"{version.root}/time_series/part-000.parquet"


@pytest.mark.parametrize(
    "relpath",
    [
        pytest.param("/outside.txt", id="absolute-posix"),
        pytest.param("../../outside.txt", id="parent-traversal"),
        pytest.param("time_series/../../outside.txt", id="nested-traversal"),
        pytest.param("C:/outside.txt", id="windows-drive"),
        pytest.param("C:outside.txt", id="windows-drive-relative"),
        pytest.param(r"\\server\share\outside.txt", id="windows-unc"),
        pytest.param(r"..\outside.txt", id="windows-traversal"),
    ],
)
def test_resolve_file_path_rejects_a_path_that_escapes_the_root(registry_root, relpath):
    version = open_local_version(_version_dir(registry_root))
    with pytest.raises(TimeFValidationError, match="dataset root"):
        version.resolve_file_path(relpath)


def test_handle_is_picklable(registry_root):
    # a torch DataLoader ships the handle to a worker process, so it must survive a pickle round-trip
    version = open_local_version(_version_dir(registry_root))
    restored = pickle.loads(pickle.dumps(version))
    assert restored.root == version.root
    assert restored.manifest.metadata.dataset_id == "demo/ecg"
    with restored.filesystem.open_input_file(restored.resolve_file_path("manifest.json")) as handle:
        assert b"demo/ecg" in handle.read()


@pytest.mark.parametrize("missing", [False, True])
def test_local_payload_source_rejects_missing_or_misplaced_manifest(registry_root, missing):
    child = open_local_version(_version_dir(registry_root))
    parent = open_local_version(registry_root / "timenet/hello-world/1.0.0")
    reference = DatasetRef.parse("timenet/hello-world@1.0.0")
    child.manifest.payload_sources[OwnerKey("parent")] = reference
    (Path(child.root) / "manifest.json").write_text(child.manifest.to_json())
    manifest_path = Path(parent.root) / "manifest.json"
    manifest = parent.manifest.to_dict()
    if missing:
        manifest_path.unlink()
    else:
        manifest["dataset_id"] = "wrong/dataset"
        manifest["metadata"]["dataset_id"] = "wrong/dataset"
        manifest_path.write_text(json.dumps(manifest))
    expected_error = TimeNetDatasetNotFoundError if missing else TimeFFormatError
    with pytest.raises(expected_error, match=r"complete release|different dataset ID"):
        open_local_version(child.root)


@pytest.mark.parametrize("corruption", [None, "truncated", "checksum"])
def test_preparation_caches_only_verified_complete_releases(registry_root, tmp_path, corruption):
    source_version = open_local_version(_version_dir(registry_root))
    part = source_version.manifest.files.time_series[0]
    source_file = Path(source_version.root) / part.path
    expected = source_file.read_bytes()
    reference = DatasetRef.parse("demo/ecg@2.0.0")
    cache = tmp_path / "cache"
    registry = LocalRegistry(registry_root)
    directory = cache / reference.dataset_id / str(reference.version)
    destination = directory / part.path

    if corruption is not None:
        damaged = expected[:-1] if corruption == "truncated" else bytes([expected[0] ^ 1]) + expected[1:]
        source_file.write_bytes(damaged)
        with pytest.raises(TimeFFormatError, match="checksum or size mismatch"):
            prepare_dependency_graph(cache, reference, fetch_from=registry)
        assert not destination.exists()
        assert not (directory / "manifest.json").exists()
        assert not list(directory.parent.glob("2.0.0.tmp-*"))
        source_file.write_bytes(expected)

    cached = prepare_dependency_graph(cache, reference, fetch_from=registry)
    assert cached.resolve_file_path(part.path) == str(destination)
    assert destination.read_bytes() == expected

    # A cached file must remain readable without contacting the registry again.
    source_file.unlink()
    cached = prepare_dependency_graph(cache, reference, fetch_from=registry)
    assert cached.resolve_file_path(part.path) == str(destination)
    assert destination.read_bytes() == expected


def test_preparation_requires_physical_sources_and_detects_cycles(registry_root, tmp_path):
    child_dir = _version_dir(registry_root)
    child_manifest_path = child_dir / "manifest.json"
    child = json.loads(child_manifest_path.read_text())
    child["payload_sources"] = {"parent": {"dataset_id": "timenet/hello-world", "version": "1.0.0"}}
    child_manifest_path.write_text(json.dumps(child))
    cache = tmp_path / "prepared"
    cached_child = cache / "demo/ecg/2.0.0"
    shutil.copytree(child_dir, cached_child)
    reference = DatasetRef.parse("demo/ecg@2.0.0")
    with pytest.raises(TimeNetDatasetNotFoundError, match="timenet/hello-world"):
        prepare_dependency_graph(cache, reference)

    def missing_remote(ref):
        raise TimeNetDatasetNotFoundError(f"no remote release {ref}")

    with pytest.raises(TimeNetDatasetNotFoundError, match="timenet/hello-world"):
        resolve_read_dependency_graph(cache, reference, missing_remote)

    prepare_dependency_graph(cache, reference, fetch_from=LocalRegistry(registry_root))
    cached_parent = cache / "timenet/hello-world/1.0.0"
    assert (cached_parent / "control.duckdb").is_file()
    parent_manifest_path = cached_parent / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text())
    parent["payload_sources"] = {"cycle": {"dataset_id": "demo/ecg", "version": "2.0.0"}}
    parent_manifest_path.write_text(json.dumps(parent))
    with pytest.raises(TimeFFormatError, match="cycle"):
        prepare_dependency_graph(cache, reference)
    with pytest.raises(TimeFFormatError, match="cycle"):
        resolve_read_dependency_graph(cache, reference, missing_remote)


@pytest.mark.parametrize("suffix", ["different/ecg/2.0.0", "demo/ecg/3.0.0", "demo/ecg/latest"])
def test_download_rejects_a_destination_for_another_release(registry_root, tmp_path, suffix):
    destination = tmp_path / "downloads" / suffix
    with pytest.raises(TimeFValidationError, match=r"expected demo/ecg@2\.0\.0|release directory"):
        LocalRegistry(registry_root).download_version("demo/ecg", "2.0.0", destination)
    assert not destination.exists()
