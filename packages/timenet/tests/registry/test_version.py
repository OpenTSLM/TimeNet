import json
from pathlib import Path
import pickle
import shutil

import pyarrow.fs as pafs
import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.manifest import PayloadSource
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry.dependencies import open_cached_or_remote, prepare_local_copy
from timenet.registry.shared import complete_copy
from timenet.types.reference_keys import SourceKey
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


def _add_source(child_dir: Path, parent_dir: Path) -> DatasetRef:
    """Make a release read the control file of another release, as a pinned payload source."""
    parent = open_local_version(parent_dir).manifest
    reference = DatasetRef(dataset_id=parent.dataset_id, version=parent.metadata.dataset_version)
    manifest_path = child_dir / "manifest.json"
    child = json.loads(manifest_path.read_text())
    pinned = PayloadSource(release=reference, files=parent.files.control)
    child["payload_sources"] = {str(reference): pinned.model_dump(mode="json")}
    manifest_path.write_text(json.dumps(child))
    return reference


@pytest.mark.parametrize("missing", [False, True])
def test_local_payload_source_rejects_missing_or_misplaced_manifest(registry_root, missing):
    child = open_local_version(_version_dir(registry_root))
    parent = open_local_version(registry_root / "timenet/hello-world/1.0.0")
    _add_source(_version_dir(registry_root), registry_root / "timenet/hello-world/1.0.0")
    manifest_path = Path(parent.root) / "manifest.json"
    manifest = parent.manifest.to_dict()
    if missing:
        manifest_path.unlink()
    else:
        manifest["dataset_id"] = "wrong/dataset"
        manifest["metadata"]["dataset_id"] = "wrong/dataset"
        manifest_path.write_text(json.dumps(manifest))
    expected_error = TimeNetDatasetNotFoundError if missing else TimeFFormatError
    expected = (
        rf"payload source timenet/hello-world@1\.0\.0 of demo/ecg@2\.0\.0 is not in {registry_root}"
        if missing
        else "different dataset ID"
    )
    with pytest.raises(expected_error, match=expected):
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
            prepare_local_copy(cache, reference, fetch_from=registry)
        assert not destination.exists()
        assert not (directory / "manifest.json").exists()
        assert not list(directory.parent.glob("2.0.0.tmp-*"))
        source_file.write_bytes(expected)

    cached = prepare_local_copy(cache, reference, fetch_from=registry)
    assert cached.resolve_file_path(part.path) == str(destination)
    assert destination.read_bytes() == expected

    # A cached file must remain readable without contacting the registry again.
    source_file.unlink()
    cached = prepare_local_copy(cache, reference, fetch_from=registry)
    assert cached.resolve_file_path(part.path) == str(destination)
    assert destination.read_bytes() == expected


def test_preparation_needs_only_the_direct_payload_sources(registry_root, tmp_path):
    child_dir = _version_dir(registry_root)
    parent_dir = registry_root / "timenet/hello-world/1.0.0"
    parent_ref = _add_source(child_dir, parent_dir)
    cache = tmp_path / "prepared"
    shutil.copytree(child_dir, cache / "demo/ecg/2.0.0")
    reference = DatasetRef.parse("demo/ecg@2.0.0")
    with pytest.raises(TimeNetDatasetNotFoundError, match=rf"payload source {parent_ref} of {reference} is not in"):
        prepare_local_copy(cache, reference)

    def missing_remote(ref):
        raise TimeNetDatasetNotFoundError(f"no remote release {ref}")

    with pytest.raises(
        TimeNetDatasetNotFoundError, match=rf"payload source {parent_ref} .* is not in remote or its cache"
    ):
        open_cached_or_remote(cache, reference, missing_remote, searched="remote")

    prepare_local_copy(cache, reference, fetch_from=LocalRegistry(registry_root))
    cached_parent = cache / "timenet/hello-world/1.0.0"
    assert (cached_parent / "control.duckdb").is_file()
    # The parent's own payload sources are not prepared: here one is missing, and one names the child.
    _add_source(cached_parent, registry_root / "demo/ecg/2.0.0")
    parent_manifest = json.loads((cached_parent / "manifest.json").read_text())
    missing = {"release": {"dataset_id": "nowhere/missing", "version": "1.0.0"}}
    missing["files"] = parent_manifest["payload_sources"]["demo/ecg@2.0.0"]["files"]
    parent_manifest["payload_sources"]["nowhere/missing@1.0.0"] = missing
    (cached_parent / "manifest.json").write_text(json.dumps(parent_manifest))
    # Rewriting the parent's manifest leaves the child's pins, which name only its control file, intact.
    prepared = prepare_local_copy(cache, reference)
    assert set(prepared.payload_sources) == {SourceKey(str(parent_ref))}
    assert prepared.resolve_payload_source(SourceKey(str(parent_ref))).payload_sources == {}
    assert open_cached_or_remote(cache, reference, missing_remote, searched="remote").payload_sources


@pytest.mark.parametrize("suffix", ["different/ecg/2.0.0", "demo/ecg/3.0.0", "demo/ecg/latest"])
def test_download_rejects_a_destination_for_another_release(registry_root, tmp_path, suffix):
    destination = tmp_path / "downloads" / suffix
    with pytest.raises(TimeFValidationError, match=r"expected demo/ecg@2\.0\.0|release directory"):
        LocalRegistry(registry_root).download_version("demo/ecg", "2.0.0", destination)
    assert not destination.exists()


@pytest.mark.parametrize("damage", [None, "missing", "truncated"])
def test_a_complete_copy_has_every_file_with_its_recorded_size(registry_root, damage):
    version = open_local_version(_version_dir(registry_root))
    path = Path(version.root) / version.manifest.files.time_series[0].path
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    assert complete_copy(Path(version.root), version.manifest) is (damage is None)
