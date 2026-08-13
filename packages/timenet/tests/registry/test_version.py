from pathlib import Path
import pickle

import pyarrow.fs as pafs
import pytest

from timenet.registry import DatasetVersion


def _version_dir(registry_root: Path) -> Path:
    return registry_root / "demo/ecg" / "2.0.0"


def test_open_local_reads_manifest_and_roots(registry_root):
    version = DatasetVersion.open_local(_version_dir(registry_root))
    assert version.manifest.metadata.dataset_id == "demo/ecg"
    assert version.root == str(_version_dir(registry_root))
    assert isinstance(version.filesystem, pafs.LocalFileSystem)


def test_open_local_without_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        DatasetVersion.open_local(tmp_path)


def test_path_joins_root_and_relpath(registry_root):
    version = DatasetVersion.open_local(_version_dir(registry_root))
    assert version.path("time_series/part-000.parquet") == f"{version.root}/time_series/part-000.parquet"


def test_store_uri_matches_path_for_local(registry_root):
    version = DatasetVersion.open_local(_version_dir(registry_root))
    assert version.store_uri("values.zarr") == version.path("values.zarr")


def test_handle_is_picklable(registry_root):
    # a torch DataLoader ships the handle to a worker process, so it must survive a pickle round-trip
    version = DatasetVersion.open_local(_version_dir(registry_root))
    restored = pickle.loads(pickle.dumps(version))
    assert restored.root == version.root
    assert restored.manifest.metadata.dataset_id == "demo/ecg"
    with restored.filesystem.open_input_file(restored.path("manifest.json")) as handle:
        assert b"demo/ecg" in handle.read()
