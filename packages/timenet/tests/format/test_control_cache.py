import pyarrow.fs as pafs

from timenet.config import is_cache_folder, settings
from timenet.format.checksums import file_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.storage import StorageSources
from timenet.format.stored_values import PackedValueWriter
from timenet.manifest import FilePart, Manifest, ManifestFiles
from timenet.registry.version import DatasetVersion

from .test_control_writer import _dataset


def test_local_control_database_uses_its_existing_path(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)
    manifest = Manifest(
        dataset_id=dataset.metadata.dataset_id,
        metadata=dataset.metadata,
        files=ManifestFiles(
            control=(FilePart(path=path.name, checksum=file_checksum(path), size=path.stat().st_size),),
        ),
        timef_format_version=1,
    )
    version = DatasetVersion(
        manifest=manifest,
        filesystem=pafs.LocalFileSystem(),
        root=str(tmp_path),
    )

    assert materialize_control(version) == path


def test_a_remote_control_database_lands_in_a_control_folder_with_the_marker(tmp_path):
    path = tmp_path / "remote" / "control.duckdb"
    path.parent.mkdir()
    dataset = _dataset()
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)
    manifest = Manifest(
        dataset_id=dataset.metadata.dataset_id,
        metadata=dataset.metadata,
        files=ManifestFiles(
            control=(FilePart(path=path.name, checksum=file_checksum(path), size=path.stat().st_size),),
        ),
        timef_format_version=1,
    )
    # Any file system but the local one counts as remote, so the database is copied into the cache.
    remote = pafs.SubTreeFileSystem(str(path.parent), pafs.LocalFileSystem())
    copy = materialize_control(DatasetVersion(manifest=manifest, filesystem=remote, root="."))
    assert copy.parent == settings().control_cache_dir
    assert copy.read_bytes() == path.read_bytes()
    assert is_cache_folder(copy.parent)
