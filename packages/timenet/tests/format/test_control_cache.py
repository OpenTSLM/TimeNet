import pyarrow.fs as pafs

from timenet.format.checksums import file_checksum
from timenet.format.control_cache import materialize_control
from timenet.format.control_writer import DuckDBControlWriter
from timenet.manifest import FilePart, Manifest, ManifestFiles
from timenet.registry.version import DatasetVersion

from .test_control_writer import _dataset


def test_local_control_database_uses_its_existing_path(tmp_path):
    path = tmp_path / "control.duckdb"
    dataset = _dataset()
    DuckDBControlWriter(path).write_hierarchy(dataset)
    manifest = Manifest(
        dataset_id=dataset.metadata.dataset_id,
        metadata=dataset.metadata,
        files=ManifestFiles(
            control=(FilePart(path.name, file_checksum(path), path.stat().st_size),),
        ),
    )
    version = DatasetVersion(
        manifest=manifest,
        filesystem=pafs.LocalFileSystem(),
        root=str(tmp_path),
    )

    assert materialize_control(version) == path
