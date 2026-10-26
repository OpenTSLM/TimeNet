import pyarrow.fs as pafs

from timenet.format.checksums import file_checksum
from timenet.format.control_cache import materialize_control
from timenet.manifest import FilePart, Manifest, ManifestFiles
from timenet.registry.version import DatasetVersion


def test_local_control_database_uses_its_existing_path(control_path, control_dataset):
    path = control_path
    dataset = control_dataset
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
        root=str(path.parent),
    )

    assert materialize_control(version) == path
