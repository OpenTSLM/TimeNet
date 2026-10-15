import pyarrow.fs as pafs
import pytest

from timenet.errors import TimeFFormatError
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
            control=(FilePart(path=path.name, checksum=file_checksum(path), size=path.stat().st_size),),
        ),
        timef_format_version=3,
    )
    version = DatasetVersion(
        manifest=manifest,
        filesystem=pafs.LocalFileSystem(),
        root=str(tmp_path),
    )

    assert materialize_control(version) == path
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(TimeFFormatError, match="manifest checksum"):
        materialize_control(version)
