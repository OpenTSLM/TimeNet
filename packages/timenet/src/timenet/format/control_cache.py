"""Materialize a verified DuckDB control file behind a local-path seam."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.config import settings
from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


def materialize_control(version: DatasetVersion) -> Path:
    """Return a local path for a version's verified DuckDB control database.

    Local versions already satisfy DuckDB's path requirement. For a remote filesystem, this function
    downloads only ``control.duckdb`` into a checksum-keyed cache. A future remote-attach implementation
    can replace this function without changing the schema or query layer.

    Returns:
        A local path suitable for a read-only DuckDB connection.

    Raises:
        TimeFFormatError: If the manifest lacks a control file or the downloaded file fails verification.
    """
    parts = version.manifest.files.control
    if len(parts) != 1:
        raise TimeFFormatError("TimeF manifest does not declare control.duckdb")
    part = parts[0]
    if isinstance(version.filesystem, pafs.LocalFileSystem):
        return Path(version.path(part.path))

    digest = part.checksum.removeprefix("sha256:")
    cache_dir = settings().cache_dir / "control"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{digest}.duckdb"
    if destination.exists() and destination.stat().st_size == part.size:
        if file_checksum(destination) == part.checksum:
            return destination
        destination.unlink()

    fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=cache_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, version.filesystem.open_input_file(version.path(part.path)) as source:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != part.size or file_checksum(temporary) != part.checksum:
            raise TimeFFormatError("downloaded control.duckdb does not match its manifest checksum")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
