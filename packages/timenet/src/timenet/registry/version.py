"""The :class:`DatasetVersion` handle: a random-access, picklable view of one committed version.

The registry hands one of these back from :meth:`~timenet.registry.BaseRegistry.open_version`, and the
reader (with the values backends) reads straight through it. It bundles the already-parsed manifest with
a pyarrow filesystem and the version's root prefix on that filesystem. A read never re-opens the registry
or re-parses ``manifest.json``. The filesystem is config-only (a
:class:`pyarrow.fs.LocalFileSystem` today, a :class:`pyarrow.fs.S3FileSystem` later), so the whole handle
pickles and a torch ``DataLoader`` can ship it to a worker.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.format.constants import MANIFEST_FILE, check_relative_path
from timenet.manifest import Manifest
from timenet.types.references import manifest_digest


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


@dataclass(frozen=True)
class DatasetVersion:
    """An opened dataset version: its manifest plus a filesystem-rooted handle to its files."""

    manifest: Manifest
    """The version's parsed manifest. The reader trusts it rather than re-reading ``manifest.json``."""
    filesystem: pafs.FileSystem
    """The filesystem the version's files live on (``LocalFileSystem`` now, ``S3FileSystem`` later). It
    MUST serve range reads. The reader pulls Parquet footers and value slices out of order through
    ``open_input_file``, so a forward-only download stream does not qualify."""
    root: str
    """The version's root prefix on :attr:`filesystem`. An example is ``/abs/ds/1.0.0``."""

    registry: "BaseRegistry | None" = field(default=None, compare=False, repr=False)
    _owners: dict[str, "DatasetVersion"] = field(default_factory=dict, compare=False, repr=False)

    def owner(self, key: str | None) -> "DatasetVersion":
        """Resolve and validate a direct physical owner once, without opening its catalog.

        Returns:
            This version or the pinned physical owner.

        Raises:
            TimeFFormatError: If the owner is unknown or its manifest changed.
        """
        if key is None:
            return self
        if key in self._owners:
            return self._owners[key]
        source = self.manifest.payload_sources.get(key)
        if source is None:
            raise TimeFFormatError(f"unknown payload owner {key!r} in {self.manifest.dataset_id}")
        registry = self.registry
        if registry is None:
            from timenet.registry.local import LocalRegistry  # noqa: PLC0415 - registry cycle

            registry = LocalRegistry(Path(self.root).parents[2])
        version = registry.open_version(source.source.dataset_id, str(source.source.version))
        if manifest_digest(version.manifest.to_dict()) != source.manifest_checksum:
            raise TimeFFormatError(f"manifest changed for payload owner {source.source}")
        self._owners[key] = version
        return version

    def path(self, relpath: str) -> str:
        """Return the filesystem path of a version-relative file.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The path to hand a pyarrow reader alongside :attr:`filesystem`.

        Raises:
            TimeFFormatError: If a downloaded artifact is absent or fails verification.
        """
        check_relative_path("relpath", relpath)
        path = f"{self.root}/{relpath}"
        if self.registry is not None and isinstance(self.filesystem, pafs.LocalFileSystem) and not Path(path).exists():
            part = next((part for part in self.manifest.files.all_files() if part.path == relpath), None)
            if part is None:
                raise TimeFFormatError(f"artifact {relpath!r} is not declared in the manifest")
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
            try:
                with (
                    os.fdopen(descriptor, "wb") as sink,
                    self.registry.open_file(
                        self.manifest.dataset_id, str(self.manifest.metadata.dataset_version), relpath
                    ) as source,
                ):
                    while chunk := source.read(1024 * 1024):
                        sink.write(chunk)
                temp_path = Path(temporary)
                if temp_path.stat().st_size != part.size or file_checksum(temp_path) != part.checksum:
                    raise TimeFFormatError(f"checksum or size mismatch for {path}")
                temp_path.replace(destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return path

    def store_uri(self, relpath: str) -> str:
        """Return a version-relative file as a store location for a store-oriented backend.

        The Parquet backend reads through ``(filesystem, path)``. The Zarr backend opens a *store*
        instead, which is what this method hands it. For a local version, the store is the plain
        filesystem path, which Zarr opens as a ``LocalStore``. An object-store version needs a
        scheme-qualified URI, such as ``s3://…``, because Zarr uses only this value and ignores
        :attr:`filesystem`. The S3 backend will supply that scheme once it exists. The two accessors
        stay distinct even though they return the same value for a local version.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The store location for ``relpath``.
        """
        check_relative_path("relpath", relpath)
        if isinstance(self.filesystem, pafs.S3FileSystem):
            return f"s3://{self.root}/{relpath}"
        if self.registry is not None:
            for part in self.manifest.files.all_files():
                if part.path.startswith(relpath + "/"):
                    self.path(part.path)
        return f"{self.root}/{relpath}"

    @classmethod
    def open_local(cls, root: str | Path) -> "DatasetVersion":
        """Open a committed version directory on the local filesystem.

        Reads the version's ``manifest.json`` and pairs it with a :class:`pyarrow.fs.LocalFileSystem`.
        A caller builds this handle when it already holds a version directory on disk, such as
        build's copy-on-write edit or a downloaded copy. It does not go through a registry's
        ``open_version``.

        Args:
            root: The version directory (``<...>/<dataset_id>/<version>``).

        Returns:
            A handle rooted at ``root``.

        Raises:
            FileNotFoundError: If ``root`` has no ``manifest.json``.
        """
        root_path = Path(root)
        manifest_path = root_path / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        return cls(manifest=manifest, filesystem=pafs.LocalFileSystem(), root=str(root_path))
