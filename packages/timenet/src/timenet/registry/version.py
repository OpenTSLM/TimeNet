"""The :class:`DatasetVersion` handle: a random-access, picklable view of one committed version.

The registry hands one of these back from :meth:`~timenet.registry.BaseRegistry.open_version`, and the
reader (with the values backends) reads straight through it. It bundles the already-parsed manifest with
a pyarrow filesystem and the version's root prefix on that filesystem, so a read never re-opens the
registry nor re-parses ``manifest.json``. The filesystem is config-only (a
:class:`pyarrow.fs.LocalFileSystem` today, a :class:`pyarrow.fs.S3FileSystem` later), so the whole handle
pickles and a torch ``DataLoader`` can ship it to a worker.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow.fs as pafs

from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest


@dataclass(frozen=True)
class DatasetVersion:
    """An opened dataset version: its manifest plus a filesystem-rooted handle to its files."""

    manifest: Manifest
    """The version's parsed manifest; the reader trusts this rather than re-reading ``manifest.json``."""
    filesystem: pafs.FileSystem
    """The filesystem the version's files live on (``LocalFileSystem`` now, ``S3FileSystem`` later). It
    MUST serve range reads: the reader pulls Parquet footers and value slices out of order through
    ``open_input_file``, so a forward-only download stream does not qualify."""
    root: str
    """The version's root prefix on :attr:`filesystem`, e.g. ``/abs/ds/1.0.0``."""

    def path(self, relpath: str) -> str:
        """Return the filesystem path of a version-relative file.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The path to hand a pyarrow reader alongside :attr:`filesystem`.
        """
        return f"{self.root}/{relpath}"

    def store_uri(self, relpath: str) -> str:
        """Return a version-relative file as a store location for a store-oriented backend.

        The Parquet backend reads through ``(filesystem, path)``; the Zarr backend opens a *store*
        instead, which is what this hands it. For a local version that is the plain filesystem path, which
        Zarr opens as a ``LocalStore``. An object-store version needs a scheme-qualified URI (``s3://…``)
        here, since Zarr drives off this alone and ignores :attr:`filesystem`; the S3 backend supplies
        that scheme when it lands, so the two accessors stay distinct even though they coincide for local.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The store location for ``relpath``.
        """
        return self.path(relpath)

    @classmethod
    def open_local(cls, root: str | Path) -> "DatasetVersion":
        """Open a committed version directory on the local filesystem.

        Reads the version's ``manifest.json`` and pairs it with a :class:`pyarrow.fs.LocalFileSystem`.
        This is the handle a caller builds when it already holds a version directory on disk (curation's
        copy-on-write edit, a downloaded copy) rather than going through a registry's ``open_version``.

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
        manifest = Manifest.from_json(manifest_path.read_text())
        return cls(manifest=manifest, filesystem=pafs.LocalFileSystem(), root=str(root_path))
