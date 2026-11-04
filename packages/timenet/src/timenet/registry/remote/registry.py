"""A registry backed by the hosted TimeNet HTTP service.

Talks to ``timenet-registry`` through :class:`~timenet.registry._http.RegistryHttpClient`. It lists and
searches the catalog, fetches manifests, streams files, downloads versions, and publishes datasets.
Artifact bytes never pass through the API. The service hands back presigned URLs.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO, cast

import httpx
import pyarrow.fs as pafs

from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetRegistryError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest, PayloadSource
from timenet.registry._paths import release_dir
from timenet.registry.lazy_read import LazyReadRegistry
from timenet.registry.remote._download import ProgressCallback, download_version_files
from timenet.registry.remote._http import RegistryHttpClient
from timenet.registry.remote.filesystem import RegistryFileSystem
from timenet.registry.remote.models import (
    DatasetDetailModel,
    DatasetListModel,
    PublishResponseModel,
    UploadGrantModel,
)
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef, is_latest
from timenet.writer import TimeFWriter, WriteProgressEvent


class RemoteRegistry(LazyReadRegistry):
    """Read and publish datasets through the hosted TimeNet HTTP service."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Open a remote registry.

        Args:
            base_url: The service root, e.g. ``https://registry.timenet.ai``.
            token: Bearer token. Defaults to ``$TIMENET_TOKEN`` (``None`` is anonymous).
            cache_dir: Where downloads are cached; defaults to the configured storage directory.
            transport: An httpx transport for testing; ``None`` uses the network.
        """
        cfg = settings()
        self._base_url = base_url
        self._http = RegistryHttpClient(base_url, token=token if token is not None else cfg.token, transport=transport)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else cfg.storage_dir

    def __str__(self) -> str:
        return f"registry {self._base_url}"

    @property
    def local_root(self) -> Path:
        """The directory that holds complete local releases."""
        return self._cache_dir

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the latest-version metadata of every dataset, sorted by id.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        payload = DatasetListModel.model_validate_json(self._http.get_text("/datasets"))
        metadatas = [
            self.get_manifest(row.dataset_id, str(row.version)).metadata if row.requires_manifest() else row.metadata()
            for row in payload.datasets
        ]
        return sorted(metadatas, key=lambda m: m.dataset_id)

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest (latest version if unspecified).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string, or ``None`` / ``"latest"`` for the latest.

        Returns:
            The dataset's manifest.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version is unknown.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the HTTP client on 404, not directly here)
        resolved = self._resolve_latest(dataset_id) if is_latest(version) else version
        text = self._http.get_text(f"/datasets/{dataset_id}/{resolved}/manifest")
        return Manifest.model_validate_json(text)

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Open one file of a dataset version for binary reading.

        Streams the presigned URL into a spooled temporary file, so the returned handle is seekable and
        self-contained (the caller closes it).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            relpath: The version-relative file path.

        Returns:
            An open, seekable binary file positioned at the start.
        """
        url = self._http.resolve_presigned(dataset_id, version, relpath)
        buffer = cast(
            BinaryIO,
            # ruff: ignore[open-file-with-context-handler] (returned to the caller)
            tempfile.SpooledTemporaryFile(max_size=8 * 2**20),
        )
        self._http.stream_to(url, buffer)
        buffer.seek(0)
        return buffer

    def _open_remote_version(self, reference: DatasetRef) -> DatasetVersion:
        """Bind one remote manifest to a range-capable filesystem.

        Returns:
            A handle without its payload sources, which ``open_version`` attaches.
        """
        manifest = self.get_manifest(reference.dataset_id, str(reference.version))
        root = f"{reference.dataset_id}/{reference.version}"
        files = {f"{root}/{part.path}": part.size for part in manifest.files.all_files()}
        filesystem = RegistryFileSystem(self._base_url, files, token=self._http.token, transport=self._http.transport)
        return DatasetVersion(manifest, pafs.PyFileSystem(pafs.FSSpecHandler(filesystem)), root)

    def _download_release(  # ruff: ignore[too-many-arguments]
        self,
        dataset_id: str,
        version: str,
        dest_dir: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Download a version's files into ``dest_dir`` (used by ``TimeNet.download`` for remotes).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            dest_dir: The target ``<...>/<id>/<version>`` directory.
            force: Re-download even if a copy already exists.
            manifest: The already-parsed manifest, passed to avoid re-fetching it; fetched if ``None``.
            progress_cb: Called with each chunk's byte count as it is written, for a progress display.
        """
        if manifest is None:
            manifest = self.get_manifest(dataset_id, version)
        download_version_files(
            self._http, manifest, dataset_id, version, Path(dest_dir), force=force, progress_cb=progress_cb
        )

    def put_dataset(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> DatasetRef:
        """Write a dataset locally and publish it to the remote registry.

        If the version already exists, return its reference unless ``force`` is set.

        Args:
            dataset: The dataset to write.
            force: Publish even if the version already exists.
            values_backend: Storage format for values (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional callback for write progress.
            pinned_by: Releases that already read this release's files, with the pins that the new files must match.

        Returns:
            A reference to the stored release.

        Raises:
            TimeNetRegistryError: If the service omits a listed file or requests a file that the write did not produce.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by put_release
        if dataset.schema is None:
            dataset.derive_schema()
        dataset_id = dataset.metadata.dataset_id
        version = str(dataset.metadata.dataset_version)
        reference = DatasetRef.parse(f"{dataset_id}@{version}")
        if not force and self.exists(dataset_id, version):
            return reference
        staging_root = Path(tempfile.mkdtemp(prefix="timenet-publish-"))
        try:
            # The writer stages locally but checks every payload source against this registry.
            with TimeFWriter(
                staging_root,
                dataset,
                values_backend=values_backend,
                progress_cb=progress_cb,
                registry=self,
                pinned_by=pinned_by,
            ) as writer:
                writer.write()
            return self.put_release(release_dir(staging_root, reference))
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def put_release(self, from_directory: Path) -> DatasetRef:
        """Publish a release directory to the remote registry.

        The directory must contain ``manifest.json`` and every file it lists.

        Args:
            from_directory: The release directory.

        Returns:
            A reference to the published release.

        Raises:
            TimeFFormatError: If ``manifest.json`` is missing, unreadable, or invalid.
            TimeNetFileError: If a listed file is missing or its size differs from the manifest.
            TimeNetRegistryError: If the service skips a listed file or requests a missing file.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _read_release
        manifest = self._read_release(from_directory)
        dataset_id = manifest.dataset_id
        version = str(manifest.metadata.dataset_version)
        manifest_bytes = (from_directory / MANIFEST_FILE).read_bytes()
        declared = {part.path for part in manifest.files.all_files()}
        publish_endpoint = f"/datasets/{dataset_id}/{version}/publish"
        publish = PublishResponseModel.model_validate_json(
            self._http.post(publish_endpoint, content=manifest_bytes).content
        )
        files = publish.files
        # Cross-check the service's upload list against the manifest before uploading. A manifest file
        # the service omits would publish an incomplete version; a requested file we did not produce is
        # not ours to upload. Both fail loudly here instead of partway through the upload.
        unrequested = sorted(declared - set(files))
        missing = [relpath for relpath in files if not (from_directory / relpath).is_file()]
        if unrequested or missing:
            raise TimeNetRegistryError(
                f"publish for {dataset_id}@{version} disagrees with the manifest: "
                f"service did not request {unrequested}, requested files not produced locally {missing}"
            )
        for relpath in files:
            self._upload_file(dataset_id, version, from_directory, relpath)
        self._http.post(f"/datasets/{dataset_id}/{version}/finalize")
        return DatasetRef(dataset_id=dataset_id, version=manifest.metadata.dataset_version)

    def _upload_file(self, dataset_id: str, version: str, version_dir: Path, relpath: str) -> None:
        """Fetch a presigned PUT URL for one file and upload its bytes.

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            version_dir: The compiled version directory holding the file.
            relpath: The version-relative file path.
        """
        endpoint = f"/datasets/{dataset_id}/{version}/publish/upload-url"
        grant = UploadGrantModel.model_validate_json(self._http.post(endpoint, json={"path": relpath}).content)
        content = (version_dir / relpath).read_bytes()
        self._http.put(grant.url, content=content, headers=grant.headers)

    def _resolve_latest(self, dataset_id: str) -> str:
        """Resolve a dataset's latest version string via the detail endpoint.

        Args:
            dataset_id: The ``org/name`` id.

        Returns:
            The latest committed version string.
        """
        detail = DatasetDetailModel.model_validate_json(self._http.get_text(f"/datasets/{dataset_id}"))
        return str(detail.version)
