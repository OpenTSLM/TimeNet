"""A registry backed by the hosted TimeNet HTTP service.

Talks to ``timenet-registry`` through :class:`~timenet.registry._http.RegistryHttpClient`: lists and
searches the catalog, fetches manifests, streams files, downloads or lazily range-reads versions, and
publishes datasets. Artifact bytes never pass through the API — the service hands back presigned URLs.
"""

from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO, cast

import httpx

from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest
from timenet.registry.remote._download import materialize_version
from timenet.registry.remote._fs import remote_version_filesystem, remote_version_root
from timenet.registry.remote._http import RegistryHttpClient
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.writer import TimeFWriter, WriteProgressEvent


class RemoteRegistry(WritableRegistry):
    """Serves datasets from the hosted TimeNet registry over HTTP(S)."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        download_mode: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Open a remote registry.

        Args:
            base_url: The service root, e.g. ``https://registry.dev.timenet.ai``.
            token: Bearer token; defaults to ``$TIMENET_TOKEN`` (``None`` is anonymous).
            cache_dir: Where downloads are cached; defaults to the configured storage directory.
            download_mode: ``"full"`` or ``"on_demand"``; defaults to ``$TIMENET_DOWNLOAD_MODE``.
            transport: An httpx transport for testing; ``None`` uses the network.
        """
        cfg = settings()
        self._base_url = base_url
        self._http = RegistryHttpClient(base_url, token=token if token is not None else cfg.token, transport=transport)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else cfg.storage_dir
        self._mode = download_mode or cfg.download_mode

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the latest-version metadata of every dataset, sorted by id.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        payload = self._http.get_json("/datasets")
        metadatas = [_metadata_from_summary(row) for row in payload["datasets"]]
        return sorted(metadatas, key=lambda m: m.dataset_id)

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest (latest version if unspecified).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string, or ``None`` / ``"latest"`` for the latest.

        Returns:
            The dataset's manifest.

        Raises:
            DatasetNotFoundError: If the dataset id or version is unknown.
        """  # noqa: DOC502 (raised by the HTTP client on 404, not directly here)
        resolved = self._resolve_latest(dataset_id) if version in {None, "", "latest"} else version
        text = self._http.get_text(f"/datasets/{dataset_id}/{resolved}/manifest")
        return Manifest.from_json(text)

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
            tempfile.SpooledTemporaryFile(max_size=8 * 2**20),  # noqa: SIM115 (returned to the caller)
        )
        self._http.stream_to(url, buffer)
        buffer.seek(0)
        return buffer

    def open_version(self, dataset_id: str, version: str | None = None, *, mode: str | None = None) -> DatasetVersion:
        """Open a committed version as a random-access handle.

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string, or ``None`` for the latest.
            mode: ``"full"`` or ``"on_demand"``; defaults to the registry's configured mode.

        Returns:
            A handle to the committed version's manifest and files.
        """
        manifest = self.get_manifest(dataset_id, version)
        resolved = str(manifest.metadata.dataset_version)
        dest = self._cache_dir / dataset_id / resolved
        # A fully downloaded version is read from local disk, like the local and S3 backends.
        if (dest / MANIFEST_FILE).exists():
            return DatasetVersion.open_local(dest)
        chosen = mode or self._mode
        # Zarr opens from a store URI it drives directly, which does not map onto presigned range reads,
        # so a remote zarr version always takes the full-download path.
        if chosen == "on_demand" and manifest.values_backend == "parquet":
            sizes = {part.path: part.size for part in manifest.files.all_files()}
            return DatasetVersion(
                manifest=manifest,
                filesystem=remote_version_filesystem(self._http, dataset_id, resolved, sizes),
                root=remote_version_root(dataset_id, resolved),
            )
        materialize_version(self._http, manifest, dataset_id, resolved, dest)
        return DatasetVersion.open_local(dest)

    def download_version(
        self,
        dataset_id: str,
        version: str,
        dest_dir: str | Path,
        *,
        force: bool = False,
        manifest: Manifest | None = None,
    ) -> None:
        """Download a version's files into ``dest_dir`` (used by ``TimeNet.download`` for remotes).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            dest_dir: The target ``<...>/<id>/<version>`` directory.
            force: Re-download even if a copy already exists.
            manifest: The already-parsed manifest, passed to avoid re-fetching it; fetched if ``None``.
        """
        if manifest is None:
            manifest = self.get_manifest(dataset_id, version)
        materialize_version(self._http, manifest, dataset_id, version, Path(dest_dir), force=force)

    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Compile a dataset locally and publish it to the remote registry.

        Compiles to a temporary directory, then runs the service publish flow: POST the manifest, PUT
        each declared file to its presigned URL, and finalize. An already-committed version is skipped
        unless ``force``.

        Args:
            dataset: The populated dataset to store.
            force: Publish even if the version is already committed.
            values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional writer progress callback.

        Returns:
            The stored version string.
        """
        if dataset.schema is None:
            dataset.derive_schema()
        dataset_id = dataset.metadata.dataset_id
        version = str(dataset.metadata.dataset_version)
        if not force and self.exists(dataset_id, version):
            return version
        staging_root = Path(tempfile.mkdtemp(prefix="timenet-publish-"))
        try:
            with TimeFWriter(staging_root, dataset, values_backend=values_backend, progress_cb=progress_cb) as writer:
                writer.write()
            version_dir = staging_root / dataset_id / version
            manifest_bytes = (version_dir / "manifest.json").read_bytes()
            files = self._http.post(f"/datasets/{dataset_id}/{version}/publish", content=manifest_bytes).json()["files"]
            for relpath in files:
                self._upload_file(dataset_id, version, version_dir, relpath)
            self._http.post(f"/datasets/{dataset_id}/{version}/finalize")
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        return version

    def _upload_file(self, dataset_id: str, version: str, version_dir: Path, relpath: str) -> None:
        """Fetch a presigned PUT URL for one file and upload its bytes.

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            version_dir: The compiled version directory holding the file.
            relpath: The version-relative file path.
        """
        grant = self._http.post(f"/datasets/{dataset_id}/{version}/publish/upload-url", json={"path": relpath}).json()
        content = (version_dir / relpath).read_bytes()
        self._http.put(grant["url"], content=content, headers=grant["headers"])

    def _resolve_latest(self, dataset_id: str) -> str:
        """Resolve a dataset's latest version string via the detail endpoint.

        Args:
            dataset_id: The ``org/name`` id.

        Returns:
            The latest committed version string.
        """
        detail = self._http.get_json(f"/datasets/{dataset_id}")
        return detail["version"]


def _metadata_from_summary(row: dict) -> DatasetMetadata:
    """Build :class:`DatasetMetadata` from a registry summary row.

    Args:
        row: A ``DatasetSummary`` mapping from the API.

    Returns:
        The reconstructed metadata (``source_url`` unknown, schema version defaulted).
    """
    return DatasetMetadata.from_dict(
        {
            "dataset_id": row["dataset_id"],
            "dataset_version": row["version"],
            "name": row["name"],
            "description": row["description"],
            "license": row["license"],
            "domains": row.get("domains", []),
            "tags": row.get("tags", []),
        }
    )
