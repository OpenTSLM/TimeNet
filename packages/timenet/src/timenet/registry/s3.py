"""A registry backed by an S3 (or S3-compatible) bucket.

An ``s3://<bucket>/<prefix>`` root holds datasets under ``datasets/<dataset_id>/<version>/``. This
matches the hosted registry's bucket layout. The manifest's relative file paths do not change. Only
the key prefix differs from a :class:`~timenet.registry.LocalRegistry`. Credentials, region, and an
optional endpoint override come from boto3's default session (``AWS_*`` vars, ``AWS_PROFILE``,
``AWS_ENDPOINT_URL``). Opening a version reads its payload sources' manifests upfront and reads remote
payloads on demand. Complete cached releases are reused. The backend has no catalog, so listing and
search are unsupported. Use it to ``store`` and to fetch by an explicit ``org/name@version``.

boto3 is optional. Install the ``timenet[s3]`` extra to use this backend.
"""

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO
from urllib.parse import urlparse
import uuid

import pyarrow.fs as pafs

from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetRegistryError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest
from timenet.registry._paths import release_dir
from timenet.registry.lazy_read import LazyReadRegistry
from timenet.registry.version import DatasetVersion
from timenet.types import DatasetMetadata, Version
from timenet.types.metadata_fields import validate_dataset_id
from timenet.types.references import DatasetRef, is_latest
from timenet.writer import TimeFWriter, WriteProgressEvent


# Dataset objects live under a ``datasets/`` prefix, matching the hosted registry's bucket layout, so
# it can sit beside other top-level prefixes (a catalog, say) without colliding.
_DATASETS_PREFIX = "datasets"


class S3Registry(LazyReadRegistry):
    """Serves and publishes datasets under an ``s3://bucket/prefix`` root."""

    def __init__(self, uri: str, *, cache_dir: str | Path | None = None) -> None:
        """Open an S3-backed registry.

        Args:
            uri: The bucket or prefix root, for example ``s3://my-bucket/registry``.
            cache_dir: Where downloads are cached; defaults to the configured storage directory.

        Raises:
            TimeNetRegistryError: If ``uri`` is not an ``s3://bucket[/prefix]`` URL.
        """
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise TimeNetRegistryError(f"S3 registry URI must be s3://bucket[/prefix], got {uri!r}")
        self._bucket = parsed.netloc
        self._prefix = parsed.path.strip("/")
        self._cache_dir = Path(cache_dir) if cache_dir is not None else settings().storage_dir

    def __str__(self) -> str:
        return f"S3 registry {self._display_root()}"

    @property
    def local_root(self) -> Path:
        """The directory that holds complete local releases."""
        return self._cache_dir

    def list_datasets(self) -> list[DatasetMetadata]:
        """Unsupported: the S3 backend has no catalog.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "S3Registry has no catalog; listing and search are unsupported. Store and download by an "
            "explicit org/name@version."
        )

    def search(self, **_kwargs: Any) -> list[DatasetMetadata]:
        """Unsupported: the S3 backend has no catalog.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "S3Registry has no catalog; listing and search are unsupported. Store and download by an "
            "explicit org/name@version."
        )

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest (latest version if unspecified).

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string, or ``None`` / ``"latest"`` for the latest.

        Returns:
            The dataset's manifest.

        Raises:
            TimeNetDatasetNotFoundError: If the dataset id or version has no manifest in the bucket.
        """
        validate_dataset_id(dataset_id)
        resolved = self._latest_version(dataset_id) if is_latest(version) else version
        if resolved is None:
            raise TimeNetDatasetNotFoundError(
                f"no committed version for dataset {dataset_id!r} under {self._display_root()}"
            )
        body = self._get_bytes(self._key(dataset_id, resolved, MANIFEST_FILE))
        if body is None:
            raise TimeNetDatasetNotFoundError(
                f"no manifest for {dataset_id!r} version {resolved!r} under {self._display_root()}"
            )
        return Manifest.model_validate_json(body.decode())

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Open one file of a dataset version for binary reading.

        Args:
            dataset_id: The ``org/name`` id.
            version: The version string.
            relpath: The version-relative file path.

        Returns:
            An open, streaming binary file object.

        Raises:
            TimeNetDatasetNotFoundError: If the object does not exist.
        """
        import botocore.exceptions  # noqa: PLC0415

        client = self._client()
        try:
            response = client.get_object(Bucket=self._bucket, Key=self._key(dataset_id, version, relpath))
        except botocore.exceptions.ClientError as exc:
            raise TimeNetDatasetNotFoundError(f"no object {relpath!r} for {dataset_id!r} version {version!r}") from exc
        return response["Body"]

    def _open_remote_version(self, reference: DatasetRef) -> DatasetVersion:
        """Bind one remote manifest to Arrow's native S3 filesystem.

        Returns:
            A handle without its payload sources, which ``open_version`` attaches.
        """
        manifest = self.get_manifest(reference.dataset_id, str(reference.version))
        root = f"{self._bucket}/{self._key(reference.dataset_id, str(reference.version))}"
        return DatasetVersion(manifest, self._s3_filesystem(), root)

    def store(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> str:
        """Compile a dataset locally and upload it under this registry's prefix.

        Uploads every file to a temporary ``<version>.tmp-<uuid>`` prefix first. Then moves each object
        into the final ``<version>/`` prefix with a server-side copy, writing ``manifest.json`` last as
        the commit marker. A failed publish then never leaves a partial version a reader can trust. An
        already-committed version is skipped unless ``force``.

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
        client = self._client()
        final_base = self._key(dataset_id, version)
        temp_base = f"{final_base}.tmp-{uuid.uuid4().hex}"
        staging = Path(tempfile.mkdtemp(prefix="timenet-s3-"))
        try:
            # The writer stages locally but checks every payload source against this registry.
            with TimeFWriter(
                staging, dataset, values_backend=values_backend, progress_cb=progress_cb, registry=self
            ) as writer:
                writer.write()
            version_dir = release_dir(staging, DatasetRef.parse(f"{dataset_id}@{version}"))
            self._delete_prefix(client, f"{final_base}.tmp-")  # sweep a prior crashed publish's temp prefix
            files = sorted(path for path in version_dir.rglob("*") if path.is_file())
            for path in files:
                relpath = path.relative_to(version_dir).as_posix()
                client.upload_file(str(path), self._bucket, f"{temp_base}/{relpath}")
            data_relpaths = [path.relative_to(version_dir).as_posix() for path in files if path.name != MANIFEST_FILE]
            for relpath in data_relpaths:
                self._move(client, f"{temp_base}/{relpath}", f"{final_base}/{relpath}")
            self._move(client, f"{temp_base}/{MANIFEST_FILE}", f"{final_base}/{MANIFEST_FILE}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self._delete_prefix(client, f"{temp_base}/")  # remove any objects left staged in the temp prefix
        return version

    @staticmethod
    def _s3_filesystem() -> pafs.S3FileSystem:
        """Build Arrow's S3 filesystem using boto3's credentials, region, and endpoint.

        Import boto3 here so local and HTTP users do not need the optional S3 dependency.

        Returns:
            A configured random-access filesystem.
        """
        import boto3  # noqa: PLC0415 - optional S3 dependency

        session = boto3.Session()
        credentials = session.get_credentials()
        kwargs: dict[str, Any] = {}
        if credentials is not None:
            frozen = credentials.get_frozen_credentials()
            kwargs.update(access_key=frozen.access_key, secret_key=frozen.secret_key, session_token=frozen.token)
        if session.region_name:
            kwargs["region"] = session.region_name
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            kwargs["endpoint_override"] = endpoint
            if endpoint.startswith("http://"):
                kwargs["scheme"] = "http"
        return pafs.S3FileSystem(**kwargs)

    @staticmethod
    def _client() -> Any:
        """Return a boto3 S3 client built from the default session (env creds/region/endpoint).

        Import boto3 here so local and HTTP users do not need the optional S3 dependency.

        Returns:
            A boto3 S3 client.
        """
        import boto3  # noqa: PLC0415 - optional S3 dependency

        return boto3.client("s3")

    def _key(self, *parts: str) -> str:
        """Build a bucket-relative key: the prefix, then ``datasets/``, then the path parts."""  # noqa: DOC201
        segments = [self._prefix, _DATASETS_PREFIX, *parts]
        return "/".join(segment.strip("/") for segment in segments if segment)

    def _display_root(self) -> str:
        """Return the ``s3://bucket/prefix`` root for error messages."""
        return f"s3://{self._bucket}/{self._prefix}" if self._prefix else f"s3://{self._bucket}"

    def _move(self, client: Any, src_key: str, dst_key: str) -> None:
        """Move one object to its final key: a server-side copy, then delete the source."""
        client.copy({"Bucket": self._bucket, "Key": src_key}, self._bucket, dst_key)
        client.delete_object(Bucket=self._bucket, Key=src_key)

    def _delete_prefix(self, client: Any, key_prefix: str) -> None:
        """Delete every object whose key starts with ``key_prefix`` (best-effort staging cleanup)."""
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=key_prefix):
            objects = [{"Key": entry["Key"]} for entry in page.get("Contents", [])]
            if objects:
                client.delete_objects(Bucket=self._bucket, Delete={"Objects": objects})

    def _get_bytes(self, key: str) -> bytes | None:
        """Return an object's bytes, or ``None`` if the object does not exist.

        Raises:
            TimeNetRegistryError: If S3 fails for a reason other than a missing object.
        """
        import botocore.exceptions  # noqa: PLC0415

        client = self._client()
        try:
            response = client.get_object(Bucket=self._bucket, Key=key)
        except botocore.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None  # a genuinely absent object
            raise TimeNetRegistryError(f"cannot read s3://{self._bucket}/{key}: {exc}") from exc
        return response["Body"].read()

    def _latest_version(self, dataset_id: str) -> str | None:
        """Return the highest committed version string for a dataset, or ``None``."""
        prefix = f"{self._key(dataset_id)}/"
        client = self._client()
        versions: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter="/"):
            for entry in page.get("CommonPrefixes", []):
                name = entry["Prefix"][len(prefix) :].strip("/")
                if _is_version(name):
                    versions.append(name)
        return max(versions, key=Version.parse) if versions else None


def _is_version(name: str) -> bool:
    """Return whether a directory name is a parseable ``major.minor.patch`` version."""
    try:
        Version.parse(name)
    except ValueError:
        return False
    return True
