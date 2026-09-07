"""The build pipeline that turns a connector into a stored dataset."""

from collections.abc import Callable
from pathlib import Path
import shutil

from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.constants import MANIFEST_FILE
from timenet.registry.writable import WritableRegistry
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS
from timenet.writer import TimeFWriter, WriteProgressEvent


def run_pipeline(  # noqa: PLR0913
    connector: BaseConnector,
    root: Path,
    *,
    cache_dir: Path | None = None,
    keep_cache: bool = False,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    force: bool = False,
    values_backend: str | None = None,
) -> Path:
    """Run one connector through the full build pipeline and return the version directory.

    This function is idempotent. If the target version is already committed, it skips the expensive
    ``download``, ``convert``, and ``store`` stages and returns the existing directory. Pass ``force``
    to rebuild it. Otherwise the stages run in order: create the cache directory, ``download`` raw
    references into it, ``convert`` them into a dataset, ``derive_schema``, ``store``, then delete
    the cache directory. The engine only writes local files. Publishing to a remote registry is a
    separate step.

    Args:
        connector: The connector to build.
        root: Output root. This function writes the dataset to ``<root>/<dataset_id>/<version>/``.
        cache_dir: Directory for downloaded artifacts (defaults to ``<TIMENET_CACHE>/<dataset_id>``).
        keep_cache: Keep the cache directory instead of removing it once the dataset is stored.
            Conversion is the only stage that needs the raw sources, so removing them frees disk
            after a successful build. The sources re-download on the next run.
        values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``). When
            ``None``, this function uses the connector's ``values_backend``, so a connector that
            needs Zarr declares it once on the class.
        progress_cb: Optional writer progress callback.
        force: Rebuild even if the version is already committed.

    Returns:
        The committed version directory.
    """
    resolved_backend = _resolve_values_backend(connector, values_backend)
    # Read only the connector's dataset.yaml card, a tiny local file, not the dataset itself. This lets
    # us resolve the version directory and skip the expensive download and convert when it exists.
    metadata = connector.metadata()
    version_dir = root / metadata.dataset_id / str(metadata.dataset_version)
    committed = (version_dir / MANIFEST_FILE).exists()
    if committed and not force:
        return version_dir

    cache = cache_dir if cache_dir is not None else settings().cache_dir / metadata.dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    raw_refs = connector.download(cache)
    dataset = connector.convert(raw_refs)
    # Derive the schema here, before the force-rebuild rmtree below. A schema failure then aborts while
    # the old committed version is still on disk. store_dataset() re-derives only if a caller reaches it
    # directly with an underived dataset. This call is not redundant with that guard.
    dataset.derive_schema()
    if committed:  # force rebuild: drop the old committed version so the writer can republish it
        shutil.rmtree(version_dir)
    store_dataset(dataset, root, values_backend=resolved_backend, progress_cb=progress_cb)
    # Only clean a cache that we created. A caller-supplied cache_dir is user-owned. We must never
    # delete it.
    if not keep_cache and cache_dir is None and cache.is_dir():
        shutil.rmtree(cache)
    return version_dir


def publish_pipeline(  # noqa: PLR0913
    connector: BaseConnector,
    registry: WritableRegistry,
    *,
    cache_dir: Path | None = None,
    keep_cache: bool = False,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    force: bool = False,
    values_backend: str | None = None,
) -> str:
    """Run one connector and publish the result through a writable registry.

    Like :func:`run_pipeline`, but the converted dataset is handed to ``registry.store`` instead of
    written to a local directory, so it also targets a remote or S3 registry. It is idempotent: an
    already-committed version skips the ``download`` and ``convert`` stages and returns unless ``force``.

    Args:
        connector: The connector to build.
        registry: The writable registry to publish into (local, remote, or S3).
        cache_dir: Directory for downloaded artifacts (defaults to ``<TIMENET_CACHE>/<dataset_id>``).
        keep_cache: Keep the cache directory instead of removing it after publishing.
        values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``). When
            ``None``, this function uses the connector's ``values_backend``.
        progress_cb: Optional writer progress callback.
        force: Republish even if the version is already committed.

    Returns:
        The published version string.
    """
    resolved_backend = _resolve_values_backend(connector, values_backend)
    metadata = connector.metadata()
    dataset_id = metadata.dataset_id
    version = str(metadata.dataset_version)
    if not force and registry.exists(dataset_id, version):
        return version

    cache = cache_dir if cache_dir is not None else settings().cache_dir / dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    dataset = connector.convert(connector.download(cache))
    dataset.derive_schema()
    registry.store(dataset, force=force, values_backend=resolved_backend, progress_cb=progress_cb)
    if not keep_cache and cache_dir is None and cache.is_dir():
        shutil.rmtree(cache)
    return version


def _resolve_values_backend(connector: BaseConnector, override: str | None) -> str:
    """Resolve and validate the backend before a build changes any files.

    Returns:
        The supported backend selected by the override or connector default.

    Raises:
        TimeFValidationError: If the selected backend is unknown.
    """
    backend = connector.values_backend if override is None else override
    if backend not in SUPPORTED_VALUES_BACKENDS:
        raise TimeFValidationError(
            f"unknown values_backend {backend!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
        )
    return backend


def store_dataset(
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    values_backend: str = "parquet",
) -> Path:
    """Serialize a populated dataset to the TimeF format under ``root``.

    If the dataset has no schema, this function derives it first. It then streams the dataset through a
    :class:`~timenet.writer.TimeFWriter`. This function lives on the engine, not on
    :class:`~timenet.connectors.BaseConnector`, because it reads only ``dataset``. This keeps the
    connector contract at fetch-and-convert and avoids a connector-to-writer dependency.

    Args:
        dataset: The populated dataset from ``convert``.
        root: Parent directory. This function creates the version directory beneath it.
        progress_cb: Optional writer progress callback.
        values_backend: Storage backend for the values plane.

    Returns:
        The committed version directory.
    """
    if dataset.schema is None:
        dataset.derive_schema()
    with TimeFWriter(root, dataset, progress_cb=progress_cb, values_backend=values_backend) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
