"""The build pipeline that turns a connector into a stored dataset."""

from collections.abc import Callable, Mapping
from pathlib import Path
import shutil

from timenet.config import settings
from timenet.connectors import BaseConnector, BuildContext
from timenet.dataset import TimeFDataset
from timenet.dataset.composition import _record_parent
from timenet.errors import TimeFValidationError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import PayloadSource
from timenet.reader import TimeFReader
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef
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
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> Path:
    """Run one connector through the full build pipeline and return the version directory.

    This function is idempotent. If the target version is already committed, it skips the expensive
    ``download``, ``convert``, and ``store`` stages and returns the existing directory. Pass ``force``
    to rebuild it. Otherwise the stages run in order: create the cache directory, open the declared
    parents from ``root``, ``download`` raw references into the cache, ``compose`` them into a
    dataset, record each declared parent in it, ``derive_schema``, ``store``, then delete the cache
    directory. Parents must already be in ``root``; the engine never fetches or builds them. A card
    that declares parents needs a connector that overrides ``compose``; otherwise the build raises
    before it opens a parent or downloads anything. The engine only writes local files. Publishing
    to a remote registry is a separate step.

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
        pinned_by: The releases that already read files of this release, with their pins. The write
            publishes only if it reproduces each pinned file, and otherwise raises.

    Returns:
        The committed version directory.

    Raises:
        TimeFValidationError: If the card declares parents and the connector does not override
            ``compose``.
    """  # noqa: DOC502 - raised by _check_composes
    resolved_backend = _resolve_values_backend(connector, values_backend)
    # Read only the connector's dataset.yaml card, a tiny local file, not the dataset itself. This lets
    # us resolve the version directory and skip the expensive download and convert when it exists.
    metadata = connector.metadata()
    version_dir = root / metadata.dataset_id / str(metadata.dataset_version)
    committed = (version_dir / MANIFEST_FILE).exists()
    if committed and not force:
        return version_dir
    _check_composes(connector, metadata)

    cache = cache_dir if cache_dir is not None else settings().cache_dir / metadata.dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    # Open the declared parents before downloading, so a missing parent fails before any raw I/O.
    with open_parents(metadata, LocalRegistry(root)) as context:
        dataset = _compose(connector, cache, context)
        # Derive the schema before writing. store_dataset() derives one only for a dataset that has none.
        dataset.derive_schema()
        # A forced rebuild replaces the committed version only once the new one is ready, so a failed
        # write keeps the old one.
        store_dataset(
            dataset,
            root,
            values_backend=resolved_backend,
            progress_cb=progress_cb,
            replace=force,
            pinned_by=pinned_by,
        )
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
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> str:
    """Run one connector and publish the result through a writable registry.

    Like :func:`run_pipeline`, but the converted dataset is handed to ``registry.store`` instead of
    written to a local directory, so it also targets a remote or S3 registry. It is idempotent: an
    already-committed version skips the ``download`` and ``convert`` stages and returns unless ``force``.
    A registry other than a local one holds open releases only, so a card that is not open fails
    before any download. The composed dataset can be a child whose card says open while a parent is
    not open. So the engine checks the dataset again with its lineage before the write.

    Args:
        connector: The connector to build.
        registry: The writable registry to publish into (local, remote, or S3).
        cache_dir: Directory for downloaded artifacts (defaults to ``<TIMENET_CACHE>/<dataset_id>``).
        keep_cache: Keep the cache directory instead of removing it after publishing.
        values_backend: Storage backend for the values plane (``"parquet"`` or ``"zarr"``). When
            ``None``, this function uses the connector's ``values_backend``.
        progress_cb: Optional writer progress callback.
        force: Republish even if the version is already committed.
        pinned_by: The releases that already read files of this release, with their pins. The store
            publishes only if it reproduces each pinned file, and otherwise raises.

    Returns:
        The published version string.

    Raises:
        TimeFValidationError: If the card declares parents and the connector does not override
            ``compose``.
        TimeNetAccessError: If the registry is not local, and the card, or the composed dataset with
            its lineage, is not open.
    """  # noqa: DOC502 - raised by _check_composes and require_open_access
    resolved_backend = _resolve_values_backend(connector, values_backend)
    metadata = connector.metadata()
    dataset_id = metadata.dataset_id
    version = str(metadata.dataset_version)
    if not force and registry.exists(dataset_id, version):
        return version
    if not isinstance(registry, LocalRegistry):
        require_open_access(metadata)
    _check_composes(connector, metadata)

    cache = cache_dir if cache_dir is not None else settings().cache_dir / dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    with open_parents(metadata, registry) as context:
        dataset = _compose(connector, cache, context)
        if not isinstance(registry, LocalRegistry):
            require_open_access(dataset.metadata, dataset.lineage.values())
        dataset.derive_schema()
        # Pass pins only when there are any, so a registry written against the earlier store()
        # signature keeps working for every other build.
        pinned = {"pinned_by": pinned_by} if pinned_by else {}
        registry.store(dataset, force=force, values_backend=resolved_backend, progress_cb=progress_cb, **pinned)
    if not keep_cache and cache_dir is None and cache.is_dir():
        shutil.rmtree(cache)
    return version


def open_parents(metadata: DatasetMetadata, registry: BaseRegistry) -> BuildContext:
    """Open each declared parent of a card from the registry a build writes to.

    A local registry must hold every parent complete, with its payload sources. A remote registry
    resolves the payload source manifests upfront, and the readers read catalogs and payloads only
    when the connector needs them. The engine never fetches or builds a parent here.

    Args:
        metadata: The child dataset card with exact parent references.
        registry: The registry the child is written to.

    Returns:
        The opened readers, keyed by parent alias. Close the context after the write.

    Raises:
        TimeNetDatasetNotFoundError: If the registry does not hold a declared parent.
    """  # noqa: DOC502 - registry preparation raises the documented error
    context = BuildContext(references=dict(metadata.parents))
    try:
        for alias, ref in metadata.parents.items():
            version = (
                prepare_local_copy(registry.root, ref)
                if isinstance(registry, LocalRegistry)
                else registry.open_version(ref.dataset_id, str(ref.version))
            )
            context.parents[alias] = TimeFReader(version)
    except BaseException:
        context.close()
        raise
    return context


def _check_composes(connector: BaseConnector, metadata: DatasetMetadata) -> None:
    """Reject a connector that declares parents but cannot receive them.

    Raises:
        TimeFValidationError: If the card declares parents and the connector does not override
            ``compose``.
    """
    if metadata.parents and type(connector).compose is BaseConnector.compose:
        raise TimeFValidationError(
            f"{metadata.dataset_id} declares parents {sorted(metadata.parents)}, so its connector must "
            "implement compose() to receive them"
        )


def _compose(connector: BaseConnector, cache: Path, context: BuildContext) -> TimeFDataset:
    """Download, compose, then record each declared parent in the dataset.

    Recording a parent sets its card and its lineage, whether or not the connector copied objects
    from it.

    Returns:
        The composed dataset.
    """
    dataset = connector.compose(connector.download(cache), context)
    for alias, parent in context.parents.items():
        _record_parent(dataset, alias, parent)
    return dataset


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


def store_dataset(  # noqa: PLR0913
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    values_backend: str = "parquet",
    replace: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
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
        replace: Replace the version if it is already committed, keeping it until the new one is
            published.
        pinned_by: The releases that already read files of this release, with their pins. The
            writer publishes only if it reproduces each pinned file.

    Returns:
        The committed version directory.
    """
    if dataset.schema is None:
        dataset.derive_schema()
    with TimeFWriter(
        root, dataset, progress_cb=progress_cb, values_backend=values_backend, replace=replace, pinned_by=pinned_by
    ) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
