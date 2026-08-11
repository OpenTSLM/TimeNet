"""The curation pipeline that turns a connector into a stored dataset."""

from collections.abc import Callable
from pathlib import Path
import shutil

from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.format.constants import MANIFEST_FILE
from timenet.writer import TimeFWriter, WriteProgressEvent


def run_pipeline(  # noqa: PLR0913
    connector: BaseConnector,
    root: Path,
    *,
    cache_dir: Path | None = None,
    clean_cache: bool = False,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    force: bool = False,
) -> Path:
    """Run one connector through the full curation pipeline and return the version directory.

    This function is idempotent. If the target version is already committed, it skips the expensive
    ``download``, ``convert``, and ``store`` stages and returns the existing directory. Pass ``force``
    to rebuild it. Otherwise the stages run in order: create the cache directory, ``download`` raw
    references into it, ``convert`` them into a dataset, ``derive_schema``, then ``store``. The engine
    only writes local files. Publishing to a remote registry is a separate step.

    Args:
        connector: The connector to curate.
        root: Output root. The dataset is written to ``<root>/<dataset_id>/<version>/``.
        cache_dir: Directory for downloaded artifacts (defaults to ``<TIMENET_CACHE>/<dataset_id>``).
        clean_cache: Remove the cache directory after the dataset is stored. The raw sources are only
            needed during conversion, so this frees disk after a successful build. The sources
            re-download on the next run.
        progress_cb: Optional writer progress callback.
        force: Rebuild even if the version is already committed.

    Returns:
        The committed version directory.
    """
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
    store_dataset(dataset, root, progress_cb=progress_cb)
    # Only clean a cache that we created. A caller-supplied cache_dir is user-owned. We must never
    # delete it.
    if clean_cache and cache_dir is None and cache.is_dir():
        shutil.rmtree(cache)
    return version_dir


def store_dataset(
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> Path:
    """Serialize a populated dataset to the TimeF format under ``root``.

    If the dataset has no schema, this function derives it first. It then streams the dataset through a
    :class:`~timenet.writer.TimeFWriter`. This function lives on the engine, not on
    :class:`~timenet.connectors.BaseConnector`, because it reads only ``dataset``. This keeps the
    connector contract at fetch-and-convert and avoids a connector-to-writer dependency.

    Args:
        dataset: The populated dataset from ``convert``.
        root: Parent directory; the version directory is created beneath it.
        progress_cb: Optional writer progress callback.

    Returns:
        The committed version directory.
    """
    if dataset.schema is None:
        dataset.derive_schema()
    with TimeFWriter(root, dataset, progress_cb=progress_cb) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
