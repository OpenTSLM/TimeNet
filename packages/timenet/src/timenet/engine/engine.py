"""The curation pipeline that turns a connector into a stored dataset."""

from collections.abc import Callable
from pathlib import Path
import shutil

from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.format.constants import MANIFEST_FILE
from timenet.writer import TimeFWriter, WriteProgressEvent


def run_pipeline(
    connector: BaseConnector,
    root: Path,
    *,
    cache_dir: Path | None = None,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    force: bool = False,
) -> Path:
    """Run one connector through the full curation pipeline and return the version directory.

    Idempotent: if the target version is already committed, the expensive ``download`` / ``convert`` /
    ``store`` stages are skipped and the existing directory is returned. Pass ``force`` to rebuild it.
    Otherwise the stages are: create the cache directory, ``download`` raw references into it,
    ``convert`` them into a dataset, ``derive_schema``, then ``store``. The engine only writes local
    files; publishing to a remote registry is a separate step.

    Args:
        connector: The connector to curate.
        root: Output root; the dataset is written to ``<root>/<dataset_id>/<version>/``.
        cache_dir: Directory for downloaded artifacts (defaults to ``<TIMENET_CACHE>/<dataset_id>``).
        progress_cb: Optional writer progress callback.
        force: Rebuild even if the version is already committed.

    Returns:
        The committed version directory.
    """
    # Reads the connector's dataset.yaml card only (a tiny local file), not the dataset itself, so we
    # can resolve the version directory and skip the expensive download/convert when it already exists.
    metadata = connector.metadata()
    version_dir = root / metadata.dataset_id / str(metadata.dataset_version)
    committed = (version_dir / MANIFEST_FILE).exists()
    if committed and not force:
        return version_dir

    cache = cache_dir if cache_dir is not None else settings().cache_dir / metadata.dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    raw_refs = connector.download(cache)
    dataset = connector.convert(raw_refs)
    # Derived here, before the force-rebuild rmtree below, so a schema failure aborts while the old
    # committed version is still on disk. store_dataset() re-derives only if a caller reaches it
    # directly with an underived dataset, so this is not redundant with that guard.
    dataset.derive_schema()
    if committed:  # force rebuild: drop the old committed version so the writer can republish it
        shutil.rmtree(version_dir)
    return store_dataset(dataset, root, progress_cb=progress_cb)


def store_dataset(
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> Path:
    """Serialize a populated dataset to the TimeF format under ``root``.

    Derives the schema first if the dataset has none, then streams it through a
    :class:`~timenet.writer.TimeFWriter`. This lives on the engine rather than on
    :class:`~timenet.connectors.BaseConnector` because it reads only ``dataset``: keeping it here
    leaves the connector contract at fetch-and-convert and avoids a connector-to-writer dependency.

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
