"""The curation pipeline that turns a connector into a stored dataset."""

from collections.abc import Callable
from pathlib import Path
import shutil

from timenet.connectors import BaseConnector
from timenet.writer import WriteProgressEvent
from timenet.writer.constants import MANIFEST_FILE


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
        cache_dir: Directory for downloaded artifacts (defaults to ``<root>/.cache/<dataset_id>``).
        progress_cb: Optional writer progress callback.
        force: Rebuild even if the version is already committed.

    Returns:
        The committed version directory.
    """
    metadata = connector.metadata()  # cheap by contract: no I/O, so we can check before downloading
    version_dir = root / metadata.dataset_id / str(metadata.dataset_version)
    committed = (version_dir / MANIFEST_FILE).exists()
    if committed and not force:
        return version_dir

    cache = cache_dir if cache_dir is not None else root / ".cache" / metadata.dataset_id
    cache.mkdir(parents=True, exist_ok=True)

    raw_refs = connector.download(cache)
    dataset = connector.convert(raw_refs)
    dataset.derive_schema()
    if committed:  # force rebuild: drop the old committed version so the writer can republish it
        shutil.rmtree(version_dir)
    return connector.store(dataset, root, progress_cb=progress_cb)
