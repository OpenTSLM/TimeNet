"""Producer-side shortcuts for curating and consuming a dataset from local code.

:func:`build` runs a connector through the engine into the shared local registry; :func:`load` does
that and reads the result back. Heavy dependencies (the writer/reader stacks) are imported lazily so
importing :mod:`timenet_connectors` stays cheap.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.dataset import TimeFDataset


def build(dataset_id: str, *, version: str | None = None, out: str | Path | None = None, force: bool = False) -> Path:
    """Curate a dataset into a local registry, resolving its connector by id.

    The producer-side one-liner over the engine. An already-curated version is reused unless ``force``
    is set. The default output is the shared local registry
    (:func:`timenet.registry.default_registry_path`), the very directory the SDK reads from, so a build
    here is immediately loadable with ``TimeNet().load(dataset_id)``.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The expected dataset version. A connector only produces its own version, so this is a
            guard: if it does not match the connector's metadata, the build is rejected before it runs.
            ``None`` builds whatever version the connector declares.
        out: Output registry directory; defaults to the shared local registry.
        force: Rebuild even if the version is already curated.

    Returns:
        The committed version directory.

    Raises:
        TimeFValidationError: If ``version`` is set and does not match the connector's declared version.
    """
    from timenet.engine import run_pipeline
    from timenet.errors import TimeFValidationError
    from timenet.registry import default_registry_path
    from timenet_connectors.discovery import resolve

    connector = resolve(dataset_id)()
    if version is not None:
        available = str(connector.metadata().dataset_version)
        if version != available:
            raise TimeFValidationError(
                f"connector for {dataset_id!r} builds version {available}, not the requested {version}"
            )
    root = Path(out).expanduser() if out is not None else default_registry_path()
    return run_pipeline(connector, root, force=force)


def load(dataset_id: str, version: str | None = None) -> "TimeFDataset":
    """Curate a dataset if needed, then load it into memory.

    The demo/one-call sugar over :func:`build` plus :meth:`timenet.client.TimeNet.load`. For a clear
    producer/consumer split, call :func:`build` and ``TimeNet().load`` yourself.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The version string, or ``None`` for the latest. Reconciled against the connector's
            declared version by :func:`build`, which rejects a mismatch.

    Returns:
        The loaded dataset.
    """
    from timenet.client import TimeNet

    build(dataset_id, version=version)
    return TimeNet().load(dataset_id, version)
