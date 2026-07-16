"""Producer-side shortcuts for curating and consuming a dataset from local code.

:func:`build` runs a connector through the engine into the shared local registry; :func:`load` does
that and reads the result back. Heavy dependencies (the writer/reader stacks) are imported lazily so
importing :mod:`timenet_connectors` stays cheap.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.dataset import TimeFDataset


def build(dataset_id: str, *, out: str | Path | None = None, force: bool = False) -> Path:
    """Curate a dataset into a local registry, resolving its connector by id.

    The producer-side one-liner over the engine. An already-curated version is reused unless ``force``
    is set. The default output is the shared local registry
    (:func:`timenet.registry.default_registry_path`), the very directory the SDK reads from, so a build
    here is immediately loadable with ``TimeNet().load(dataset_id)``.

    Args:
        dataset_id: The dataset id (``org/name``).
        out: Output registry directory; defaults to the shared local registry.
        force: Rebuild even if the version is already curated.

    Returns:
        The committed version directory.
    """
    from timenet.engine import run_pipeline
    from timenet.registry import default_registry_path
    from timenet_connectors.discovery import resolve

    root = Path(out).expanduser() if out is not None else default_registry_path()
    return run_pipeline(resolve(dataset_id)(), root, force=force)


def load(dataset_id: str, version: str | None = None) -> "TimeFDataset":
    """Curate a dataset if needed, then load it into memory.

    The demo/one-call sugar over :func:`build` plus :meth:`timenet.client.TimeNet.load`. For a clear
    producer/consumer split, call :func:`build` and ``TimeNet().load`` yourself.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The version string, or ``None`` for the latest.

    Returns:
        The loaded dataset.
    """
    from timenet.client import TimeNet

    build(dataset_id)
    return TimeNet().load(dataset_id, version)
