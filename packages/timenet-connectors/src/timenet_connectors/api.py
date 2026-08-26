"""Producer-side shortcuts to build and load a dataset from local code.

:func:`build` runs a connector through the engine into the shared local registry. :func:`load` runs
build and reads the result back. The writer and reader stacks are heavy, so they load lazily. This
keeps the import of :mod:`timenet_connectors` cheap.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.dataset import TimeFDataset


def build(
    dataset_id: str,
    *,
    version: str | None = None,
    out: str | Path | None = None,
    force: bool = False,
    keep_cache: bool = False,
) -> Path:
    """Build a dataset into a local registry by connector id.

    This is the producer-side one-liner over the engine. The build reuses an already-built version
    unless the caller sets ``force``. The default output is the shared local registry
    (:func:`timenet.registry.default_registry_path`). The SDK reads from that directory, so a build
    here loads at once with ``TimeNet().load(dataset_id)``.

    Like ``timenet-build build``, this runs the connector in an environment built from its
    ``requirements.txt``. Set ``TIMENET_ISOLATION=off`` to run it in this interpreter instead, which
    is what you want while writing a connector.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The expected dataset version. A connector produces only its own version, so this
            value is a guard. If it does not match the connector's metadata, the build stops before it
            runs. ``None`` builds the version that the connector declares.
        out: Output registry directory. Defaults to the shared local registry.
        force: Rebuild even if the registry already has a built version.
        keep_cache: Keep the raw download cache. A successful build deletes it by default.

    Returns:
        The committed version directory.

    Raises:
        TimeFValidationError: If the caller sets ``version`` and it does not match the connector's
            declared version.
    """  # noqa: DOC502 (raised by _check_version, not directly here)
    from timenet.config import settings  # noqa: PLC0415
    from timenet.registry import default_registry_path  # noqa: PLC0415

    if version is not None:
        _check_version(dataset_id, version)
    root = Path(out).expanduser() if out is not None else default_registry_path()
    if settings().isolation == "on":
        from timenet_connectors.builder.env import run_isolated  # noqa: PLC0415

        return Path(run_isolated(dataset_id, root, force=force, keep_cache=keep_cache))

    from timenet.engine import run_pipeline  # noqa: PLC0415
    from timenet_connectors.discovery import resolve  # noqa: PLC0415

    return run_pipeline(resolve(dataset_id)(), root, force=force, keep_cache=keep_cache)


def _check_version(dataset_id: str, version: str) -> None:
    """Reject a requested version the connector does not build.

    Deliberately import-free, like :func:`timenet_connectors.discovery.requirements_for`. This guard
    runs before the isolated build, so asking the connector for its metadata would need the very
    dependencies that only the child environment has. It reads the card beside the connector instead.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The requested dataset version.

    Raises:
        TimeFValidationError: If the connector declares a different version.
        LookupError: If no connector exists for the id.
        TimeNetInvalidCardError: If the connector's card is missing or invalid.
    """  # noqa: DOC502 (LookupError and TimeNetInvalidCardError come from the calls below)
    from timenet.errors import TimeFValidationError  # noqa: PLC0415
    from timenet.types import DatasetMetadata  # noqa: PLC0415
    from timenet_connectors.discovery import connector_dir  # noqa: PLC0415

    card = DatasetMetadata.from_yaml(connector_dir(dataset_id) / "dataset.yaml")
    available = str(card.dataset_version)
    if version != available:
        raise TimeFValidationError(
            f"connector for {dataset_id!r} builds version {available}, not the requested {version}"
        )


def load(dataset_id: str, version: str | None = None) -> "TimeFDataset":
    """Build a dataset if needed, then load it into memory.

    This is the one-call sugar over :func:`build` plus :meth:`timenet.client.TimeNet.load`. For a
    clear producer and consumer split, call :func:`build` and ``TimeNet().load`` yourself.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The version string, or ``None`` for the latest. :func:`build` checks this value
            against the connector's declared version and rejects a mismatch.

    Returns:
        The loaded dataset.
    """
    from timenet.client import TimeNet  # noqa: PLC0415

    build(dataset_id, version=version)
    return TimeNet().load(dataset_id, version)
