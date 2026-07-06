"""Lazy, convention-based connector discovery.

Each connector lives in its own folder at ``datasets/<org>/<name>/`` (or ``datasets/<name>/`` for a
flat id): the package's ``__init__.py`` exposes a ``CONNECTOR`` class and a ``dataset.yaml`` card sits
beside it. A dataset id maps to that package by convention, so there is no central registry to maintain
and only the requested connector is imported. Each connector declares its own id via ``metadata()``.
"""

import importlib
import importlib.util
import pkgutil

from timenet.connectors import BaseConnector
import timenet_connectors.datasets as _datasets


_DATASETS = _datasets.__name__


def resolve(dataset_id: str) -> type[BaseConnector]:
    """Return the connector class for a dataset id, importing only its module.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` (``chengsenwang/tsqa``) id.

    Returns:
        The connector class.

    Raises:
        LookupError: If no module exists for the id, or it exposes no ``CONNECTOR``.
        ModuleNotFoundError: If a connector module for the id exists but fails to import a dependency
            of its own (surfaced instead of being masked as an unknown id).
    """
    module_name = _module_name(dataset_id)
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        # find_spec imports the parent packages. An unknown id fails to find one of *our* modules; a
        # broken import inside an existing connector fails to find something else, so surface that real
        # error instead of masking it as "unknown connector".
        if exc.name and (module_name + ".").startswith(exc.name + "."):
            spec = None  # a missing parent package (e.g. an unknown org)
        else:
            raise
    if spec is None:
        raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(available()) or '(none)'}")
    connector = getattr(importlib.import_module(module_name), "CONNECTOR", None)
    if connector is None:
        raise LookupError(f"module {module_name!r} defines no CONNECTOR")
    return connector


def available() -> list[str]:
    """List the dataset ids of every discoverable connector.

    Imports each connector module (used for listings and error messages, not the build hot path).

    Returns:
        The declared dataset ids, sorted.
    """
    ids: set[str] = set()
    for info in pkgutil.walk_packages(_datasets.__path__, _DATASETS + "."):
        # A connector package re-exports CONNECTOR from its ``connector`` submodule, so both the
        # package and that submodule surface it; a set dedupes them. Org packages expose none.
        connector = getattr(importlib.import_module(info.name), "CONNECTOR", None)
        if connector is not None:
            ids.add(connector().metadata().dataset_id)
    return sorted(ids)


def _module_name(dataset_id: str) -> str:
    """Map a dataset id to its connector module path.

    Python packages are lowercase (to avoid case-insensitive-filesystem clashes) and hyphens become
    underscores, so ``chengsenwang/tsqa`` -> ``...datasets.chengsenwang.tsqa``.

    Returns:
        The dotted module path of the connector.
    """
    org, slash, name = dataset_id.partition("/")
    if slash:
        return f"{_DATASETS}.{_segment(org)}.{_segment(name)}"
    return f"{_DATASETS}.{_segment(org)}"


def _segment(part: str) -> str:
    """Normalize one id segment to its Python package name: lowercase, hyphens to underscores.

    Args:
        part: An org or name segment of a dataset id.

    Returns:
        The importable package/module name for that segment.
    """
    return part.replace("-", "_").lower()
