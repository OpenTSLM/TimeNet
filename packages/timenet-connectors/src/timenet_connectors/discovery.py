"""Lazy, convention-based connector discovery.

Concrete connectors live at ``datasets/<org>/<name>.py`` (or ``datasets/<name>.py`` for a flat id) and
expose a module-level ``CONNECTOR`` class. A dataset id maps to that module by convention, so there is
no central registry to maintain and only the requested connector is imported. Each connector declares
its own id via ``metadata()``.
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
    """
    module_name = _module_name(dataset_id)
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None  # a missing parent package (e.g. an unknown org)
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
    ids: list[str] = []
    for info in pkgutil.walk_packages(_datasets.__path__, _DATASETS + "."):
        if info.ispkg:
            continue
        connector = getattr(importlib.import_module(info.name), "CONNECTOR", None)
        if connector is not None:
            ids.append(connector().metadata().dataset_id)
    return sorted(ids)


def _module_name(dataset_id: str) -> str:
    """Map a dataset id to its connector module path.

    Python packages are lowercase (to avoid case-insensitive-filesystem clashes) and hyphens in the
    name become underscores, so ``chengsenwang/tsqa`` -> ``...datasets.chengsenwang.tsqa``.

    Returns:
        The dotted module path of the connector.
    """
    org, slash, name = dataset_id.partition("/")
    org = org.lower()
    leaf = (name if slash else org).replace("-", "_").lower()
    return f"{_DATASETS}.{org}.{leaf}" if slash else f"{_DATASETS}.{leaf}"
