"""Discover connectors lazily by convention.

Each connector lives in its own folder at ``datasets/<org>/<name>/``, or ``datasets/<name>/`` for a
flat id. The package's ``__init__.py`` exposes a ``CONNECTOR`` class. A ``dataset.yaml`` card sits
beside it. A dataset id maps to that package by convention, so there is no central registry. Discovery
imports only the requested connector. Each connector declares its own id through ``metadata()``.
"""

import importlib
import importlib.util
import pkgutil

from timenet.connectors import BaseConnector
import timenet_connectors.datasets as _datasets


_DATASETS = _datasets.__name__


def resolve(dataset_id: str) -> type[BaseConnector]:
    """Return the connector class for a dataset id and import only its module.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` (``chengsenwang/tsqa``) id.

    Returns:
        The connector class.

    Raises:
        LookupError: If no module exists for the id, or it exposes no ``CONNECTOR``.
        ModuleNotFoundError: If a connector module for the id exists but fails to import one of its own
            dependencies. This surfaces the real error instead of masking it as an unknown id.
    """
    module_name = _module_name(dataset_id)
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        # find_spec imports the parent packages. An unknown id fails to find one of *our* modules. A
        # broken import inside an existing connector fails to find something else. Surface that real
        # error instead of masking it as "unknown connector".
        if exc.name and (module_name + ".").startswith(exc.name + "."):
            spec = None  # a missing parent package, for example an unknown org
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

    This imports each connector module. Use it for listings and error messages, not the build hot path.

    Returns:
        The declared dataset ids, sorted.
    """
    ids: set[str] = set()
    for info in pkgutil.walk_packages(_datasets.__path__, _DATASETS + "."):
        # Skip each connector's co-located ``tests`` package of test modules and fixtures. It lives
        # beside the connector but is not a connector module. Importing a test module pulls
        # test-only deps such as pytest into this producer path. Match the ``tests`` dir, not a
        # ``test_``-prefixed name, so this still finds a connector like ``timenet/test-mean``.
        if "tests" in info.name.split("."):
            continue
        # A connector package re-exports CONNECTOR from its ``connector`` submodule, so both the
        # package and that submodule surface it. A set dedupes them. Org packages expose none.
        connector = getattr(importlib.import_module(info.name), "CONNECTOR", None)
        if connector is not None:
            ids.add(connector().metadata().dataset_id)
    return sorted(ids)


def _module_name(dataset_id: str) -> str:
    """Map a dataset id to its connector module path.

    Python packages are lowercase to avoid clashes on case-insensitive filesystems, and hyphens become
    underscores. So ``chengsenwang/tsqa`` maps to ``...datasets.chengsenwang.tsqa``.

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
