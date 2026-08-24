"""Discover connectors lazily by convention.

Each connector lives in its own folder at ``datasets/<org>/<name>/``, or ``datasets/<name>/`` for a
flat id. The package's ``__init__.py`` exposes a ``CONNECTOR`` class. A ``dataset.yaml`` card sits
beside it. A dataset id maps to that package by convention, so there is no central registry. Discovery
imports only the requested connector. Each connector declares its own id through ``metadata()``.
"""

import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import pkgutil

import yaml

from timenet.connectors import BaseConnector
import timenet_connectors.datasets as _datasets


_DATASETS = _datasets.__name__


def connector_dir(dataset_id: str) -> Path:
    """Return the directory of a dataset id's connector package.

    Deliberately import-free. This runs before the connector's environment
    exists, so importing the connector module (which is what :func:`resolve`
    does) would fail on the very dependencies its ``requirements.txt``
    declares. ``find_spec`` imports the parent packages only, never the leaf.

    A connector directory holds a ``dataset.yaml`` card. An org namespace package
    (``physionet``) also has ``submodule_search_locations`` but no card, so it is
    not a connector and is rejected like any unknown id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The connector package's directory.

    Raises:
        LookupError: If no connector with a ``dataset.yaml`` card exists for the id.
    """
    spec = _connector_spec(dataset_id)
    if spec and spec.submodule_search_locations:
        path = Path(next(iter(spec.submodule_search_locations)))
        if (path / "dataset.yaml").is_file():
            return path
    raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(_known_ids()) or '(none)'}")


def has_connector(dataset_id: str) -> bool:
    """Report whether a connector package exists for an id, without importing it.

    Ask this instead of catching :func:`connector_dir`'s ``LookupError``: a miss is an ordinary
    answer here, not an exception.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        Whether a connector package exists for the id.
    """
    try:
        connector_dir(dataset_id)
    except LookupError:
        return False
    return True


def requirements_for(dataset_id: str) -> Path | None:
    """Return the connector's ``requirements.txt``, or ``None`` if it declares none.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The path to the connector's ``requirements.txt``, or ``None``.

    Raises:
        LookupError: If no module exists for the id.
    """  # noqa: DOC502 (raised by connector_dir, not directly here)
    path = connector_dir(dataset_id) / "requirements.txt"
    return path if path.is_file() else None


def _known_ids() -> list[str]:
    """List the dataset ids declared by the connector cards on disk, importing nothing.

    :func:`available` answers the same question by importing every connector. The unknown-id error
    path cannot afford that: those dependencies live in each connector's own environment, not this
    one.

    Returns:
        The declared dataset ids, sorted.
    """
    ids: set[str] = set()
    for root in _datasets.__path__:
        # A flat id sits at ``datasets/<name>/``, an ``org/name`` id one level deeper.
        for card in (*Path(root).glob("*/dataset.yaml"), *Path(root).glob("*/*/dataset.yaml")):
            declared = (yaml.safe_load(card.read_text(encoding="utf-8")) or {}).get("dataset_id")
            if declared:
                ids.add(declared)
    return sorted(ids)


def _connector_spec(dataset_id: str) -> importlib.machinery.ModuleSpec | None:
    """Find a connector module's spec without importing the module itself.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The module spec, or ``None`` when the id names no known module.

    Raises:
        ModuleNotFoundError: If a connector module for the id exists but fails to import a
            dependency of its own (surfaced instead of being masked as an unknown id).
    """
    module_name = _module_name(dataset_id)
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        # find_spec imports the parent packages. An unknown id fails to find one of *our* modules. A
        # broken import inside an existing connector fails to find something else. Surface that real
        # error instead of masking it as "unknown connector".
        if exc.name and (module_name + ".").startswith(exc.name + "."):
            return None  # a missing parent package (for example, an unknown org)
        raise


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
    """  # noqa: DOC502 (raised by _connector_spec, not directly here)
    spec = _connector_spec(dataset_id)
    if spec is None:
        raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(available()) or '(none)'}")
    module_name = _module_name(dataset_id)
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
