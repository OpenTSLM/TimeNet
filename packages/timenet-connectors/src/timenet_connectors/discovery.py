"""Discover connectors lazily by convention.

Each connector lives in its own folder at ``datasets/<org>/<name>/``, or ``datasets/<name>/`` for a
flat id. The package's ``__init__.py`` exposes a ``CONNECTOR`` class. A ``dataset.yaml`` card sits
beside it. A dataset id maps to that package by convention, so there is no central registry. Discovery
imports only the requested connector. Each connector declares its own id through ``metadata()``.

The built-in ``timenet_connectors.datasets`` package is the first datasets root. Other installed
packages add roots under the ``timenet_connectors.datasets`` entry-point group. The value of each
entry point is the module path of a datasets package with the same layout.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from importlib.metadata import entry_points
import importlib.util
from pathlib import Path
import pkgutil
import warnings

import yaml

from timenet.connectors import BaseConnector
from timenet.errors import ConnectorCardWarning
import timenet_connectors.datasets as _datasets


DATASETS_ENTRY_POINT_GROUP = "timenet_connectors.datasets"


@dataclass(frozen=True)
class _Root:
    """One datasets package that holds connectors."""

    prefix: str
    """The module path of the package, for example ``timenet_connectors.datasets``."""
    paths: Sequence[str]
    """The directories of the package."""
    distribution: str | None
    """The distribution that registers the package, or ``None`` for the built-in root."""


def _roots() -> list[_Root]:
    """Return the built-in datasets root, then each root registered by an entry point.

    Returns:
        The datasets roots.
    """
    roots = [_Root(_datasets.__name__, list(_datasets.__path__), None)]
    for entry in entry_points(group=DATASETS_ENTRY_POINT_GROUP):
        if any(root.prefix == entry.value for root in roots):
            continue
        # find_spec imports the parent packages only, so no connector is imported here.
        try:
            spec = importlib.util.find_spec(entry.value)
        except ImportError:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            warnings.warn(
                f"datasets entry point {entry.name!r} names no package: {entry.value!r}",
                ConnectorCardWarning,
                stacklevel=2,
            )
            continue
        owner = None if entry.dist is None else entry.dist.name
        roots.append(_Root(entry.value, list(spec.submodule_search_locations), owner))
    return roots


def _relative_parts(dataset_id: str) -> list[str]:
    """Return the package path segments of a dataset id below a datasets root."""
    return [_segment(part) for part in dataset_id.split("/", 1)]


def _pick_root(dataset_id: str) -> tuple[_Root, Path]:
    """Find the one root that holds a connector card for a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The root and the connector directory in it.

    Raises:
        LookupError: If no root, or more than one root, holds a card for the id.
    """
    parts = _relative_parts(dataset_id)
    found: list[tuple[_Root, Path]] = []
    for root in _roots():
        for path in root.paths:
            directory = Path(path).joinpath(*parts)
            if (directory / "dataset.yaml").is_file():
                found.append((root, directory))
                break
    if not found:
        raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(known_ids()) or '(none)'}")
    if len(found) > 1:
        owners = ", ".join(root.prefix for root, _ in found)
        raise LookupError(f"more than one datasets package has a connector for {dataset_id!r}: {owners}")
    return found[0]


def owning_distribution(dataset_id: str) -> str | None:
    """Return the distribution that registers the datasets root of a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The distribution name, or ``None`` when the connector is in the built-in root.

    Raises:
        LookupError: If no root, or more than one root, holds a card for the id.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _pick_root, not directly here)
    return _pick_root(dataset_id)[0].distribution


def connector_dir(dataset_id: str) -> Path:
    """Return the directory of a dataset id's connector package.

    Deliberately import-free. This runs before the connector's environment
    exists, so importing the connector module (which is what :func:`resolve`
    does) would fail on the very dependencies its ``requirements.txt``
    declares.

    A connector directory holds a ``dataset.yaml`` card. An org folder
    (``physionet``) has no card, so it is not a connector and is rejected like any unknown id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The connector package's directory.

    Raises:
        LookupError: If no connector with a ``dataset.yaml`` card exists for the id, or more than
            one datasets package has one.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _pick_root, not directly here)
    return _pick_root(dataset_id)[1]


def requirements_for(dataset_id: str) -> Path | None:
    """Find a connector's requirements file.

    Args:
        dataset_id: The dataset ID, as ``org/name``.

    Returns:
        The path to ``requirements.txt``, or ``None`` if the file is absent.

    Raises:
        LookupError: If no connector with a ``dataset.yaml`` card exists for the ID.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by connector_dir, not directly here)
    path = connector_dir(dataset_id) / "requirements.txt"
    return path if path.is_file() else None


def known_ids() -> list[str]:
    """List dataset IDs from connector cards without importing connectors.

    A card that cannot be read, is not a YAML mapping, or lacks ``dataset_id`` is skipped with a
    :class:`~timenet.errors.ConnectorCardWarning`, so it fails only its own dataset. An id that
    more than one datasets package declares is skipped with the same warning.

    Returns:
        The sorted dataset IDs, with duplicates removed.
    """
    owners: dict[str, set[str]] = {}
    for root in _roots():
        for path in root.paths:
            # A flat id sits at ``datasets/<name>/``, an ``org/name`` id one level deeper.
            for card in (*Path(path).glob("*/dataset.yaml"), *Path(path).glob("*/*/dataset.yaml")):
                declared = _declared_id(card)
                if declared is not None:
                    owners.setdefault(declared, set()).add(root.prefix)
    for dataset_id, prefixes in owners.items():
        if len(prefixes) > 1:
            warnings.warn(
                f"more than one datasets package declares {dataset_id!r} ({', '.join(sorted(prefixes))}); "
                "its dataset is skipped",
                ConnectorCardWarning,
                stacklevel=2,
            )
    return sorted(dataset_id for dataset_id, prefixes in owners.items() if len(prefixes) == 1)


def _declared_id(card: Path) -> str | None:
    """Read the dataset ID of a card, or warn and return ``None`` for a card that cannot be read.

    Args:
        card: The ``dataset.yaml`` file.

    Returns:
        The declared dataset ID, or ``None``.
    """
    try:
        raw = yaml.safe_load(card.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        problem = f"could not read dataset card {card}: {error}"
    else:
        if not isinstance(raw, dict):
            problem = f"dataset card {card} must be a YAML mapping, got {type(raw).__name__}"
        elif not raw.get("dataset_id"):
            problem = f"dataset card {card} must declare a dataset_id"
        else:
            return raw["dataset_id"]
    warnings.warn(f"{problem}; its dataset is skipped", ConnectorCardWarning, stacklevel=3)
    return None


def resolve(dataset_id: str) -> type[BaseConnector]:
    """Return the connector class for a dataset id and import only its module.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` (``chengsenwang/tsqa``) id.

    Returns:
        The connector class.

    Raises:
        LookupError: If no connector card exists for the id, more than one datasets package has one,
            or the module exposes no ``CONNECTOR``.
        ModuleNotFoundError: If the connector module fails to import one of its own dependencies.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _module_name and the import, not directly here)
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
    infos = (info for root in _roots() for info in pkgutil.walk_packages(root.paths, root.prefix + "."))
    for info in infos:
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
    underscores. So ``chengsenwang/tsqa`` maps to ``...datasets.chengsenwang.tsqa`` in the root
    that holds its card.

    Returns:
        The dotted module path of the connector.

    Raises:
        LookupError: If no root, or more than one root, holds a card for the id.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _pick_root, not directly here)
    root, _ = _pick_root(dataset_id)
    return ".".join((root.prefix, *_relative_parts(dataset_id)))


def _segment(part: str) -> str:
    """Normalize one id segment to its Python package name: lowercase, hyphens to underscores.

    Args:
        part: An org or name segment of a dataset id.

    Returns:
        The importable package/module name for that segment.
    """
    return part.replace("-", "_").lower()
