"""Discover connectors lazily by convention.

Each connector lives in its own folder at ``datasets/<org>/<name>/``, or ``datasets/<name>/`` for a
flat id. The package's ``__init__.py`` exposes a ``CONNECTOR`` class. A ``dataset.yaml`` card sits
beside it. Discovery reads every ``dataset.yaml`` under the built-in root and under the roots
registered in the ``timenet_connectors.datasets`` entry-point group, and imports only the requested
connector.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import functools
import importlib
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
import warnings

import yaml

from timenet.connectors import BaseConnector
from timenet.errors import ConnectorCardWarning
import timenet_connectors.datasets as _datasets


@dataclass(frozen=True)
class _ConnectorCard:
    """One connector card in a datasets root."""

    directory: Path
    """The connector directory, which holds the ``dataset.yaml`` card."""
    module_name: str
    """The module path of the connector package."""
    package_name: str | None
    """The distribution that registers the datasets root, or ``None`` for the built-in root."""


@functools.cache
def _get_connector_cards_by_id() -> Mapping[str, _ConnectorCard]:
    """Find the datasets roots once per process and read every connector card in them.

    A card that cannot be read gives one :class:`~timenet.errors.ConnectorCardWarning`. When more
    than one root declares an id, the first root wins, so the built-in root wins over the others.

    Returns:
        The connector card for each declared dataset id.
    """
    cards: dict[str, _ConnectorCard] = {}
    for datasets_module, package_name in _iter_datasets_roots():
        for path in datasets_module.__path__:
            # A flat id sits at ``datasets/<name>/``, an ``org/name`` id one level deeper.
            for card_path in (*Path(path).glob("*/dataset.yaml"), *Path(path).glob("*/*/dataset.yaml")):
                declared = _declared_id(card_path)
                if declared is None or declared in cards:
                    continue
                module_name = ".".join((datasets_module.__name__, *card_path.parent.relative_to(path).parts))
                cards[declared] = _ConnectorCard(card_path.parent, module_name, package_name)
    return cards


def _iter_datasets_roots() -> Iterator[tuple[ModuleType, str | None]]:
    """Yield the built-in datasets package, then each package registered by an entry point.

    Yields:
        Each datasets package with the distribution that registers it, or ``None`` for the built-in one.
    """
    yield _datasets, None
    for entry in entry_points(group="timenet_connectors.datasets"):
        yield entry.load(), entry.dist.name  # ty: ignore[unresolved-attribute]


def _get_connector_card(dataset_id: str) -> _ConnectorCard:
    """Find the connector card for a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id, as its card declares it.

    Returns:
        The connector card.

    Raises:
        LookupError: If no root holds a card for the id.
    """
    card = _get_connector_cards_by_id().get(dataset_id)
    if card is None:
        raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(known_ids()) or '(none)'}")
    return card


def get_connector_package_name(dataset_id: str) -> str | None:
    """Return the installed package that holds the connector of a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The distribution name of the package, or ``None`` when the connector is built in.

    Raises:
        LookupError: If no root holds a card for the id.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _get_connector_card, not directly here)
    return _get_connector_card(dataset_id).package_name


def connector_dir(dataset_id: str) -> Path:
    """Return the directory of a dataset id's connector package.

    Reads only the cards and never imports the connector, because this runs before the connector's
    environment exists.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The connector package's directory.

    Raises:
        LookupError: If no connector with a ``dataset.yaml`` card exists for the id.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _get_connector_card, not directly here)
    return _get_connector_card(dataset_id).directory


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
    :class:`~timenet.errors.ConnectorCardWarning`, so it fails only its own dataset.

    Returns:
        The sorted dataset IDs.
    """
    return sorted(_get_connector_cards_by_id())


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
        LookupError: If no connector card exists for the id, or the module exposes no ``CONNECTOR``.
        ModuleNotFoundError: If the connector module fails to import one of its own dependencies.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _get_connector_card and the import, not directly here)
    module_name = _get_connector_card(dataset_id).module_name
    connector = getattr(importlib.import_module(module_name), "CONNECTOR", None)
    if connector is None:
        raise LookupError(f"module {module_name!r} defines no CONNECTOR")
    return connector
