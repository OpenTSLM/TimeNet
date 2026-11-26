"""Discover connectors lazily by convention.

Each connector lives in its own folder at ``datasets/<org>/<name>/``, or ``datasets/<name>/`` for a
flat id. The package's ``__init__.py`` exposes a ``CONNECTOR`` class. A ``dataset.yaml`` card sits
beside it. A datasets root is a datasets package: the built-in one, or one that an installed
package registers in the ``timenet_connectors.datasets`` entry-point group. Discovery reads every
``dataset.yaml`` in each root. It imports only the requested connector.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import functools
import importlib
from importlib.metadata import entry_points
import os
from pathlib import Path
from types import ModuleType
import warnings

import yaml

from timenet.connectors import BaseConnector
from timenet.errors import ConnectorCardWarning, TimeNetBuildError, TimeNetDuplicateDatasetError
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
def _get_connector_cards_by_id() -> Mapping[str, list[_ConnectorCard]]:
    """Find the datasets roots once per process and read every connector card in them.

    A card that cannot be read gives one :class:`~timenet.errors.ConnectorCardWarning`.

    Returns:
        The connector cards for each declared dataset id. More than one card means a conflict.
    """
    cards: dict[str, list[_ConnectorCard]] = {}
    for datasets_module, package_name in _iter_datasets_roots():
        for path in dict.fromkeys(os.path.realpath(path) for path in datasets_module.__path__):
            for card_path in find_card_paths(Path(path)):
                declared = _declared_id(card_path)
                if declared is None:
                    continue
                module_name = ".".join((datasets_module.__name__, *card_path.parent.relative_to(path).parts))
                cards.setdefault(declared, []).append(_ConnectorCard(card_path.parent, module_name, package_name))
    return cards


def find_card_paths(root: Path) -> list[Path]:
    """Find the connector cards in a datasets directory.

    A flat id sits at ``<root>/<name>/``, an ``org/name`` id one level deeper.

    Args:
        root: The datasets directory.

    Returns:
        The ``dataset.yaml`` paths, sorted.
    """
    return sorted((*root.glob("*/dataset.yaml"), *root.glob("*/*/dataset.yaml")))


def _iter_datasets_roots() -> Iterator[tuple[ModuleType, str | None]]:
    """Yield the built-in datasets package, then each package registered by an entry point.

    Yields:
        Each datasets package with the distribution that registers it, or ``None`` for the built-in one.

    Raises:
        TimeNetBuildError: If an entry point names no installed package, or its module cannot be imported.
    """
    yield _datasets, None
    for entry in entry_points(group="timenet_connectors.datasets"):
        if entry.dist is None:
            raise TimeNetBuildError(f"the datasets entry point {entry.name} = {entry.value} names no installed package")
        try:
            package = entry.load()
        except ImportError as exc:
            raise TimeNetBuildError(
                f"cannot import the datasets entry point {entry.name} = {entry.value} of {entry.dist.name}: {exc}"
            ) from exc
        yield package, entry.dist.name


def _get_connector_card(dataset_id: str) -> _ConnectorCard:
    """Find the connector card for a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id, as its card declares it.

    Returns:
        The connector card.

    Raises:
        LookupError: If no root holds a card for the id.
        TimeNetDuplicateDatasetError: If more than one card declares the id.
    """
    cards = _get_connector_cards_by_id().get(dataset_id)
    if cards is None:
        raise LookupError(f"no connector for {dataset_id!r}; known: {', '.join(known_ids()) or '(none)'}")
    if len(cards) > 1:
        directories = ", ".join(str(card.directory) for card in cards)
        raise TimeNetDuplicateDatasetError(f"more than one connector declares {dataset_id!r}: {directories}")
    return cards[0]


def get_connector_package_name(dataset_id: str) -> str | None:
    """Return the installed package that holds the connector of a dataset id.

    Args:
        dataset_id: A flat (``hello_world``) or ``org/name`` id.

    Returns:
        The distribution name of the package, or ``None`` when the connector is built in.

    Raises:
        LookupError: If no root holds a card for the id.
        TimeNetDuplicateDatasetError: If more than one card declares the id.
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
        TimeNetDuplicateDatasetError: If more than one card declares the id.
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
        TimeNetDuplicateDatasetError: If more than one card declares the ID.
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
        TimeNetDuplicateDatasetError: If more than one card declares the id.
        ModuleNotFoundError: If the connector module fails to import one of its own dependencies.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _get_connector_card and the import, not directly here)
    module_name = _get_connector_card(dataset_id).module_name
    connector = getattr(importlib.import_module(module_name), "CONNECTOR", None)
    if connector is None:
        raise LookupError(f"module {module_name!r} defines no CONNECTOR")
    return connector
