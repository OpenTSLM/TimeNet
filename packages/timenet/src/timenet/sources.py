"""Find dataset releases across registries and installed connectors."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeAlias, Union

from timenet import builders
from timenet._warnings import warn_at_caller
from timenet.errors import ConnectorCardWarning, TimeNetAccessError, TimeNetInvalidCardError
from timenet.types import DatasetMetadata
from timenet.types.access import Access, AccessRequirement
from timenet.types.references import LATEST_VERSION, DatasetRef
from timenet.types.version import Version


if TYPE_CHECKING:
    from timenet.manifest import Manifest
    from timenet.registry.base import BaseRegistry


# The registry module imports this module, so the alias names the registry as a forward reference.
ReleaseSource: TypeAlias = Union["BaseRegistry", "InstalledConnectors"]
"""A source that can serve or build a dataset release: a registry or the installed connectors."""


@dataclass(frozen=True)
class FoundInRegistry:
    """A result for a release that a registry serves.

    Attributes:
        version: The exact version.
        registry: The registry that serves the release.
        manifest: The release manifest that the registry read, or ``None``.
    """

    version: Version
    registry: "BaseRegistry"
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class FoundInConnector:
    """A result for a release that an installed connector builds.

    Attributes:
        builder: The backend and card that the lookup read.
    """

    builder: builders.InstalledBuilder

    @property
    def version(self) -> Version:
        """The version that the card declares."""
        return self.builder.card.dataset_version


Found: TypeAlias = FoundInRegistry | FoundInConnector
"""A result for a release that a source serves or builds."""


@dataclass(frozen=True)
class NotFound:
    """A result for a release that the source cannot find."""


@dataclass(frozen=True)
class NotServed:
    """A result for a listed release whose files the source cannot serve.

    Attributes:
        version: The exact version that the source lists.
        access: The strictest access level of the release's card and the cards in its lineage.
        requirements: Each card that is not open access, with its access level and access page.
        manifest: The release manifest, or ``None``.
    """

    version: Version
    access: Access
    requirements: tuple[AccessRequirement, ...]
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)

    def access_error(self, dataset_id: str) -> TimeNetAccessError:
        """Create an error with the release's access requirements.

        Args:
            dataset_id: The dataset ID.

        Returns:
            An error with the release, its access level, and its access pages.
        """
        release = DatasetRef(dataset_id=dataset_id, version=self.version)
        return TimeNetAccessError(release, self.access, self.requirements)


ReleaseLookup: TypeAlias = Found | NotFound | NotServed
"""The result of looking up a release in a source."""


@dataclass(frozen=True)
class InstalledConnectors:
    """A release source backed by installed dataset connectors.

    Each connector provides only the version in its card.
    All instances compare equal.
    """

    has_catalog: ClassVar[bool] = True
    """Whether :meth:`list_datasets` works. It always does for installed connectors."""

    def __str__(self) -> str:
        if not tuple(builders.entry_points(group=builders.ENTRY_POINT_GROUP)):
            return "installed connectors (no connector package is installed)"
        return "installed connectors"

    def resolve(self, dataset_id: str, version: Version | None) -> ReleaseLookup:  # ruff: ignore[no-self-use] (every source has it)
        """Check whether an installed connector can build a release.

        The lookup uses the first entry point that claims the dataset.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None`` for the latest version.

        Returns:
            ``FoundInConnector`` if a connector builds the requested version, or ``NotFound``.

        Raises:
            TimeNetInvalidCardError: If the connector's card cannot be read.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
        builder = builders.installed_builder(dataset_id)
        if builder is None or (version is not None and version != builder.card.dataset_version):
            return NotFound()
        return FoundInConnector(builder)

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the card of each dataset that an installed connector builds, without a build.

        If two packages claim a dataset, the first entry point wins, as for a build. The call leaves out
        a dataset whose card cannot be read, with a :class:`~timenet.errors.ConnectorCardWarning`.

        Returns:
            The cards, sorted by dataset ID.

        Raises:
            TimeNetInvalidCardError: If a backend cannot list its dataset IDs.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
        return self._cards()

    def _cards(self) -> list[DatasetMetadata]:  # ruff: ignore[no-self-use] (the listing of this source)
        """Return the readable cards of installed connectors, sorted by dataset ID."""
        cards: list[DatasetMetadata] = []
        for claim in builders.claims():
            try:
                card = claim.backend.card(claim.dataset_id)
            except TimeNetInvalidCardError as error:
                warn_at_caller(
                    ConnectorCardWarning(
                        f"{claim.dataset_id}: the card of its installed connector cannot be read, "
                        f"so the listing leaves it out: {error}"
                    )
                )
                continue
            if card is not None:
                cards.append(card)
        return sorted(cards, key=lambda card: card.dataset_id)


Ask: TypeAlias = Callable[[ReleaseSource, str, Version | None], ReleaseLookup]
"""A function that asks one source for a release: ``ask(source, dataset_id, version)``."""


def resolve_release(
    sources: Sequence[ReleaseSource], dataset_id: str, version: str | None, *, ask: Ask | None = None
) -> ReleaseLookup:
    """Find a source that provides the requested dataset release.

    An exact version comes from the first source that serves it, and no version means the newest version of the first
    source that serves the dataset. ``"latest"`` is the highest version of any source, source order breaks ties, and
    source errors stop the search.

    Args:
        sources: The sources, in order.
        dataset_id: The dataset ID.
        version: An exact version, ``"latest"``, or ``None`` or ``""`` for no version.
        ask: The function that asks one source, or ``None`` to call the source's ``resolve()``.

    Returns:
        A ``Found`` result if available, otherwise the first ``NotServed`` result or ``NotFound``.

    Raises:
        TimeFValidationError: If ``version`` is neither ``"latest"`` nor a ``major.minor.patch`` version.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by Version.parse
    ask = ask or _ask
    if version == LATEST_VERSION:
        return _resolve_latest(sources, dataset_id, ask)
    return _resolve_first(sources, dataset_id, Version.parse(version) if version else None, ask)


def _ask(source: ReleaseSource, dataset_id: str, version: Version | None) -> ReleaseLookup:
    """Return the answer of a source for a release."""
    return source.resolve(dataset_id, version)


def _resolve_first(
    sources: Sequence[ReleaseSource], dataset_id: str, version: Version | None, ask: Ask
) -> ReleaseLookup:
    """Return the answer of the first source that serves ``version``, or its newest for ``None``, else the first listing."""
    listed: NotServed | None = None
    for source in sources:
        answer = ask(source, dataset_id, version)
        if isinstance(answer, Found):
            return answer
        if isinstance(answer, NotServed) and listed is None:
            listed = answer
    return NotFound() if listed is None else listed


def _resolve_latest(sources: Sequence[ReleaseSource], dataset_id: str, ask: Ask) -> ReleaseLookup:
    """Return the highest version that any source provides, from the first source that provides it."""
    best: Found | None = None
    listed: list[tuple[ReleaseSource, NotServed]] = []
    before_best = 0
    for source in sources:
        answer = ask(source, dataset_id, None)
        if isinstance(answer, NotServed):
            listed.append((source, answer))
        elif isinstance(answer, Found) and (best is None or answer.version > best.version):
            best, before_best = answer, len(listed)
    if best is None:
        return listed[0][1] if listed else NotFound()
    return _ask_again(listed[:before_best], dataset_id, best.version, ask) or best


def _ask_again(
    listed: Sequence[tuple[ReleaseSource, NotServed]], dataset_id: str, version: Version, ask: Ask
) -> Found | None:
    """Ask each source that lists a higher version without serving it whether it serves ``version``.

    Returns:
        The answer of the first such source that serves ``version``, or ``None``.
    """
    for source, answer in listed:
        if answer.version > version:
            again = ask(source, dataset_id, version)
            if isinstance(again, Found):
                return again
    return None
