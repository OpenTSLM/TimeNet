"""Find dataset releases across registries and installed connectors."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable
import warnings

from timenet import builders
from timenet.errors import ConnectorCardWarning, TimeFValidationError, TimeNetAccessError, TimeNetInvalidCardError
from timenet.types import DatasetMetadata, Domain, License, Task
from timenet.types.access import Access, AccessRequirement
from timenet.types.metadata import matches_metadata
from timenet.types.references import DatasetRef, is_latest
from timenet.types.version import Version


if TYPE_CHECKING:
    from timenet.manifest import Manifest


Answer: TypeAlias = "Found | NotFound | NotServed"
"""The result of looking up a release in a source."""


@runtime_checkable
class ReleaseSource(Protocol):
    """A source that can serve or build a dataset release.

    A :class:`~pathlib.Path` also passes runtime ``isinstance`` checks because it has a ``resolve`` method.
    Check for ``str`` and ``os.PathLike`` first.
    """

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Check whether this source can provide a dataset release.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None``, ``""``, or ``"latest"`` for the latest version.

        Returns:
            A :class:`Found`, :class:`NotFound`, or :class:`NotServed` result.
        """
        ...


@dataclass(frozen=True)
class Found:
    """A result that identifies an available release and its source.

    Attributes:
        version: The exact version.
        entry: The source that answered for the release.
        manifest: The release manifest, or ``None``.
    """

    version: Version
    entry: ReleaseSource
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class NotFound:
    """A result for a release that the source cannot find."""


@dataclass(frozen=True)
class NotServed:
    """A result for a listed release whose files the source cannot serve.

    Attributes:
        version: The exact version that the source lists.
        access: The strictest access level of the release's card and the cards in its lineage.
        access_url: The access page of the first card that is not open access, or ``None``.
        requirements: Each card that is not open access, with its access level and access page.
        manifest: The release manifest, or ``None``.
    """

    version: Version
    access: Access
    access_url: str | None
    requirements: tuple[AccessRequirement, ...] = ()
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)

    def access_error(self, dataset_id: str) -> TimeNetAccessError:
        """Create an error with the release's access requirements.

        Args:
            dataset_id: The dataset ID.

        Returns:
            An error with the release, its access level, and its access pages.
        """
        release = DatasetRef(dataset_id=dataset_id, version=self.version)
        requirements = self.requirements
        if not requirements and self.access_url is not None:
            requirements = (AccessRequirement(release, self.access, self.access_url),)
        return TimeNetAccessError(release, self.access, requirements)


@dataclass(frozen=True)
class InstalledConnectors:
    """A release source backed by installed dataset connectors.

    Each connector provides only the version in its card.
    All instances compare equal.
    """

    def __str__(self) -> str:
        if not tuple(builders.entry_points(group=builders.ENTRY_POINT_GROUP)):
            return "installed connectors (no connector package is installed)"
        return "installed connectors"

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Check whether an installed connector can build a release.

        The lookup uses the first entry point that claims the dataset.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None``, ``""``, or ``"latest"`` for the latest version.

        Returns:
            ``Found`` if a connector provides the requested version, or ``NotFound``.

        Raises:
            TimeNetInvalidCardError: If the connector's card cannot be read.
            TimeFValidationError: If a connector claims the dataset and ``version`` is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend and by Version.parse
        card = builders.find_card(dataset_id)
        wanted = None if is_latest(version) else version
        if card is None or (wanted is not None and Version.parse(wanted) != card.dataset_version):
            return NotFound()
        return Found(card.dataset_version, self)

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

    def search(  # ruff: ignore[too-many-arguments]
        self,
        *,
        query: str | list[str] | None = None,
        domain: Domain | list[Domain] | None = None,
        task: type[Task] | list[type[Task]] | None = None,
        license: License | list[License] | None = None,
        time_series_spec: str | list[str] | None = None,
        dataset_id: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        limit: int = 100,
    ) -> list[DatasetMetadata]:
        """Return the cards of :meth:`list_datasets` that match every filter.

        A card has no schema, so a ``task`` or ``time_series_spec`` filter matches no card.

        Args:
            query: Match any term in the name, description, or tags, ignoring case.
            domain: Match any of these domains.
            task: Match any of these task classes in the schema.
            license: Match any of these licenses.
            time_series_spec: Match all of these ``spec_type`` values.
            dataset_id: Match any of these IDs.
            tag: Match all of these tags.
            limit: The maximum number of results, defaulting to 100.

        Returns:
            The matching cards, sorted by dataset ID, with at most ``limit`` entries.

        Raises:
            TimeFValidationError: If ``limit`` is negative.
            TimeNetInvalidCardError: If a backend cannot list its dataset IDs.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend
        if limit < 0:
            raise TimeFValidationError(f"limit must be 0 or more, got {limit}")
        # None or an empty list skips a filter. A type filter reads the schema, which a card does not have.
        if task not in (None, []) or time_series_spec not in (None, []):
            return []
        cards = self._cards()
        return [
            card
            for card in cards
            if matches_metadata(card, query=query, domain=domain, license=license, dataset_id=dataset_id, tag=tag)
        ][:limit]

    def _cards(self) -> list[DatasetMetadata]:  # ruff: ignore[no-self-use] (the listing of this source)
        """Return the readable cards of installed connectors, sorted by dataset ID."""
        claimed: dict[str, DatasetMetadata | None] = {}
        for entry in builders.entry_points(group=builders.ENTRY_POINT_GROUP):
            backend = entry.load()()
            for dataset_id in backend.dataset_ids():
                if dataset_id in claimed:
                    continue
                try:
                    claimed[dataset_id] = backend.card(dataset_id)
                except TimeNetInvalidCardError as error:
                    claimed[dataset_id] = None
                    warnings.warn(
                        ConnectorCardWarning(
                            f"{dataset_id}: the card of its installed connector cannot be read, "
                            f"so the listing leaves it out: {error}"
                        ),
                        stacklevel=5,
                    )
        cards = {dataset_id: card for dataset_id, card in claimed.items() if card is not None}
        return [cards[dataset_id] for dataset_id in sorted(cards)]


def resolve_release(sources: Sequence[ReleaseSource], dataset_id: str, version: str | None) -> Answer:
    """Find a source that provides the requested dataset release.

    Exact requests use the first matching source, and latest requests use the highest available version.
    Source order breaks ties, and source errors stop the search.

    Args:
        sources: The sources, in order.
        dataset_id: The dataset ID.
        version: An exact version, or ``None``, ``""``, or ``"latest"`` for the latest version.

    Returns:
        ``Found`` if available, otherwise the first ``NotServed`` result or ``NotFound``.
    """
    version = None if is_latest(version) else version
    if version is not None:  # rule 1
        listed: NotServed | None = None
        for source in sources:
            answer = source.resolve(dataset_id, version)
            if isinstance(answer, Found):
                return answer
            if isinstance(answer, NotServed) and listed is None:
                listed = answer
        return NotFound() if listed is None else listed
    answers = [(source, source.resolve(dataset_id, None)) for source in sources]  # rule 2
    found = [answer.version for _, answer in answers if isinstance(answer, Found)]
    if not found:
        return next((answer for _, answer in answers if isinstance(answer, NotServed)), NotFound())
    best = max(found)
    first = next(
        index for index, (_, answer) in enumerate(answers) if isinstance(answer, Found) and answer.version == best
    )
    # A source before the first one that serves ``best`` can still serve it, if it listed a higher version.
    for source, answer in answers[:first]:
        if isinstance(answer, NotServed) and answer.version > best:
            again = source.resolve(dataset_id, str(best))
            if isinstance(again, Found):
                return again
    return answers[first][1]
