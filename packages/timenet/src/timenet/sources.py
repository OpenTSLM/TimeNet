"""Find dataset releases across registries and installed connectors."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from timenet import builders
from timenet.errors import TimeNetAccessError
from timenet.types.access import Access, AccessRequirement
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
