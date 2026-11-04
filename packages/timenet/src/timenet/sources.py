"""Sources of releases, and the rules that pick the source of a release.

A source says whether it can provide a release, and at which version. A registry serves the files of
the releases that it holds. :class:`InstalledConnectors` builds a release with the installed connector
that claims its dataset.

:func:`resolve_release` asks an ordered list of sources and follows three rules:

1. An exact version comes from the first source that provides it.
2. "Latest" is the highest version that any source provides. The order then decides only which
   source provides that version.
3. Only :class:`NotFound` and :class:`NotServed` fall through to the next source. An error of a
   source stops the resolution.

This module imports nothing from :mod:`timenet.registry` at run time, so that the registries can
import it.
"""

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
"""The answer of a source for one release."""


@runtime_checkable
class ReleaseSource(Protocol):
    """Something that can provide releases: a registry or :class:`InstalledConnectors`.

    A :class:`~pathlib.Path` also passes an ``isinstance`` check against this protocol, because it has
    a ``resolve`` method. So a caller that takes paths and sources checks for ``str`` and
    ``os.PathLike`` first.
    """

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Say whether this source can provide a release.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, such as ``"1.2.0"``, or ``None``, ``""`` or ``"latest"`` for the
                latest version.

        Returns:
            :class:`Found`, :class:`NotFound` or :class:`NotServed`.
        """
        ...


@dataclass(frozen=True)
class Found:
    """The source can provide the release.

    Attributes:
        version: The exact version.
        entry: The source that answered for the release.
        manifest: The manifest that the source read to give this answer, so that a planner does not
            fetch it again; ``None`` if the source read none.
    """

    version: Version
    entry: ReleaseSource
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class NotFound:
    """The source does not know the release."""


@dataclass(frozen=True)
class NotServed:
    """The source lists the release, but it does not serve its files, for example for a credentialed release.

    Attributes:
        version: The exact version that the source lists.
        access: The effective access of the release: the strictest access of its card and of the
            cards in its lineage.
        access_url: The page of the first card that is not open, where users get access, or ``None``.
        requirements: Each card that is not open, with its access and its page, in the order of
            :func:`~timenet.types.access.access_requirements`.
        manifest: The manifest that the source read to give this answer, so that a planner does not
            fetch it again; ``None`` if the source read none.
    """

    version: Version
    access: Access
    access_url: str | None
    requirements: tuple[AccessRequirement, ...] = ()
    manifest: "Manifest | None" = field(default=None, compare=False, repr=False)

    def access_error(self, dataset_id: str) -> TimeNetAccessError:
        """Return the error that tells the caller where to get access to the release.

        Args:
            dataset_id: The dataset ID of the release.

        Returns:
            The error. It names the release, its effective access, and the page of each card that is
            not open. If ``requirements`` is empty and ``access_url`` is set, it names that page for
            the release.
        """
        release = DatasetRef(dataset_id=dataset_id, version=self.version)
        requirements = self.requirements
        if not requirements and self.access_url is not None:
            requirements = (AccessRequirement(release, self.access, self.access_url),)
        return TimeNetAccessError(release, self.access, requirements)


@dataclass(frozen=True)
class InstalledConnectors:
    """Every builder backend that an installed package registers under the ``timenet.builders`` group.

    A connector builds only the version that its card declares, so this source answers with that
    version. If no connector package is installed, it answers :class:`NotFound` to every request.
    All instances are equal.
    """

    def __str__(self) -> str:
        if not tuple(builders.entry_points(group=builders.ENTRY_POINT_GROUP)):
            return "installed connectors (no connector package is installed)"
        return "installed connectors"

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Say whether an installed connector builds the release.

        If two packages claim the dataset, the first entry point wins, as it does for a build.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, such as ``"1.2.0"``, or ``None``, ``""`` or ``"latest"`` for the
                latest version.

        Returns:
            ``Found`` with the version that the card declares, if ``version`` is the latest version or
            that version. Else ``NotFound()``. For a dataset that no connector claims, the answer is
            ``NotFound()``, whatever ``version`` is.

        Raises:
            TimeNetInvalidCardError: If the card of the connector cannot be read.
            TimeFValidationError: If ``version`` is not a valid version. This error is raised only when
                a connector claims the dataset.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the backend and by Version.parse
        card = builders.find_card(dataset_id)
        wanted = None if is_latest(version) else version
        if card is None or (wanted is not None and Version.parse(wanted) != card.dataset_version):
            return NotFound()
        return Found(card.dataset_version, self)


def resolve_release(sources: Sequence[ReleaseSource], dataset_id: str, version: str | None) -> Answer:
    """Find the source that provides a release, by the three rules of this module.

    For an exact version, the sources are asked in order until one of them answers :class:`Found`.
    For "latest", every source is asked. The highest version of the :class:`Found` answers wins. A
    source that answered :class:`NotServed` with a higher version is asked again for the winning
    version, because it can still serve that version. Then the first source in order that provides
    the winning version answers.

    A source that answers "latest" with :class:`NotServed` hides the older versions that it serves,
    because a registry cannot list the versions that it serves.

    Args:
        sources: The sources, in order.
        dataset_id: The dataset ID.
        version: An exact version, such as ``"1.2.0"``, or ``None``, ``""`` or ``"latest"`` for the
            latest version.

    Returns:
        The :class:`Found` answer of the source that provides the release. If no source provides it,
        the first :class:`NotServed` answer, so that the caller can name the access page. Else
        :class:`NotFound`.
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
