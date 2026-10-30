"""The :class:`Access` enum: how a user obtains a dataset's data.

Access (what you must do to get the bytes) is a separate axis from the license (how you may
redistribute them). A dataset can be openly licensed yet credentialed, or the reverse.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.types.metadata import DatasetMetadata
    from timenet.types.parent_dataset_card import ParentDatasetCard


@unique
class Access(StrEnum):
    """How a user obtains the source data."""

    OPEN = "open"
    """Anyone can fetch the data; no account or agreement is needed."""
    CREDENTIALED = "credentialed"
    """A credentialed account and a signed agreement are needed, for example a PhysioNet DUA."""
    RESTRICTED = "restricted"
    """Access is granted case by case, for example an AWS-IAM grant or a direct request."""


_STRICTNESS = {Access.OPEN: 0, Access.CREDENTIALED: 1, Access.RESTRICTED: 2}


@dataclass(frozen=True)
class AccessRequirement:
    """A release whose card declares an access that is not open, and where to obtain access to it."""

    release: DatasetRef
    """The release whose card declares the access."""
    access: Access
    """The access level that the card declares. It is never ``OPEN``."""
    access_url: str
    """Where to obtain access, from the card of the release."""


def access_requirements(
    metadata: "DatasetMetadata", lineage: "Iterable[ParentDatasetCard]" = ()
) -> tuple[AccessRequirement, ...]:
    """Return each card whose declared access is not open: the card of the release, then its lineage.

    Each card keeps the access level and the ``access_url`` that its author declared. An import
    does not change them. A card whose access is not open must set an ``access_url``.

    Args:
        metadata: The card or the manifest metadata of the release.
        lineage: The card of each ancestor release, as a manifest or a dataset holds them. A card
            that is not built yet has no lineage.

    Returns:
        One requirement per card, in the order of ``metadata`` and then ``lineage``.
    """
    requirements = []
    for card in (metadata, *(ancestor.metadata for ancestor in lineage)):
        if card.access is not Access.OPEN and card.access_url:
            release = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
            requirements.append(AccessRequirement(release, card.access, card.access_url))
    return tuple(requirements)


def effective_access(metadata: "DatasetMetadata", lineage: "Iterable[ParentDatasetCard]" = ()) -> Access:
    """Return the strictest access level of a release card and of the cards in its lineage.

    A child that reads the data of a credentialed parent needs credentialed access too, also when
    its own card says open. So this level, not the level of the card alone, applies to the data of
    the release.

    Args:
        metadata: The card or the manifest metadata of the release.
        lineage: The card of each ancestor release, as a manifest or a dataset holds them. A card
            that is not built yet has no lineage.

    Returns:
        The strictest level.
    """
    cards = (metadata, *(ancestor.metadata for ancestor in lineage))
    return max((card.access for card in cards), key=_STRICTNESS.__getitem__)
