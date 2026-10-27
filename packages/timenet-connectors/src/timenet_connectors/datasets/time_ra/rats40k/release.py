"""The RATS40K release as published: its Hub pin, its two files, their splits, and their pinned inventory."""

from collections.abc import Mapping
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import Split


HF_REPO = "Time-RA/RATs40K"
SOURCE_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"


@unique
class Variant(StrEnum):
    """The release's two files: univariate and multivariate series, each with its own prompt."""

    UNI = "uni"
    MULTI = "multi"

    @property
    def source_file(self) -> str:
        """Return the file's name in the Hub repository, such as ``RATs-Uni-TSImage_Reason.json``."""
        return f"RATs-{self.value.capitalize()}-TSImage_Reason.json"


@unique
class SourceSplit(StrEnum):
    """A top-level key of a file: the authors' split, with the TimeNet split it maps to."""

    TRAIN = "TSAD_train"
    TEST = "TSAD_test"

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the key's rows: ``TSAD_train`` is ``train``."""
        return Split(self.value.removeprefix("TSAD_"))


class PinnedInventory(NamedTuple):
    """One file's pinned inventory at :data:`SOURCE_REVISION`."""

    rows: Mapping[Split, int]
    """Usable rows per split, without the null entries."""
    null_keys: Mapping[Split, frozenset[str]]
    """The keys of the published null entries per split, which the connector skips."""

    def check(self, path: Path, rows: Mapping[Split, int], null_keys: Mapping[Split, frozenset[str]]) -> None:
        """Compare a file's counted rows and null keys with this inventory.

        Raises:
            TimeFValidationError: If the file isn't the pinned revision's.
        """
        if dict(rows) != dict(self.rows) or dict(null_keys) != dict(self.null_keys):
            raise TimeFValidationError(f"{path}: pinned row counts or null entry keys differ")


# The eleven null entries are univariate test keys that the authors' labeling/rank_models.py starts as
# None and never fills.
PINNED_INVENTORY = {
    Variant.UNI: PinnedInventory(
        rows={Split.TRAIN: 30266, Split.TEST: 6034},
        null_keys={
            Split.TRAIN: frozenset(),
            Split.TEST: frozenset({"106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"}),
        },
    ),
    Variant.MULTI: PinnedInventory(
        rows={Split.TRAIN: 2728, Split.TEST: 546},
        null_keys={Split.TRAIN: frozenset(), Split.TEST: frozenset()},
    ),
}
