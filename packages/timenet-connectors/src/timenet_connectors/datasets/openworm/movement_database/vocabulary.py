"""The closed sets of the movement database that the connector reads: a worm's sex and its ventral side."""

from enum import StrEnum, unique


@unique
class Sex(StrEnum):
    """A worm's sex, as a video's ``experiment_info`` writes it."""

    HERMAPHRODITE = "hermaphrodite"
    MALE = "male"

    @property
    def karyotype(self) -> str:
        """Return the sex chromosomes the authors' class names use: ``XX`` or ``XO``."""
        return _KARYOTYPES[self]


_KARYOTYPES = {Sex.HERMAPHRODITE: "XX", Sex.MALE: "XO"}


@unique
class VentralSide(StrEnum):
    """Which way the worm's ventral side turns, seen in the video, from head to tail.

    The skeleton's points run from head to tail, so this side tells the ventral from the dorsal bends.
    """

    CLOCKWISE = "clockwise"
    ANTICLOCKWISE = "anticlockwise"
    UNKNOWN = "unknown"
    """The source doesn't state the side; Zenodo lists 7 of the benchmark's videos so."""
