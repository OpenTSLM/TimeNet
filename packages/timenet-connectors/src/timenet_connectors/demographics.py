"""Demographic values that connectors share, so every dataset spells them the same way.

A source codes a person's sex its own way, such as ``is_male``, ``F=1`` or ``0`` for male. A connector
decodes that code into :class:`Sex` and stores the enum's value, so a ``sex`` annotation or target reads the
same in every dataset.
"""

from enum import StrEnum, unique


@unique
class Sex(StrEnum):
    """A person's sex, as a dataset stores it once its source's code is decoded."""

    MALE = "male"
    FEMALE = "female"
