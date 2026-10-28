"""The closed sets of CAPTURE-24: raw labels, label schemes, and the sexes and age bands of participants.

The 206 raw labels map to a class in each of the six schemes. ``metadata.csv`` writes the sexes and the age
bands.
"""

import csv
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints, ValidationError

from timenet.errors import TimeFFormatError


# A strict, non-blank string, as RATs40K and TRACE define it: a blank field must not pass as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]


@unique
class LabelScheme(StrEnum):
    """A column of the label dictionary: one published grouping of the raw labels into classes.

    The value is the column's name without its ``label:`` prefix, in the file's column order.
    """

    WILLETTS_SPECIFIC_2018 = "WillettsSpecific2018"
    WILLETTS_MET_2018 = "WillettsMET2018"
    DOHERTY_SPECIFIC_2018 = "DohertySpecific2018"
    WILLETTS_2018 = "Willetts2018"
    DOHERTY_2018 = "Doherty2018"
    WALMSLEY_2020 = "Walmsley2020"

    @property
    def column(self) -> str:
        """Return the scheme's column in the label dictionary, such as ``label:Walmsley2020``."""
        return f"label:{self.value}"

    @property
    def stem(self) -> str:
        """Return the scheme in the spelling of an annotation key, such as ``walmsley_2020``."""
        return _STEMS[self]


_STEMS = {
    LabelScheme.WILLETTS_SPECIFIC_2018: "willetts_specific_2018",
    LabelScheme.WILLETTS_MET_2018: "willetts_met_2018",
    LabelScheme.DOHERTY_SPECIFIC_2018: "doherty_specific_2018",
    LabelScheme.WILLETTS_2018: "willetts_2018",
    LabelScheme.DOHERTY_2018: "doherty_2018",
    LabelScheme.WALMSLEY_2020: "walmsley_2020",
}

NO_LABEL = -1
"""The code of an unlabelled sample, which the source writes as an empty field."""


@unique
class Sex(StrEnum):
    """A participant's sex, as ``metadata.csv`` writes it."""

    FEMALE = "F"
    MALE = "M"


@unique
class AgeBand(StrEnum):
    """A participant's age band, as ``metadata.csv`` writes it.

    The authors binned ages into these four for anonymity, so the release holds no exact age.
    """

    AGE_18_29 = "18-29"
    AGE_30_37 = "30-37"
    AGE_38_52 = "38-52"
    AGE_53_PLUS = "53+"


class DictionaryRow(BaseModel):
    """One row of the label dictionary: a raw label, and its class in each scheme.

    Each class field is named by its scheme's ``stem`` and read from the scheme's ``label:`` column.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    annotation: Text
    """The raw label, a Compendium of Physical Activity entry."""
    willetts_specific_2018: Text = Field(alias=LabelScheme.WILLETTS_SPECIFIC_2018.column)
    willetts_met_2018: Text = Field(alias=LabelScheme.WILLETTS_MET_2018.column)
    doherty_specific_2018: Text = Field(alias=LabelScheme.DOHERTY_SPECIFIC_2018.column)
    willetts_2018: Text = Field(alias=LabelScheme.WILLETTS_2018.column)
    doherty_2018: Text = Field(alias=LabelScheme.DOHERTY_2018.column)
    walmsley_2020: Text = Field(alias=LabelScheme.WALMSLEY_2020.column)

    @classmethod
    def parse(cls, header: list[str], row: list[str], context: str) -> Self:
        """Validate one CSV row against the header's columns.

        Args:
            header: The file's header.
            row: The row's fields; a blank line is an empty row.
            context: The file and line, for the error.

        Returns:
            The row.

        Raises:
            TimeFFormatError: If the row doesn't have one field per column, or a field is blank.
        """
        if len(row) != len(header):
            raise TimeFFormatError(f"{context}: expected {len(header)} fields, got {len(row)}")
        try:
            return cls.model_validate_strings(dict(zip(header, row, strict=True)))
        except ValidationError as exc:
            raise TimeFFormatError(f"{context}: {exc}") from exc

    def class_in(self, scheme: LabelScheme) -> str:
        """Return the raw label's class in one scheme.

        Returns:
            The class, as the dictionary writes it.
        """
        return getattr(self, scheme.stem)


@dataclass(frozen=True)
class LabelDictionary:
    """The label dictionary: every raw label, and its class in each scheme, in file order."""

    rows: tuple[DictionaryRow, ...]
    """The validated rows, in file order, which every view below keeps."""

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and check the dictionary.

        Returns:
            The dictionary.

        Raises:
            TimeFFormatError: If the header isn't ``annotation`` and the six scheme columns, or a raw label
                is empty or repeated, or a class is empty.
        """
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        header = ["annotation", *(scheme.column for scheme in LabelScheme)]
        if not rows or rows[0] != header:
            raise TimeFFormatError(f"{path}: expected the header {','.join(header)}")
        dictionary_rows = [DictionaryRow.parse(header, row, f"{path}:{line}") for line, row in enumerate(rows[1:], 2)]
        dictionary = cls(rows=tuple(dictionary_rows))
        if len(set(dictionary.raw_labels)) != len(dictionary.raw_labels):
            raise TimeFFormatError(f"{path}: a raw label is listed twice")
        return dictionary

    @property
    def raw_labels(self) -> tuple[str, ...]:
        """Return every raw label, in file order.

        Each is a Compendium of Physical Activity entry, such as ``7030 sleeping;MET 0.95``.
        """
        return tuple(row.annotation for row in self.rows)

    def classes(self, scheme: LabelScheme) -> tuple[str, ...]:
        """Return one scheme's class of each raw label, in the order of :attr:`raw_labels`.

        Returns:
            One class per raw label, as the dictionary writes it.
        """
        return tuple(row.class_in(scheme) for row in self.rows)

    def vocabulary(self, scheme: LabelScheme) -> tuple[str, ...]:
        """Return a scheme's classes, sorted.

        Sorted, a class's index breaks the authors' ties: ``Series.mode`` sorts the modes it finds, and
        ``make_windows`` takes the first.

        Returns:
            The distinct classes, in Python string order.
        """
        return tuple(sorted(set(self.classes(scheme))))

    def class_codes(self, scheme: LabelScheme) -> np.ndarray:
        """Return each raw label's class as an index into :meth:`vocabulary`, in the order of :attr:`raw_labels`.

        Returns:
            One int16 code per raw label.
        """
        code_of = {name: code for code, name in enumerate(self.vocabulary(scheme))}
        return np.array([code_of[name] for name in self.classes(scheme)], dtype=np.int16)
