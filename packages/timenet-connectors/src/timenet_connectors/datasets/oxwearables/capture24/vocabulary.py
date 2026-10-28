"""The closed sets of CAPTURE-24: its 206 raw labels, its six label schemes, and the map between them."""

import csv
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Self

import numpy as np

from timenet.errors import TimeFFormatError


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

BENCHMARK_SCHEMES = (LabelScheme.WILLETTS_SPECIFIC_2018, LabelScheme.WALMSLEY_2020)
"""The schemes ``benchmark.py`` scores, in its default ``--annots`` order. Each becomes a task per window.

So every window has two tasks: one in ``WillettsSpecific2018`` (10 classes) and one in ``Walmsley2020``
(4 classes). The other four schemes (``WillettsMET2018``, ``DohertySpecific2018``, ``Willetts2018`` and
``Doherty2018``) have no tasks; the dataset still holds their vocabulary and mapping as annotations.
"""

NO_LABEL = -1
"""The code of an unlabelled sample, which the source writes as an empty field."""


@dataclass(frozen=True)
class LabelDictionary:
    """The label dictionary: every raw label, and its class in each scheme."""

    raw_labels: tuple[str, ...]
    """Every raw label, in file order: a Compendium of Physical Activity entry, such as
    ``7030 sleeping;MET 0.95``."""
    classes: dict[LabelScheme, tuple[str, ...]]
    """Each scheme's class of each raw label, in the order of :attr:`raw_labels`."""

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
        body = rows[1:]
        raw_labels = tuple(row[0] for row in body)
        if any(len(row) != len(header) or not all(row) for row in body):
            raise TimeFFormatError(f"{path}: every row needs a raw label and a class in each scheme")
        if len(set(raw_labels)) != len(raw_labels):
            raise TimeFFormatError(f"{path}: a raw label is listed twice")
        classes = {scheme: tuple(row[index] for row in body) for index, scheme in enumerate(LabelScheme, 1)}
        return cls(raw_labels=raw_labels, classes=classes)

    def vocabulary(self, scheme: LabelScheme) -> tuple[str, ...]:
        """Return a scheme's classes, sorted.

        Sorted, a class's index breaks the authors' ties: ``Series.mode`` sorts the modes it finds, and
        ``make_windows`` takes the first.

        Returns:
            The distinct classes, in Python string order.
        """
        return tuple(sorted(set(self.classes[scheme])))

    def class_codes(self, scheme: LabelScheme) -> np.ndarray:
        """Return each raw label's class as an index into :meth:`vocabulary`, in the order of :attr:`raw_labels`.

        Returns:
            One int16 code per raw label.
        """
        index = {name: code for code, name in enumerate(self.vocabulary(scheme))}
        return np.array([index[name] for name in self.classes[scheme]], dtype=np.int16)
