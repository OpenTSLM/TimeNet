"""The label schemes of the CAPTURE-24 label dictionary, and the label sets of the benchmarked ones.

A label scheme groups the raw labels into activity classes. The dictionary gives each raw label's class
in six schemes. The authors' benchmark.py scores two of them, so only these two have label sets and tasks.
"""

from enum import StrEnum, unique

from timenet.types import Label, LabelSet


_SCHEME_COLUMN_PREFIX = "label:"


@unique
class Scheme(StrEnum):
    """A label scheme of the label dictionary, in the file's column order."""

    WILLETTS_SPECIFIC_2018 = "WillettsSpecific2018"
    WILLETTS_MET_2018 = "WillettsMET2018"
    DOHERTY_SPECIFIC_2018 = "DohertySpecific2018"
    WILLETTS_2018 = "Willetts2018"
    DOHERTY_2018 = "Doherty2018"
    WALMSLEY_2020 = "Walmsley2020"

    @property
    def column(self) -> str:
        """The dictionary column that holds each raw label's class in this scheme."""
        return f"{_SCHEME_COLUMN_PREFIX}{self.value}"


# The label sets of the two schemes that benchmark.py scores, with the classes as the label dictionary
# names them. Each set is in string order, because a tie between classes goes to the first in string
# order.
WILLETTS_SPECIFIC_2018 = LabelSet(
    name="willetts_specific_2018",
    labels=tuple(
        Label(name=name)
        for name in (
            "bicycling",
            "household-chores",
            "manual-work",
            "mixed-activity",
            "sitting",
            "sleep",
            "sports",
            "standing",
            "vehicle",
            "walking",
        )
    ),
)
WALMSLEY_2020 = LabelSet(
    name="walmsley_2020",
    labels=tuple(Label(name=name) for name in ("light", "moderate-vigorous", "sedentary", "sleep")),
)

LABEL_SETS = {
    Scheme.WILLETTS_SPECIFIC_2018: WILLETTS_SPECIFIC_2018,
    Scheme.WALMSLEY_2020: WALMSLEY_2020,
}
"""The label set of each scheme that the authors' benchmark.py scores. Only these schemes have tasks."""
