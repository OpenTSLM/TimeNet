"""The closed sets of CODE-15%: the leads and the six abnormalities.

A patient's sex is the shared :class:`~timenet_connectors.demographics.Sex`.
"""

from enum import StrEnum, unique


@unique
class Lead(StrEnum):
    """A lead, named and ordered as the release's tracings hold them.

    The order is the Zenodo record's and the authors' ``preprocess.py`` ``all_leads``. The tracings agree:
    where DIII, AVR, AVL and AVF are derived from DI and DII, this order is the one they fit.
    """

    DI = "DI"
    DII = "DII"
    DIII = "DIII"
    AVR = "AVR"
    AVL = "AVL"
    AVF = "AVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

    @property
    def column(self) -> int:
        """Return the lead's index on the last axis of the tracings."""
        return _COLUMNS[self]


_COLUMNS = {lead: column for column, lead in enumerate(Lead)}


@unique
class Abnormality(StrEnum):
    """One of the six abnormalities, by its ``exams.csv`` column, in the authors' model output order.

    The order is the ``automatic-ecg-diagnosis`` README's output order, which the abnormality tasks' targets
    and vocabulary follow. ``exams.csv`` writes ST before AF.
    """

    FIRST_DEGREE_AV_BLOCK = "1dAVb"
    RIGHT_BUNDLE_BRANCH_BLOCK = "RBBB"
    LEFT_BUNDLE_BRANCH_BLOCK = "LBBB"
    SINUS_BRADYCARDIA = "SB"
    ATRIAL_FIBRILLATION = "AF"
    SINUS_TACHYCARDIA = "ST"

    @property
    def full_name(self) -> str:
        """Return the abnormality's name, as the Zenodo record and the authors' README write it."""
        return _FULL_NAMES[self]


_FULL_NAMES = {
    Abnormality.FIRST_DEGREE_AV_BLOCK: "1st degree AV block",
    Abnormality.RIGHT_BUNDLE_BRANCH_BLOCK: "right bundle branch block",
    Abnormality.LEFT_BUNDLE_BRANCH_BLOCK: "left bundle branch block",
    Abnormality.SINUS_BRADYCARDIA: "sinus bradycardia",
    Abnormality.ATRIAL_FIBRILLATION: "atrial fibrillation",
    Abnormality.SINUS_TACHYCARDIA: "sinus tachycardia",
}
