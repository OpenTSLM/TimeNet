"""The closed sets of the MIT-BIH Arrhythmia Database: annotation codes, rhythm labels, signal quality and sex.

The codes are the WFDB annotation mnemonics the reference ``.atr`` files write. Which of them is a beat comes
from WFDB's ``isqrs`` table (``ecgmap.h``), which the authors' beat-by-beat comparator ``bxb`` uses. The rhythm
labels are those the authors' ``dbnotes.c`` names, and the signal quality states those it decodes.
"""

from enum import StrEnum, unique


@unique
class BeatCode(StrEnum):
    """A beat label: an annotation code WFDB's ``isqrs`` counts as a QRS complex, and the release writes.

    The order is the columns of the authors' table of beat types. ``isqrs`` also counts ``B``, ``n``, ``r``
    and the learning code, which the release never writes.
    """

    NORMAL = "N"
    LEFT_BUNDLE_BRANCH_BLOCK = "L"
    RIGHT_BUNDLE_BRANCH_BLOCK = "R"
    ATRIAL_PREMATURE = "A"
    ABERRATED_ATRIAL_PREMATURE = "a"
    NODAL_PREMATURE = "J"
    SUPRAVENTRICULAR_PREMATURE = "S"
    PREMATURE_VENTRICULAR_CONTRACTION = "V"
    FUSION_OF_VENTRICULAR_AND_NORMAL = "F"
    VENTRICULAR_FLUTTER_WAVE = "!"
    """``isqrs`` counts it as a beat; the authors' notes call it a non-beat. It occurs only in record 207,
    inside its episodes of ventricular flutter, which ``bxb`` skips."""
    ATRIAL_ESCAPE = "e"
    NODAL_ESCAPE = "j"
    VENTRICULAR_ESCAPE = "E"
    PACED = "/"
    FUSION_OF_PACED_AND_NORMAL = "f"
    UNCLASSIFIABLE = "Q"


@unique
class NonBeatCode(StrEnum):
    """An annotation code the release writes that is not a beat."""

    RHYTHM_CHANGE = "+"
    """Starts a rhythm episode; its auxiliary text is the rhythm label."""
    SIGNAL_QUALITY_CHANGE = "~"
    """Starts a signal quality episode; its subtype states each signal's quality."""
    COMMENT = '"'
    """A comment; its auxiliary text is the comment."""
    BLOCKED_ATRIAL_PREMATURE = "x"
    """A non-conducted P-wave. The authors' table of beat types counts it, as a non-beat."""
    ARTIFACT = "|"
    """An isolated QRS-like artifact."""
    VENTRICULAR_FLUTTER_START = "["
    VENTRICULAR_FLUTTER_END = "]"


EVENT_CODES = (
    NonBeatCode.BLOCKED_ATRIAL_PREMATURE,
    NonBeatCode.ARTIFACT,
    NonBeatCode.VENTRICULAR_FLUTTER_START,
    NonBeatCode.VENTRICULAR_FLUTTER_END,
)
"""The non-beats kept as point annotations of their own: every non-beat with no auxiliary text or subtype
meaning of its own."""


@unique
class RhythmLabel(StrEnum):
    """A rhythm label, as the auxiliary text of a rhythm change writes it.

    These are the rhythms the authors' ``dbnotes.c`` names, in its order. The release writes 15 of them; it
    writes no ``(B22``, ``(B3``, ``(SAB`` or ``(EAR``.
    """

    NORMAL_SINUS_RHYTHM = "(N"
    SINUS_BRADYCARDIA = "(SBR"
    SECOND_DEGREE_HEART_BLOCK = "(BII"
    SECOND_DEGREE_HEART_BLOCK_MOBITZ_II = "(B22"
    THIRD_DEGREE_HEART_BLOCK = "(B3"
    SINO_ATRIAL_BLOCK = "(SAB"
    PRE_EXCITATION = "(PREX"
    ATRIAL_BIGEMINY = "(AB"
    ECTOPIC_ATRIAL_RHYTHM = "(EAR"
    SUPRAVENTRICULAR_TACHYARRHYTHMIA = "(SVTA"
    ATRIAL_FLUTTER = "(AFL"
    ATRIAL_FIBRILLATION = "(AFIB"
    PACED_RHYTHM = "(P"
    NODAL_RHYTHM = "(NOD"
    VENTRICULAR_BIGEMINY = "(B"
    VENTRICULAR_TRIGEMINY = "(T"
    IDIOVENTRICULAR_RHYTHM = "(IVR"
    VENTRICULAR_TACHYCARDIA = "(VT"
    VENTRICULAR_FLUTTER = "(VFL"


@unique
class SignalQuality(StrEnum):
    """One signal's quality, as the authors' ``dbnotes.c`` decodes a signal quality change."""

    CLEAN = "clean"
    NOISY = "noisy"
    UNREADABLE = "unreadable"


@unique
class Sex(StrEnum):
    """A subject's sex, as the first header comment writes it."""

    FEMALE = "F"
    MALE = "M"
