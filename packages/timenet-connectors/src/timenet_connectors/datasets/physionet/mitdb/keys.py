"""The names the MIT-BIH connector writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    SOURCE_SHA256 = "source_sha256"
    AGE = "age"
    """The subject's age in years, from the first header comment. Absent where the header writes -1."""
    SEX = "sex"
    MEDICATIONS = "medications"
    """The second header comment, as written, such as ``Digoxin, Lasix`` or ``None``."""
    BEAT = "beat"
    """A beat label at its fiducial point, valued by its code."""
    RHYTHM = "rhythm"
    """A rhythm episode: from a rhythm change to the next one, or to the end of the record."""
    SIGNAL_QUALITY = "signal_quality"
    """One signal's quality, from a signal quality change to the next one, or to the end of the record."""
    EVENT = "event"
    """A non-beat point label, valued by its code: ``x``, ``|``, ``[`` or ``]``."""
    COMMENT = "comment"
    """A comment at a point, valued by its text: ``MISSB``, ``PSE`` or ``TS``."""
    BEAT_VOCABULARY = "beat_vocabulary"
    """The beat codes; the beat classification tasks' ``target_schema``."""
    RHYTHM_VOCABULARY = "rhythm_vocabulary"


SUBTYPE = "subtype"
"""The key, in an annotation's metadata, of the WFDB subtype the source writes where it is not 0."""
