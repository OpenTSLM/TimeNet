"""The annotation keys this connector writes.

They live here because ``connector.py`` writes them and the tests read them, and a key spelled two
ways is a bug nobody sees.
"""

from enum import StrEnum


class SlipEvalKey(StrEnum):
    """The keys of the annotations a SLIP evaluation record or task carries."""

    SOURCE_BENCHMARK = "source_benchmark"
    """The folder this window came from, e.g. ``wisdm``."""

    SPLIT = "split"
    """``train`` or ``test``, as the release cut it."""

    VOCABULARY = "vocabulary"
    """One folder's whole closed set of classes. A task's ``target_schema`` is this annotation's id."""

    LABEL = "label"
    """One class of one folder's vocabulary. Registered once and referenced by every task with it."""
