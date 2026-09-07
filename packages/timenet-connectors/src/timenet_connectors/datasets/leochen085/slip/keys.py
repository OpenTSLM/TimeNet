"""The annotation keys this connector writes.

They live here because ``connector.py`` writes them and the tests read them, and a key spelled two
ways is a bug nobody sees.
"""

from enum import StrEnum


class SlipKey(StrEnum):
    """The keys of the annotations a SLIP record carries."""

    SOURCE_DATASET = "source_dataset"
    """The corpus this row was drawn from, as the ``dataset`` column states it."""

    DOMAIN = "domain"
    """The domain that corpus belongs to, as the ``category`` column states it."""

    SOURCE_URL = "source_url"
    """Where that corpus was published, from ``meta.csv``."""

    ALL_NAN_SIGNALS = "all_nan_signals"
    """The names of this record's signals that hold no numbers. Written only when there are some."""
