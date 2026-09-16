"""The annotation keys this connector writes.

``connector.py`` is the only module that writes them. They are named here so that each key has one
spelling, and so the README's table of what a record carries has something to match.

A caption annotation is not here. Its key is the shard column the caption came from, so
``connector.py`` writes it from the same tuple it reads the columns with.
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
