"""Identifiers for values-plane storage backends supported by TimeF."""

from enum import StrEnum


class ValuesBackend(StrEnum):
    """Storage backend for the time-series values plane, recorded as the manifest ``values_backend`` tag.

    The member *values* are the on-disk tags; being a :class:`~enum.StrEnum`, each member is also a
    plain ``str``, so it compares and serializes exactly like the bare tag it replaces.
    """

    PARQUET = "parquet"


SUPPORTED_VALUES_BACKENDS: frozenset[ValuesBackend] = frozenset(ValuesBackend)
"""Every backend a writer may target and a reader can resolve."""
