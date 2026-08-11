"""Identifiers for values-plane storage backends supported by TimeF."""

from enum import StrEnum


class ValuesBackend(StrEnum):
    """Storage backend for the time-series values plane, recorded as the manifest ``values_backend`` tag.

    Each member value is the on-disk tag. This class is a :class:`~enum.StrEnum`, so each member is
    also a plain ``str``. Each member compares and serializes like its bare tag.
    """

    PARQUET = "parquet"
    ZARR = "zarr"


SUPPORTED_VALUES_BACKENDS: frozenset[ValuesBackend] = frozenset(ValuesBackend)
"""Every backend a writer may target and a reader can resolve."""
