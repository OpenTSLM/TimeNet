"""Identifiers for values-plane storage backends supported by TimeF."""

PARQUET_VALUES_BACKEND = "parquet"
ZARR_VALUES_BACKEND = "zarr"
SUPPORTED_VALUES_BACKENDS = frozenset({PARQUET_VALUES_BACKEND, ZARR_VALUES_BACKEND})
