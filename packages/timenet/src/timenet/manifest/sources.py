"""Flat physical storage dependencies, distinct from the build input graph."""

from timenet.manifest.files import FilePart
from timenet.types.references import InputLock
from timenet.values_backends import ValuesBackend


class PayloadSource(InputLock):
    """One immutable physical owner and the files needed from that release."""

    files: tuple[FilePart, ...] = ()
    values_backend: ValuesBackend = ValuesBackend.PARQUET
