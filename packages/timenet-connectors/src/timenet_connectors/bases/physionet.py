"""A reusable base class for connectors that read WFDB records from PhysioNet.

A subclass downloads a PhysioNet database archive with :func:`~timenet_connectors.download.ensure_archive`
and reads the records with `wfdb <https://wfdb.readthedocs.io>`_. The base class imports ``wfdb`` lazily,
so a base user who only curates offline datasets does not need it. To use ``wfdb``, install the
``physionet`` extra. If ``wfdb`` is missing, the base class raises an error that states the fix.
"""

from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa

from timenet.connectors import BaseConnector


TRaw = TypeVar("TRaw")


class BasePhysioNetConnector(BaseConnector[TRaw], ABC):
    """The base class for PhysioNet-backed connectors that read WFDB records."""

    @staticmethod
    def _wfdb() -> Any:
        """Import ``wfdb`` lazily. Raise an error that states the fix if the extra is missing.

        Returns:
            The imported ``wfdb`` module.

        Raises:
            ImportError: If ``wfdb`` (the ``physionet`` extra) is not installed.
        """
        try:
            import wfdb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "reading PhysioNet records needs the physionet extra: pip install 'timenet-connectors[physionet]'"
            ) from exc
        return wfdb

    def _read_header(self, record_base: Path) -> Any:
        """Read a WFDB record header. This read is fast because it does not decode the signal.

        Args:
            record_base: The record path without the ``.dat`` and ``.hea`` extensions.

        Returns:
            The ``wfdb`` header record. The header exposes ``fs``, ``sig_len``, and ``sig_name``.
        """
        return self._wfdb().rdheader(str(record_base))

    def _lead_loader(self, record_base: Path, lead_idx: int) -> Callable[[], pa.Array]:
        """Build a lazy loader for the samples of one lead. The loader returns a float32 Arrow array in physical units.

        Args:
            record_base: The record path without the ``.dat`` and ``.hea`` extensions.
            lead_idx: The zero-based lead index within the record.

        Returns:
            A loader that takes no arguments. The loader returns the physical signal of the lead
            as a float32 Arrow array.
        """

        def load() -> pa.Array:
            signal, _ = self._wfdb().rdsamp(str(record_base), channels=[lead_idx])
            return pa.array(signal[:, 0].astype("float32"))

        return load
