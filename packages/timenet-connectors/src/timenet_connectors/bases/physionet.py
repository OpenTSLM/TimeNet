"""A reusable base for connectors that read WFDB records from PhysioNet.

Subclasses download a PhysioNet database archive (with
:func:`~timenet_connectors.download.ensure_archive`) and read its records with
`wfdb <https://wfdb.readthedocs.io>`_. ``wfdb`` is imported lazily so base users who only curate offline
datasets don't need it (install the ``physionet`` extra); a missing library raises an actionable error.
"""

from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa

from timenet.connectors import BaseConnector


TRaw = TypeVar("TRaw")


class BasePhysioNetConnector(BaseConnector[TRaw], ABC):
    """Base class for PhysioNet-backed connectors: WFDB record I/O."""

    @staticmethod
    def _wfdb() -> Any:
        """Import ``wfdb`` lazily, with an actionable error when the extra is missing.

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
        """Read a WFDB record header (cheap: no signal decode).

        Args:
            record_base: The record path without the ``.dat`` / ``.hea`` extension.

        Returns:
            The ``wfdb`` header record, exposing ``fs``, ``sig_len``, and ``sig_name``.
        """
        return self._wfdb().rdheader(str(record_base))

    def _lead_loader(self, record_base: Path, lead_idx: int) -> Callable[[], pa.Array]:
        """Build a lazy loader for one lead's samples as a float32 Arrow array (physical units).

        Args:
            record_base: The record path without the ``.dat`` / ``.hea`` extension.
            lead_idx: The zero-based lead index within the record.

        Returns:
            A no-argument loader returning the lead's physical signal as a float32 Arrow array.
        """

        def load() -> pa.Array:
            signal, _ = self._wfdb().rdsamp(str(record_base), channels=[lead_idx])
            return pa.array(signal[:, 0].astype("float32"))

        return load
