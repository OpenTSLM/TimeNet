"""A reusable base for connectors that read WFDB records from PhysioNet.

Subclasses download a PhysioNet database archive (a zip served from PhysioNet's open ``physionet-open``
S3 bucket, or any HTTP URL) and read its records with `wfdb <https://wfdb.readthedocs.io>`_. ``wfdb`` and
``boto3`` are imported lazily so base users who only curate offline datasets don't need them (install the
``physionet`` extra); a missing library raises an actionable error.
"""

from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet_connectors.bases import http


TRaw = TypeVar("TRaw")


class BasePhysioNetConnector(BaseConnector[TRaw], ABC):
    """Base class for PhysioNet-backed connectors: WFDB record I/O plus archive caching."""

    @staticmethod
    def _ensure_archive(url: str, cache_dir: Path, sentinel: str, *, filename: str | None = None) -> Path:
        """Download and extract a zip archive into ``cache_dir`` once.

        Thin delegate to :func:`timenet_connectors.bases.http.ensure_archive`, kept as a method so
        subclasses call it as ``self._ensure_archive(...)``. See that function for the caching and
        atomic-download semantics; ``s3://`` URLs are routed to the S3 helper, any other URL is
        streamed over HTTP.

        Args:
            url: The archive URL — an ``s3://bucket/key`` object or an ``http(s)://`` URL.
            cache_dir: Directory the archive is downloaded and extracted into.
            sentinel: A path relative to ``cache_dir`` marking extraction as complete.
            filename: Overrides the cached archive name for URLs whose path has no usable filename.

        Returns:
            ``cache_dir`` (the extraction root).
        """
        return http.ensure_archive(url, cache_dir, sentinel, filename=filename)

    @staticmethod
    def _stream_download(url: str, dest: Path) -> None:
        """Stream a URL to ``dest`` atomically.

        Thin delegate to :func:`timenet_connectors.bases.http.download_file`, kept for subclasses
        that call it directly.

        Args:
            url: The source URL.
            dest: The destination file path.
        """
        http.download_file(url, dest)

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
