"""A reusable base for connectors that read WFDB records from PhysioNet.

Subclasses download a PhysioNet database archive (a zip served from PhysioNet's open S3 bucket) and
read its records with `wfdb <https://wfdb.readthedocs.io>`_. ``wfdb`` is imported lazily so base users
who only curate offline datasets don't need it (install the ``physionet`` extra); a missing library
raises an actionable error.
"""

from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
import zipfile

import pyarrow as pa

from timenet.connectors import BaseConnector


TRaw = TypeVar("TRaw")

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB streamed per write when fetching an archive


class BasePhysioNetConnector(BaseConnector[TRaw], ABC):
    """Base class for PhysioNet-backed connectors: WFDB record I/O plus archive caching."""

    def _ensure_archive(self, url: str, cache_dir: Path, sentinel: str) -> Path:
        """Download and extract a PhysioNet zip archive into ``cache_dir`` once.

        Idempotent: if ``cache_dir / sentinel`` already exists the download and extraction are
        skipped, so re-running a build reuses the cache.

        Args:
            url: The archive URL (PhysioNet open S3 bucket).
            cache_dir: Directory the archive is downloaded and extracted into.
            sentinel: A path relative to ``cache_dir`` whose presence means extraction is complete.

        Returns:
            ``cache_dir`` (the extraction root).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        if (cache_dir / sentinel).exists():
            return cache_dir
        zip_path = cache_dir / url.rsplit("/", 1)[-1]
        if not zip_path.exists():
            self._stream_download(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(cache_dir)
        return cache_dir

    @staticmethod
    def _stream_download(url: str, dest: Path) -> None:
        """Stream a URL to ``dest`` in chunks (kept out of memory for multi-GB archives).

        Args:
            url: The source URL.
            dest: The destination file path.

        Raises:
            ImportError: If ``requests`` (pulled by the ``physionet`` extra) is not installed.
        """
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised via the wfdb-missing path
            raise ImportError(
                "downloading from PhysioNet needs the physionet extra: pip install 'timenet-connectors[physionet]'"
            ) from exc
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)

    @staticmethod
    def _wfdb() -> Any:
        """Import ``wfdb`` lazily, with an actionable error when the extra is missing.

        Returns:
            The imported ``wfdb`` module.

        Raises:
            ImportError: If ``wfdb`` (the ``physionet`` extra) is not installed.
        """
        try:
            import wfdb
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
