"""A reusable base for connectors that pull rows from a HuggingFace Hub dataset.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``.
Downloads read the Hub's auto-generated parquet ref. The Hub produces that ref for public and gated datasets.
``huggingface_hub`` is declared in the connector's ``requirements.txt`` and installed into the environment the
build runs in; the base imports it lazily, which is what keeps ``--no-isolation`` usable. The base reads
``HF_TOKEN`` from the environment, so gated datasets work with no extra wiring.
Fully private datasets have no auto-parquet ref, and this base does not support them.
"""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import TimeNetDatasetNotFoundError


_BATCH_ROWS = 65536  # Parquet rows decoded per batch.


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for HuggingFace-backed connectors. Rows are plain dicts (one per dataset row)."""

    HF_REPO: ClassVar[str]
    # The Hub auto-converts every public dataset to parquet on this ref, whatever the source format
    # (CSV, JSON, ...). Reading it keeps this base format-agnostic and needs no heavy ``datasets`` dep.
    PARQUET_REVISION: ClassVar[str] = "refs/convert/parquet"

    def _hub_functions(self) -> tuple[Any, Any]:
        """Import and return the Hub file-listing and download functions.

        Returns:
            The ``hf_hub_download`` and ``list_repo_files`` functions, in that order.

        Raises:
            ImportError: If the connector-local ``huggingface_hub`` dependency is unavailable.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc
        return hf_hub_download, list_repo_files

    def _parquet_filenames(self) -> list[str]:
        """List Parquet filenames at the configured Hub revision.

        Returns:
            Every Parquet filename in stable order.
        """
        _, list_repo_files = self._hub_functions()
        return [
            filename
            for filename in sorted(list_repo_files(self.HF_REPO, repo_type="dataset", revision=self.PARQUET_REVISION))
            if filename.endswith(".parquet")
        ]

    def _read_parquet_rows(self, cache_dir: Path, filenames: list[str]) -> list[dict[str, Any]]:
        """Download selected Parquet files and decode their rows.

        Args:
            cache_dir: Directory where the connector caches Hub files.
            filenames: Parquet filenames at :attr:`PARQUET_REVISION` to download.

        Returns:
            One dict per decoded row, in filename and source-row order.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415

        rows: list[dict[str, Any]] = []
        for path in self._download_parquet_files(cache_dir, filenames):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS):
                rows.extend(batch.to_pylist())
        return rows

    def _download_parquet_files(self, cache_dir: Path, filenames: list[str]) -> list[Path]:
        """Download selected Parquet files and return their cached paths.

        Args:
            cache_dir: Directory where the connector caches Hub files.
            filenames: Parquet filenames at :attr:`PARQUET_REVISION` to download.

        Returns:
            Cached local paths in ``filenames`` order.
        """
        hf_hub_download, _ = self._hub_functions()
        return [
            Path(
                hf_hub_download(
                    self.HF_REPO,
                    filename,
                    repo_type="dataset",
                    revision=self.PARQUET_REVISION,
                    cache_dir=str(cache_dir),
                )
            )
            for filename in filenames
        ]

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download the repo's auto-converted parquet file(s) and return their rows.

        Reads the Hub's ``refs/convert/parquet`` branch (see :attr:`PARQUET_REVISION`), so it handles any
        source format the same way. For very large datasets the Hub conversion can be partial.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One dict per row across all parquet files.

        Raises:
            TimeNetDatasetNotFoundError: If the revision holds no parquet files, or they hold no rows.
        """
        revision = self.PARQUET_REVISION
        filenames = self._parquet_filenames()
        if not filenames:
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} has no parquet files on {revision!r}; the Hub publishes that ref only "
                "for public and gated datasets, not fully private ones"
            )
        rows = self._read_parquet_rows(cache_dir, filenames)
        if not rows:
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} returned no rows from {len(filenames)} parquet file(s)"
            )
        return rows
