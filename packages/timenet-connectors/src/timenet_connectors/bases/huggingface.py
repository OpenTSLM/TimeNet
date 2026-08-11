"""A reusable base for connectors that pull rows from a HuggingFace Hub dataset.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``.
Downloads read the Hub's auto-generated parquet ref. The Hub produces that ref for public and gated datasets.
``huggingface_hub`` is imported lazily, so base users do not need it. Install the ``huggingface`` extra to get
it. The base reads ``HF_TOKEN`` from the environment, so gated datasets work with no extra wiring. Fully private
datasets have no auto-parquet ref, and this base does not support them.
"""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import DatasetNotFoundError


_BATCH_ROWS = 65536  # parquet rows decoded per batch


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for HuggingFace-backed connectors. Rows are plain dicts (one per dataset row)."""

    HF_REPO: ClassVar[str]
    # The Hub auto-converts every public dataset to parquet on this ref, whatever the source format
    # (CSV, JSON, ...). Reading it keeps this base format-agnostic and needs no heavy ``datasets`` dep.
    PARQUET_REVISION: ClassVar[str] = "refs/convert/parquet"

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download the repo's auto-converted parquet file(s) and return their rows.

        Reads the Hub's ``refs/convert/parquet`` branch (see :attr:`PARQUET_REVISION`), so it handles any
        source format the same way. For very large datasets the Hub conversion can be partial.

        Args:
            cache_dir: Where Hub files are cached.

        Returns:
            One dict per row across all parquet files.

        Raises:
            ImportError: If the ``huggingface`` extra (``huggingface_hub``) is not installed.
            DatasetNotFoundError: If the revision holds no parquet files, or they hold no rows.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs the huggingface extra: pip install 'timenet-connectors[huggingface]'"
            ) from exc
        import pyarrow.parquet as pq  # noqa: PLC0415

        revision = self.PARQUET_REVISION
        filenames = [
            f
            for f in sorted(list_repo_files(self.HF_REPO, repo_type="dataset", revision=revision))
            if f.endswith(".parquet")
        ]
        if not filenames:
            raise DatasetNotFoundError(
                f"{self.HF_REPO!r} has no parquet files on {revision!r}; the Hub publishes that ref only "
                "for public and gated datasets, not fully private ones"
            )
        rows: list[dict[str, Any]] = []
        for filename in filenames:
            path = hf_hub_download(
                self.HF_REPO, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir)
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS):
                rows.extend(batch.to_pylist())
        if not rows:
            raise DatasetNotFoundError(f"{self.HF_REPO!r} returned no rows from {len(filenames)} parquet file(s)")
        return rows
