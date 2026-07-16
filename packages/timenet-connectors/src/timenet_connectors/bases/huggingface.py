"""A reusable base for connectors that pull rows from a HuggingFace Hub dataset.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``. Real
downloads read the Hub's auto-generated parquet ref, which the Hub produces for public datasets (gated
ones included). ``huggingface_hub`` is imported lazily so base users don't need it (install the
``huggingface`` extra) and reads ``HF_TOKEN`` from the environment, so gated datasets work with no extra
wiring. Fully private datasets have no auto-parquet ref and aren't supported by this base.
"""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from timenet.connectors import BaseConnector


_BATCH_ROWS = 65536  # parquet rows decoded per batch


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for HuggingFace-backed connectors. Rows are plain dicts (one per dataset row)."""

    HF_REPO: ClassVar[str]
    # The Hub auto-converts every public dataset to parquet on this ref, regardless of its source format
    # (CSV, JSON, ...). Reading it keeps this base format-agnostic and needs no heavy ``datasets`` dep.
    PARQUET_REVISION: ClassVar[str] = "refs/convert/parquet"

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download the repo's auto-converted parquet file(s) and return their rows.

        Reads the Hub's ``refs/convert/parquet`` branch (see :attr:`PARQUET_REVISION`) so any source
        format is handled uniformly. For very large datasets the Hub's conversion can be partial.

        Args:
            cache_dir: Where Hub files are cached.

        Returns:
            One dict per row across all parquet files.

        Raises:
            ImportError: If the ``huggingface`` extra (``huggingface_hub``) is not installed.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs the huggingface extra: pip install 'timenet-connectors[huggingface]'"
            ) from exc
        import pyarrow.parquet as pq

        revision = self.PARQUET_REVISION
        rows: list[dict[str, Any]] = []
        for filename in sorted(list_repo_files(self.HF_REPO, repo_type="dataset", revision=revision)):
            if not filename.endswith(".parquet"):
                continue
            path = hf_hub_download(
                self.HF_REPO, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir)
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS):
                rows.extend(batch.to_pylist())
        return rows
