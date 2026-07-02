"""A reusable base for connectors that pull rows from a HuggingFace Hub dataset.

Subclasses set ``HF_REPO`` and a ``FIXTURE`` (a small checked-in JSON used in testing mode), and
implement ``metadata()`` and ``convert()``. Real downloads lazily import ``huggingface_hub`` so base
users don't need it (install the ``huggingface`` extra); ``huggingface_hub`` reads ``HF_TOKEN`` from the
environment itself, so gated/private repos work with no extra wiring.
"""

from abc import ABC
import json
import os
from pathlib import Path
from typing import Any, ClassVar

from timenet.connectors import BaseConnector


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for HuggingFace-backed connectors. Rows are plain dicts (one per dataset row)."""

    HF_REPO: ClassVar[str]
    FIXTURE: ClassVar[Path]
    # The Hub auto-converts every public dataset to parquet on this ref, regardless of its source format
    # (CSV, JSON, ...). Reading it keeps this base format-agnostic and needs no heavy ``datasets`` dep.
    PARQUET_REVISION: ClassVar[str] = "refs/convert/parquet"

    def __init__(self) -> None:
        """Read testing mode and an optional row limit from the environment."""
        self.testing = os.environ.get("TIMENET_TESTING") == "1"
        limit = os.environ.get("TIMENET_ROW_LIMIT")
        self.row_limit = int(limit) if limit else None

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Return the dataset's rows: the fixture in testing mode, else fetched from the Hub.

        Args:
            cache_dir: Where Hub files are cached.

        Returns:
            One dict per row.
        """
        rows = self._fixture_rows() if self.testing else self._download_rows(cache_dir)
        return rows if self.row_limit is None else rows[: self.row_limit]

    def _fixture_rows(self) -> list[dict[str, Any]]:
        """Load the checked-in fixture rows (testing mode).

        Returns:
            The fixture rows as dicts.
        """
        return json.loads(self.FIXTURE.read_text())

    def _download_rows(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Download the repo's auto-converted parquet file(s) and read their rows.

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
            rows.extend(pq.read_table(path).to_pylist())
            if self.row_limit is not None and len(rows) >= self.row_limit:
                break
        return rows
