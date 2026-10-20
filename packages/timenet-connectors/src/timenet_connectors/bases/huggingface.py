"""Reusable bases for connectors that read a HuggingFace Hub dataset.

:class:`BaseHuggingFaceConnector` reads the Hub's auto-converted parquet rows.
:class:`BaseHubSnapshotConnector` downloads the original files at a pinned revision instead and hands
``convert()`` one :class:`HubFileRef` per file, for sources with no parquet ref or that a connector
reads lazily from the files themselves.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``.
Downloads read the Hub's auto-generated parquet ref. The Hub produces that ref for public and gated datasets.
``huggingface_hub`` is declared in the connector's ``requirements.txt`` and installed into the environment the
build runs in; the base imports it lazily, which is what keeps ``--no-isolation`` usable. The base reads
``HF_TOKEN`` from the environment, so gated datasets work with no extra wiring.
Fully private datasets have no auto-parquet ref, and this base does not support them.
"""

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import TimeNetDatasetNotFoundError
from timenet_connectors.sources.huggingface_hub import hub_snapshot


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
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One dict per row across all parquet files.

        Raises:
            ImportError: If ``huggingface_hub``, declared in this connector's requirements, is missing.
            TimeNetDatasetNotFoundError: If the revision holds no parquet files, or they hold no rows.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc
        import pyarrow.parquet as pq  # noqa: PLC0415

        revision = self.PARQUET_REVISION
        filenames = [
            f
            for f in sorted(list_repo_files(self.HF_REPO, repo_type="dataset", revision=revision))
            if f.endswith(".parquet")
        ]
        if not filenames:
            raise TimeNetDatasetNotFoundError(
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
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} returned no rows from {len(filenames)} parquet file(s)"
            )
        return rows


@dataclass(frozen=True)
class HubFileRef:
    """One local source file and the part of the dataset it holds."""

    path: Path
    part: str
    """The connector's key for this file, such as a variant or a source partition."""
    expected_rows: int | None = None
    """Pinned row count to check the file against; ``None`` skips the check, as local fixtures do."""


class BaseHubSnapshotConnector(BaseConnector[HubFileRef], ABC):
    """Base class for connectors that read original files from a pinned Hub dataset revision.

    Subclasses set :attr:`HF_REPO`, :attr:`SOURCE_REVISION`, and :attr:`SOURCE_FILES`, and implement
    ``convert()`` over the returned :class:`HubFileRef` list. Gated datasets work through the same
    ``HF_TOKEN`` the Hub client reads from the environment.
    """

    HF_REPO: ClassVar[str]
    SOURCE_REVISION: ClassVar[str]
    """An immutable commit: a mutable branch can silently change rebuilt datasets."""
    SOURCE_FILES: ClassVar[Mapping[str, str]]
    """Each part's repository-relative filename, in the order ``convert()`` receives them."""
    SOURCE_ROWS: ClassVar[Mapping[str, int]] = {}
    """Optional pinned row count per part, passed on as :attr:`HubFileRef.expected_rows`."""

    def download(self, cache_dir: Path) -> list[HubFileRef]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION`.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One reference per part, pointing into the local snapshot.
        """
        snapshot = hub_snapshot(self.HF_REPO, self.SOURCE_REVISION, cache_dir, tuple(self.SOURCE_FILES.values()))
        return [
            HubFileRef(snapshot / filename, part, self.SOURCE_ROWS.get(part))
            for part, filename in self.SOURCE_FILES.items()
        ]
