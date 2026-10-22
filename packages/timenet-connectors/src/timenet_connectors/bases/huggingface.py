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
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetDownloadError
from timenet_connectors.sources.huggingface_hub import hub_file, hub_files, hub_snapshot


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
            TimeNetDatasetNotFoundError: If the revision holds no parquet files, or they hold no rows.
        """
        import pyarrow.parquet as pq  # noqa: PLC0415

        revision = self.PARQUET_REVISION
        filenames = [f for f in sorted(hub_files(self.HF_REPO, revision)) if f.endswith(".parquet")]
        if not filenames:
            raise TimeNetDatasetNotFoundError(
                f"{self.HF_REPO!r} has no parquet files on {revision!r}; the Hub publishes that ref only "
                "for public and gated datasets, not fully private ones"
            )
        rows: list[dict[str, Any]] = []
        for filename in filenames:
            path = hub_file(self.HF_REPO, revision, cache_dir, filename)
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
    filename: str
    """The source file's repository-relative filename, as declared in ``SOURCE_FILES``."""
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

        Raises:
            TimeNetDownloadError: If the snapshot lacks a declared file. The Hub matches
                :attr:`SOURCE_FILES` as patterns, so a name absent from the revision downloads nothing
                instead of failing.
        """
        snapshot = hub_snapshot(self.HF_REPO, self.SOURCE_REVISION, cache_dir, tuple(self.SOURCE_FILES.values()))
        refs = [
            HubFileRef(snapshot / filename, part, filename, self.SOURCE_ROWS.get(part))
            for part, filename in self.SOURCE_FILES.items()
        ]
        missing = [ref.filename for ref in refs if not ref.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"{self.HF_REPO}@{self.SOURCE_REVISION} has no {', '.join(missing)}")
        return refs
