"""Reusable bases for connectors that read a HuggingFace Hub dataset.

:class:`BaseHuggingFaceConnector` reads the Hub's auto-converted parquet rows.
:class:`BaseHubSnapshotConnector` downloads the original files at a pinned revision instead and hands
``convert()`` one :class:`HubFileRef` per file, for sources with no parquet ref or that a connector
reads lazily from the files themselves.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``.
``huggingface_hub`` is declared in the connector's ``requirements.txt`` and installed into the environment the
build runs in; :mod:`timenet_connectors.sources.huggingface_hub` imports it lazily, which is what keeps
``--no-isolation`` usable. The Hub client reads ``HF_TOKEN`` from the environment, so gated datasets work with
no extra wiring.
"""

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet_connectors.sources.huggingface_hub import hub_file, hub_files


_BATCH_ROWS = 65536  # parquet rows decoded per batch
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for HuggingFace-backed connectors. Rows are plain dicts (one per dataset row).

    Downloads read the Hub's auto-generated parquet ref, which the Hub produces for public and gated
    datasets. Fully private datasets have no auto-parquet ref, and this base does not support them.
    """

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
            TimeNetDownloadError: If the revision cannot be listed or a parquet file cannot be downloaded.
        """  # noqa: DOC502 - TimeNetDownloadError is raised by hub_files and hub_file
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
    """The connector's own name for this file, such as ``"hhar_v2"``: the key in ``SOURCE_FILES`` and
    ``SOURCE_ROWS``, which ``convert()`` branches on and builds ids from. ``SOURCE_FILES[part]`` gives the
    file's repository-relative filename."""
    expected_rows: int | None = None
    """Pinned row count to check the file against; ``None`` skips the check, as local fixtures do."""

    def check_rows(self, count: int) -> None:
        """Check the number of rows ``convert()`` read from this file against its pinned inventory.

        Args:
            count: Rows read from the file.

        Raises:
            TimeFValidationError: If the file held no rows, or a count other than :attr:`expected_rows`.
        """
        if not count:
            raise TimeFValidationError(f"{self.path} ({self.part}): no rows")
        if self.expected_rows is not None and count != self.expected_rows:
            raise TimeFValidationError(f"{self.path} ({self.part}): expected {self.expected_rows} rows, found {count}")


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
    """Optional pinned row count per part, passed on as :attr:`HubFileRef.expected_rows`.

    ``convert()`` checks each count with :meth:`HubFileRef.check_rows`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Validate the declaration when the class is defined, so a fixture-only convert() test catches a
        # bad pin that download() would otherwise report only during a real build. Intermediate bases may
        # leave the attributes unset.
        super().__init_subclass__(**kwargs)
        revision = getattr(cls, "SOURCE_REVISION", None)
        if revision is not None and not _COMMIT_SHA.fullmatch(revision):
            raise TimeFValidationError(
                f"{cls.__name__}.SOURCE_REVISION must be a 40-character commit, got {revision!r}"
            )
        files = getattr(cls, "SOURCE_FILES", None)
        if files is not None:
            # A misspelled part would otherwise read as "no pinned count" and skip the row check.
            unknown = sorted(cls.SOURCE_ROWS.keys() - files.keys())
            if unknown:
                raise TimeFValidationError(
                    f"{cls.__name__}.SOURCE_ROWS names parts absent from SOURCE_FILES: {', '.join(unknown)}"
                )

    def download(self, cache_dir: Path) -> list[HubFileRef]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION`.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One reference per part, pointing into the local Hub cache.

        Raises:
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # noqa: DOC502 - TimeNetDownloadError is raised by hub_file
        return [
            HubFileRef(
                hub_file(self.HF_REPO, self.SOURCE_REVISION, cache_dir, filename),
                part,
                self.SOURCE_ROWS.get(part),
            )
            for part, filename in self.SOURCE_FILES.items()
        ]
