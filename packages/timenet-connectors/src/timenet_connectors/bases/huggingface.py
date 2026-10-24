"""Reusable bases for connectors that read a HuggingFace Hub dataset.

:class:`BaseHuggingFaceConnector` reads the parquet rows the Hub converts.
:class:`BaseHubSnapshotConnector` downloads the original files at a pinned commit instead. Use it when
the dataset has no parquet conversion or the connector reads the files itself.

Subclasses set ``HF_REPO``, ship a ``dataset.yaml`` card beside the connector, and implement ``convert()``.
``huggingface_hub`` is declared in the connector's ``requirements.txt`` and installed into the environment the
build runs in; :mod:`timenet_connectors.sources.huggingface_hub` imports it lazily, which is what keeps
``--no-isolation`` usable. The Hub client reads ``HF_TOKEN`` from the environment, so gated datasets work with
no extra wiring.
"""

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.types import Annotation
from timenet_connectors.sources.huggingface_hub import hub_file, hub_files


_BATCH_ROWS = 65536  # parquet rows decoded per batch
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class BaseHuggingFaceConnector(BaseConnector[dict[str, Any]], ABC):
    """Base class for connectors that read the parquet rows the Hub converts a dataset to.

    Subclasses set :attr:`HF_REPO` and implement ``convert()`` over the returned rows, one plain dict
    per dataset row. Use :class:`BaseHubSnapshotConnector` instead when the dataset has no parquet
    conversion or the connector needs the original files at a pinned commit.
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
            ImportError: If ``huggingface_hub``, declared in this connector's requirements, is missing.
            TimeNetDatasetNotFoundError: If the revision holds no parquet files, or they hold no rows.
            TimeNetDownloadError: If the revision cannot be listed or a parquet file cannot be downloaded.
        """  # noqa: DOC502 - ImportError and TimeNetDownloadError are raised by hub_files and hub_file
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
            path = hub_file(self.HF_REPO, filename, revision=revision, cache_dir=cache_dir)
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
    """The connector's own name for this file, such as ``"hhar_v2"``: the key in ``SOURCE_FILES``, which
    ``convert()`` branches on and builds ids from."""


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
    """Each part's repository-relative filename, in the order ``download()`` returns them."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Check the declaration when the class is defined
        revision = getattr(cls, "SOURCE_REVISION", None)
        if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
            raise TimeFValidationError(
                f"{cls.__name__}.SOURCE_REVISION must be a 40-character commit, got {revision!r}"
            )
        if not getattr(cls, "SOURCE_FILES", None):
            raise TimeFValidationError(f"{cls.__name__}.SOURCE_FILES must declare at least one file")

    def check_refs(self, raw_refs: Sequence[HubFileRef]) -> None:
        """Check that ``raw_refs`` is nonempty and names each part once, only parts :attr:`SOURCE_FILES` declares.

        Args:
            raw_refs: The references ``convert()`` received.

        Raises:
            TimeFValidationError: If ``raw_refs`` is empty, repeats a part, or names an undeclared part.
        """
        parts = [ref.part for ref in raw_refs]
        if not parts or len(set(parts)) != len(parts):
            raise TimeFValidationError(f"{type(self).__name__} requires nonempty, distinct part references")
        unknown = sorted(set(parts) - self.SOURCE_FILES.keys())
        if unknown:
            raise TimeFValidationError(f"{type(self).__name__} has no source part {', '.join(unknown)}")

    def new_dataset(self, raw_refs: Sequence[HubFileRef], id_prefix: str) -> TimeFDataset:
        """Check ``raw_refs``, then start a dataset annotated with its Hub repo and pinned revision.

        Args:
            raw_refs: The references ``convert()`` received.
            id_prefix: Prefix for the annotation ids, such as ``"sensorcaps"``.

        Returns:
            An empty dataset carrying ``source_repository`` and ``source_revision`` annotations.

        Raises:
            TimeFValidationError: If ``raw_refs`` is empty, repeats a part, or names an undeclared part.
        """  # noqa: DOC502 - TimeFValidationError is raised by check_refs
        self.check_refs(raw_refs)
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", self.SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value, id=f"{id_prefix}-{key}"))
        return dataset

    def download(self, cache_dir: Path) -> list[HubFileRef]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION`.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One reference per part, pointing into the local Hub cache.

        Raises:
            ImportError: If ``huggingface_hub``, declared in this connector's requirements, is missing.
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # noqa: DOC502 - ImportError and TimeNetDownloadError are raised by hub_file
        revision = self.SOURCE_REVISION
        return [
            HubFileRef(hub_file(self.HF_REPO, filename, revision=revision, cache_dir=cache_dir), part)
            for part, filename in self.SOURCE_FILES.items()
        ]
