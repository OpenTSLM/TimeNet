"""A reusable base for connectors that read files from a pinned HuggingFace Hub dataset revision.

Subclasses pass ``hf_repo``, ``source_revision``, and ``source_files`` as class keywords, ship a
``dataset.yaml`` card beside the connector, and implement ``convert_files()``. ``download()`` returns a
:class:`HubFileRef` list. The base's ``convert()`` checks that list, then passes
``convert_files()`` the path of each part. The base also fills ``source_url`` and ``source_revision``
in the dataset metadata from ``HF_REPO`` and ``SOURCE_REVISION``.
:mod:`timenet_connectors.sources.huggingface_hub` describes how the files download. For the Hub's
parquet conversion, pin the commit ``refs/convert/parquet`` points to.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.types import DatasetMetadata
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class HubFileRef:
    """A downloaded file (or a pattern's directory) and the part it belongs to."""

    path: Path
    part: str
    """The connector's own name for this file, such as ``"hhar_v2"``: the key in ``SOURCE_FILES`` and in the
    mapping that ``convert_files()`` receives."""


class BaseHuggingFaceConnector(BaseConnector[list[HubFileRef]], ABC):
    """Base class for connectors that read files from a pinned Hub dataset revision.

    Subclasses pass ``hf_repo``, ``source_revision``, and ``source_files`` as class keywords, and
    implement :meth:`convert_files`. The base stores the keywords as :attr:`HF_REPO`,
    :attr:`SOURCE_REVISION`, and :attr:`SOURCE_FILES`. A subclass that passes no keywords keeps the
    values of its parent. :meth:`convert` runs :meth:`check_refs` first.
    """

    HF_REPO: ClassVar[str]
    SOURCE_REVISION: ClassVar[str]
    """An immutable commit: a mutable branch can silently change rebuilt datasets."""
    SOURCE_FILES: ClassVar[Mapping[str, str]]
    """The repo-relative filename of each part, in download-order. A filename can end
    in a glob like ``data/*.parquet``; the part's path is then the directory of the matching files."""

    def __init_subclass__(
        cls,
        *,
        hf_repo: str | None = None,
        source_revision: str | None = None,
        source_files: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if hf_repo is not None:
            cls.HF_REPO = hf_repo
        if source_revision is not None:
            cls.SOURCE_REVISION = source_revision
        if source_files is not None:
            cls.SOURCE_FILES = MappingProxyType(dict(source_files))
        if not getattr(cls, "HF_REPO", None):
            raise TimeFValidationError(f"{cls.__name__} must pass hf_repo")
        revision = getattr(cls, "SOURCE_REVISION", None)
        if not isinstance(revision, str) or not _COMMIT_SHA.fullmatch(revision):
            raise TimeFValidationError(
                f"{cls.__name__} source_revision must be a 40-character commit, got {revision!r}"
            )
        if not getattr(cls, "SOURCE_FILES", None):
            raise TimeFValidationError(f"{cls.__name__} source_files must declare at least one file")

    def metadata(self) -> DatasetMetadata:
        """Return the card, with ``source_url`` and ``source_revision`` from :attr:`HF_REPO` and :attr:`SOURCE_REVISION`.

        The card can omit both fields. If it sets one, the value must match the connector.

        Returns:
            The dataset's :class:`~timenet.types.DatasetMetadata`.

        Raises:
            TimeFValidationError: If the card sets ``source_url`` or ``source_revision`` to another value.
        """
        card = super().metadata()
        source = {
            "source_url": f"https://huggingface.co/datasets/{self.HF_REPO}",
            "source_revision": self.SOURCE_REVISION,
        }
        for field, value in source.items():
            declared = getattr(card, field)
            if declared is not None and declared != value:
                raise TimeFValidationError(
                    f"{type(self).__name__} card sets {field} to {declared!r}, but the connector reads {value!r}"
                )
        return card.model_copy(update=source)

    def check_refs(self, raw_refs: Sequence[HubFileRef]) -> None:
        """Check that ``raw_refs`` is nonempty and names each part once, only parts :attr:`SOURCE_FILES` declares.

        Args:
            raw_refs: The references :meth:`convert` received.

        Raises:
            TimeFValidationError: If ``raw_refs`` is empty, repeats a part, or names an undeclared part.
        """
        parts = [ref.part for ref in raw_refs]
        if not parts or len(set(parts)) != len(parts):
            raise TimeFValidationError(f"{type(self).__name__} requires nonempty, distinct part references")
        unknown = sorted(set(parts) - self.SOURCE_FILES.keys())
        if unknown:
            raise TimeFValidationError(f"{type(self).__name__} has no source part {', '.join(unknown)}")

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Check ``raw_refs``, then convert the files they name.

        Args:
            raw_refs: The references from :meth:`download_async`.

        Returns:
            The dataset from :meth:`convert_files`.

        Raises:
            TimeFValidationError: If ``raw_refs`` is empty, repeats a part, or names an undeclared part.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_refs
        self.check_refs(raw_refs)
        return self.convert_files({ref.part: ref.path for ref in raw_refs})

    @abstractmethod
    def convert_files(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Convert the downloaded files into a dataset.

        Args:
            files: The local path of each part: a file, or the directory of a pattern part.

        Returns:
            The populated dataset.
        """

    async def download_async(self, cache_dir: Path) -> list[HubFileRef]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION` concurrently.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One reference per part, in :attr:`SOURCE_FILES` order.

        Raises:
            ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # ruff: ignore[docstring-extraneous-exception] - also raised by hub_snapshot
        root = await hub_snapshot(self.HF_REPO, self.SOURCE_REVISION, cache_dir, tuple(self.SOURCE_FILES.values()))
        refs = []
        for part, name in self.SOURCE_FILES.items():
            path = root / name if "*" not in name else root / Path(name).parent
            if not path.exists():
                raise TimeNetDownloadError(f"{self.HF_REPO}@{self.SOURCE_REVISION} has no {name}")
            refs.append(HubFileRef(path, part))
        return refs
