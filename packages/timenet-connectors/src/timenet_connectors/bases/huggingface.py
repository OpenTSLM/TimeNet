"""A reusable base for connectors that read files from a pinned HuggingFace Hub dataset revision.

Subclasses pass ``hf_repo``, ``source_revision``, and ``source_files`` as class keywords, ship a
``dataset.yaml`` card beside the connector, and implement ``convert()``. ``download()`` returns a mapping
from each part in ``SOURCE_FILES`` to its local path. ``convert()`` receives that mapping. The base also
fills ``source_url`` and ``source_revision`` in the dataset metadata from ``HF_REPO`` and ``SOURCE_REVISION``.
:mod:`timenet_connectors.sources.huggingface_hub` describes how the files download. For the Hub's
parquet conversion, pin the commit ``refs/convert/parquet`` points to.
"""

from abc import ABC
from collections.abc import Mapping
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.types import DatasetMetadata
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class BaseHuggingFaceConnector(BaseConnector[Mapping[str, Path]], ABC):
    """Base class for connectors that read files from a pinned Hub dataset revision.

    Subclasses pass ``hf_repo``, ``source_revision``, and ``source_files`` as class keywords, and
    implement :meth:`convert`, which receives the local path of each part. The base stores the keywords
    as :attr:`HF_REPO`, :attr:`SOURCE_REVISION`, and :attr:`SOURCE_FILES`. A subclass that passes no
    keywords keeps the values of its parent.
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

    async def download_async(self, cache_dir: Path) -> dict[str, Path]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION` concurrently.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            The local path of each part, in :attr:`SOURCE_FILES` order: a file, or the directory of a pattern part.

        Raises:
            ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # ruff: ignore[docstring-extraneous-exception] - also raised by hub_snapshot
        root = await hub_snapshot(self.HF_REPO, self.SOURCE_REVISION, cache_dir, tuple(self.SOURCE_FILES.values()))
        files = {}
        for part, name in self.SOURCE_FILES.items():
            path = root / name if "*" not in name else root / Path(name).parent
            if not path.exists():
                raise TimeNetDownloadError(f"{self.HF_REPO}@{self.SOURCE_REVISION} has no {name}")
            files[part] = path
        return files
