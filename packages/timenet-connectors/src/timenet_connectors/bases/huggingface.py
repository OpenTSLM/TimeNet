"""A reusable base for connectors that read files from a pinned HuggingFace Hub dataset revision.

Subclasses set ``HF_REPO``, ``SOURCE_REVISION``, and ``SOURCE_FILES``, ship a ``dataset.yaml`` card
beside the connector, and implement ``convert()`` over the :class:`HubFileRef` list ``download()``
returns. The base checks that list before the subclass's ``convert()`` runs.
:mod:`timenet_connectors.sources.huggingface_hub` describes how the files download. For the Hub's
parquet conversion, pin the commit ``refs/convert/parquet`` points to.
"""

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import functools
from pathlib import Path
import re
from typing import Any, ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet_connectors.sources.huggingface_hub import download_hub_files


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class HubFileRef:
    """One local source file and the part of the dataset it holds."""

    path: Path
    part: str
    """The connector's own name for this file, such as ``"hhar_v2"``: the key in ``SOURCE_FILES``, which
    ``convert()`` branches on and builds ids from."""


class BaseHuggingFaceConnector(BaseConnector[HubFileRef], ABC):
    """Base class for connectors that read files from a pinned Hub dataset revision.

    Subclasses set :attr:`HF_REPO`, :attr:`SOURCE_REVISION`, and :attr:`SOURCE_FILES`, and implement
    ``convert()`` over the returned :class:`HubFileRef` list. The base wraps each subclass's ``convert()``
    to run :meth:`check_refs` first.
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
        # Every subclass's convert() checks its refs first, so no connector has to call check_refs itself
        convert = cls.__dict__.get("convert")
        if convert is not None:

            @functools.wraps(convert)
            def checked(self: BaseHuggingFaceConnector, raw_refs: list[HubFileRef]) -> TimeFDataset:
                self.check_refs(raw_refs)
                return convert(self, raw_refs)

            cls.convert = checked

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

    async def download_async(self, cache_dir: Path) -> list[HubFileRef]:
        """Download :attr:`SOURCE_FILES` at :attr:`SOURCE_REVISION` concurrently.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One reference per part, in :attr:`SOURCE_FILES` order.

        Raises:
            ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
            TimeNetDownloadError: If a declared file is absent from the revision or cannot be downloaded.
        """  # noqa: DOC502 - both are raised by download_hub_files
        paths = await download_hub_files(
            self.HF_REPO, list(self.SOURCE_FILES.values()), revision=self.SOURCE_REVISION, cache_dir=cache_dir
        )
        return [HubFileRef(path, part) for part, path in zip(self.SOURCE_FILES, paths, strict=True)]
