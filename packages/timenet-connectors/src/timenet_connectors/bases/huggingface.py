"""A reusable base for connectors that read files from a pinned HuggingFace Hub dataset revision.

Subclasses set ``HF_REPO``, ``SOURCE_REVISION``, and ``SOURCE_FILES``, ship a ``dataset.yaml`` card
beside the connector, and implement ``convert()`` over the :class:`HubFileRef` list ``download()``
returns. The base checks that list before the subclass's ``convert()`` runs. It also fills
``source_url`` and ``source_revision`` in the dataset metadata from ``HF_REPO`` and ``SOURCE_REVISION``.
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
from timenet.errors import TimeFValidationError, TimeNetConnectorError, TimeNetDownloadError
from timenet.types import DatasetMetadata
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class HubFileRef:
    """A downloaded file (or a pattern's directory) and the part it belongs to."""

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
    """The repo-relative filename of each part, in download-order. A filename can end
    in a glob like ``data/*.parquet``; the part's path is then the directory of the matching files."""

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

    def ref_paths(self, raw_refs: Sequence[HubFileRef], *parts: str) -> dict[str, Path]:
        """Map each part in ``raw_refs`` to its local path.

        Args:
            raw_refs: The references ``convert()`` received.
            *parts: The parts the caller needs.

        Returns:
            The path of each part: a file, or the directory of a pattern part.

        Raises:
            TimeNetConnectorError: If one of ``parts`` is not in ``raw_refs``.
        """
        paths = {ref.part: ref.path for ref in raw_refs}
        missing = [part for part in parts if part not in paths]
        if missing:
            raise TimeNetConnectorError(f"{self.HF_REPO} refs lack {', '.join(missing)}")
        return paths

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
