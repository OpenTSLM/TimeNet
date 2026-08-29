"""The interface every format implements, and the artifact it produces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel


class Artifact(BaseModel):
    """What a format wrote, and what it cost on disk."""

    format: str
    """The name of the format that wrote it."""
    path: Path
    """The file or directory the format wrote."""
    size_bytes: int
    """Total bytes on disk, summed over every file when the artifact is a directory."""


class Format(Protocol):
    """One library's storage. It reads the release its own way, then writes and reads its artifact.

    Every format starts at the same directory of raw files and ends at values in memory. What
    happens in between is that library's own path, because that is the path its user would run.
    A format never reads another format's artifact. Parquet is not a rival to TimeF here; it is
    pandas' own on-disk format, in the same way ``.pt`` is torch's.

    The read of the release is inside :meth:`write` on purpose. An EDF file is not a format that
    pandas or torch can open, so a reference loader stands between the release and the frame, and
    what it costs is part of what that format costs.
    """

    name: str

    def write(self, source: Path, out: Path) -> Artifact:
        """Read the release and write it in this format.

        Args:
            source: The directory the release was extracted into. Every format reads the same one.
            out: A directory this format may write into. It is created if absent.

        Returns:
            The artifact written, carrying its size on disk.
        """
        ...

    def read_all(self, path: Path) -> list[np.ndarray]:
        """Read every value back.

        Args:
            path: The artifact path returned by :meth:`write`.

        Returns:
            One array for each run of values the format stores: an epoch for pandas and torch, a
            time series for TimeF. The arrays are not stacked, because a recording holds channels
            that were sampled at different rates and so differ in length.
        """
        ...


def directory_size(path: Path) -> int:
    """Total the bytes an artifact occupies.

    A directory reports the sum of every file beneath it, not the size of its own directory
    entry, which says nothing about the data.

    Args:
        path: A file or a directory.

    Returns:
        The size in bytes.
    """
    if path.is_file():
        return path.stat().st_size

    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
