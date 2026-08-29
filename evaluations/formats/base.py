"""The interface every format implements, and the artifact it produces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
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
    """One library's storage. It writes its own artifact and reads that artifact back.

    A format never reads another format's artifact. Parquet is not a rival to TimeF here; it is
    pandas' own on-disk format, in the same way ``.pt`` is torch's.
    """

    name: str

    def write(self, frame: pd.DataFrame, out: Path) -> Artifact:
        """Write the shared frame in this format.

        Args:
            frame: The one in-memory dataset, as produced by ``evaluations.source.load_frame``.
            out: A directory this format may write into. It is created if absent.

        Returns:
            The artifact written, carrying its size on disk.
        """
        ...

    def read_all(self, path: Path) -> np.ndarray:
        """Read every value back.

        Args:
            path: The artifact path returned by :meth:`write`.

        Returns:
            The signals, shaped ``(n_epochs, n_channels, n_samples)``.
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
