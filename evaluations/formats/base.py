"""The interface every format implements."""

from __future__ import annotations

from enum import StrEnum, unique
from pathlib import Path
from typing import Protocol

import numpy as np


@unique
class FormatName(StrEnum):
    """The formats under comparison. One member for each library that stores the dataset."""

    # Parquet, the on-disk format of pandas.
    PANDAS = "pandas"
    # A .pt file, the on-disk format of torch.
    TORCH = "torch"
    # A TimeF version, as the connector of the dataset builds it.
    TIMEF = "timef"


class Format(Protocol):
    """One library's storage. It reads the release its own way, then writes and reads it back.

    Every format starts at the same directory of raw files and ends at values in memory. Between
    those two points it runs its own library's path, because that is the path its user would run.
    No format reads another format's artifact. Parquet is not a rival to TimeF. It is the on-disk
    format of pandas, in the same way that ``.pt`` is the one of torch.

    The read of the release is inside :meth:`write` on purpose. Neither pandas nor torch can open
    an EDF file, so a loader stands between the release and the frame. What that loader costs is
    part of what the format costs.
    """

    name: FormatName

    def write(self, source: Path, out: Path) -> Path:
        """Read the release and write it in this format.

        A format states no size. It gives back what it wrote, and the caller measures that, so
        one rule covers all three and no format can claim a size its own files disagree with.

        Args:
            source: The directory the release was extracted into. Every format reads the same one.
            out: A directory this format may write into. It is created if absent.

        Returns:
            The file or directory written.
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
