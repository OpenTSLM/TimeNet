"""Calendar fields of a pandas DatetimeIndex, read through one typed helper.

pandas generates ``DatetimeIndex.month`` and its siblings at runtime, so a static type checker cannot
see them. Reading them here keeps the rest of the code free of type-checker suppressions.
"""

from operator import methodcaller

import numpy as np
import pandas as pd


def field(index: pd.Index, name: str) -> np.ndarray:
    """Read one calendar field of a DatetimeIndex.

    Args:
        index: A DatetimeIndex.
        name: ``year``, ``month``, ``day``, ``hour``, ``day_of_week`` or ``day_of_year``.

    Returns:
        The field, one integer per entry of the index.
    """
    return np.asarray(getattr(pd.DatetimeIndex(index), name))


def floor_hour(index: pd.Index) -> pd.DatetimeIndex:
    """Round each timestamp of a DatetimeIndex down to its hour.

    Returns:
        The rounded index.
    """
    return pd.DatetimeIndex(methodcaller("floor", "h")(pd.DatetimeIndex(index)))


def normalize(index: pd.Index) -> pd.DatetimeIndex:
    """Set each timestamp of a DatetimeIndex to midnight of its day.

    Returns:
        The normalized index.
    """
    return pd.DatetimeIndex(methodcaller("normalize")(pd.DatetimeIndex(index)))
