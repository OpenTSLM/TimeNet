"""What the release's card states about each of the eleven evaluation folders.

The folder fixes a record's shape: how many signals, at what rate, over what vocabulary. None of
that is in the parquet files, so it comes from the card's table and lives here as data.

The rate is the one fact the data does not state anywhere. It is used as the card gives it, even
where multiplying it by the measured window length gives an odd duration. Three folders do; the
README says which, and says so rather than substituting a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Folder:
    """One evaluation folder, as the release's card describes it."""

    name: str  # the directory name, which is also the task's target_schema
    rate_hz: Fraction  # the sampling rate the card states; the data states none
    signals: tuple[str, ...]  # one name per signal, in the order the X column stores them
    classes: int  # how many classes the card states; the build checks the data against it


def _numbered(prefix: str, count: int) -> tuple[str, ...]:
    """Give ``count`` signal names for a folder whose card names no individual channel.

    Args:
        prefix: What the card calls the sensor, e.g. ``acc``.
        count: How many signals the folder ships.

    Returns:
        One name per signal, numbered from zero.
    """
    return tuple(f"{prefix}{index}" for index in range(count))


FOLDERS: tuple[Folder, ...] = (
    Folder("AsphaltObstacles", Fraction(100), ("acc_magnitude",), 4),
    # "Hourly" and "Minute" in the card, written as the rates they are.
    Folder("Beijing_AQI", Fraction(1, 3600), _numbered("env", 7), 4),
    Folder("PPG_CVA", Fraction(65), ("ppg",), 2),
    Folder("PPG_DM", Fraction(65), ("ppg",), 2),
    Folder("PPG_HTN", Fraction(65), ("ppg",), 4),
    Folder("ptbxl", Fraction(100), ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"), 5),
    Folder("sleepEDF", Fraction(100), ("eeg0", "eeg1"), 5),
    Folder("studentlife", Fraction(1, 60), _numbered("sensor", 10), 3),
    Folder("uci_har", Fraction(50), (*_numbered("acc", 3), *_numbered("gyro", 3)), 7),
    Folder("wesad", Fraction(700), _numbered("ch", 13), 3),
    Folder("wisdm", Fraction(30), ("acc_x", "acc_y", "acc_z"), 18),
)
"""Every folder the release ships, in the order the connector walks them."""

PPG_FOLDERS: tuple[str, ...] = ("PPG_CVA", "PPG_DM", "PPG_HTN")
"""The three folders that hold the same 650 windows under three diagnoses. One record each, three
tasks, rather than the same values stored three times."""

BY_NAME: dict[str, Folder] = {folder.name: folder for folder in FOLDERS}
"""Every folder, keyed by its directory name."""
