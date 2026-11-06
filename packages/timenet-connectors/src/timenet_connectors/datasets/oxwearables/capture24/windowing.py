"""CAPTURE-24 segmentation, label runs, and benchmark window selection.

These functions operate on validated timestamps and raw label codes. Recording I/O stays in source.py.
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from timenet.errors import TimeNetConnectorError
from timenet.types import US_PER_S
from timenet_connectors.datasets.oxwearables.capture24.schemes import LABEL_SETS, Scheme


SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD_US = US_PER_S // SAMPLE_RATE_HZ
WINDOW_US = 10 * US_PER_S
_WINDOW_SAMPLES = WINDOW_US // SAMPLE_PERIOD_US
NO_LABEL = -1
"""The raw code of an unlabeled sample, written as an empty source field."""


@dataclass(frozen=True)
class Segment:
    """A run of samples exactly one period apart, between gaps."""

    first_row: int
    n_values: int
    offset_us: int
    """The time between the recording's first sample and this segment's first sample."""


@dataclass(frozen=True)
class LabelRuns:
    """A segment's runs of one raw label: consecutive samples with the same non-empty label."""

    starts_us: np.ndarray
    """The time between the recording's first sample and each run's first sample."""
    ends_us: np.ndarray
    """The end of each run, one sample period after its last sample. Each run covers ``[start, end)``."""
    codes: np.ndarray
    """Each run's raw label, as an index into the dictionary."""


@dataclass(frozen=True)
class Windows:
    """The windows kept from a recording."""

    first_rows: np.ndarray
    """Each kept window's first row."""
    classes: dict[Scheme, np.ndarray]
    """Each kept window's class per benchmarked scheme, as an index into that scheme's label set."""


def make_segments(timestamps_us: np.ndarray) -> tuple[Segment, ...]:
    """Split the samples wherever the step to the next timestamp is not exactly one sample period.

    Returns:
        The segments, in time order.
    """
    bounds = [0, *(np.flatnonzero(np.diff(timestamps_us) != SAMPLE_PERIOD_US) + 1).tolist(), len(timestamps_us)]
    return tuple(
        Segment(first_row=start, n_values=end - start, offset_us=int(timestamps_us[start] - timestamps_us[0]))
        for start, end in pairwise(bounds)
    )


def label_runs(raw_codes: np.ndarray, segment: Segment) -> LabelRuns:
    """Find a segment's runs of one raw label. Unlabeled samples belong to no run.

    Runs stay within the segment; sample positions determine their time offsets.

    Returns:
        The segment's runs, in time order.
    """
    codes = raw_codes[segment.first_row : segment.first_row + segment.n_values]
    changes = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(codes)]))
    labeled = codes[starts] != NO_LABEL
    starts, ends = starts[labeled], ends[labeled]
    return LabelRuns(
        starts_us=segment.offset_us + starts * SAMPLE_PERIOD_US,
        ends_us=segment.offset_us + ends * SAMPLE_PERIOD_US,
        codes=codes[starts],
    )


def _full_bin_starts(offsets_us: np.ndarray) -> np.ndarray:
    """Return the first row of each 10-second bin containing exactly 1,000 samples."""
    bins = offsets_us // WINDOW_US
    bin_starts = np.flatnonzero(np.diff(bins, prepend=bins[0] - 1))
    return bin_starts[np.diff(bin_starts, append=len(bins)) == _WINDOW_SAMPLES]


def _check_window_gaps(path: Path, first_rows: np.ndarray, segments: tuple[Segment, ...]) -> None:
    """Reject kept windows whose first and last samples belong to different segments.

    Raises:
        TimeNetConnectorError: If a kept window crosses a gap.
    """
    segment_first_rows = np.array([segment.first_row for segment in segments])
    first_segment = np.searchsorted(segment_first_rows, first_rows, side="right")
    last_segment = np.searchsorted(segment_first_rows, first_rows + _WINDOW_SAMPLES - 1, side="right")
    crossing = first_segment != last_segment
    if crossing.any():
        # Recording CSVs have one header line; row indices start at zero.
        line = int(first_rows[crossing][0]) + 2
        raise TimeNetConnectorError(
            f"{path}:{line}: the window that starts here holds {_WINDOW_SAMPLES} samples across a gap"
        )


def _majority_classes(window_codes: np.ndarray, class_indices: dict[Scheme, np.ndarray]) -> dict[Scheme, np.ndarray]:
    """Return the majority class per window and scheme; alphabetical order breaks ties."""
    classes = {}
    for scheme, class_index_by_raw_code in class_indices.items():
        sample_class_indices = class_index_by_raw_code[window_codes]
        # The unlabeled sentinel lies beyond the label set, so it contributes to no class count.
        counts = np.stack(
            [(sample_class_indices == index).sum(axis=1) for index in range(len(LABEL_SETS[scheme].labels))], axis=1
        )
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return classes


def make_windows(
    path: Path,
    offsets_us: np.ndarray,
    raw_codes: np.ndarray,
    segments: tuple[Segment, ...],
    class_indices: dict[Scheme, np.ndarray],
) -> Windows:
    """Keep full, labeled bins on the recording's original 10-second grid.

    Gaps do not reset the grid. A full bin can still cross a gap shorter than one extra sample
    period; reject it if it has labels. Each scheme votes over labeled samples only.

    Returns:
        The first row and class indices of each kept window.

    Raises:
        TimeNetConnectorError: If a kept window crosses a gap.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _check_window_gaps
    full_window_starts = _full_bin_starts(offsets_us)
    window_codes = raw_codes[full_window_starts[:, None] + np.arange(_WINDOW_SAMPLES)]
    has_label = (window_codes != NO_LABEL).any(axis=1)
    first_rows = full_window_starts[has_label]
    window_codes = window_codes[has_label]
    _check_window_gaps(path, first_rows, segments)
    return Windows(first_rows=first_rows, classes=_majority_classes(window_codes, class_indices))
