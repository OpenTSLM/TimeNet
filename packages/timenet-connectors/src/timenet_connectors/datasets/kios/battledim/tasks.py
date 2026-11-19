"""Cut the recording into two-week windows and state what the leak list says about each one.

A window is 14 days: a context week, then a target week. It starts every 7 days, so consecutive
windows share a week. Each window gives four tasks about the target week, all scoped to the window:
the types of the leaks that start in it, when they start, the mean leak outflow, and a short text
that describes every leak that starts or is repaired in it. The answers come from the leak list,
the leak tables and the network model by fixed rules. No model writes them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from itertools import pairwise

from timenet.dataset import Record
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping, JsonValue
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
)
from timenet.types.splits import Split
from timenet_connectors.datasets.kios.battledim.source import STEP, Leak, Release
from timenet_connectors.splitting import StratifiedSplitter


WINDOW = timedelta(days=14)
TARGET = timedelta(days=7)
STRIDE = timedelta(days=7)
# Four ISO weeks make one calendar block. Windows whose target weeks share a block share a partition
# in both years, because 2019 repeats most of the 2018 demand series (see README.md).
WEEKS_PER_BLOCK = 4
# Below this outflow, in m3/h, the text keeps the leak tables' second decimal.
_SMALL_OUTFLOW = 0.1

NEW_LEAK_TYPES = LabelSet(
    name="new_leak_types",
    labels=(
        Label(name="abrupt", description="A leak that appears at full size at its start."),
        Label(name="incipient", description="A leak that grows from its start until its peak time."),
    ),
    multi_label=True,
)
EXPLAIN_PROMPT = (
    "Describe every leak that started or was repaired in the last 7 days of this two-week window: "
    "the pipe, when, how it grew, its size, and the nearest pressure sensor."
)
LOCATE_PROMPT = "Locate the start of every leak that began in the last 7 days of this two-week window."
CLASSIFY_PROMPT = "Which types of leak began in the last 7 days of this two-week window?"
OUTFLOW_PROMPT = "What was the mean total leak outflow in the last 7 days of this two-week window?"


@dataclass(frozen=True)
class Window:
    """One two-week window and what the leak list says about its target week."""

    index: int
    start: datetime
    new: tuple[Leak, ...]
    repaired: tuple[Leak, ...]

    @property
    def target_start(self) -> datetime:
        """The first step of the target week."""
        return self.start + WINDOW - TARGET

    @property
    def end(self) -> datetime:
        """The exclusive end of the window."""
        return self.start + WINDOW

    @property
    def block(self) -> int:
        """The calendar block of the target week: its ISO week, counted in blocks of four."""
        return (self.target_start.isocalendar().week - 1) // WEEKS_PER_BLOCK


def windows(release: Release) -> list[Window]:
    """List every window that fits in the recording, in time order.

    Returns:
        The windows, each with the leaks that start or are repaired in its target week.

    Raises:
        TimeNetConnectorError: If a leak starts or is repaired outside every target week, so that no
            task would mention it.
    """
    found = []
    start = release.start
    while start + WINDOW <= release.end + STEP:
        frame = Window(index=len(found), start=start, new=(), repaired=())
        found.append(
            replace(
                frame,
                new=tuple(leak for leak in release.leaks if frame.target_start <= leak.start < frame.end),
                # The release writes a repair time as the leak's configured end.
                repaired=tuple(
                    leak for leak in release.leaks if leak.repaired and frame.target_start <= leak.end < frame.end
                ),
            )
        )
        start += STRIDE
    for leak in release.leaks:
        if sum(leak in window.new for window in found) != 1:
            raise TimeNetConnectorError(f"leak {leak.pipe} does not start in exactly one target week")
        if leak.repaired and sum(leak in window.repaired for window in found) != 1:
            raise TimeNetConnectorError(f"the repair of leak {leak.pipe} is not in exactly one target week")
    return found


def split_windows(found: list[Window]) -> dict[int, Split]:
    """Assign each window to a partition by the calendar block of its target week.

    Returns:
        The partition of each window, keyed by window index.
    """
    splitter = StratifiedSplitter(
        train_folds=8,
        validation_folds=1,
        test_folds=1,
        stratify_by=lambda window: bool(window.new),
        group_by=lambda window: window.block,
        seed=42,
    )
    return {window.index: split for split, members in splitter.split(found).items() for window in members}


def embargoed(found: list[Window], split_of: dict[int, Split]) -> set[int]:
    """Find the windows whose context week is a target week of another held-out partition.

    Windows overlap by one week: a window's context week is the target week of the window before
    it. A training window after a test target week would show training the signals that the test
    tasks ask about, so it gets no tasks. The same holds between validation and test.

    Returns:
        The indices of the windows that get no tasks.
    """
    held_out = {Split.VALIDATION, Split.TEST}
    return {
        window.index
        for previous, window in pairwise(found)
        if split_of[previous.index] in held_out and split_of[previous.index] != split_of[window.index]
    }


def _rate(outflow: float) -> str:
    """Format an outflow in m3/h so that the leak tables' smallest values stay visible."""
    return f"{outflow:.2f}" if outflow < _SMALL_OUTFLOW else f"{outflow:.1f}"


def _outflow(release: Release, leak: Leak, start: datetime, end: datetime) -> float:
    """The largest outflow of ``leak`` in ``[start, end)``, in m3/h."""
    return float(release.leak_outflow[leak.pipe][release.step_of(start) : release.step_of(end)].max())


def _hops(count: int) -> str:
    return f"{count} link{'s' if count != 1 else ''}"


def explain(release: Release, window: Window) -> str:
    """Write the text answer for one window from the leak list, the leak tables and the network.

    Returns:
        One sentence per new or repaired leak, then the leaks still active at the end of the window.
    """
    sentences = []
    for leak in window.new:
        sensor, hops = release.network.nearest(leak.pipe, release.sensors["pressure"])
        if leak.leak_type == "abrupt":
            kind, growth = "A sudden leak", ""
        elif leak.peak < window.end:
            kind, growth = "A gradually growing leak", f" and peaked on {leak.peak:%Y-%m-%d %H:%M}"
        else:
            # A later peak is not in the window, so the text does not state it.
            kind, growth = "A gradually growing leak", " and was still growing at the end of this window"
        outflow = _outflow(release, leak, leak.start, window.end)
        # The leak tables round an incipient leak's first hours to zero outflow.
        flow = f"an outflow of up to {_rate(outflow)} m3/h" if outflow > 0 else "no outflow in the leak tables yet"
        sentences.append(
            f"{kind} started on pipe {leak.pipe} on {leak.start:%Y-%m-%d %H:%M}{growth}, "
            f"with a hole of {leak.diameter_m * 1000:.1f} mm and {flow} in this window. "
            f"The nearest pressure sensor is {sensor}, {_hops(hops)} away."
        )
    for leak in window.repaired:
        last_hour = leak.end + STEP - timedelta(hours=1)
        sentences.append(
            f"The leak on pipe {leak.pipe} was repaired on {leak.end:%Y-%m-%d %H:%M}; "
            f"its outflow before the repair was {_outflow(release, leak, last_hour, leak.end + STEP):.1f} m3/h."
        )
    if not sentences:
        sentences.append("No leak started or was repaired in the last 7 days.")
    # Active means the leak list's interval holds the last step, whatever the rounded outflow says.
    last = window.end - STEP
    active = [leak.pipe for leak in release.leaks if leak.start <= last <= leak.end]
    total = sum(float(release.leak_outflow[pipe][release.step_of(last)]) for pipe in active)
    sentences.append(
        f"At the end of the window {len(active)} leak{'s are' if len(active) != 1 else ' is'} active, "
        f"with a total outflow of {total:.1f} m3/h."
    )
    return " ".join(sentences)


def window_tasks(release: Release, record: Record, window: Window, split: Split) -> list[Task]:
    """Build the four tasks of one window, all scoped to the window.

    Returns:
        The classification, localization, outflow and text tasks, in that order.
    """
    scope = TimeInterval.micros(release.micros(window.start), release.micros(window.end))
    target = slice(release.step_of(window.target_start), release.step_of(window.end))
    outflow = release.total_outflow[target]
    new_leaks: list[JsonValue] = [leak.pipe for leak in window.new]
    repaired_leaks: list[JsonValue] = [leak.pipe for leak in window.repaired]
    metadata: JsonMapping = {
        "window_index": window.index,
        "window_start": f"{window.start:%Y-%m-%d %H:%M}",
        "target_week_start": f"{window.target_start:%Y-%m-%d %H:%M}",
        "calendar_block": window.block,
        "new_leaks": new_leaks,
        "repaired_leaks": repaired_leaks,
    }
    inputs = (record,)
    return [
        ClassificationTask(
            inputs=inputs,
            scope=scope,
            split=split,
            metadata=metadata,
            prompt=CLASSIFY_PROMPT,
            targets=tuple(sorted({leak.leak_type for leak in window.new})),
            label_set=NEW_LEAK_TYPES,
        ),
        TemporalLocalizationTask(
            inputs=inputs,
            scope=scope,
            split=split,
            metadata=metadata,
            prompt=LOCATE_PROMPT,
            mode=LocalizationMode.EXHAUSTIVE,
            targets=tuple(TimePoint.micros(release.micros(leak.start)) for leak in window.new),
        ),
        ScalarPredictionTask(
            inputs=inputs,
            scope=scope,
            split=split,
            metadata=metadata,
            prompt=OUTFLOW_PROMPT,
            targets=(round(float(outflow.mean()), 3),),
            unit="m^3/h",
            target_name="mean_leak_outflow",
        ),
        AnswerTask(
            inputs=inputs,
            scope=scope,
            split=split,
            metadata=metadata,
            prompt=EXPLAIN_PROMPT,
            targets=(explain(release, window),),
        ),
    ]
