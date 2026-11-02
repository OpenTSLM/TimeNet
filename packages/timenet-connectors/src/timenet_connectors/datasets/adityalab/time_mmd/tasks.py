"""Forecasting tasks over one record, following the protocol the Time-MMD paper evaluates with.

The paper's library, MM-TSFlib, inherits the data loader of the Time-Series-Library. That loader
cuts a series into three chronological stretches, 70 % of the steps for training, 10 % for
validation and 20 % for testing, and slides a window of ``lookback + horizon`` steps over each
stretch with a stride of one step. The validation and the test stretch each begin ``lookback``
steps early, so that the first window of each has the end of the stretch before it as context.
A window belongs to the stretch that holds it whole, so no target step lies in two splits.

Each window is one :class:`~timenet.types.ForecastingTask`. Its ``scope`` is the context interval
over every signal of the record, so a model sees the other columns as covariates alongside the
target. Its one target is the horizon interval scoped to the target signal alone, which is the
only column the paper forecasts. Its ``input_annotations`` are the record's text annotations that
overlap the context interval and end at or before the forecast boundary, so no text about the
horizon reaches the model. The tasks are generated and never held as a list: a daily series yields
tens of thousands of them.
"""

from bisect import bisect_right
from collections.abc import Iterable, Iterator

from timenet.dataset import Record, Signal
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ForecastingTask, Split, TimeInterval


# The chronological shares of the Time-Series-Library loader. Validation takes what is left.
TRAIN_FRACTION = 0.7
TEST_FRACTION = 0.2


def split_stretches(n_steps: int, lookback: int) -> dict[Split, tuple[int, int]]:
    """Cut a series into the three chronological stretches a window may lie in.

    Each stretch is a half-open range of steps. The training stretch starts at the first step.
    The validation and the test stretch each start ``lookback`` steps before their own share
    begins, so that their first window's context reaches back into the stretch before.

    Args:
        n_steps: How many steps the series holds.
        lookback: The context length of a window, in steps.

    Returns:
        The stretch of each split.

    Raises:
        TimeFValidationError: If the training share is shorter than the context, so that the
            validation stretch would start before the series does.
    """
    n_train = int(n_steps * TRAIN_FRACTION)
    n_test = int(n_steps * TEST_FRACTION)
    if n_train < lookback:
        raise TimeFValidationError(
            f"a series of {n_steps} steps puts {n_train} in the training share, fewer than the "
            f"{lookback} steps of context a window needs"
        )
    return {
        Split.TRAIN: (0, n_train),
        Split.VALIDATION: (n_train - lookback, n_steps - n_test),
        Split.TEST: (n_steps - n_test - lookback, n_steps),
    }


def iter_windows(n_steps: int, lookback: int, horizon: int) -> Iterator[tuple[Split, int]]:
    """Slide a window over each stretch and give the split and first step of every position.

    Args:
        n_steps: How many steps the series holds.
        lookback: The context length of a window, in steps.
        horizon: The forecast length of a window, in steps.

    Yields:
        The split and the first context step of each window, split by split and then in step order.
    """
    for split, (first, last) in split_stretches(n_steps, lookback).items():
        # the last window is the one that ends at the last step of the series
        for start in range(first, last - lookback - horizon + 1):
            yield split, start


class ContextIndex:
    """The interval annotations of a record, ordered by their end, for lookup by context window.

    A window may use an annotation that overlaps its context interval and ends at or before its
    forecast boundary. An annotation that runs past the boundary tells about the horizon, so it is
    left out even where it starts inside the context. Ordering by end makes each window's
    annotations one contiguous slice, found by bisection.
    """

    def __init__(self, annotations: Iterable[Annotation]) -> None:
        ended = [
            (annotation.span.end_us, annotation)
            for annotation in annotations
            if isinstance(annotation.span, TimeInterval)
        ]
        ended.sort(key=lambda pair: pair[0])
        self._ends = [end_us for end_us, _ in ended]
        self._annotations = tuple(annotation for _, annotation in ended)

    def within(self, start_us: int, boundary_us: int) -> tuple[Annotation, ...]:
        """Return the annotations a window may use, in order of their end.

        Args:
            start_us: The start of the context interval, relative to the record origin.
            boundary_us: The end of the context interval and start of the horizon.

        Returns:
            The annotations that end after ``start_us`` and at or before ``boundary_us``.
        """
        return self._annotations[bisect_right(self._ends, start_us) : bisect_right(self._ends, boundary_us)]


def iter_forecasting_tasks(
    record: Record, signal: Signal, *, lookback: int, horizons: tuple[int, ...]
) -> Iterator[ForecastingTask]:
    """Give one forecasting task for each window of each horizon of one target signal.

    The scope covers every signal of the record over the context interval; the target names the
    target signal alone over the horizon interval. The input annotations are the record's interval
    annotations that overlap the context interval and end at or before the forecast boundary.

    Args:
        record: The record the signal belongs to, which the task names as its one input.
        signal: The target signal, on a regular axis. Its cadence counts the steps.
        lookback: The context length of a window, in steps.
        horizons: The forecast lengths to window with, in steps.

    Yields:
        One task per window, horizon by horizon.

    Raises:
        TimeFValidationError: If the signal has no regular cadence to count steps on.
    """
    axis = signal.time_axis
    if not isinstance(axis, RegularAxis):
        raise TimeFValidationError(f"signal {signal.name!r} needs a regular cadence to window in steps")
    context = ContextIndex(record.annotations)
    for horizon in horizons:
        for split, start in iter_windows(signal.n_values, lookback, horizon):
            boundary = start + lookback
            start_us, boundary_us = axis.time_offset_us(start), axis.time_offset_us(boundary)
            yield ForecastingTask(
                inputs=(record,),
                split=split,
                # the scope covers every signal of the record over the context interval
                scope=TimeInterval.micros(start_us, boundary_us),
                # the text that overlaps the context and ends at or before the boundary
                input_annotations=context.within(start_us, boundary_us),
                # the target names the target signal alone over the horizon interval
                targets=(
                    TimeInterval.micros(boundary_us, axis.time_offset_us(boundary + horizon), time_series=(signal,)),
                ),
                metadata={"context_steps": lookback, "horizon_steps": horizon, "first_context_step": start},
            )
