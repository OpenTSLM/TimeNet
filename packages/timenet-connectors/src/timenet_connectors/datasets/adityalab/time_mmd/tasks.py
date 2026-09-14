"""Build the tasks of one record from the series and the annotations it already carries.

Nothing here opens a file. ``connector.py`` builds the records, and every function here takes
values and gives tasks. The writer calls the stream more than once, so the caller passes a
callable that gives a fresh iterator each time.

Two questions are asked. The first is the one the release was built for: forecast the target
series from everything before an origin. The second is the one a report answers: describe
what the series did over the days the report covers. Both are stated with a ``scope`` on the
one record of the domain, so the record ships the raw series and no window is cut out of it.

No task carries a prompt. The release states no question in words.
"""

from collections.abc import Iterator, Sequence
from fractions import Fraction

from timenet.dataset import Record, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, AnswerTask, ForecastingTask, Task, TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd.domains import DomainShape
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, record_id_of


# The share of a series the Time-MMD protocol holds out as its test split, at the end. The
# Time-Series-Library convention that MM-TSFlib extends splits a series 70/10/20 in time order.
# Its code writes ``int(n * 0.2)`` for the size of the test split, and a fraction keeps that
# exact.
TEST_SHARE = Fraction(1, 5)


def test_origin(n_values: int) -> int:
    """Give the index of the first value of the test split, which is the first forecast origin.

    Args:
        n_values: How many values the series holds.

    Returns:
        The index. Every value from it on is a test target.
    """
    return n_values - n_values * TEST_SHARE.numerator // TEST_SHARE.denominator


def iter_forecasting_tasks(record_id: str, target: TimeSeries, horizons: Sequence[int]) -> Iterator[ForecastingTask]:
    """Give one forecasting task for each origin of the test split and each horizon.

    The context is everything the record holds from the first value of the target series to the
    origin: every series, and every annotation whose span lies in that stretch. A text before the
    first value stays in the record and lies outside every scope, because a scope that started
    before the series would name steps the series does not have. The target is the next
    ``horizon`` values of the target series alone, because that is the series the release names
    as the target. The other series are context, and a consumer who wants them forecast too
    widens the target span.

    An origin steps one value at a time through the test split, the way the protocol's sliding
    window does, and a horizon that would reach past the last value gets no task.

    Args:
        record_id: The record the tasks are about.
        target: The series to forecast.
        horizons: How many values each task predicts, one set of tasks for each.

    Yields:
        The tasks, by horizon and then by origin.

    Raises:
        TimeFFormatError: If the target series has no regular axis. A horizon counts values, and
            only a cadence turns that count into a span.
    """
    axis = target.time_axis
    if not isinstance(axis, RegularAxis):
        raise TimeFFormatError(f"{target.time_series_id}: has no regular cadence, so a horizon of values names no span")

    first_value = axis.time_offset_us(0)
    first = test_origin(target.n_values)
    for horizon in horizons:
        for origin in range(first, target.n_values - horizon + 1):
            start = axis.time_offset_us(origin)
            yield ForecastingTask(
                scope=TimeInterval.micros(first_value, start),
                target_span=TimeInterval.micros(
                    start, axis.time_offset_us(origin + horizon), time_series_ids=(target.time_series_id,)
                ),
                record_ids=(record_id,),
                id=f"{record_id}-forecast-h{horizon}-o{origin}",
            )


def iter_caption_tasks(record_id: str, target: TimeSeries, annotations: Sequence[Annotation]) -> Iterator[AnswerTask]:
    """Give one caption task for each report fact whose days hold a value of the target series.

    The task carries no prompt, so it is a caption: given the record from the first value of the
    target series to the end of the days the report covers, state what the report states. The
    answer is the annotation itself, by reference. A search fact gets no task, because a search result is often about something
    other than the series. A report whose days lie past the last value gets none either,
    because nothing can ask a record to describe values it does not hold.

    Args:
        record_id: The record the tasks are about.
        target: The series the record forecasts, whose window decides which reports get a task.
        annotations: The annotations of the record, from which the report facts are picked.

    Yields:
        The tasks, in the order of the annotations.

    Raises:
        TimeFFormatError: If the target series has no timeline, so no report can be placed
            against it.
    """
    window = target.span_us
    if window is None:
        raise TimeFFormatError(f"{target.time_series_id}: has no timeline, so no report can be placed against it")

    first, end = window
    for annotation in annotations:
        if annotation.key != AnnotationKey.REPORT_FACT or annotation.span is None:
            continue
        if annotation.span.exclusive_end <= first or annotation.span.start_us >= end:
            continue
        yield AnswerTask(
            scope=TimeInterval.micros(first, annotation.span.exclusive_end),
            target_annotation_ids=(annotation.id,),
            record_ids=(record_id,),
            id=f"{annotation.id}-caption",
        )


def find_target(record: Record, shape: DomainShape) -> TimeSeries:
    """Find the series of a record that holds the target the shape names.

    Args:
        record: The record of the domain.
        shape: What the domain holds, and which signal is its target.

    Returns:
        The target series.

    Raises:
        TimeFFormatError: If the record holds no series by that signal name, or more than one.
    """
    matches = [one for one in record.time_series if one.signal == shape.target]
    if len(matches) != 1:
        raise TimeFFormatError(
            f"{record.record_id}: holds {len(matches)} series named {shape.target!r}, and the target is one of them"
        )

    return matches[0]


def iter_tasks(records: Sequence[Record], shapes: Sequence[DomainShape]) -> Iterator[Task]:
    """Give every task of a build, one record at a time.

    Args:
        records: The records of the build, each carrying its series and annotations.
        shapes: The domains the records were built from, matched to them by record id.

    Yields:
        The forecasting tasks of each record, then its caption tasks.

    Raises:
        TimeFValidationError: If a record belongs to no domain of the shapes.
    """
    by_record_id = {record_id_of(shape.name): shape for shape in shapes}
    for record in records:
        shape = by_record_id.get(record.record_id)
        if shape is None:
            raise TimeFValidationError(f"record {record.record_id!r} belongs to no domain of this connector")

        target = find_target(record, shape)
        yield from iter_forecasting_tasks(record.record_id, target, shape.horizons)
        yield from iter_caption_tasks(record.record_id, target, record.annotations)
