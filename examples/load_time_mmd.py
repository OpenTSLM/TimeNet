"""Load the built Time-MMD dataset and read one forecast and one caption through their spans.

Build the dataset first, then run this script::

    uv run timenet-build build adityalab/time-mmd
    uv run python examples/load_time_mmd.py

``TimeNet().load(...)`` returns a ``TimeFDataset`` of two records, one for each domain. Every task
is stated with a scope on its record, so this reads the context of a forecast and its ground truth
through ``step_range`` and ``read_steps``, and a caption's target through the annotation it names.
Pass a registry directory as the first argument to read a build made with ``--out``.
"""

from datetime import UTC, datetime, timedelta
import sys

from timenet.client import TimeNet
from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, ForecastingTask, TimeInterval, unix_us


# How many values of a context to print. A context reaches back to the start of the record.
_TAIL = 8


def _day(record: Record, offset_us: int) -> str:
    """Render a time offset of an anchored record as the calendar day it falls on.

    Returns:
        The ISO date.

    Raises:
        TimeFValidationError: If the record has no wall-clock anchor.
    """
    if record.start_time is None:
        raise TimeFValidationError(f"{record.record_id} has no start_time, so an offset names no day")
    anchor = datetime.fromtimestamp(unix_us(record.start_time) / 1_000_000, tz=UTC)
    return (anchor + timedelta(microseconds=offset_us)).date().isoformat()


def _forecast(dataset: TimeFDataset, record: Record) -> None:
    """Print the first forecasting task of a record: the end of its context and its ground truth.

    Raises:
        TimeFValidationError: If the task is not stated as time spans on its record.
    """
    task = next(
        one for one in dataset.tasks if isinstance(one, ForecastingTask) and one.record_ids == (record.record_id,)
    )
    if not isinstance(task.scope, TimeInterval) or not isinstance(task.target_span, TimeInterval):
        raise TimeFValidationError(f"{task.id} is not stated as time spans on its record")
    named = task.target_span.time_series_ids or ()
    target = next(one for one in record.time_series if one.time_series_id in named)
    start, stop = target.step_range(task.scope)
    context = target.read_steps(max(start, stop - _TAIL), stop).to_pylist()
    truth = target.read_steps(*target.step_range(task.target_span)).to_pylist()
    print(f"  forecast {task.id}")
    print(
        f"    context ends {_day(record, task.scope.end_us)}; its last {len(context)} values of {target.signal}: {context}"
    )
    print(f"    horizon {_day(record, task.target_span.start_us)} to {_day(record, task.target_span.end_us)}: {truth}")


def _caption(dataset: TimeFDataset, record: Record) -> None:
    """Print the first caption task of a record: where its context ends and the fact it asks for.

    Raises:
        TimeFValidationError: If the task is not stated as a time span on its record.
    """
    task = next(one for one in dataset.tasks if isinstance(one, AnswerTask) and one.record_ids == (record.record_id,))
    if not isinstance(task.scope, TimeInterval):
        raise TimeFValidationError(f"{task.id} is not stated as a time span on its record")
    fact = next(one for one in record.annotations if one.id == task.target_annotation_ids[0])
    print(f"  caption {task.id}")
    print(f"    context ends {_day(record, task.scope.end_us)}; target: {str(fact.value)[:150]}")


def main() -> None:
    """Load the dataset, print its describe() summary, and show one task of each kind per record."""
    registry = sys.argv[1] if len(sys.argv) > 1 else None
    dataset = TimeNet(registry=registry).load("adityalab/time-mmd")
    dataset.describe()
    for record in dataset.records:
        print(f"\n{record.record_id}")
        _forecast(dataset, record)
        _caption(dataset, record)


if __name__ == "__main__":
    main()
