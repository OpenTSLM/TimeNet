"""Load a curated TimeF dataset and view it with Apache Arrow (no pandas needed).

Build the dataset first, then run this script::

    uv run timenet-curate build "chengsenwang/tsqa"
    uv run python examples/load_tsqa_arrow.py

``TimeNet().load(...)`` returns a ``TimeFDataset`` whose per-series values load lazily as Arrow arrays.
The catalog below is built from metadata only, so it never touches the value shards and scales to large
datasets; actual values are read for a single sample at the end. Pass a different dataset id as the first
argument to inspect another dataset.
"""

import sys

import pyarrow as pa
from rich.console import Console
from rich.table import Table as RichTable

from timenet.client import TimeNet
from timenet.dataset import Sample, TimeFDataset, TimeSeries
from timenet.types import QATask


def _annotation(sample: Sample, key: str) -> str | None:
    """Return the value of the sample's static annotation with this key.

    Args:
        sample: The sample to inspect.
        key: The annotation key to look up.

    Returns:
        The annotation's value as a string, or None if the sample has no annotation with that key.
    """
    return next((str(annotation.value) for annotation in sample.annotations if annotation.key == key), None)


def _first_qa(sample: Sample, tasks_by_id: dict[str, object]) -> QATask | None:
    """Return the sample's first QA task, or None.

    Args:
        sample: The sample whose tasks to scan.
        tasks_by_id: The dataset's tasks keyed by id.

    Returns:
        The first :class:`~timenet.types.QATask` attached to the sample, or None.
    """
    for task_id in sample.task_ids:
        task = tasks_by_id.get(task_id)
        if isinstance(task, QATask):
            return task
    return None


def _point_count(series: TimeSeries) -> int | None:
    """Return a series' number of points from its span metadata, without loading any values.

    Args:
        series: The series to measure.

    Returns:
        ``round((t_end_s - t_start_s) * sampling_rate_hz)``, or None if the series has no ``t_end_s``.
    """
    if series.t_end_s is None:
        return None
    return round((series.t_end_s - series.t_start_s) * series.sampling_rate_hz)


def catalog(dataset: TimeFDataset) -> pa.Table:
    """Build a one-row-per-sample Arrow table from metadata only (no series values are loaded).

    Columns are assembled directly with an explicit schema, and point counts come from each first
    channel's span, so the whole thing stays O(samples) with no per-sample disk reads.

    Args:
        dataset: The loaded dataset.

    Returns:
        An Arrow table with one row per sample.
    """
    tasks_by_id: dict[str, object] = {task.id: task for task in dataset.tasks}
    ids: list[str] = []
    tasks: list[str | None] = []
    questions: list[str | None] = []
    answers: list[str | None] = []
    channels: list[int] = []
    points: list[int | None] = []
    for sample in dataset.samples:
        qa = _first_qa(sample, tasks_by_id)
        head = qa.question.split("\n", 1)[0] if qa else None
        ids.append(sample.sample_id)
        tasks.append(_annotation(sample, "task"))
        questions.append(f"{head[:48]}…" if head and len(head) > 48 else head)
        answers.append(qa.answer if qa else None)
        channels.append(len(sample.time_series))
        points.append(_point_count(sample.time_series[0]))
    return pa.table(
        {
            "sample_id": pa.array(ids, pa.string()),
            "task": pa.array(tasks, pa.string()),
            "question": pa.array(questions, pa.string()),
            "answer": pa.array(answers, pa.string()),
            "channels": pa.array(channels, pa.int32()),
            "points": pa.array(points, pa.int64()),
        }
    )


def series_table(sample: Sample) -> pa.Table:
    """Build a ``(t_seconds, value)`` Arrow table for a sample's first channel.

    Args:
        sample: The sample whose first channel to tabulate.

    Returns:
        An Arrow table with time and value columns.
    """
    series = sample.time_series[0]
    values = series.to_arrow()
    times = pa.array([index / series.sampling_rate_hz for index in range(len(values))], type=pa.float64())
    return pa.table({"t_seconds": times, "value": values})


def render(table: pa.Table, title: str) -> None:
    """Print an Arrow table row-wise as a rich table.

    Args:
        table: The Arrow table to render (usually a small ``slice``).
        title: The heading shown above the table.
    """
    rich_table = RichTable(title=title, title_justify="left", title_style="bold")
    for name in table.column_names:
        rich_table.add_column(name)
    for row in table.to_pylist():
        rich_table.add_row(*(str(row[name]) for name in table.column_names))
    Console().print(rich_table)


def main() -> None:
    """Load the dataset and print an Arrow catalog plus one sample's series."""
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else "chengsenwang/tsqa"
    dataset = TimeNet().load(dataset_id)
    print(f"{dataset_id}: {len(dataset.samples)} samples, {len(dataset.tasks)} tasks")
    if not dataset.samples:
        print(f'no samples — build it first: uv run timenet-curate build "{dataset_id}"')
        return

    table = catalog(dataset)
    print(f"catalog is a {type(table).__module__}.{type(table).__name__}; schema:\n{table.schema}\n")
    render(table.slice(0, 8), f"{dataset_id} — first samples (of {table.num_rows})")
    render(series_table(dataset.samples[0]).slice(0, 10), "sample[0] series (first channel)")


if __name__ == "__main__":
    main()
