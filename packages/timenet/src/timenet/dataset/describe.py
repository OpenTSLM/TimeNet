"""Plain-text summary of a :class:`~timenet.dataset.TimeFDataset` (the ``describe()`` implementation).

Kept out of ``dataset.py`` to keep the model lean. Uses only the dataset's public surface and stdlib, so
it pulls in no CLI/rich dependency. Counts are recomputed from the in-memory samples/tasks (a loaded
dataset drops the manifest's counts block); the sample preview reads only span metadata, and value dtypes
are sampled from one series per spec.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.dataset.dataset import TimeFDataset
    from timenet.dataset.time_series import TimeSeries


def point_count(series: TimeSeries) -> int:
    """Return a series' value count, without loading values.

    Args:
        series: The series to measure.

    Returns:
        The series' value count.
    """
    return series.n_values


def describe_text(dataset: TimeFDataset, *, rows: int) -> str:
    """Build the multi-section plain-text summary for :meth:`TimeFDataset.describe`.

    Args:
        dataset: The dataset to summarize.
        rows: Number of samples to show in the preview.

    Returns:
        The formatted summary string.
    """
    samples = dataset.samples
    unique_series = {ts.time_series_id: ts for sample in samples for ts in sample.time_series}

    blocks = [
        _identity(dataset),
        _counts(dataset, unique_series),
        _specs(unique_series),
        _preview(dataset, rows),
    ]
    return "\n\n".join(block for block in blocks if block)


def _identity(dataset: TimeFDataset) -> str:
    meta = dataset.metadata
    lines = [f"{meta.dataset_id} @ {meta.dataset_version}"]
    fields = {"name": meta.name, "license": str(meta.license)}
    if meta.domains:
        fields["domains"] = ", ".join(str(domain) for domain in meta.domains)
    if meta.tags:
        fields["tags"] = ", ".join(meta.tags)
    width = max(len(key) for key in fields)
    lines += [f"  {key.ljust(width)}  {value}" for key, value in fields.items()]
    return "\n".join(lines)


def _counts(dataset: TimeFDataset, unique_series: dict[str, TimeSeries]) -> str:
    task_counts = Counter(str(task.task_type) for task in dataset.tasks)
    annotation_ids = {ann.id for sample in dataset.samples for ann in sample.annotations}
    spec_counts = Counter(ts.spec.spec_type for ts in unique_series.values())
    fields = {
        "samples": str(len(dataset.samples)),
        "series": _histogram(spec_counts),
        "annotations": str(len(annotation_ids)),
        "tasks": _histogram(task_counts),
    }
    width = max(len(key) for key in fields)
    lines = ["counts"] + [f"  {key.ljust(width)}  {value}" for key, value in fields.items()]
    return "\n".join(lines)


def _specs(unique_series: dict[str, TimeSeries]) -> str:
    if not unique_series:
        return ""
    representative: dict[str, TimeSeries] = {}
    for series in unique_series.values():
        representative.setdefault(series.spec.spec_type, series)
    header = ("spec", "name", "value", "dtype")
    table_rows = [
        (
            spec_type,
            series.spec.name,
            str(series.spec.unit_value),
            str(series.to_arrow().type),  # one bounded load per spec, for the value dtype
        )
        for spec_type, series in sorted(representative.items())
    ]
    return "specs\n" + _fixed_width(header, table_rows)


def _preview(dataset: TimeFDataset, rows: int) -> str:
    samples = dataset.samples
    if not samples:
        return ""
    header = ("sample_id", "channels", "length", "tasks", "annotations")
    table_rows = []
    for sample in samples[:rows]:
        length = point_count(sample.time_series[0]) if sample.time_series else None
        table_rows.append(
            (
                sample.sample_id,
                str(len(sample.time_series)),
                "?" if length is None else str(length),
                str(len(sample.task_ids)),
                str(len(sample.annotations)),
            )
        )
    heading = f"samples (first {min(rows, len(samples))} of {len(samples)})"
    return f"{heading}\n" + _fixed_width(header, table_rows)


def _histogram(counter: Counter[str]) -> str:
    """Render a counter as ``a=1, b=2`` sorted by key, or ``-`` when empty.

    Args:
        counter: The counts to render.

    Returns:
        The ``key=count`` pairs joined by commas, or ``-`` if empty.
    """
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter)) or "-"


def _fixed_width(header: tuple[str, ...], table_rows: Sequence[tuple[str, ...]]) -> str:
    """Render a header + rows as a 2-space-padded fixed-width table, indented two spaces.

    Args:
        header: The column headers.
        table_rows: The data rows.

    Returns:
        The formatted table text.
    """
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *table_rows, strict=True)]
    lines = [header, *table_rows]
    return "\n".join(
        "  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in lines
    )
