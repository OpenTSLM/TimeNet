"""Build the TSAQA Records from checked rows, and the dataset annotations that the task stream uses."""

from collections.abc import Sequence

from timenet.dataset import Record, Source, TimeFDataset
from timenet.json import JsonValue
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import INPUT_ROLES
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import Row, SeriesLoader


def _records(row: Row, loader: SeriesLoader) -> tuple[Record, ...]:
    """Build a row's Records in the order its task gives them to a model, without reading any values.

    Returns:
        The row's Records.
    """
    roles = (*INPUT_ROLES[: row.n_inputs], *row.printed_roles)
    records = []
    for position, (role, signal) in enumerate(zip(roles, loader.signals(row), strict=True)):
        metadata: dict[str, JsonValue] = {"file": row.file, "row": row.index, "role": str(role)}
        record = Record(sources=(Source(name="series", signals=(signal,)),), metadata=metadata)
        if position == 0:
            record.add_annotations(
                {key: value if isinstance(value, str) else list(value) for key, value in row.contexts.on_input.items()}
            )
        records.append(record)
    return tuple(records)


def add_records(dataset: TimeFDataset, rows: Sequence[Row], loader: SeriesLoader) -> list[tuple[Record, ...]]:
    """Build the Records of every row and add them to the dataset, in row order.

    Returns:
        The Records of each row, in the same order as ``rows``.
    """
    records = [_records(row, loader) for row in rows]
    for row_records in records:
        for record in row_records:
            dataset.add_record(record=record)
    return records
