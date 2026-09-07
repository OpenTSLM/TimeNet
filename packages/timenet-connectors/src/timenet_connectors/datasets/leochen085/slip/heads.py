"""Readable heads of every kind of file the SLIP pretraining corpus ships.

A head opens one file, reads a small part of it, and gives that part back as text. It writes
nothing. Run these against a new release of the dataset and a changed shape shows at once.

The corpus is parquet, so the bounded unit of a read is a record batch, not a file. Every function
here asks the reader for the rows it prints and nothing more.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq


if TYPE_CHECKING:
    import pyarrow as pa

_CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")


def _clip(text: str, width: int = 88) -> str:
    """Give ``text`` on one line, cut to ``width`` characters with an ellipsis.

    Args:
        text: The value to show.
        width: How many characters to keep.

    Returns:
        A single line no longer than ``width`` plus the ellipsis.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= width else f"{flat[:width]}…"


def _first_batch(path: Path, rows: int) -> pa.RecordBatch:
    """Read the first ``rows`` rows of a parquet file and nothing else.

    Args:
        path: The parquet file to open.
        rows: How many rows to read.

    Returns:
        The first record batch, holding at most ``rows`` rows.
    """
    return next(pq.ParquetFile(path).iter_batches(batch_size=rows))


def head_shard(path: Path, rows: int = 2) -> str:
    """Give the schema of one corpus shard and its first rows.

    Args:
        path: One ``data/train-*.parquet`` shard.
        rows: How many rows to print.

    Returns:
        The declared columns and their types, the shard's row count, and for each printed row its
        category, its source dataset, the shape of its series, and its captions.
    """
    reader = pq.ParquetFile(path)
    lines = [
        f"file:       {path.name}",
        f"rows:       {reader.metadata.num_rows}",
        f"row groups: {reader.metadata.num_row_groups}",
        "columns:",
    ]
    lines += [f"  {field.name:<12} {field.type}" for field in reader.schema_arrow]

    batch = _first_batch(path, rows)
    for index, row in enumerate(batch.to_pylist()):
        series = row["time_series"]
        lengths = {len(channel) for channel in series}
        lines += [
            "",
            f"row {index}",
            f"  category    {row['category']!r}",
            f"  dataset     {row['dataset']!r}",
            f"  time_series {len(series)} series, lengths {sorted(lengths)}",
            f"  first value {series[0][0] if series and series[0] else None!r}",
        ]
        lines += [f"  {name:<11} {_clip(row[name] or '')}" for name in _CAPTION_COLUMNS]
    return "\n".join(lines)


def head_sources(path: Path, rows: int = 5) -> str:
    """Give the header and first rows of the table describing the corpus's source datasets.

    Args:
        path: The release's ``meta.csv``.
        rows: How many data rows to print.

    Returns:
        The header row, the printed data rows one field per line, and the total row count.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        printed = [row for _, row in zip(range(rows), reader, strict=False)]
        remaining = sum(1 for _ in reader)

    lines = [f"file:   {path.name}", f"rows:   {len(printed) + remaining}", f"header: {header}"]
    for index, row in enumerate(printed):
        lines.append("")
        lines.append(f"row {index}")
        lines += [f"  {name:<12} {_clip(str(value))}" for name, value in zip(header, row, strict=False)]
    return "\n".join(lines)


def head_declared_schema(path: Path) -> str:
    """Give the columns the release *says* the corpus has.

    The release states its schema in ``dataset_info.json``. A shard states its own. Printing the
    declared one makes the difference between them visible.

    Args:
        path: The release's ``dataset_info.json``.

    Returns:
        Each declared feature with its type, and the file's other top-level keys.
    """
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    lines = [f"file:     {path.name}", f"features: {len(features)}"]
    lines += [f"  {name:<12} {json.dumps(spec, separators=(',', ':'))}" for name, spec in features.items()]
    lines.append("other keys:")
    lines += [f"  {key:<12} {value!r}" for key, value in info.items() if key != "features"]
    return "\n".join(lines)
