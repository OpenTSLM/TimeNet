"""Readable heads of the files the SLIP evaluation folders ship.

Every folder holds the same kind of file — a parquet of fixed windows with one label each — so
there is one head. It opens one file, reads a small part of it, and gives that part back as text.
It writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


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


def head_windows(path: Path, rows: int = 2) -> str:
    """Give the schema of one evaluation split and its first rows.

    Args:
        path: One ``<folder>/{train,test}-*.parquet`` file.
        rows: How many rows to print.

    Returns:
        The declared columns and their types, the split's row count, and for each printed row the
        shape of its window, its label in both encodings, its participant, and its prompt.
    """
    reader = pq.ParquetFile(path)
    lines = [
        f"file:       {path.parent.name}/{path.name}",
        f"rows:       {reader.metadata.num_rows}",
        f"row groups: {reader.metadata.num_row_groups}",
        "columns:",
    ]
    lines += [f"  {field.name:<15} {field.type}" for field in reader.schema_arrow]

    batch = next(reader.iter_batches(batch_size=rows))
    for index, row in enumerate(batch.to_pylist()):
        window = row["X"]
        lengths = {len(signal) for signal in window}
        lines += [
            "",
            f"row {index}",
            f"  X              {len(window)} signals, lengths {sorted(lengths)}",
            f"  first values   {[round(signal[0], 4) for signal in window[:4]]}",
        ]
        # Not every folder ships every column, so print what each row actually holds.
        for name in ("label", "text_label", "participant_id", "prompt"):
            shown = _clip(str(row[name])) if name in row else "— absent —"
            lines.append(f"  {name:<14} {shown}")
    return "\n".join(lines)
