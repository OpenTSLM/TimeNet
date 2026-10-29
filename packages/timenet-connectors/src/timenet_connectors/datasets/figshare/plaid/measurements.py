"""Pair each PLAID CSV file with its metadata entry, and read its values.

A CSV file holds one row per sample, current in ampere then voltage in volt, with no header and no time
column: the time of a row is its index over the rate. The files are read twice, apart.
:func:`read_measurements` counts each file's rows when the release is parsed, which is all a Record needs
until it is written. :func:`read_values` reads the values when a signal is written.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.figshare.plaid.entries import AggregatedEntry, SubmeteredEntry
from timenet_connectors.datasets.figshare.plaid.release import SAMPLE_RATE_HZ, KnownDefects, Subset


COLUMNS = ("current", "voltage")
"""The CSV columns, in file order, as the authors' ``read_data_given_id`` names them. They name the signals."""

_READ_BLOCK_BYTES = 1 << 24
_LENGTH_TOLERANCE_S = 0.005


@dataclass(frozen=True)
class Measurement:
    """One CSV file, its metadata entry, and its number of rows."""

    subset: Subset
    path: Path
    n_values: int
    entry: SubmeteredEntry | AggregatedEntry

    @property
    def file(self) -> str:
        """Return the file's path in the release, such as ``submetered/1.csv``."""
        return f"{self.subset.directory}/{self.path.name}"


def count_rows(path: Path) -> int:
    """Count a CSV file's rows without parsing them.

    Returns:
        The number of lines, counting a last line that has no newline.

    Raises:
        TimeFFormatError: If the file is empty.
    """
    rows, last = 0, b""
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            rows += block.count(b"\n")
            last = block[-1:]
    if not last:
        raise TimeFFormatError(f"{path}: holds no samples")
    return rows + (last != b"\n")


def read_values(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    """Read a CSV file's current and voltage columns as float64.

    Args:
        path: The file.
        n_rows: The rows :func:`count_rows` found when the release was parsed, which the file must still hold.

    Returns:
        One array per column of :data:`COLUMNS`.

    Raises:
        TimeFFormatError: If a row hasn't exactly two values, a value is missing, unreadable or not finite,
            or the row count changed.
    """
    try:
        table = pa_csv.read_csv(
            path,
            read_options=pa_csv.ReadOptions(column_names=list(COLUMNS)),
            # Only an empty field is missing, so a written nan reaches the finite check below.
            convert_options=pa_csv.ConvertOptions(column_types=dict.fromkeys(COLUMNS, pa.float64()), null_values=[""]),
        )
    except pa.ArrowInvalid as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows != n_rows:
        raise TimeFFormatError(f"{path}: holds {table.num_rows} rows, {n_rows} when it was parsed")
    values = {}
    for column in COLUMNS:
        array = table[column].combine_chunks()
        if array.null_count:
            raise TimeFFormatError(f"{path}: {array.null_count} samples have no {column} value")
        values[column] = array.to_numpy()
        if not np.isfinite(values[column]).all():
            raise TimeFFormatError(f"{path}: a {column} value is not finite")
    return values


def read_measurements(
    root: Path,
    subset: Subset,
    entries: tuple[SubmeteredEntry, ...] | tuple[AggregatedEntry, ...],
    defects: KnownDefects,
) -> tuple[Measurement, ...]:
    """Pair each entry with its CSV file under ``root``, and count the file's rows.

    Args:
        root: The directory holding the subset's extracted directory.
        subset: The subset the entries describe.
        entries: The subset's metadata entries, in number order.
        defects: The release's known defects, which must be exactly the ones found.

    Returns:
        One measurement per entry, in number order.

    Raises:
        TimeFFormatError: If the CSV files other than the subset's pinned unlisted ones don't match the entries'
            numbers, the entries whose rows don't last their stated duration are not the pinned ones, or an
            aggregated event falls outside its file.
    """
    directory = root / subset.directory
    unlisted = defects.unlisted_files[subset]
    if not unlisted <= {path.name for path in directory.glob("*.csv")}:
        raise TimeFFormatError(f"{directory}: expected the unlisted files {sorted(unlisted)}")
    files = {path.stem: path for path in directory.glob("*.csv") if path.name not in unlisted}
    numbers = {str(entry.number) for entry in entries}
    if set(files) != numbers:
        missing = sorted(numbers - set(files), key=int)
        extra = sorted(set(files) - numbers)
        raise TimeFFormatError(f"{directory}: CSV files missing for entries {missing}, files with no entry {extra}")
    measurements = []
    misstated = set()
    for entry in entries:
        path = files[str(entry.number)]
        measurement = Measurement(subset=subset, path=path, n_values=count_rows(path), entry=entry)
        # The entry states the duration to hundredths of a second, so the rows may differ from it by half of one.
        if abs(measurement.n_values / SAMPLE_RATE_HZ - entry.length_s) > _LENGTH_TOLERANCE_S:
            misstated.add(entry.number)
        if isinstance(entry, AggregatedEntry):
            outside = [event.sample_index for event in entry.events() if event.sample_index >= measurement.n_values]
            if outside:
                raise TimeFFormatError(
                    f"{measurement.file}: events at samples {outside} fall outside its {measurement.n_values} rows"
                )
        measurements.append(measurement)
    if misstated != defects.misstated_lengths[subset]:
        raise TimeFFormatError(
            f"{directory}: the entries whose rows don't last their stated length are {sorted(misstated)}, "
            f"the pinned release's are {sorted(defects.misstated_lengths[subset])}"
        )
    return tuple(measurements)
