"""The HAI connector: every data file of the HAI security dataset, as published.

HAI is a security dataset of a hardware-in-the-loop power plant testbed: a boiler, a turbine and a
water treatment process, coupled through a power grid simulator. Its authors recorded the plant four
times (versions 20.07, 21.03, 22.04 and 23.05), one row per second, and staged attacks on its
control loops. This connector reads the 28 data files of the four versions at one pinned commit.

Each data file becomes one Record on a one-second axis, with two Sources. ``HAI data points`` holds
one Signal for each data point of the file, under its column name, in ``float64``. The spec of each
signal comes from the manual's table of data points (:mod:`.points`): the quantity it measures and
its unit. ``HAI attack flags`` holds HAI's own labels as published: ``attack`` and ``attack_P1`` to
``attack_P3`` in 20.07 and 21.03, ``Attack`` in 22.04, and ``label`` from the two label files of
23.05. HAI publishes no labels for the 23.05 training files, so their records have no flag.

The connector adds no task. Child datasets such as ``icsdataset/hai-loop-attacks`` import the
records or signals they need.
"""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
import gzip
from pathlib import Path
import threading
from typing import IO, Self

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet_connectors.datasets.icsdataset.hai.files import DATA_FILES, LABEL_FILES, SourceFile
from timenet_connectors.datasets.icsdataset.hai.points import ATTACK_FLAG, manual_point, spec_of
from timenet_connectors.download import Artifact, download_files


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
"""How HAI prints its times: local wall-clock time, without a time zone."""
POINTS_SOURCE = "HAI data points"
FLAGS_SOURCE = "HAI attack flags"
EVERY_SECOND = RegularAxis.from_rate_hz(1)
_EPOCH = datetime(1970, 1, 1)
"""Day zero of the seconds that Arrow reads HAI's naive wall-clock times into."""


def _open_text(path: Path) -> IO[str]:
    """Open a plain or gzip-compressed CSV file as text, keeping its line ends for the csv module.

    Returns:
        The open text stream.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


@dataclass(frozen=True)
class DataFile:
    """A downloaded HAI data file."""

    source: SourceFile
    path: Path

    def read_header(self) -> tuple[str, ...]:
        """Read the column names.

        Returns:
            The names, in file order.
        """
        with _open_text(self.path) as stream:
            return tuple(next(csv.reader(stream, delimiter=self.source.format.delimiter), ()))

    def read_columns(self, points: tuple[str, ...]) -> dict[str, pa.Array]:
        """Read the data points and the attack flags of the file in one parse.

        Args:
            points: The data point columns, read as ``float64``. The flags are read as ``int8``.

        Returns:
            Each column by name.

        Raises:
            TimeNetConnectorError: If the file cannot be parsed into those types.
        """
        flags = self.source.format.flag_columns
        types = dict.fromkeys(points, pa.float64()) | dict.fromkeys(flags, pa.int8())
        try:
            table = pa_csv.read_csv(
                self.path,
                parse_options=pa_csv.ParseOptions(delimiter=self.source.format.delimiter),
                convert_options=pa_csv.ConvertOptions(include_columns=list(types), column_types=types),
            )
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self.source.path}: {exc}") from exc
        return {name: table.column(name).combine_chunks() for name in types}


def _read_times(path: Path, where: str, column: str, delimiter: str, formats: list[str]) -> np.ndarray:
    """Read one time column as whole seconds since the epoch.

    Returns:
        The times.

    Raises:
        TimeNetConnectorError: If the file has no such column or a time does not parse.
    """
    try:
        table = pa_csv.read_csv(
            path,
            parse_options=pa_csv.ParseOptions(delimiter=delimiter),
            convert_options=pa_csv.ConvertOptions(
                include_columns=[column], column_types={column: pa.timestamp("s")}, timestamp_parsers=formats
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{where}: {exc}") from exc
    times = table.column(column)
    if times.null_count:
        raise TimeNetConnectorError(f"{where} has an empty time")
    return times.cast(pa.int64()).to_numpy()


@dataclass(frozen=True)
class Timeline:
    """The time column and the header of a data file, checked to step one second per row."""

    first: datetime
    n_rows: int
    columns: tuple[str, ...]
    seconds: np.ndarray
    """Each row's time, in whole seconds since the epoch."""

    @classmethod
    def read(cls, data_file: DataFile) -> Self:
        """Read and check the time column of a file.

        Returns:
            The first time, the number of rows, the column names and the times.

        Raises:
            TimeNetConnectorError: If the file has no time column or no row, a time does not parse,
                or two rows are not one second apart. The message names the file and the row.
        """
        columns = data_file.read_header()
        name, where = data_file.source.format.time_column, data_file.source.path
        if name not in columns:
            raise TimeNetConnectorError(f"{where} has no {name!r} column")
        seconds = _read_times(data_file.path, where, name, data_file.source.format.delimiter, [TIME_FORMAT])
        if not len(seconds):
            raise TimeNetConnectorError(f"{where} has no data row")
        gaps = np.flatnonzero(np.diff(seconds) != 1)
        if gaps.size:
            row = int(gaps[0])
            raise TimeNetConnectorError(
                f"{where}: data rows {row + 1} and {row + 2} are {int(seconds[row + 1] - seconds[row])} s apart, not 1 s"
            )
        # Arrow counts the seconds of the printed wall-clock time from 1970-01-01, so adding them to
        # that naive date gives back the printed time.
        first = _EPOCH + timedelta(seconds=int(seconds[0]))
        return cls(first=first, n_rows=len(seconds), columns=columns, seconds=seconds)


@dataclass(frozen=True)
class LabelFile:
    """A downloaded 23.05 label file: one HAI attack label per row of its data file."""

    source: SourceFile
    path: Path

    def read_labels(self, timeline: Timeline) -> pa.Array:
        """Read the labels and check that they line up with the rows of their data file.

        ``label-test1.csv`` prints its times to the second, and each row must name the time of its
        data row. ``label-test2.csv`` prints them to the minute, such as ``2022-08-17 0:00``, so
        there each row must name the minute of its data row.

        Returns:
            The labels, as ``int8``.

        Raises:
            TimeNetConnectorError: If the file has another number of rows than its data file, a label
                is not 0 or 1, or a row names another time than its data row.
        """
        where = self.source.path
        seconds = _read_times(self.path, where, "timestamp", ",", [TIME_FORMAT, "%Y-%m-%d %H:%M"])
        if len(seconds) != timeline.n_rows:
            raise TimeNetConnectorError(f"{where} has {len(seconds)} rows, its data file {timeline.n_rows}")
        # Either every row names the second of its data row, or every row names the minute.
        wrong = np.flatnonzero(seconds != timeline.seconds)
        if wrong.size:
            wrong = np.flatnonzero(seconds != timeline.seconds - timeline.seconds % 60)
        if wrong.size:
            raise TimeNetConnectorError(f"{where}: data row {int(wrong[0]) + 1} names another time than its data row")
        try:
            table = pa_csv.read_csv(
                self.path,
                convert_options=pa_csv.ConvertOptions(include_columns=["label"], column_types={"label": pa.int8()}),
            )
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{where}: {exc}") from exc
        labels = table.column("label").combine_chunks()
        if labels.null_count or not set(labels.to_numpy().tolist()) <= {0, 1}:
            raise TimeNetConnectorError(f"{where}: a label is not 0 or 1")
        return labels


class _ColumnReader:
    """Read the columns of each data file in one parse, and hand out each column once.

    The writer reads the signals spec by spec, so it comes back to one file for every spec, a few
    columns at a time. This reader parses a file once, keeps its columns, and drops each column once
    it hands it out. A column asked for a second time is parsed again. Memory holds at most the
    columns not yet written.
    """

    def __init__(self) -> None:
        self._held: dict[str, dict[str, pa.Array]] = {}
        self._lock = threading.Lock()

    def __call__(self, data_file: DataFile, points: tuple[str, ...], column: str, n_rows: int) -> pa.Array:
        with self._lock:
            held = self._held.get(data_file.source.path)
            if held is None or column not in held:
                held = self._held[data_file.source.path] = data_file.read_columns(points)
            values = held.pop(column)
            if not held:
                del self._held[data_file.source.path]
        if len(values) != n_rows:
            raise TimeNetConnectorError(
                f"{data_file.source.path} has {len(values)} rows, not the {n_rows} it had at conversion"
            )
        return values


def create_record(
    data_file: DataFile, timeline: Timeline, reader: _ColumnReader, label_file: LabelFile | None = None
) -> Record:
    """Create the record of one data file: one signal per data point and one per attack flag.

    Args:
        data_file: The file.
        timeline: Its checked time column and header.
        reader: The reader the signals load their values through.
        label_file: The 23.05 label file of the data file, if HAI publishes one.

    Returns:
        The record, not yet added to a dataset.

    Raises:
        TimeNetConnectorError: If the file lacks an attack flag of its version, or a column repeats.
    """
    form, where = data_file.source.format, data_file.source.path
    missing = [flag for flag in form.flag_columns if flag not in timeline.columns]
    if missing:
        raise TimeNetConnectorError(f"{where} has no attack flags {missing}")
    if len(set(timeline.columns)) != len(timeline.columns):
        raise TimeNetConnectorError(f"{where} names a column twice")
    points = tuple(name for name in timeline.columns if name != form.time_column and name not in form.flag_columns)
    n_rows = timeline.n_rows

    def lazy(name: str) -> partial[pa.Array]:
        return partial(reader, data_file, points, name, n_rows)

    signals = []
    for name in points:
        point = manual_point(name)
        metadata: JsonMapping = {"manual_point": None if point is None else point.name}
        signals.append(
            Signal.from_loader(
                spec=spec_of(name),
                time_axis=EVERY_SECOND,
                name=name,
                loader=lazy(name),
                n_values=n_rows,
                metadata=metadata,
            )
        )
    flags = [
        Signal.from_loader(spec=ATTACK_FLAG, time_axis=EVERY_SECOND, name=flag, loader=lazy(flag), n_values=n_rows)
        for flag in form.flag_columns
    ]
    if label_file is not None:
        # Read the labels now: a label file that does not line up with its data file fails the conversion.
        labels = label_file.read_labels(timeline)
        flags.append(
            Signal.from_loader(
                spec=ATTACK_FLAG, time_axis=EVERY_SECOND, name="label", loader=lambda: labels, n_values=n_rows
            )
        )
    sources = [Source(name=POINTS_SOURCE, signals=signals)]
    if flags:
        sources.append(Source(name=FLAGS_SOURCE, signals=flags))
    record_metadata: JsonMapping = {
        "version": data_file.source.version,
        "file": data_file.source.stem,
        "path": where,
        # HAI prints local wall-clock times and states no time zone, so the record keeps the text.
        "start_time": timeline.first.strftime(TIME_FORMAT),
        "n_seconds": n_rows,
    }
    if label_file is not None:
        record_metadata["label_file"] = label_file.source.path
    return Record(sources=sources, metadata=record_metadata)


@dataclass(frozen=True)
class HaiRelease:
    """The downloaded data files, and the label files by the stem of the data file they label."""

    files: tuple[DataFile, ...]
    labels: Mapping[str, LabelFile]


class HaiConnector(BaseConnector[HaiRelease]):
    """Connector for the HAI security dataset: every data file of four versions, as published."""

    async def download_async(self, cache_dir: Path) -> HaiRelease:  # ruff: ignore[no-self-use]
        """Download the 28 pinned data files and the two 23.05 label files.

        Args:
            cache_dir: Directory for the downloaded files, laid out as in the HAI repository.

        Returns:
            The files.
        """
        label_paths = await download_files(
            Artifact(one.url, cache_dir / one.path, sha256=one.sha256) for one in LABEL_FILES.values()
        )
        data_paths = await download_files(
            Artifact(one.url, cache_dir / one.path, sha256=one.sha256) for one in DATA_FILES
        )
        labels = {
            stem: LabelFile(source, path) for (stem, source), path in zip(LABEL_FILES.items(), label_paths, strict=True)
        }
        files = tuple(DataFile(one, path) for one, path in zip(DATA_FILES, data_paths, strict=True))
        return HaiRelease(files=files, labels=labels)

    def convert(self, release: HaiRelease) -> TimeFDataset:
        """Create one record for each data file.

        Args:
            release: The release from :meth:`download_async`.

        Returns:
            The populated dataset, without tasks.

        Raises:
            TimeNetConnectorError: If a file appears twice, or a label file names no data file of
                the release.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        reader = _ColumnReader()
        stems = {(one.source.version, one.source.stem) for one in release.files}
        if len(stems) != len(release.files):
            raise TimeNetConnectorError("a data file appears twice in the release")
        unknown = sorted(stem for stem in release.labels if (release.labels[stem].source.version, stem) not in stems)
        if unknown:
            raise TimeNetConnectorError(f"label files name no data file of the release: {unknown}")
        for data_file in release.files:
            label_file = release.labels.get(data_file.source.stem) if data_file.source.version == "23.05" else None
            dataset.add_record(create_record(data_file, Timeline.read(data_file), reader, label_file))
        return dataset


CONNECTOR = HaiConnector
