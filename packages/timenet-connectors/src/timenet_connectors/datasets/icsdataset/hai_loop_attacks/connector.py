"""The HAI loop-attacks connector: 300-second windows of one control loop, labelled attack or normal.

HAI is a security dataset of a hardware-in-the-loop power plant testbed. Its authors recorded the
plant four times (versions 20.07, 21.03, 22.04 and 23.05), one row per second, and staged attacks
on its control loops. This connector reads the 17 data files that hold examples. Each file becomes
one Record, with one Signal for each series that a loop example uses, on a one-second axis.

A frozen table, ``examples.csv.gz``, names the examples: which file, which 300-second window, which
loop and which of its series, the label, the split, and the attack interval. It holds no series
value. The connector builds the series from the raw files and gives each example a
ClassificationTask scoped to its window and series. Each attack example also gets a
TemporalLocalizationTask for the attack interval inside the window.

The table derives from HAI, so it is licensed CC BY-SA 4.0 like HAI and is not part of this MIT
package. The connector downloads it from its own repository, pinned by commit and SHA-256. Set
``HAI_LOOP_ATTACKS_DIR`` to a local folder holding ``examples.csv.gz`` (a checkout of that
repository) to read it from there instead. The pin applies either way.
"""

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
import gzip
import hashlib
import os
from pathlib import Path
from typing import IO, Self

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimeSeriesSpec,
)
from timenet_connectors.download import Artifact, download_files


HAI_COMMIT = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
"""The commit of https://github.com/icsdataset/hai that every file comes from."""
_RAW_URL = f"https://raw.githubusercontent.com/icsdataset/hai/{HAI_COMMIT}"
# 22.04 and 23.05 sit in Git LFS. Their raw link returns a 133-byte pointer, not the data.
_LFS_URL = f"https://media.githubusercontent.com/media/icsdataset/hai/{HAI_COMMIT}"

# The repository that publishes the examples table, pinned by commit. HAI_LOOP_ATTACKS_DIR can
# name a local checkout of it instead.
EXAMPLES_OWNER = "vid-t"
EXAMPLES_REPO = "hai-loop-attacks-data"
EXAMPLES_COMMIT = "0e569b70a270908a86feba8bdc3b4d4f3bce0f4c"
EXAMPLES_FILENAME = "examples.csv.gz"
"""The frozen table of loop examples: one row per example, no series values."""
EXAMPLES_SHA256 = "3c49b4dc67871cccd2d3bf32721e76c4809e388bd1dba2df4b1ed36f2ef3d48a"
"""SHA-256 of the table file as published, gzip bytes included."""
EXAMPLES_DIR_ENV = "HAI_LOOP_ATTACKS_DIR"
"""The environment variable that names a local folder holding the table, instead of the download."""
EXAMPLE_COLUMNS = (
    "example_id",
    "window_id",
    "version",
    "file",
    "split",
    "window_kind",
    "anchor",
    "scored",
    "start",
    "loop",
    "label",
    "attack_ids",
    "loop_uncertain",
    "attack_start_s",
    "attack_end_s",
    "no_visible_change",
    "reacting",
    "train_use",
    "tags",
    "roles",
    "source_tags",
    "answer",
)
WINDOW_SECONDS = 300
"""Every example covers 300 rows, one per second."""
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class _VersionFormat:
    """How one HAI version publishes its data files."""

    url: str
    delimiter: str
    time_column: str


_FORMATS = {
    "20.07": _VersionFormat(_RAW_URL, ";", "time"),
    "21.03": _VersionFormat(_RAW_URL, ",", "time"),
    "22.04": _VersionFormat(_LFS_URL, ",", "timestamp"),
    "23.05": _VersionFormat(_LFS_URL, ",", "timestamp"),
}


@dataclass(frozen=True)
class SourceFile:
    """One pinned HAI data file."""

    version: str
    stem: str
    """The file name without its extension, as the examples table names it."""
    filename: str
    sha256: str

    @property
    def path(self) -> str:
        """The path of the file in the HAI repository, such as ``hai-21.03/test1.csv.gz``."""
        return f"hai-{self.version}/{self.filename}"

    @property
    def url(self) -> str:
        """The pinned download URL."""
        return f"{_FORMATS[self.version].url}/{self.path}"


SOURCE_FILES = (
    SourceFile("20.07", "test1", "test1.csv.gz", "68cc18af44efcd756e9eaaa81790855836e8aee3f2ce9c4d9dd95a0ada36c2e4"),
    SourceFile("20.07", "test2", "test2.csv.gz", "427841d8240d5fa557fb28ed58da21bb93d0a3c2b30e43b82ab2d816330185e2"),
    SourceFile("20.07", "train1", "train1.csv.gz", "376e8288a5e3cf69bc5c506f75d6fb9cebb30ff43326c60ed675ccea37700844"),
    SourceFile("21.03", "test1", "test1.csv.gz", "532de41bed372206941609d05b763958f59abecfd63c8c123649f823452e445e"),
    SourceFile("21.03", "test2", "test2.csv.gz", "c5572e0ed19906107ceb56c4b4f40e165153fc7ba87b3506230d53d2478c29bb"),
    SourceFile("21.03", "test3", "test3.csv.gz", "bff835894baff0c73871bc01ece4f5cf759d455a049e808b012128767420caaf"),
    SourceFile("21.03", "test4", "test4.csv.gz", "51380fd28fd985b94a10cae3d8b80926becb3af9f233f2d3a38ab8be9482a482"),
    SourceFile("21.03", "test5", "test5.csv.gz", "f0d59308b167ed95d0e81fb3e27c906682c9b21d3dba0ab4c4d696565df35179"),
    SourceFile("21.03", "train1", "train1.csv.gz", "fdd0162315b32817ba665e2cc8990eb09c47c4edc29683a86b11dc9df062b3a9"),
    SourceFile("22.04", "test1", "test1.csv", "19627f5dac40c3a5039e7468cfd6c6e4ccd37a85055671e8c349b6bce5f50ed1"),
    SourceFile("22.04", "test2", "test2.csv", "b2ba5d6665deee1c86870a3970417b51a3e640a2cc18b9f8de2d6d8a21979923"),
    SourceFile("22.04", "test3", "test3.csv", "f253f1c78a4a4654d843f8f670a24caa7a72ac91e4418b4505009822b3c55940"),
    SourceFile("22.04", "test4", "test4.csv", "2d2e99ed3364d75d27ab84f0a6523496f7295decf363c90b8a1b9b84e1070376"),
    SourceFile("22.04", "train1", "train1.csv", "63383690ffc5344ba68a019c7000854a4d341a22d743ac035f5207072a197b33"),
    SourceFile(
        "23.05", "hai-test1", "hai-test1.csv", "78c7f1d4de1f2ab9ccc2f8c719f80f831033543adb0c81d0d78f84f40838d4be"
    ),
    SourceFile(
        "23.05", "hai-test2", "hai-test2.csv", "b2b8dd295aefd87e39260fe43cb4c73ee86d6264b0ac4b0761e7efb0c2b545c3"
    ),
    SourceFile(
        "23.05", "hai-train3", "hai-train3.csv", "bfcec2dc05adea103e7491546b0e28268faaa26d3cc717d10f4595c94b81e85d"
    ),
)
"""The 17 data files that hold examples. The 23.05 label files are not needed: the table holds the
labels."""

LOOP_STATE = LabelSet(
    [
        Label(
            name="normal",
            description=(
                "No attack that the HAI manual assigns to this control loop is flagged in any second of the"
                " window. Another loop of the window may be under attack."
            ),
        ),
        Label(
            name="attack",
            description=(
                "An attack that the HAI manual assigns to this control loop is flagged in at least one second"
                " of the window."
            ),
        ),
    ],
    annotations=[Annotation("name", value="hai_loop_state")],
)
# HAI publishes no unit per column in its data files, so every spec leaves the unit unknown.
_SPECS = {
    "target": TimeSeriesSpec(spec_type="control_setpoint", name="Control setpoint", unit_value=None, dtype="float64"),
    "measurement": TimeSeriesSpec(
        spec_type="process_measurement", name="Process measurement", unit_value=None, dtype="float64"
    ),
    "command": TimeSeriesSpec(spec_type="control_command", name="Control command", unit_value=None, dtype="float64"),
}
_EVERY_SECOND = RegularAxis.from_rate_hz(1)


def _open_text(path: Path) -> IO[str]:
    """Open a plain or gzip-compressed CSV file as text, keeping its line ends for the csv module.

    Returns:
        The open text stream.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _parse_bool(value: str, column: str) -> bool:
    """Parse a ``True`` or ``False`` cell.

    Returns:
        The value.

    Raises:
        TimeNetConnectorError: If the cell holds anything else.
    """
    if value not in {"True", "False"}:
        raise TimeNetConnectorError(f"{column} {value!r} is not True or False")
    return value == "True"


def _parse_int(value: str, column: str) -> int:
    """Parse a whole-number cell.

    Returns:
        The value.

    Raises:
        TimeNetConnectorError: If the cell is not a whole number.
    """
    try:
        return int(value)
    except ValueError as exc:
        raise TimeNetConnectorError(f"{column} {value!r} is not a whole number") from exc


@dataclass(frozen=True)
class SeriesTag:
    """One series of a loop: its standard (22.04) name, its column in the raw file, and its role."""

    tag: str
    source_tag: str
    loop: str
    role: str


@dataclass(frozen=True)
class Example:
    """One row of the examples table: a window of one file seen through one control loop."""

    example_id: str
    window_id: str
    version: str
    file: str
    split: Split
    window_kind: str
    anchor: str
    scored: bool
    start: datetime
    loop: str
    label: str
    attack_ids: tuple[str, ...]
    loop_uncertain: bool
    attack_start_s: int | None
    attack_end_s: int | None
    no_visible_change: bool
    reacting: bool
    train_use: bool
    series: tuple[SeriesTag, ...]
    answer: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> Self:
        """Parse and check one row of the table.

        Returns:
            The example.

        Raises:
            TimeNetConnectorError: If a cell does not hold what the table promises, or two cells
                disagree.
        """
        version, label = row["version"], row["label"]
        if version not in _FORMATS:
            raise TimeNetConnectorError(f"version {version!r} is not one of {sorted(_FORMATS)}")
        if label not in LOOP_STATE.names:
            raise TimeNetConnectorError(f"label {label!r} is not one of {LOOP_STATE.names}")
        try:
            split = Split(row["split"])
            # HAI prints local wall-clock times and states no time zone, so the time stays naive.
            start = datetime.strptime(row["start"], _TIME_FORMAT)
        except ValueError as exc:
            raise TimeNetConnectorError(f"split {row['split']!r} or start {row['start']!r} is invalid") from exc
        tags, roles, sources = (row[column].split() for column in ("tags", "roles", "source_tags"))
        if not tags or len(tags) != len(roles) or len(tags) != len(sources):
            raise TimeNetConnectorError("tags, roles and source_tags must name the same one or more series")
        unknown_roles = sorted(set(roles) - set(_SPECS))
        if unknown_roles:
            raise TimeNetConnectorError(f"roles {unknown_roles} are not among {sorted(_SPECS)}")
        window_id = f"{version}/{row['file']}/{start:%Y%m%dT%H%M%S}"
        if row["window_id"] != window_id or row["example_id"] != f"{window_id}/{row['loop']}":
            raise TimeNetConnectorError(
                f"example_id {row['example_id']!r} does not match its version, file, start and loop"
            )
        if not row["answer"].endswith(f"Answer: {label}"):
            raise TimeNetConnectorError(f"answer does not end with 'Answer: {label}'")
        example = cls(
            example_id=row["example_id"],
            window_id=window_id,
            version=version,
            file=row["file"],
            split=split,
            window_kind=row["window_kind"],
            anchor=row["anchor"],
            scored=_parse_bool(row["scored"], "scored"),
            start=start,
            loop=row["loop"],
            label=label,
            attack_ids=tuple(row["attack_ids"].split()),
            loop_uncertain=_parse_bool(row["loop_uncertain"], "loop_uncertain"),
            attack_start_s=_attack_second(row["attack_start_s"], "attack_start_s", label),
            attack_end_s=_attack_second(row["attack_end_s"], "attack_end_s", label),
            no_visible_change=_parse_bool(row["no_visible_change"], "no_visible_change"),
            reacting=_parse_bool(row["reacting"], "reacting"),
            train_use=_parse_bool(row["train_use"], "train_use"),
            series=tuple(
                SeriesTag(tag, source, row["loop"], role)
                for tag, source, role in zip(tags, sources, roles, strict=True)
            ),
            answer=row["answer"],
        )
        example.check_attack()
        return example

    def check_attack(self) -> None:
        """Check that the attack cells and flags fit the label.

        An attack example names its attack and its interval, and a normal one neither. Only an attack
        example can show no visible change, and only a normal example can be a reacting neighbour.
        The training sample holds neither kind.

        Raises:
            TimeNetConnectorError: If the attack cells or flags do not fit the label.
        """
        if self.train_use and (self.no_visible_change or self.reacting):
            raise TimeNetConnectorError("train_use is set on a no_visible_change or reacting example")
        if self.label == "normal":
            if self.attack_ids:
                raise TimeNetConnectorError("a normal example names attack_ids")
            if self.no_visible_change:
                raise TimeNetConnectorError("a normal example has no_visible_change set")
            return
        if self.reacting:
            raise TimeNetConnectorError("an attack example has reacting set")
        start, end = self.attack_start_s, self.attack_end_s
        if not self.attack_ids or start is None or end is None:
            raise TimeNetConnectorError("an attack example names no attack or no attack interval")
        # Every attack in the table starts inside its window. None starts before it.
        if not 0 <= start < WINDOW_SECONDS or end < start:
            raise TimeNetConnectorError(
                f"attack interval {start}..{end} s does not start inside the {WINDOW_SECONDS} s window"
            )

    def task_metadata(self) -> JsonMapping:
        """Return the metadata of the example's classification task.

        Returns:
            The JSON-compatible metadata.
        """
        return {
            "example_id": self.example_id,
            "window_id": self.window_id,
            "version": self.version,
            "file": self.file,
            "loop": self.loop,
            "tags": [one.tag for one in self.series],
            "source_tags": [one.source_tag for one in self.series],
            "roles": [one.role for one in self.series],
            "window_kind": self.window_kind,
            "anchor": self.anchor,
            "scored": self.scored,
            "no_visible_change": self.no_visible_change,
            "reacting": self.reacting,
            "train_use": self.train_use,
            "attack_ids": list(self.attack_ids),
            "loop_uncertain": self.loop_uncertain,
            "attack_start_s": self.attack_start_s,
            "attack_end_s": self.attack_end_s,
        }


def _attack_second(value: str, column: str, label: str) -> int | None:
    """Parse the attack start or end second of an example, where -1 means no attack.

    Returns:
        The second, or ``None`` for a normal example.

    Raises:
        TimeNetConnectorError: If a normal example has a second other than -1.
    """
    second = _parse_int(value, column)
    if label == "normal":
        if second != -1:
            raise TimeNetConnectorError(f"a normal example has {column} {second}, not -1")
        return None
    return second


def check_examples_table(path: Path) -> None:
    """Check that the examples table is the one the connector pins.

    Raises:
        TimeNetConnectorError: If the SHA-256 of the file differs from ``EXAMPLES_SHA256``.
    """
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != EXAMPLES_SHA256:
        raise TimeNetConnectorError(f"{path} has SHA-256 {digest}, not the pinned {EXAMPLES_SHA256}")


def examples_url(owner: str = EXAMPLES_OWNER, repo: str = EXAMPLES_REPO, commit: str = EXAMPLES_COMMIT) -> str:
    """Return the pinned download URL of the examples table.

    Returns:
        The URL of the table at the pinned commit of its GitHub repository.

    Raises:
        TimeNetConnectorError: If the owner, repository or commit is a placeholder such as ``<owner>``.
    """
    if any(part.startswith("<") for part in (owner, repo, commit)):
        raise TimeNetConnectorError(
            f"the examples table URL names a placeholder ({owner}/{repo}@{commit}); set {EXAMPLES_DIR_ENV}"
            f" to a local folder that holds {EXAMPLES_FILENAME}"
        )
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{EXAMPLES_FILENAME}"


def local_examples_table() -> Path | None:
    """Return the examples table in the folder that ``HAI_LOOP_ATTACKS_DIR`` names, if it is set.

    Returns:
        The path of the table, or ``None`` when the variable is unset or empty.

    Raises:
        TimeNetConnectorError: If the folder holds no table, or not the pinned one.
    """
    folder = os.environ.get(EXAMPLES_DIR_ENV, "")
    if not folder:
        return None
    path = Path(folder).expanduser() / EXAMPLES_FILENAME
    if not path.is_file():
        raise TimeNetConnectorError(f"{EXAMPLES_DIR_ENV} is set, but {path} does not exist")
    check_examples_table(path)
    return path


def read_examples(path: Path) -> list[Example]:
    """Read every example of the table, in file order.

    Returns:
        The examples.

    Raises:
        TimeNetConnectorError: If the header is not the expected one, a row is malformed, or an
            example id repeats. The message names the data row.
    """
    with _open_text(path) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXAMPLE_COLUMNS:
            raise TimeNetConnectorError(f"{path.name} has columns {reader.fieldnames}, not {list(EXAMPLE_COLUMNS)}")
        examples, seen = [], set()
        for number, row in enumerate(reader, start=1):
            try:
                example = Example.from_row(row)
                if example.example_id in seen:
                    raise TimeNetConnectorError(f"example_id {example.example_id!r} repeats")
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{path.name} data row {number}: {exc}") from exc
            seen.add(example.example_id)
            examples.append(example)
    return examples


def series_by_version(examples: Iterable[Example]) -> dict[str, dict[str, SeriesTag]]:
    """Collect the series each version uses, in the order the examples first name them.

    Returns:
        For each version, its series by standard tag.

    Raises:
        TimeNetConnectorError: If two examples of one version give one tag another column, loop or
            role, or two tags share one column.
    """
    found: dict[str, dict[str, SeriesTag]] = {}
    for example in examples:
        tags = found.setdefault(example.version, {})
        for one in example.series:
            known = tags.setdefault(one.tag, one)
            if known != one:
                raise TimeNetConnectorError(f"{example.example_id}: {one} disagrees with {known}")
    for version, tags in found.items():
        columns = [one.source_tag for one in tags.values()]
        if len(set(columns)) != len(columns):
            raise TimeNetConnectorError(f"version {version}: two tags share one source column")
    return found


@dataclass(frozen=True)
class DataFile:
    """A downloaded HAI data file."""

    source: SourceFile
    path: Path

    @property
    def delimiter(self) -> str:
        """The column separator: ``;`` in 20.07, ``,`` in the later versions."""
        return _FORMATS[self.source.version].delimiter

    @property
    def time_column(self) -> str:
        """The time column: ``time`` in 20.07 and 21.03, ``timestamp`` in 22.04 and 23.05."""
        return _FORMATS[self.source.version].time_column

    def read_header(self) -> tuple[str, ...]:
        """Read the column names.

        Returns:
            The names, in file order.
        """
        with _open_text(self.path) as stream:
            return tuple(next(csv.reader(stream, delimiter=self.delimiter), ()))


@dataclass(frozen=True)
class Timeline:
    """The time column of a data file, checked to step one second per row."""

    first: datetime
    n_rows: int
    columns: tuple[str, ...]

    @classmethod
    def read(cls, data_file: DataFile) -> Self:
        """Read and check the time column of a file.

        Returns:
            The first time, the number of rows and the column names.

        Raises:
            TimeNetConnectorError: If the file has no time column, a time does not parse, or two
                rows are not one second apart. The message names the file and the row.
        """
        columns = data_file.read_header()
        name, where = data_file.time_column, data_file.source.path
        if name not in columns:
            raise TimeNetConnectorError(f"{where} has no {name!r} column")
        try:
            table = pa_csv.read_csv(
                data_file.path,
                parse_options=pa_csv.ParseOptions(delimiter=data_file.delimiter),
                convert_options=pa_csv.ConvertOptions(
                    include_columns=[name],
                    column_types={name: pa.timestamp("s")},
                    timestamp_parsers=[_TIME_FORMAT],
                ),
            )
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{where}: {exc}") from exc
        times = table.column(name)
        if times.null_count or len(times) < WINDOW_SECONDS:
            raise TimeNetConnectorError(f"{where} has an empty time or fewer than {WINDOW_SECONDS} rows")
        seconds = times.cast(pa.int64()).to_numpy()
        gaps = np.flatnonzero(np.diff(seconds) != 1)
        if gaps.size:
            row = int(gaps[0])
            raise TimeNetConnectorError(
                f"{where}: data rows {row + 1} and {row + 2} are {int(seconds[row + 1] - seconds[row])} s apart, not 1 s"
            )
        first = times[0].as_py()
        return cls(first=first, n_rows=len(seconds), columns=columns)

    def offset_of(self, example: Example) -> int:
        """Return the row where an example's window starts.

        Returns:
            The row, counted from zero, which is also the second from the first row.

        Raises:
            TimeNetConnectorError: If the window does not lie inside the file.
        """
        offset = int((example.start - self.first).total_seconds())
        if offset < 0 or offset + WINDOW_SECONDS > self.n_rows:
            raise TimeNetConnectorError(f"{example.example_id}: the window leaves its file of {self.n_rows} rows")
        return offset


class _ColumnReader:
    """Read the series columns of one data file lazily, holding one file's columns at a time.

    The writer reads the series of one record together, once for each spec type, so each file is
    parsed once per role.
    """

    def __init__(self) -> None:
        self._table = lru_cache(maxsize=1)(self._read)

    @staticmethod
    def _read(data_file: DataFile, columns: tuple[str, ...]) -> pa.Table:
        try:
            return pa_csv.read_csv(
                data_file.path,
                parse_options=pa_csv.ParseOptions(delimiter=data_file.delimiter),
                convert_options=pa_csv.ConvertOptions(
                    include_columns=list(columns), column_types=dict.fromkeys(columns, pa.float64())
                ),
            )
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{data_file.source.path}: {exc}") from exc

    def __call__(self, data_file: DataFile, columns: tuple[str, ...], column: str, n_rows: int) -> pa.Array:
        table = self._table(data_file, columns)
        if table.num_rows != n_rows:
            raise TimeNetConnectorError(
                f"{data_file.source.path} has {table.num_rows} rows, not the {n_rows} it had at conversion"
            )
        return table.column(column).combine_chunks()


@dataclass(frozen=True)
class FileRecord:
    """The record of one data file, its timeline, and its signals by standard tag."""

    record: Record
    timeline: Timeline
    signals: Mapping[str, Signal]


def _record(
    data_file: DataFile, timeline: Timeline, series: Mapping[str, SeriesTag], reader: _ColumnReader
) -> FileRecord:
    """Create the record of one data file, with one lazy signal for each series its version uses.

    Returns:
        The record, its timeline and its signals.

    Raises:
        TimeNetConnectorError: If the file lacks a column that a series needs.
    """
    columns = tuple(one.source_tag for one in series.values())
    missing = sorted(set(columns) - set(timeline.columns))
    if missing:
        raise TimeNetConnectorError(f"{data_file.source.path} has no columns {missing}")
    signals = {
        one.tag: Signal.from_loader(
            spec=_SPECS[one.role],
            time_axis=_EVERY_SECOND,
            name=one.source_tag,
            loader=partial(reader, data_file, columns, one.source_tag, timeline.n_rows),
            n_values=timeline.n_rows,
            metadata={"tag": one.tag, "loop": one.loop, "role": one.role},
        )
        for one in series.values()
    }
    source = data_file.source
    metadata: JsonMapping = {
        "version": source.version,
        "file": source.stem,
        "path": source.path,
        # HAI prints local wall-clock times and states no time zone, so the record keeps the text.
        "start_time": timeline.first.strftime(_TIME_FORMAT),
        "n_seconds": timeline.n_rows,
    }
    record = Record(sources=(Source(name="HAI control loops", signals=tuple(signals.values())),), metadata=metadata)
    return FileRecord(record=record, timeline=timeline, signals=signals)


def _tasks(example: Example, file_record: FileRecord) -> list[Task]:
    """Create the classification task of an example and, for an attack, its localization task.

    Args:
        example: The example.
        file_record: The record of the example's file.

    Returns:
        The tasks.
    """
    record, offset = file_record.record, file_record.timeline.offset_of(example)
    series = tuple(file_record.signals[one.tag] for one in example.series)
    scope = TimeInterval.seconds(offset, offset + WINDOW_SECONDS, time_series=series)
    classify = ClassificationTask(
        inputs=(record,),
        targets=(example.label,),
        label_set=LOOP_STATE,
        scope=scope,
        split=example.split,
        # The rule-written text that reads the series and ends with the label.
        rationale=example.answer,
        metadata=example.task_metadata(),
    )
    if example.attack_start_s is None or example.attack_end_s is None:
        return [classify]
    # The table gives the last flagged second, inclusive. The interval ends one second later, or at
    # the end of the window if the attack outlasts it.
    end = min(example.attack_end_s + 1, WINDOW_SECONDS)
    locate = TemporalLocalizationTask(
        inputs=(record,),
        targets=(TimeInterval.seconds(offset + example.attack_start_s, offset + end, time_series=series),),
        scope=scope,
        split=example.split,
        mode=LocalizationMode.SPARSE,
        from_tasks=(classify,),
        metadata={
            "example_id": example.example_id,
            "attack_ids": list(example.attack_ids),
            "attack_start_s": example.attack_start_s,
            "attack_end_s": example.attack_end_s,
            "cut_at_window_end": example.attack_end_s + 1 > WINDOW_SECONDS,
        },
    )
    return [classify, locate]


@dataclass(frozen=True)
class HaiRelease:
    """The downloaded data files and the examples table."""

    files: tuple[DataFile, ...]
    examples_path: Path


class HaiLoopAttacksConnector(BaseConnector[HaiRelease]):
    """Connector for 300-second control-loop windows of the HAI security dataset."""

    async def download_async(self, cache_dir: Path) -> HaiRelease:  # ruff: ignore[no-self-use]
        """Download the 17 pinned data files and the pinned examples table.

        The table comes from the folder that ``HAI_LOOP_ATTACKS_DIR`` names when it is set,
        and from its repository otherwise. Both are checked against ``EXAMPLES_SHA256``.

        Args:
            cache_dir: Directory for the downloaded files, laid out as in the HAI repository.

        Returns:
            The files and the table.
        """
        # Resolve the table first, so that a missing or wrong table fails before 650 MB of downloads.
        table = local_examples_table()
        artifacts = [Artifact(one.url, cache_dir / one.path, sha256=one.sha256) for one in SOURCE_FILES]
        if table is None:
            destination = cache_dir / "examples" / EXAMPLES_COMMIT / EXAMPLES_FILENAME
            artifacts.append(Artifact(examples_url(), destination, sha256=EXAMPLES_SHA256))
        paths = await download_files(artifacts)
        data_paths = paths[: len(SOURCE_FILES)]
        files = tuple(DataFile(one, path) for one, path in zip(SOURCE_FILES, data_paths, strict=True))
        return HaiRelease(files=files, examples_path=table or paths[-1])

    def convert(self, release: HaiRelease) -> TimeFDataset:
        """Create one record for each data file and the tasks of every example.

        Args:
            release: The release from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeNetConnectorError: If an example names a file the release lacks, or the series of a
                version do not fit its files.
        """
        examples = read_examples(release.examples_path)
        series = series_by_version(examples)
        dataset = TimeFDataset(metadata=self.metadata())
        reader = _ColumnReader()
        records: dict[tuple[str, str], FileRecord] = {}
        for data_file in release.files:
            key = (data_file.source.version, data_file.source.stem)
            if key in records:
                raise TimeNetConnectorError(f"{data_file.source.path} appears twice")
            file_record = _record(data_file, Timeline.read(data_file), series.get(key[0], {}), reader)
            dataset.add_record(record=file_record.record)
            records[key] = file_record
        tasks: list[Task] = []
        for example in examples:
            found = records.get((example.version, example.file))
            if found is None:
                raise TimeNetConnectorError(f"{example.example_id}: the release has no file {example.file!r}")
            tasks.extend(_tasks(example, found))
        dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = HaiLoopAttacksConnector
