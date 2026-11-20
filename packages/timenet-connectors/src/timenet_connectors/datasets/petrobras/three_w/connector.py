"""Connector for Petrobras 3W, a dataset of rare undesirable events in offshore oil wells.

3W v2.0.0 holds 2,228 instances: 1,119 real ones from 40 wells, 1,089 simulated and 20 hand-drawn.
Each instance is one Parquet file with 27 process variables sampled at 1 Hz and a per-second
``class`` label that marks normal operation and the transient and steady phases of one event.

One instance becomes one record. A variable that the file leaves empty for the whole instance
becomes no signal, because that well has no such sensor. A variable that is empty only in part
keeps its gaps. A variable that holds one value for the whole instance keeps it and carries a
``constant_value`` annotation, because a frozen reading usually means a stuck sensor. Nothing is
filled in or corrected.

The release has no official split. Real instances are split by well, so no well appears in two
partitions. Simulated and hand-drawn instances come from no well and always go to training.
"""

from collections.abc import Callable, Iterator
import csv
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    LocalizationMode,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
)
from timenet_connectors.datasets.petrobras.three_w.events import EVENT_TYPES, EVENTS_BY_CODE, Event, Phase, phase_of
from timenet_connectors.datasets.petrobras.three_w.variables import SOURCE_ORDER, VARIABLES, Variable
from timenet_connectors.download import Artifact, download_files
from timenet_connectors.splitting import StratifiedSplitter


SOURCE_REPO = "petrobras/3W"
SOURCE_REVISION = "c86cc7c70063646910be7292de8a54a6fb2a8399"
"""The 3W commit every build reads (dataset version 2.0.0)."""
MANIFEST = Path(__file__).with_name("files.csv")
"""Every source file at :data:`SOURCE_REVISION`, with its size and SHA-256."""

_SAMPLING_RATE_HZ = 1
_TIMESTAMP = "timestamp"
_LABEL = "class"
_CONSTANT = Annotation(
    "constant_value",
    value=True,
    description="The variable holds one value for the whole instance, which usually means a stuck sensor.",
)
_PROVENANCES = {"WELL": "real", "SIMULATED": "simulated", "DRAWN": "drawn"}


@dataclass(frozen=True)
class InstanceFile:
    """One downloaded 3W instance."""

    path: Path
    """The local Parquet file."""
    event_code: int
    """The event type of the instance, from its folder."""

    @property
    def instance(self) -> str:
        """The instance name, which is the file name without its extension."""
        return self.path.stem

    @property
    def provenance(self) -> str:
        """The provenance, ``real``, ``simulated`` or ``drawn``, from the file name prefix.

        Raises:
            TimeNetConnectorError: If the prefix is not one 3W uses.
        """
        prefix = self.instance.split("_", 1)[0].split("-", 1)[0]
        if prefix not in _PROVENANCES:
            raise TimeNetConnectorError(f"{self.instance}: unknown instance prefix {prefix!r}")
        return _PROVENANCES[prefix]

    @property
    def well(self) -> str | None:
        """The well id (for example ``WELL-00002``) of a real instance, or ``None``."""
        return self.instance.split("_", 1)[0] if self.provenance == "real" else None


def read_manifest(path: Path = MANIFEST) -> list[tuple[str, int, str]]:
    """Read the pinned file list.

    Args:
        path: The manifest CSV with ``path``, ``size`` and ``sha256`` columns.

    Returns:
        One ``(path, size, sha256)`` triple per source file, paths relative to the 3W repo root.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["path"], int(row["size"]), row["sha256"]) for row in csv.DictReader(handle)]


def _column_loader(path: Path, column: str) -> Callable[[], pa.Array]:
    """Return a loader that reads one column of one instance only when it is called.

    Args:
        path: The Parquet file.
        column: The column to read.

    Returns:
        A callable giving the column as one Arrow array.
    """

    def load() -> pa.Array:
        return pq.read_table(path, columns=[column]).column(column).combine_chunks().cast(pa.float64())

    return load


def _label_runs(labels: list[int | None]) -> Iterator[tuple[int | None, int, int]]:
    """Group consecutive equal labels.

    Args:
        labels: The per-second labels, ``None`` where 3W gives none.

    Yields:
        ``(label, start, end)`` for each run, with ``end`` exclusive, in seconds.
    """
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            yield labels[start], start, index
            start = index


def _signals(file: InstanceFile, metadata: pq.FileMetaData) -> list[tuple[Variable, Signal]]:
    """Build the signals of one instance from the Parquet column statistics.

    The statistics give each column's null count and range, so no value is read here.

    Args:
        file: The instance.
        metadata: Its Parquet metadata.

    Returns:
        One ``(variable, signal)`` pair per variable the instance records, in variable order. A
        signal that holds one value for the whole instance carries the ``constant_value`` annotation.

    Raises:
        TimeNetConnectorError: If a column is missing or has no statistics.
    """
    n_rows = metadata.num_rows
    columns = {metadata.schema.column(j).name: j for j in range(metadata.num_columns)}
    built: list[tuple[Variable, Signal]] = []
    for variable in VARIABLES:
        if variable.column not in columns:
            raise TimeNetConnectorError(f"{file.instance}: has no column {variable.column!r}")
        nulls, lowest, highest = 0, None, None
        for group in range(metadata.num_row_groups):
            stats = metadata.row_group(group).column(columns[variable.column]).statistics
            if stats is None or not stats.has_null_count:
                raise TimeNetConnectorError(f"{file.instance}: column {variable.column!r} has no statistics")
            nulls += stats.null_count
            if stats.has_min_max:
                lowest = stats.min if lowest is None else min(lowest, stats.min)
                highest = stats.max if highest is None else max(highest, stats.max)
        if nulls == n_rows:
            continue
        signal = Signal.from_loader(
            spec=variable.spec,
            time_axis=RegularAxis.from_rate_hz(_SAMPLING_RATE_HZ),
            name=variable.column,
            loader=_column_loader(file.path, variable.column),
            n_values=n_rows,
            null_count=nulls,
            annotations=[_CONSTANT] if lowest == highest else [],
            metadata={"description": variable.description},
        )
        built.append((variable, signal))
    return built


def _start(table: pa.Table, file: InstanceFile) -> str:
    """Return the first timestamp of an instance, checking that it is sampled at exactly 1 Hz.

    3W stores the timestamps without a time zone, and the release does not state one. So the
    connector keeps the first timestamp as written, as text in the record metadata, and gives the
    record no absolute start time.

    Args:
        table: The instance's ``timestamp`` column.
        file: The instance.

    Returns:
        The first timestamp in ISO 8601 form, without a time zone.

    Raises:
        TimeNetConnectorError: If the timestamps are not one second apart.
    """
    timestamps = table.column(_TIMESTAMP).cast(pa.timestamp("s"))
    seconds = timestamps.cast(pa.int64()).to_pylist()
    if any(later - earlier != 1 for earlier, later in pairwise(seconds)):
        raise TimeNetConnectorError(f"{file.instance}: timestamps are not one second apart")
    return timestamps[0].as_py().isoformat()


@dataclass
class _Built:
    """A record with the facts its tasks and its split need."""

    record: Record
    file: InstanceFile
    event: Event
    transients: tuple[TimeInterval, ...]


def _build_record(dataset: TimeFDataset, file: InstanceFile) -> _Built:
    """Build and add the record of one instance with its annotations.

    Args:
        dataset: The dataset being built.
        file: The instance.

    Returns:
        The record and the facts its tasks need.

    Raises:
        TimeNetConnectorError: If the folder names no 3W event or a label does not belong to it.
    """
    if file.event_code not in EVENTS_BY_CODE:
        raise TimeNetConnectorError(f"{file.instance}: folder {file.event_code} is not a 3W event")
    event = EVENTS_BY_CODE[file.event_code]
    metadata = pq.read_metadata(file.path)
    signals = _signals(file, metadata)
    table = pq.read_table(file.path, columns=[_TIMESTAMP, _LABEL])
    start = _start(table, file)

    sources = [
        Source(name, signals=[signal for variable, signal in signals if variable.source == name])
        for name in SOURCE_ORDER
        if any(variable.source == name for variable, _ in signals)
    ]
    annotations = [
        Annotation("provenance", value=file.provenance),
        Annotation("event_type", value=event.name),
    ]
    if file.well is not None:
        annotations.append(Annotation("well", value=file.well))

    transients: list[TimeInterval] = []
    labels = table.column(_LABEL).to_pylist()
    for label, begin, end in _label_runs(labels):
        if label is None:
            continue
        phase = phase_of(label, event)
        if phase is None:
            raise TimeNetConnectorError(f"{file.instance}: label {label} at {begin} s is not part of {event.name!r}")
        span = TimeInterval.micros(begin * US_PER_S, end * US_PER_S)
        annotations.append(Annotation("event_phase", value=str(phase), span=span))
        if phase is Phase.TRANSIENT:
            transients.append(span)

    record = Record(
        sources=sources,
        time_span=TimeInterval.micros(0, metadata.num_rows * US_PER_S),
        metadata={
            "instance": file.instance,
            "well": file.well,
            "provenance": file.provenance,
            "event_code": event.code,
            "start": start,
        },
        annotations=annotations,
    )
    dataset.add_record(record)
    return _Built(record=record, file=file, event=event, transients=tuple(transients))


class ThreeWConnector(BaseConnector[list[InstanceFile]]):
    """Connector for Petrobras 3W v2.0.0 (``petrobras/three-w``)."""

    # ruff: ignore[no-self-use] (overrides the instance method BaseConnector.download_async)
    async def download_async(self, cache_dir: Path) -> list[InstanceFile]:
        """Fetch every instance at the pinned commit and check its SHA-256.

        Args:
            cache_dir: The directory to download into.

        Returns:
            One reference per instance, in manifest order.
        """
        entries = read_manifest()
        artifacts = [
            Artifact(
                url=f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_REVISION}/{relative}",
                dest=cache_dir / relative,
                sha256=sha256,
            )
            for relative, _size, sha256 in entries
        ]
        paths = await download_files(artifacts)
        return [InstanceFile(path=path, event_code=int(path.parent.name)) for path in paths]

    def convert(self, instances: list[InstanceFile]) -> TimeFDataset:
        """Build one record per instance, then its classification and localization tasks.

        Args:
            instances: The instances from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        built = [_build_record(dataset, file) for file in instances]
        splits = _assign_splits(built)
        dataset.add_tasks(list(_tasks(built, splits)))
        return dataset


def _assign_splits(built: list[_Built]) -> dict[str, Split]:
    """Split real instances by well and put every other instance in training.

    Args:
        built: Every record with its instance.

    Returns:
        The split of each instance, by instance name.
    """
    splits = {item.file.instance: Split.TRAIN for item in built if item.file.well is None}
    real = [item for item in built if item.file.well is not None]
    if real:
        splitter: StratifiedSplitter[_Built] = StratifiedSplitter(
            train_folds=3,
            validation_folds=1,
            test_folds=1,
            stratify_by=lambda item: item.event.code,
            group_by=lambda item: item.file.well,
        )
        for split, items in splitter.split(real).items():
            for item in items:
                splits[item.file.instance] = split
    return splits


def _tasks(built: list[_Built], splits: dict[str, Split]) -> Iterator[Task]:
    """Give the tasks of every record.

    Every instance gets a classification task for its event type. An instance whose event has a
    labeled transient phase also gets a localization task for that phase.

    Args:
        built: Every record with its facts.
        splits: The split of each instance.

    Yields:
        The tasks, record by record.
    """
    for item in built:
        split = splits[item.file.instance]
        yield ClassificationTask(
            inputs=[item.record],
            targets=item.event.name,
            label_set=EVENT_TYPES,
            split=split,
        )
        if item.transients:
            yield TemporalLocalizationTask(
                inputs=[item.record],
                targets=list(item.transients),
                mode=LocalizationMode.EXHAUSTIVE,
                split=split,
            )


CONNECTOR = ThreeWConnector
