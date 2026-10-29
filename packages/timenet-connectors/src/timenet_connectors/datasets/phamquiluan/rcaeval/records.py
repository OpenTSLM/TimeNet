"""One Record per case: a Source of lazy Signals per telemetry file, its inject time, and its system."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import partial

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.json import JsonMapping
from timenet.types import US_PER_S, Annotation, InputModality, TimeInterval, TimeOrigin, TimePoint, TimeSeriesSpec, ureg
from timenet_connectors.datasets.phamquiluan.rcaeval import benchmark
from timenet_connectors.datasets.phamquiluan.rcaeval.cases import Case
from timenet_connectors.datasets.phamquiluan.rcaeval.keys import (
    CLOCKS,
    COLUMN_ORDER,
    SOURCE_FILE,
    AnnotationKey,
    SourceName,
)
from timenet_connectors.datasets.phamquiluan.rcaeval.release import Suite
from timenet_connectors.datasets.phamquiluan.rcaeval.tables import (
    LOG_SUB_MICROSECOND,
    TRACE_NUMBERS,
    TRACE_SOURCE_ROW,
    EventKind,
    EventTable,
    GridTable,
    TableReader,
    event_signal_columns,
    read_events,
    read_grid,
)
from timenet_connectors.metadata import RecordMetadata


DERIVED_PERIOD_S = 15
"""The authors' derived series have one row every 15 s, which the loader checks."""
METRIC_KINDS = frozenset(
    {"cpu", "mem", "diskio", "socket", "workload", "latency", "latency-50", "latency-90", "error", "load"}
)
"""The metrics the files hold, the part of a column name after its service: ``<service>_<metric>``."""

_DERIVED_FILES = {
    SourceName.LOGTS: "logts.csv",
    SourceName.TRACETS_ERR: "tracets_err.csv",
    SourceName.TRACETS_LAT: "tracets_lat.csv",
}
_EVENT_SOURCES = {EventKind.LOGS: SourceName.LOGS, EventKind.TRACES: SourceName.TRACES}


class RCAEvalRecordMetadata(RecordMetadata):
    """An RCAEval Record's metadata: the suite its case belongs to.

    The shared ``subject_id`` is the system the case ran on, such as ``Online Boutique``. The case's
    directory names its root cause and fault, so it is task metadata, not Record metadata.
    """

    suite: Suite


def record_metadata(record: Record) -> RCAEvalRecordMetadata:
    """Read an RCAEval Record's typed metadata back.

    Returns:
        The suite and system of the Record's case.

    Raises:
        TimeFValidationError: If the Record's metadata is not RCAEval metadata.
    """  # noqa: DOC502 - raised by from_json
    return RCAEvalRecordMetadata.from_json(record.metadata)


@dataclass(frozen=True)
class CaseTelemetry:
    """The shapes of a case's telemetry files, read and checked once."""

    metrics: GridTable
    logs: EventTable | None
    traces: EventTable | None
    header_only_traces: bool
    """Whether the case has a ``traces.csv`` that holds only its header, which counts as no traces."""
    derived: Mapping[SourceName, GridTable]
    """The authors' derived series the case has: ``logts``, ``tracets_err`` and ``tracets_lat``."""

    @classmethod
    def read(cls, case: Case) -> "CaseTelemetry":
        """Read and check every telemetry file of a case.

        Returns:
            The files' shapes.

        Raises:
            TimeFFormatError: If a metric column isn't ``<service>_<metric>`` with a known metric.
        """
        metrics = read_grid(case.member(case.suite.metrics_file), benchmark.METRICS_PERIOD_S)
        unknown = [column for column in metrics.columns if column.partition("_")[2] not in METRIC_KINDS]
        if unknown:
            raise TimeFFormatError(f"{case.path}: metric columns {unknown} are not <service>_<metric>")
        logs = read_events(case.member("logs.csv"), EventKind.LOGS) if "logs.csv" in case.files else None
        traces = read_events(case.member("traces.csv"), EventKind.TRACES) if "traces.csv" in case.files else None
        derived = {
            name: read_grid(case.member(filename), DERIVED_PERIOD_S)
            for name, filename in _DERIVED_FILES.items()
            if filename in case.files
        }
        return cls(
            metrics=metrics,
            logs=logs,
            traces=traces,
            header_only_traces="traces.csv" in case.files and traces is None,
            derived=derived,
        )

    def origin_us(self) -> int:
        """Return the Record's origin: the earliest time in any of its files, in Unix microseconds."""
        firsts = [self.metrics.first_us(), *(table.first_us() for table in self.derived.values())]
        firsts += [table.first_us for table in (self.logs, self.traces) if table is not None]
        return min(firsts)

    def metrics_span(self, origin_us: int) -> TimeInterval:
        """Return the span of the metrics' main clock, ``time``, on the Record's timeline.

        Returns:
            From the first row to one period past the last.
        """
        clock = self.metrics.clocks["time"]
        start = clock.first_s * US_PER_S - origin_us
        return TimeInterval.micros(start, start + clock.n_rows * self.metrics.period_s * US_PER_S)


def build_record(case: Case, telemetry: CaseTelemetry, reader: TableReader) -> Record:
    """Build a case's Record: one Source per telemetry file, and its inject time and system as annotations.

    Args:
        case: The case.
        telemetry: The shapes of its files.
        reader: The reader the Signals reread their files through.

    Returns:
        The Record, whose values stay in the zip until read.

    Raises:
        TimeFFormatError: If the inject time has no metric row before it, or none from it.
    """
    origin_us = telemetry.origin_us()
    metrics = telemetry.metrics_span(origin_us)
    inject_us = case.inject_s * US_PER_S - origin_us
    if not metrics.start_us < inject_us < metrics.end_us:
        raise TimeFFormatError(f"{case.path}: inject time {case.inject_s} is not inside the metrics")
    sources = [_grid_source(SourceName.METRICS, telemetry.metrics, origin_us, reader, _metric_spec)]
    for table in (telemetry.logs, telemetry.traces):
        if table is not None:
            sources.append(_event_source(table, origin_us, reader))
    for name, table in telemetry.derived.items():
        sources.append(_grid_source(name, table, origin_us, reader, partial(_derived_spec, name)))
    metadata = RCAEvalRecordMetadata(subject_id=case.suite.system.display_name, suite=case.suite)
    record = Record(start_time=TimeOrigin(origin_us), sources=tuple(sources), metadata=metadata.to_json())
    record.add_annotations(
        (
            Annotation(
                key=AnnotationKey.INJECT_TIME,
                value=case.inject_s,
                span=TimePoint.micros(inject_us),
                description="The fault injection time, in Unix seconds, from inject_time.txt; main.py gives it "
                "to every method",
            ),
            Annotation(key=AnnotationKey.SYSTEM, value=case.suite.system.display_name),
        )
    )
    return record


def _grid_source(
    name: SourceName,
    table: GridTable,
    origin_us: int,
    reader: TableReader,
    spec_of: Callable[[str], TimeSeriesSpec],
) -> Source:
    signals = []
    for column, clock_column in table.columns.items():
        clock = table.clocks[clock_column]
        signals.append(
            Signal.from_loader(
                name=column,
                spec=spec_of(column),
                time_axis=RegularAxis(
                    period_us=Fraction(table.period_s * US_PER_S), offset_us=clock.first_s * US_PER_S - origin_us
                ),
                n_values=clock.n_rows,
                loader=partial(reader.grid_values, table, column),
            )
        )
    metadata: JsonMapping = {SOURCE_FILE: table.member.name.rpartition("/")[2], COLUMN_ORDER: list(table.columns)}
    if len(table.clocks) > 1:
        # A RE1-OB data.csv can hold two tables side by side, each timed by its own clock.
        metadata[CLOCKS] = {
            clock: [column for column, timed_by in table.columns.items() if timed_by == clock] for clock in table.clocks
        }
    return Source(name=name, signals=tuple(signals), metadata=metadata)


def _event_source(table: EventTable, origin_us: int, reader: TableReader) -> Source:
    axis = IrregularAxis(first_us=table.first_us - origin_us, last_us=table.last_us - origin_us)
    offsets = partial(reader.event_offsets, table, origin_us)
    columns = event_signal_columns(table)
    signals = tuple(
        Signal.from_loader(
            name=column,
            spec=_event_spec(table.kind, column),
            time_axis=axis,
            time_offsets_loader=offsets,
            n_values=table.n_rows,
            loader=partial(reader.event_values, table, column),
        )
        for column in columns
    )
    return Source(
        name=_EVENT_SOURCES[table.kind],
        signals=signals,
        metadata={SOURCE_FILE: table.member.name.rpartition("/")[2], COLUMN_ORDER: list(table.columns)},
    )


# The release states no units, so values have none. TimeNet reads None as unknown, not dimensionless.
def _metric_spec(column: str) -> TimeSeriesSpec:
    metric = column.partition("_")[2]
    return TimeSeriesSpec(
        spec_type=f"rcaeval_metric_{metric.replace('-', '_')}",
        name=f"RCAEval {metric} metric",
        unit_value=None,
        dtype="float64",
        nullable=True,
    )


def _derived_spec(name: SourceName, _column: str) -> TimeSeriesSpec:
    return TimeSeriesSpec(
        spec_type=f"rcaeval_{name}",
        name=f"RCAEval {name} series",
        unit_value=None,
        dtype="float64",
        nullable=True,
    )


def _event_spec(kind: EventKind, column: str) -> TimeSeriesSpec:
    spec_type = f"rcaeval_{kind}_{column}"
    name = f"RCAEval {kind} {column}"
    if column == LOG_SUB_MICROSECOND:
        return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=ureg.nanosecond, dtype="uint16")
    if column == TRACE_SOURCE_ROW:
        return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=None, dtype="uint32")
    if kind is EventKind.TRACES and column in TRACE_NUMBERS:
        return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=None, dtype="float64", nullable=True)
    return TimeSeriesSpec(
        spec_type=spec_type,
        name=name,
        unit_value=None,
        dtype="str",
        nullable=True,
        modality=InputModality.TEXT,
    )
