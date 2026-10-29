"""Load RCAEval's zips into one lazy Record per case, and the case's service and indicator tasks."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.dataset import Record, Signal
from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, TimeInterval
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.phamquiluan.rcaeval import benchmark
from timenet_connectors.datasets.phamquiluan.rcaeval.cases import ROOT_CAUSE_FILE, Case, list_cases
from timenet_connectors.datasets.phamquiluan.rcaeval.keys import SourceName
from timenet_connectors.datasets.phamquiluan.rcaeval.records import CaseTelemetry, build_record
from timenet_connectors.datasets.phamquiluan.rcaeval.release import (
    PINNED_ARCHIVES,
    PINNED_UNTIMED_ROWS,
    FoundArchive,
    PinnedArchive,
    Suite,
    UntimedRows,
)
from timenet_connectors.datasets.phamquiluan.rcaeval.tables import TableReader
from timenet_connectors.datasets.phamquiluan.rcaeval.tasks import RCAEvalIndicatorTask, RCAEvalServiceTask, RCAEvalTask


@dataclass(frozen=True)
class _ParsedCase:
    """One case after parsing: the case, its lazy Record, its window, and its indicator, if its fault has one."""

    case: Case
    record: Record
    scope: TimeInterval
    indicator: tuple[Signal, str] | None
    """The indicator's Signal, and the indicator as ``main.py`` names it."""


class RCAEvalDatasetLoader(DatasetLoader[RCAEvalTask]):
    """RCAEval's zips, read and checked: one lazy Record per case, and its tasks at both levels.

    RCAEval defines one task, root cause analysis, scored at two levels, each a kind here:

    - Service (:class:`RCAEvalServiceTask`): name the root cause service. Every case has one.
    - Indicator (:class:`RCAEvalIndicatorTask`): name the root cause indicator, a metric of that
      service. Every case whose fault ``main.py`` scores an indicator for has one: RE1 and RE2.

    No task has a split. Select one kind with ``tasks_of``.

    The loader keeps each file's shape, not its values. The Signals reread the zips, so the zips must stay
    available until writing ends.
    """

    TASK_KINDS = (RCAEvalServiceTask, RCAEvalIndicatorTask)

    def __init__(self, cases: tuple[_ParsedCase, ...]) -> None:
        self._cases = cases

    @classmethod
    def parse_from(
        cls,
        refs: Iterable[Path],
        *,
        archives: Mapping[Suite, PinnedArchive] = PINNED_ARCHIVES,
        untimed: Mapping[tuple[str, str], UntimedRows] = PINNED_UNTIMED_ROWS,
    ) -> Self:
        """Read and check every case of each zip once, and build one lazy Record per case.

        Args:
            refs: The suites' zips, each named as in the record, such as ``RE2-OB.zip``, each at most once.
            archives: Each zip's pinned inventory.
            untimed: The pinned rows without a time, by case and clock column; any other such row fails.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a file isn't a suite's zip, a suite comes twice, or a zip's inventory or
                rows without a time differ from the pinned ones.
        """
        paths = list(refs)
        by_name = {suite.archive_filename: suite for suite in Suite}
        zips = [(path, by_name[path.name]) for path in paths if path.name in by_name]
        if not paths or len(zips) != len(paths) or len({suite for _, suite in zips}) != len(zips):
            raise TimeFValidationError(f"RCAEval loads each suite's zip at most once, got {[p.name for p in paths]}")
        reader = TableReader()
        parsed: list[_ParsedCase] = []
        found_untimed: dict[tuple[str, str], UntimedRows] = {}
        for path, suite in zips:
            found = {"cases": 0, "logs": 0, "traces": 0, "header_only_traces": 0, "derived": 0, "root_cause_files": 0}
            for case in list_cases(path, suite):
                telemetry = CaseTelemetry.read(case)
                for clock in telemetry.metrics.clocks.values():
                    if clock.untimed_rows:
                        found_untimed[case.path, clock.column] = UntimedRows(clock.untimed_rows, clock.untimed_values)
                parsed.append(_parse_case(case, telemetry, reader))
                found["cases"] += 1
                found["logs"] += telemetry.logs is not None
                found["traces"] += telemetry.traces is not None
                found["header_only_traces"] += telemetry.header_only_traces
                found["derived"] += SourceName.LOGTS in telemetry.derived
                found["root_cause_files"] += ROOT_CAUSE_FILE in case.files
            FoundArchive(**found).check(path, archives[suite])
        loaded = {suite.value for _, suite in zips}
        expected = {key: rows for key, rows in untimed.items() if key[0].split("/")[0] in loaded}
        if found_untimed != expected:
            raise TimeFValidationError(
                f"metric rows without a time are {found_untimed}, the pinned zips hold {expected}"
            )
        return cls(tuple(parsed))

    def records(self) -> Iterator[Record]:
        """Return one Record per case, in zip then path order, whose values stay in the zips until written."""
        return (parsed.record for parsed in self._cases)

    def tasks(self) -> Iterator[RCAEvalTask]:
        """Return each case's service task, then its indicator task if its fault has one, in Record order.

        Yields:
            The tasks.
        """
        for parsed in self._cases:
            yield RCAEvalServiceTask(case=parsed.case, record=parsed.record, scope=parsed.scope)
            if parsed.indicator is not None:
                signal, authors_name = parsed.indicator
                yield RCAEvalIndicatorTask(
                    case=parsed.case,
                    record=parsed.record,
                    scope=parsed.scope,
                    indicator=signal,
                    authors_indicator=authors_name,
                )


def _parse_case(case: Case, telemetry: CaseTelemetry, reader: TableReader) -> _ParsedCase:
    record = build_record(case, telemetry, reader)
    origin_us = telemetry.origin_us()
    metrics = telemetry.metrics_span(origin_us)
    inject_us = case.inject_s * US_PER_S - origin_us
    scope = benchmark.window(inject_us, metrics.start_us, metrics.end_us)
    indicator = None
    metric = case.fault.indicator_metric(case.suite)
    if metric is not None:
        column = benchmark.indicator_column(list(telemetry.metrics.columns), case.service, metric, context=case.path)
        (source,) = (source for source in record.sources if source.name == SourceName.METRICS)
        (signal,) = (signal for signal in source.signals if signal.name == column)
        indicator = (signal, f"{case.service}_{metric}")
    return _ParsedCase(case=case, record=record, scope=scope, indicator=indicator)
