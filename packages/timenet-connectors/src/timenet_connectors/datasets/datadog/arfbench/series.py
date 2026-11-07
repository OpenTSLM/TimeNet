"""The metric series of ARFBench: one file per metric and rollup interval, one Record per metric."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import Record, RegularAxis, Signal, Source
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, MissingReason, MissingValues, TimeOrigin, TimeSeriesSpec
from timenet.types.clock import US_PER_S


UNTAGGED = "untagged"  # the source name for the empty tag label of a metric with no grouping

# The release states no unit for any metric, and it stores values as float64. A rollup file lists
# the tag groups of a metric on one time grid. A step at which the metric reported nothing for a
# group holds a null, or the file skips it. Inside a group's observed window both are a missing
# timestep. In the 1.0.0 release, 42 of 142 metrics have such gaps, and a sparse metric misses up
# to 99% of the steps of a fine grid. The release records each series' null count.
METRIC = TimeSeriesSpec(
    spec_type="metric",
    name="Observability metric",
    unit_value=None,
    dtype="float64",
    missing=MissingValues(
        reason=MissingReason.NOT_RECORDED,
        description=(
            "The metric reported no data point for this step of the rollup grid. A sparse metric can "
            "miss almost every step of a fine rollup."
        ),
    ),
)


@dataclass(frozen=True)
class TagGroupSeries:
    """One tag group's values on its rollup grid, from the group's first value to its last."""

    start_us: int
    """The timestamp of the first value, in microseconds since the epoch."""
    values: pa.Array
    """The value at each grid step, with a null at each step without a value."""

    @classmethod
    def from_rows(cls, epochs_us: np.ndarray, values: np.ndarray, missing: np.ndarray, *, period_us: int) -> Self:
        """Place the rows of one group on the grid of its rollup.

        The rows must be in time order and start and end with a value. A step that the rows skip
        is a missing timestep, like a row that holds a null.

        Args:
            epochs_us: The timestamp of each row, in microseconds since the epoch.
            values: The value of each row. The value of a row that holds a null is undefined.
            missing: Whether each row holds a null.
            period_us: The rollup interval in microseconds.

        Returns:
            The group's series on the grid.

        Raises:
            TimeNetConnectorError: If a row is off the grid, or two rows share a step.
        """
        steps = epochs_us - epochs_us[0]
        if np.any(steps % period_us) or np.any(np.diff(steps) <= 0):
            raise TimeNetConnectorError(f"rows are off the {period_us // US_PER_S} s grid or repeat a step")
        steps //= period_us
        grid_values = np.full(int(steps[-1]) + 1, np.nan)
        grid_values[steps] = values
        present = np.zeros(len(grid_values), dtype=bool)
        present[steps[~missing]] = True
        return cls(start_us=int(epochs_us[0]), values=pa.array(grid_values, mask=~present))

    def to_signal(self, *, period_us: int, origin_us: int, name: str) -> Signal:
        """Build the signal for this group at its rollup interval.

        Args:
            period_us: The rollup interval in microseconds.
            origin_us: The record's start, which the axis offset counts from.
            name: The signal name.

        Returns:
            The signal on a regular axis, with a missing timestep at each step without a value.
        """
        axis = RegularAxis(period_us=Fraction(period_us), offset_us=self.start_us - origin_us)
        return Signal.from_values(self.values, spec=METRIC, name=name, time_axis=axis)


@dataclass(frozen=True)
class RollupFile:
    """One series file: the values of one metric at one rollup interval."""

    metric: str
    """The metric ID, such as ``35928_0``."""
    interval_s: int
    """The rollup interval in seconds."""
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Parse the metric ID and the interval from a file name such as ``35928_0_10.parquet``.

        Returns:
            The rollup file.

        Raises:
            TimeNetConnectorError: If the file name does not end in an interval.
        """
        metric, _, interval_s = path.stem.rpartition("_")
        if not metric or not interval_s.isdigit():
            raise TimeNetConnectorError(f"{path.name} is not named {{metric}}_{{interval}}.parquet")
        return cls(metric=metric, interval_s=int(interval_s), path=path)

    @property
    def period_us(self) -> int:
        """The rollup interval in microseconds."""
        return self.interval_s * US_PER_S

    def read_tag_group_series(self) -> dict[str, TagGroupSeries]:
        """Read the file and cut each tag group to its observed window.

        The file lists the tag groups of the metric on one time grid, with a null where the metric
        reported nothing. Most files list every group at every step. Steps before a group's first
        value and after its last one are grid padding, so each group is cut to its observed window.
        A group with no value is left out. Rows are sorted by group and time, because the file
        order is arbitrary.

        Returns:
            Each tag label mapped to its series on the grid.

        Raises:
            TimeNetConnectorError: If a group has rows off the grid. The message names the file
                and the group.
        """
        table = pq.read_table(self.path, columns=["epoch", "group", "value"])
        table = table.sort_by([("group", "ascending"), ("epoch", "ascending")])
        labels = table.column("group").to_numpy(zero_copy_only=False)
        epochs = table.column("epoch").to_numpy().astype("datetime64[us]").astype("int64")
        values = table.column("value").to_numpy(zero_copy_only=False)
        missing = table.column("value").is_null().to_numpy(zero_copy_only=False)
        bounds = [0, *(np.flatnonzero(labels[1:] != labels[:-1]) + 1), len(labels)]
        groups = {}
        for start, stop in pairwise(bounds):
            observed = np.flatnonzero(~missing[start:stop]) + start
            if not len(observed):
                continue
            window = slice(int(observed[0]), int(observed[-1]) + 1)
            label = str(labels[start])
            try:
                groups[label] = TagGroupSeries.from_rows(
                    epochs[window], values[window], missing[window], period_us=self.period_us
                )
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{self.path.name} group {label!r}: {exc}") from exc
        return groups

    def read_query_name(self) -> str:
        """Read the ``query_name`` column, which holds one value per file.

        Returns:
            The identifier that the release gives the metric. It differs from the metric ID for
            most metrics, and the release does not explain it.
        """
        return pq.read_table(self.path, columns=["query_name"]).column("query_name")[0].as_py()


@dataclass(frozen=True)
class MetricSeriesFiles:
    """The series files of one metric, one per rollup interval."""

    metric: str
    """The metric ID, such as ``35928_0``."""
    rollups: tuple[RollupFile, ...]
    """The metric's rollup files, finest interval first."""

    @classmethod
    def from_series_dir(cls, series_dir: Path) -> list[Self]:
        """Group the series files of a directory by metric.

        Returns:
            The metrics in ID order, each with its rollups finest first.
        """
        by_metric: dict[str, list[RollupFile]] = {}
        for path in sorted(series_dir.glob("*.parquet")):
            rollup = RollupFile.from_path(path)
            by_metric.setdefault(rollup.metric, []).append(rollup)
        return [
            cls(metric=metric, rollups=tuple(sorted(rollups, key=lambda rollup: rollup.interval_s)))
            for metric, rollups in sorted(by_metric.items())
        ]

    @property
    def finest(self) -> RollupFile:
        """The rollup file with the smallest interval."""
        return self.rollups[0]

    def create_record(self) -> Record:
        """Create the metric's Record with a Source for each tag group.

        Each source holds the group's series at every rollup where the group has a value. A group
        with no value at any rollup gets no source. The record starts at the earliest observation.

        Returns:
            The Record, not yet added to a dataset, with the metric IDs and the finest interval as
            annotations.
        """
        series_by_rollup = {rollup: rollup.read_tag_group_series() for rollup in self.rollups}
        origin_us = min(group.start_us for groups in series_by_rollup.values() for group in groups.values())
        sources = []
        for label in sorted({label for groups in series_by_rollup.values() for label in groups}):
            signals = tuple(
                groups[label].to_signal(period_us=rollup.period_us, origin_us=origin_us, name=f"{rollup.interval_s}s")
                for rollup, groups in series_by_rollup.items()
                if label in groups
            )
            sources.append(Source(name=label or UNTAGGED, signals=signals))
        record = Record(
            metadata={"source_record_id": f"arfbench-{self.metric}"},
            start_time=TimeOrigin(origin_us),
            sources=tuple(sources),
        )
        record.add_annotations(
            [
                Annotation(key="metric_id", value=self.metric),
                Annotation(key="incident_id", value=self.metric.partition("_")[0]),
                Annotation(key="query_name", value=self.finest.read_query_name()),
                Annotation(key="finest_interval_s", value=self.finest.interval_s, unit="second"),
            ]
        )
        return record
