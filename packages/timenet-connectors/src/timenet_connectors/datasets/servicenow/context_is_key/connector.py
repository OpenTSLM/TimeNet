"""The Context is Key connector: a context-aided forecasting benchmark from the HuggingFace Hub.

Source repo ``ServiceNow/context-is-key``. Each row is one forecasting instance: a history and the
future values to forecast, as two JSON objects that map a column name to ``{timestamp: value}``, plus
three text fields (``background``, ``scenario``, ``constraints``) and the fields the benchmark's
metric reads. Each row becomes a history record, a future record, and a
:class:`~timenet.types.ForecastingTask` from the first to the second.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
import itertools
import json
import math
import re
from typing import ClassVar

from pydantic import ConfigDict

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import ForecastingTask, TimeSeriesSpec, ureg
from timenet.types.clock import TimeOrigin, offset_us
from timenet.types.splits import Split
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.readers import DataRow, ParquetDataReader


_SPEC = TimeSeriesSpec(
    spec_type="cik_series",
    name="Context is Key Series",
    unit_value=ureg.dimensionless,
    dtype="float64",
)


# A monthly series names each step as "2022-07", which datetime.fromisoformat does not read.
_MONTH = re.compile(r"\d{4}-\d{2}")
_MONTH_FORMAT = "%Y-%m"


class _SourceRow(DataRow):
    """Columns consumed from a Context is Key Parquet row."""

    model_config = ConfigDict(extra="ignore")

    name: str
    seed: int
    weight: str
    context_sources: list[str]
    skills: list[str]
    background: str
    scenario: str
    constraints: str
    seasonal_period: int
    past_time: str
    future_time: str
    metric_scaling: float
    region_of_interest: list[int]
    constraint_min: float
    constraint_max: float
    constraint_variable_max_index: list[int]
    constraint_variable_max_values: list[float]


def _parse_moment(text: str) -> datetime:
    """Read one source timestamp as UTC. A ``YYYY-MM`` month reads as its first day.

    Args:
        text: The timestamp as the source writes it.

    Returns:
        The timezone-aware moment.

    Raises:
        TimeNetConnectorError: If ``text`` is in none of the source's timestamp formats.
    """
    try:
        moment = datetime.strptime(text, _MONTH_FORMAT) if _MONTH.fullmatch(text) else datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimeNetConnectorError(f"Context is Key timestamp {text!r} is in no known format") from exc
    return moment.replace(tzinfo=UTC)


@dataclass(frozen=True)
class _Column:
    """One column of a history or a future: its moments and values, in source order."""

    name: str
    moments: tuple[datetime, ...]
    values: tuple[float, ...]
    monthly: bool
    """Whether the source names the steps as months. Months differ in length, so such a series is
    irregular even when the months of one short window happen to be equally long."""


def _parse_frame(encoded: str) -> list[_Column]:
    """Decode a ``{column: {timestamp: value}}`` JSON object into its columns."""
    frame: dict[str, dict[str, float]] = json.loads(encoded)
    return [
        _Column(
            name=name,
            moments=tuple(_parse_moment(stamp) for stamp in cells),
            values=tuple(float(value) for value in cells.values()),
            monthly=any(_MONTH.fullmatch(stamp) for stamp in cells),
        )
        for name, cells in frame.items()
    ]


def _signal(column: _Column, origin: datetime) -> Signal:
    """Build one signal on the timeline that starts at ``origin``.

    A series with a constant step gets a regular axis. A monthly series, or one with uneven steps,
    gets an irregular axis.

    Args:
        column: The column to convert.
        origin: The absolute time of the record's relative zero.

    Returns:
        The signal, named as the source names the column.
    """
    offsets = [offset_us(moment, origin) for moment in column.moments]
    steps = {later - earlier for earlier, later in itertools.pairwise(offsets)}
    if len(steps) == 1 and not column.monthly:
        axis = RegularAxis(period_us=Fraction(steps.pop()), offset_us=offsets[0])
        return Signal.from_values(column.values, spec=_SPEC, name=column.name, time_axis=axis)
    return Signal.from_irregular(column.values, time_offsets_us=offsets, spec=_SPEC, name=column.name)


def _finite(value: float) -> float | None:
    """Return ``value``, or ``None`` for the infinity the source uses to say "no bound"."""
    return value if math.isfinite(value) else None


class ContextIsKeyConnector(BaseHuggingFaceConnector):
    """Connector for the Context is Key benchmark (Hub repo ``ServiceNow/context-is-key``)."""

    HF_REPO = "ServiceNow/context-is-key"
    SOURCE_REVISION = "32da467ade142a16d099675e78b1129745c5ef7d"
    # The repo also holds data/ICML2025-*.parquet, the version the paper reports on.
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {"test": "data/test-00000-of-00001.parquet"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Build a history record, a future record, and a forecasting task per row.

        Args:
            raw_refs: The parquet file from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            for row in ParquetDataReader(ref.path, _SourceRow).iter_rows():
                history, future = self._make_records(row)
                dataset.add_record(record=history)
                dataset.add_record(record=future)
                dataset.add_task(task=self._make_task(row, history, future))
        return dataset

    @staticmethod
    def _make_records(row: _SourceRow) -> tuple[Record, Record]:
        """Build the history and future records of one row on a shared time origin.

        Args:
            row: The source row.

        Returns:
            The history record and the future record, in this order.

        Raises:
            TimeNetConnectorError: If the history is empty.
        """
        past = _parse_frame(row.past_time)
        ahead = _parse_frame(row.future_time)
        starts = [column.moments[0] for column in past if column.moments]
        if not starts:
            raise TimeNetConnectorError(f"Context is Key instance {row.name} (seed {row.seed}) has no history")
        start = min(starts)
        origin = TimeOrigin(timestamp=start)
        instance = {"task_name": row.name, "seed": row.seed}

        def record(part: str, columns: list[_Column]) -> Record:
            signals = tuple(_signal(column, start) for column in columns)
            return Record(
                sources=(Source(name="Context is Key series", signals=signals),),
                start_time=origin,
                metadata={**instance, "part": part},
            )

        return record("history", past), record("future", ahead)

    @staticmethod
    def _make_task(row: _SourceRow, history: Record, future: Record) -> ForecastingTask:
        """Build the forecasting task of one row, with the text context as its prompt."""
        parts = (row.background, row.scenario, row.constraints)
        # The list displays let the checker read each list as JSON values: a list[str] is not a list[JsonValue].
        metadata: JsonMapping = {
            "task_name": row.name,
            "seed": row.seed,
            "weight": row.weight,
            "context_sources": [*row.context_sources],
            "skills": [*row.skills],
            "background": row.background,
            "scenario": row.scenario,
            "constraints": row.constraints,
            "seasonal_period": row.seasonal_period,
            "metric_scaling": row.metric_scaling,
            "region_of_interest": [*row.region_of_interest],
            "constraint_min": _finite(row.constraint_min),
            "constraint_max": _finite(row.constraint_max),
            "constraint_variable_max_index": [*row.constraint_variable_max_index],
            "constraint_variable_max_values": [*row.constraint_variable_max_values],
        }
        return ForecastingTask(
            inputs=(history,),
            targets=(future,),
            prompt="\n\n".join(part for part in parts if part),
            split=Split.TEST,
            metadata=metadata,
        )


CONNECTOR = ContextIsKeyConnector
