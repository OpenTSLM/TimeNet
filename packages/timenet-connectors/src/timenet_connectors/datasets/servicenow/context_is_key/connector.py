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
import math
import re
from typing import Annotated, ClassVar, Literal, NamedTuple, Self, TypeAlias

from jaxtyping import Float64, Int64
import numpy as np
from pydantic import BeforeValidator, ConfigDict, Field, Json, StrictFloat, StrictInt, StrictStr, model_validator

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis, time_offsets_from_datetimes
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import ForecastingTask, TimeSeriesSpec, ureg
from timenet.types.clock import TimeOrigin
from timenet.types.splits import Split
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.readers import DataRow, ParquetDataReader
from timenet_connectors.time_axes import axis_for_offsets


_SPEC = TimeSeriesSpec(
    spec_type="cik_series",
    name="Context is Key Series",
    unit_value=ureg.dimensionless,
    dtype="float64",
)


# A monthly series names each step as "2022-07", which datetime.fromisoformat does not read.
_MONTH = re.compile(r"\d{4}-\d{2}")
_MONTH_FORMAT = "%Y-%m"

FiniteFloat: TypeAlias = Annotated[StrictFloat, Field(allow_inf_nan=False)]
"""A float that is neither infinite nor NaN."""

ColumnName: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
"""The name the source gives a column, such as ``"0"`` or ``"GHI"``. It becomes the signal name."""

SourceTimestamp: TypeAlias = Annotated[
    StrictStr, Field(pattern=r"^\d{4}-\d{2}(-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{3})?)?)?$")
]
"""A naive timestamp in one of the source's three formats: ``"2024-07-01T00:00:00.000"``,
``"2006-10-08 23:50"``, or ``"2022-07"`` for a month."""

SourceValue: TypeAlias = FiniteFloat | StrictInt
"""One observed value. The source writes whole values as JSON integers. A null or a NaN is invalid."""

SeriesCells: TypeAlias = Annotated[dict[SourceTimestamp, SourceValue], Field(min_length=1)]
"""One column: at least one ``{timestamp: value}`` pair. The time axis rejects pairs out of time order."""

SeriesFrame: TypeAlias = Annotated[dict[ColumnName, SeriesCells], Field(min_length=1)]
"""A history or a future: ``{column name: {timestamp: value}}``, with at least one column."""

ContextSource: TypeAlias = Literal["c_i", "c_h", "c_f", "c_cov", "c_causal"]
"""A kind of text context, as the paper names them: intemporal, historical, future, covariate, causal."""

LowerBound: TypeAlias = Annotated[
    FiniteFloat | None, BeforeValidator(lambda value: None if value == -math.inf else value)
]
"""A lower bound on the forecast. The source writes "no bound" as -inf, which JSON cannot hold, so it
becomes ``None``."""

UpperBound: TypeAlias = Annotated[
    FiniteFloat | None, BeforeValidator(lambda value: None if value == math.inf else value)
]
"""An upper bound on the forecast. The source writes "no bound" as +inf, which becomes ``None``."""

TimestepIndex: TypeAlias = Annotated[StrictInt, Field(ge=0)]
"""An index into the future values."""


class _SourceRow(DataRow):
    """Columns consumed from a Context is Key Parquet row."""

    model_config = ConfigDict(extra="ignore")

    name: Annotated[StrictStr, Field(min_length=1)]
    seed: Annotated[StrictInt, Field(ge=0)]
    weight: Annotated[StrictStr, Field(pattern=r"^\d+(/\d+)?$")]
    """The instance's weight in the benchmark score, as a fraction such as ``"1/3"``."""
    context_sources: Annotated[list[ContextSource], Field(min_length=1)]
    skills: list[Annotated[StrictStr, Field(min_length=1)]]
    background: StrictStr
    scenario: StrictStr
    constraints: StrictStr
    seasonal_period: Literal[-1] | Annotated[StrictInt, Field(ge=1)]
    """The season length in timesteps. The source writes -1 for a series without one."""
    past_time: Json[SeriesFrame]
    future_time: Json[SeriesFrame]
    metric_scaling: Annotated[FiniteFloat, Field(gt=0)]
    region_of_interest: list[TimestepIndex]
    constraint_min: LowerBound
    constraint_max: UpperBound
    constraint_variable_max_index: list[TimestepIndex]
    constraint_variable_max_values: list[FiniteFloat]

    @model_validator(mode="after")
    def _check_indices(self) -> Self:
        """Check that every timestep index points into the future and that each indexed bound has a value.

        Returns:
            The validated row.

        Raises:
            ValueError: If an index is past the shortest future column, or the two
                ``constraint_variable_max_*`` lists differ in length. Pydantic reports it as a validation error.
        """
        horizon = min(len(cells) for cells in self.future_time.values())
        indices = [*self.region_of_interest, *self.constraint_variable_max_index]
        if any(index >= horizon for index in indices):
            raise ValueError(f"a timestep index is past the {horizon} future values")
        if len(self.constraint_variable_max_index) != len(self.constraint_variable_max_values):
            raise ValueError("constraint_variable_max_index and constraint_variable_max_values differ in length")
        return self


def _parse_moment(text: str) -> datetime:
    """Read one source timestamp as UTC. A ``YYYY-MM`` month reads as its first day.

    Args:
        text: The timestamp as the source writes it.

    Returns:
        The timezone-aware moment.

    Raises:
        TimeNetConnectorError: If ``text`` names no real date, such as a 13th month.
    """
    try:
        moment = datetime.strptime(text, _MONTH_FORMAT) if _MONTH.fullmatch(text) else datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimeNetConnectorError(f"Context is Key timestamp {text!r} is not a valid date") from exc
    return moment.replace(tzinfo=UTC)


@dataclass(frozen=True)
class _Column:
    """One column of a history or a future: its moments and values, in source order."""

    name: str
    moments: tuple[datetime, ...]
    values: Float64[np.ndarray, " time"]
    monthly: bool
    """Whether the source names the steps as months. Months differ in length, so such a series is
    irregular even when the months of one short window happen to be equally long."""


def _columns(frame: SeriesFrame) -> list[_Column]:
    """Turn a validated frame into its columns.

    Args:
        frame: A history or a future, as the row schema validated it.

    Returns:
        One column per source column, in source order.
    """
    return [
        _Column(
            name=name,
            moments=tuple(_parse_moment(stamp) for stamp in cells),
            values=np.fromiter(cells.values(), dtype=np.float64, count=len(cells)),
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
    offsets: Int64[np.ndarray, " time"] = time_offsets_from_datetimes(column.moments, start_time=origin)
    axis = None if column.monthly else axis_for_offsets(offsets)
    if isinstance(axis, RegularAxis):
        return Signal.from_values(column.values, spec=_SPEC, name=column.name, time_axis=axis)
    return Signal.from_irregular(column.values, time_offsets_us=offsets, spec=_SPEC, name=column.name)


class _InstanceRecords(NamedTuple):
    """The two records of one forecasting instance, on one time origin."""

    history: Record
    """The values the model sees."""
    future: Record
    """The values the model forecasts."""


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
                records = self._make_records(row)
                dataset.add_record(record=records.history)
                dataset.add_record(record=records.future)
                dataset.add_task(task=self._make_task(row, records))
        return dataset

    @staticmethod
    def _make_records(row: _SourceRow) -> _InstanceRecords:
        """Build the history and future records of one row on a shared time origin.

        The origin is the earliest moment of the history, so the future sits after the history on the
        same timeline.

        Args:
            row: The source row.

        Returns:
            The history record and the future record.
        """
        past = _columns(row.past_time)
        ahead = _columns(row.future_time)
        start = min(moment for column in past for moment in column.moments)
        origin = TimeOrigin(timestamp=start)
        instance = {"task_name": row.name, "seed": row.seed}

        def record(part: str, columns: list[_Column]) -> Record:
            signals = tuple(_signal(column, start) for column in columns)
            return Record(
                sources=(Source(name="Context is Key series", signals=signals),),
                start_time=origin,
                metadata={**instance, "part": part},
            )

        return _InstanceRecords(history=record("history", past), future=record("future", ahead))

    @staticmethod
    def _make_task(row: _SourceRow, records: _InstanceRecords) -> ForecastingTask:
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
            "constraint_min": row.constraint_min,
            "constraint_max": row.constraint_max,
            "constraint_variable_max_index": [*row.constraint_variable_max_index],
            "constraint_variable_max_values": [*row.constraint_variable_max_values],
        }
        return ForecastingTask(
            inputs=(records.history,),
            targets=(records.future,),
            prompt="\n\n".join(part for part in parts if part),
            split=Split.TEST,
            metadata=metadata,
        )


CONNECTOR = ContextIsKeyConnector
