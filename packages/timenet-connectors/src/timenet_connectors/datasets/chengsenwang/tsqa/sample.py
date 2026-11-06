"""Parsed TSQA rows and the keys used for dataset splitting."""

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, Json

from timenet.types import TimeSeriesSpec
from timenet_connectors.sources.rows import NonBlankText, SourceRow, validate_row
from timenet_connectors.sources.signals import numeric_signal


_NumericSeries = Annotated[list[int | float], Field(min_length=1)]
_SPEC = TimeSeriesSpec(spec_type="series", name="Series", unit_value=None, dtype="float64")


class _TSQARow(SourceRow):
    """Published TSQA columns; series may be flat or multichannel JSON."""

    task: NonBlankText = Field(alias="Task")
    question: NonBlankText = Field(alias="Question")
    answer: NonBlankText = Field(alias="Answer")
    label: str | None = Field(default=None, alias="Label")
    size: int | None = Field(default=None, alias="Size")
    series: Json[_NumericSeries | Annotated[list[_NumericSeries], Field(min_length=1)]] = Field(alias="Series")


@dataclass(frozen=True)
class TSQASample:
    """One parsed source row."""

    category: str
    question: str
    answer: str
    label: str | None
    signals: tuple[tuple[float, ...], ...]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TSQASample":
        """Parse a row once, normalizing flat series to one channel.

        Returns:
            A sample whose numeric signal values also serve as its grouping key.
        """
        parsed = validate_row(row, _TSQARow, location="TSQA row")
        series = parsed.series
        channels = series if isinstance(series[0], list) else [series]
        return cls(
            category=parsed.task,
            question=parsed.question,
            answer=parsed.answer,
            label=parsed.label,
            signals=tuple(
                tuple(numeric_signal(channel, f"Series[{index}]", spec=_SPEC).to_pylist())
                for index, channel in enumerate(channels)
            ),
        )


def get_category(sample: TSQASample) -> str:
    """Return the source question category used for stratification.

    Returns:
        The row's category, without a hardcoded category list.
    """
    return sample.category


def get_signal_id(sample: TSQASample) -> tuple[tuple[float, ...], ...]:
    """Identify identical signals by their parsed values and channel layout.

    Use this key to keep questions about the same numeric series together across
    train, validation, and test. Row IDs identify questions and cannot prevent
    signal leakage. Parsing normalizes JSON whitespace and numeric notation.

    This detects exact duplicate series, not overlapping windows or transformed
    copies. Those require a shared source-recording or subject ID when available.

    Returns:
        A content key shared by questions about the same numeric series.
    """
    return sample.signals
