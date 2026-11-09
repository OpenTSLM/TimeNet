"""Parsed TSQA rows and the keys used for dataset splitting."""

from dataclasses import dataclass
import json

from pydantic import Field

from timenet_connectors.sources.readers.base import DataRow


class TSQASourceRow(DataRow):
    """The published columns used to construct a TSQA sample."""

    category: str = Field(alias="Task")
    question: str = Field(alias="Question")
    answer: str = Field(alias="Answer")
    label: str | None = Field(default=None, alias="Label")
    series: str = Field(alias="Series")


@dataclass(frozen=True)
class TSQASample:
    """One parsed source row."""

    category: str
    question: str
    answer: str
    label: str | None
    signals: tuple[tuple[float, ...], ...]

    @classmethod
    def from_row(cls, row: TSQASourceRow) -> "TSQASample":
        """Parse a row once, normalizing flat series to one channel.

        Returns:
            A sample whose numeric signal values also serve as its grouping key.
        """
        series = json.loads(row.series)
        channels = series if series and isinstance(series[0], list) else [series]
        return cls(
            category=row.category,
            question=row.question,
            answer=row.answer,
            label=row.label,
            signals=tuple(tuple(float(value) for value in channel) for channel in channels),
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
