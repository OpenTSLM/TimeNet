"""Parsed TSQA rows and the keys used for dataset splitting."""

from dataclasses import dataclass
import json
from typing import Any

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.fields import as_list, numbers, text, text_field


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

        Raises:
            TimeNetConnectorError: If the JSON or a decoded field is invalid.
        """
        try:
            series = as_list(json.loads(text_field(row, "Series")), "Series", nonempty=True)
        except json.JSONDecodeError as exc:
            raise TimeNetConnectorError(f"Series: invalid JSON: {exc}") from exc
        channels = series if series and isinstance(series[0], list) else [series]
        return cls(
            category=text_field(row, "Task"),
            question=text_field(row, "Question"),
            answer=text_field(row, "Answer"),
            label=None if row.get("Label") is None else text(row["Label"], "Label", allow_blank=True),
            signals=tuple(
                tuple(numbers(channel, f"Series[{index}]").tolist()) for index, channel in enumerate(channels)
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
