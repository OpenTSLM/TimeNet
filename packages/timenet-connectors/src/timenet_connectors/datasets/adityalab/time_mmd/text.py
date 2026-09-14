"""Turn the textual files of one domain into annotations on the timeline of its record.

The release pairs each domain with two textual files. ``*_report.csv`` holds what a language
model extracted from the reports the authors selected for the domain, and ``*_search.csv``
holds the same for the web search results of each week. Both share one shape: a row index, a
``start_date``, an ``end_date``, a ``fact`` and a ``preds`` field, which the README of the release
spells ``pred``.

A ``fact`` is the objective statement the model extracted. ``preds`` holds the predictions it
extracted for the long term and the short term, joined by a semicolon. The model was told to
write ``NA`` where it found nothing, and Appendix F of the Time-MMD paper states that prompt.

Each field that states something becomes one annotation, over the days its row states. A field
that holds ``NA``, or nothing, states that nothing was found, and it becomes no annotation.
Nothing here rewrites, splits or re-dates a text. The README states why.
"""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

from timenet.errors import TimeFFormatError
from timenet.types import Annotation
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, TextKind
from timenet_connectors.datasets.adityalab.time_mmd.timeline import Timeline


# The columns every textual file of the release carries, after its unnamed row index. The files
# spell the predictions column ``preds``, and the README of the release spells it ``pred``. A file
# holds exactly one of the two.
START_COLUMN = "start_date"
END_COLUMN = "end_date"
FACT_COLUMN = "fact"
PREDICTION_COLUMNS = ("preds", "pred")

# The token the prompt told the model to write where it found nothing, on its own or with a
# note after it. The lookahead rules out a word that starts with the same letters, such as NASA.
_NOT_AVAILABLE = re.compile(r"^\s*N/?A(?![A-Za-z0-9])")

# The separator between the long-term and the short-term prediction of a ``preds`` field.
_PREDICTION_SEPARATOR = ";"

# The model that wrote every text of the release (Time-MMD paper, section 2.2).
_MODEL = "Llama3-70B"

_FACT_KEYS = {TextKind.REPORT: AnnotationKey.REPORT_FACT, TextKind.SEARCH: AnnotationKey.SEARCH_FACT}
_PREDICTION_KEYS = {
    TextKind.REPORT: AnnotationKey.REPORT_PREDICTION,
    TextKind.SEARCH: AnnotationKey.SEARCH_PREDICTION,
}

# What each key states. Every annotation of one key states the same thing, so the description
# belongs to the key and the schema holds it once.
_DESCRIPTIONS = {
    AnnotationKey.REPORT_FACT: (
        "An objective fact the model extracted from a report the release selected for the domain, over the "
        "days the span covers. As the release writes it."
    ),
    AnnotationKey.REPORT_PREDICTION: (
        "The long-term and then the short-term prediction the model extracted from that report, joined by a "
        "semicolon. As the release writes them."
    ),
    AnnotationKey.SEARCH_FACT: (
        "An objective fact the model extracted from the web search results of the week the span covers. As the "
        "release writes it."
    ),
    AnnotationKey.SEARCH_PREDICTION: (
        "The long-term and then the short-term prediction the model extracted from those search results, joined "
        "by a semicolon. As the release writes them."
    ),
}


@dataclass(frozen=True)
class TextRow:
    """One row of a textual file: the days it covers, and the two texts it states."""

    row: int
    """The position of the row in its file, counted from 0. The release writes the same number in
    its first column, and this connector counts instead of reading it."""
    start: date
    """The first day the row covers."""
    end: date
    """The last day it covers, inclusive."""
    fact: str
    """The ``fact`` field, verbatim."""
    predictions: str
    """The ``preds`` field, verbatim."""


def read_rows(path: Path) -> tuple[TextRow, ...]:
    """Read one textual file.

    Args:
        path: The file.

    Returns:
        Its rows, in file order. A file with no row gives an empty tuple.

    Raises:
        TimeFFormatError: If the file lacks a column, holds both spellings of the predictions column
            or neither, holds a row with a field missing, states a date that does not parse, or
            states an end before its start.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or ()
        missing = [column for column in (START_COLUMN, END_COLUMN, FACT_COLUMN) if column not in header]
        if missing:
            raise TimeFFormatError(f"{path.name}: lacks the columns {missing}")

        predictions = [column for column in PREDICTION_COLUMNS if column in header]
        if len(predictions) != 1:
            raise TimeFFormatError(
                f"{path.name}: holds the prediction columns {predictions}, and a file holds one of {list(PREDICTION_COLUMNS)}"
            )

        needed = (START_COLUMN, END_COLUMN, FACT_COLUMN, predictions[0])
        rows: list[TextRow] = []
        for index, row in enumerate(reader):
            line = index + 2  # the header is line 1
            fields = {column: row.get(column) for column in needed}
            absent = [column for column, text in fields.items() if text is None]
            if absent:
                raise TimeFFormatError(f"{path.name} line {line}: holds no {absent} field")

            start = _parse_date(fields[START_COLUMN] or "", path, START_COLUMN, line)
            end = _parse_date(fields[END_COLUMN] or "", path, END_COLUMN, line)
            if end < start:
                raise TimeFFormatError(
                    f"{path.name} line {line}: ends on {end.isoformat()}, before it starts on {start.isoformat()}"
                )

            rows.append(
                TextRow(
                    row=index,
                    start=start,
                    end=end,
                    fact=fields[FACT_COLUMN] or "",
                    predictions=fields[predictions[0]] or "",
                )
            )
    return tuple(rows)


def states_nothing(text: str) -> bool:
    """Say whether a field states nothing: it is empty, or holds the ``NA`` the prompt allowed.

    Args:
        text: The field.

    Returns:
        ``True`` for an empty field, or one that starts with ``NA`` or ``N/A`` as a word.
    """
    return not text.strip() or _NOT_AVAILABLE.match(text) is not None


def predictions_state_nothing(text: str) -> bool:
    """Say whether a ``preds`` field states nothing in any of its parts.

    A field that holds ``NA`` for the long term and a prediction for the short term still
    states something, so it is kept whole. Only a field whose every part states nothing is
    dropped.

    Args:
        text: The field.

    Returns:
        ``True`` when every semicolon-separated part states nothing.
    """
    return all(states_nothing(part) for part in text.split(_PREDICTION_SEPARATOR))


def annotation_id(record_id: str, kind: TextKind, field: str, row: int) -> str:
    """Give the id of the annotation one field of one row becomes.

    Args:
        record_id: The record the annotation belongs to.
        kind: Which textual file the row is in.
        field: ``fact`` or ``prediction``.
        row: The position of the row in its file, counted from 0.

    Returns:
        The id, which two builds of one release give alike.
    """
    return f"{record_id}-{kind}-{field}-{row}"


def build_annotations(
    record_id: str, kind: TextKind, rows: tuple[TextRow, ...], timeline: Timeline
) -> list[Annotation]:
    """Turn the rows of one textual file into annotations, over the days each row states.

    Args:
        record_id: The record the annotations belong to.
        kind: Which textual file the rows are from, which picks their keys.
        rows: The rows, from :func:`read_rows`.
        timeline: Where the record's zero sits, which places the days.

    Returns:
        The annotations, in row order, a fact before the predictions of the same row. A field
        that states nothing gives none.
    """
    built: list[Annotation] = []
    for row in rows:
        span = timeline.interval(row.start, row.end)
        if not states_nothing(row.fact):
            key = _FACT_KEYS[kind]
            built.append(
                Annotation(
                    key=key,
                    value=row.fact,
                    span=span,
                    description=_DESCRIPTIONS[key],
                    source=_MODEL,
                    id=annotation_id(record_id, kind, "fact", row.row),
                )
            )
        if not predictions_state_nothing(row.predictions):
            key = _PREDICTION_KEYS[kind]
            built.append(
                Annotation(
                    key=key,
                    value=row.predictions,
                    span=span,
                    description=_DESCRIPTIONS[key],
                    source=_MODEL,
                    id=annotation_id(record_id, kind, "prediction", row.row),
                )
            )
    return built


def _parse_date(text: str, path: Path, column: str, line: int) -> date:
    """Parse one ISO date.

    Args:
        text: The field.
        path: The file, for the error message.
        column: The header of the field, for the error message.
        line: The line of the row in the file, counted from 1, for the error message.

    Returns:
        The date.

    Raises:
        TimeFFormatError: If the field is not a date of the form ``YYYY-MM-DD``.
    """
    try:
        return date.fromisoformat(text.strip())
    except ValueError as exc:
        raise TimeFFormatError(
            f"{path.name} line {line}: {column!r} holds {text!r}, which is not a YYYY-MM-DD date"
        ) from exc
