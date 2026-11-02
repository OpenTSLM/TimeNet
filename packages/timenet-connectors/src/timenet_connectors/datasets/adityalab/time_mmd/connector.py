"""The Time-MMD connector: government time series paired with dated text, for two domains.

Time-MMD aligns a numerical series of each of nine domains with text the authors collected for
it, then filtered and summarized with Llama3-70B. The text comes from two kinds of source. A
*report* is a publication the authors selected for the domain, such as the EIA's Weekly Petroleum
Status Report. A *search* row is a web search result for the domain's keywords. Each row of text
is dated by the period it covers, and the release splits it into the objective facts it states
and the predictions it makes, so that a forecaster can read the facts without the predictions.

This connector converts the Energy and the Environment domain. Each is one record. The numerical
columns are its signals, on a regular weekly or daily cadence. Each row of text is an interval
annotation on the record, spanning the days the row covers, with the facts, the long-term
prediction, and the short-term prediction under separate keys. The record's origin is the earliest date any file states, so that text from
before the numerical series starts still lands on the timeline.

The forecasting tasks follow the protocol the paper evaluates with. See
:mod:`~timenet_connectors.datasets.adityalab.time_mmd.tasks`.
:mod:`~timenet_connectors.datasets.adityalab.time_mmd.sources` declares the domains and reads
their files.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, ForecastingTask, TimeInterval, TimeOrigin, unix_us
from timenet_connectors.datasets.adityalab.time_mmd import sources, tasks
from timenet_connectors.datasets.adityalab.time_mmd.sources import (
    DOMAINS,
    ONE_DAY,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    TARGET_COLUMN,
    Column,
    DomainSource,
    TextKind,
    TextRow,
)
from timenet_connectors.download import Artifact, download_files


_RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_REVISION}"

_TEXT_KINDS: tuple[TextKind, ...] = ("report", "search")

# A text row has two columns. The fact column is one statement. The prediction column holds a
# long-term prediction, a ``;``, then a short-term one, and each half becomes its own annotation.
# The key of an annotation is the kind of source and the field, joined: ``report_fact``,
# ``search_prediction_short``.
_FACT_FIELD = ("fact", "The objective facts the release extracted from the text, verbatim.")
_PREDICTION_FIELDS = (
    ("prediction_long", "The long-term prediction the release extracted from the text, verbatim."),
    ("prediction_short", "The short-term prediction the release extracted from the text, verbatim."),
)
_PREDICTION_SEPARATOR = ";"

# The marker the release writes in a cell, or in one half of a prediction cell, when it extracted
# nothing. A half that holds nothing else states nothing.
_NA = "NA"

_TEXT_SOURCES = {
    "report": "a report the authors selected for this domain",
    "search": "a web search result for this domain's keywords",
}


@dataclass(frozen=True)
class TimeMmdDomainFiles:
    """The three downloaded files of one domain, which ``convert`` reads."""

    domain: str
    """The domain name the release uses, such as ``"Energy"``."""
    numerical: Path
    """The numerical CSV."""
    report: Path
    """The report text CSV."""
    search: Path
    """The search text CSV."""

    def text_path(self, kind: TextKind) -> Path:
        """Give the path of one of the two text files.

        Args:
            kind: Which text file.

        Returns:
            Its downloaded path.
        """
        return self.report if kind == "report" else self.search


def _midnight_us(day: date) -> int:
    """Give the Unix microsecond of midnight UTC that starts a day.

    The release dates its rows by calendar day and states no time zone. Midnight UTC is one fixed
    reading of each date, so the same date lands on the same microsecond in every build.

    Args:
        day: The calendar day.

    Returns:
        Unix microseconds.
    """
    return unix_us(datetime(day.year, day.month, day.day, tzinfo=UTC))


def _signal(column: Column, values: list[float | int | str], axis: RegularAxis) -> Signal:
    """Wrap the parsed cells of one column as a signal.

    Args:
        column: The column declaration.
        values: Its parsed cells, one per row.
        axis: The cadence the rows keep.

    Returns:
        The signal.
    """
    dtype = column.spec.dtype
    data = values if dtype in {"str", "enum"} else np.asarray(values, dtype=np.dtype(dtype))
    return Signal.from_values(data, spec=column.spec, name=column.signal_name, time_axis=axis)


def _states_nothing(text: str) -> bool:
    """Tell whether a text cell, or one half of a prediction cell, carries no content.

    Args:
        text: The cell or half, verbatim.

    Returns:
        ``True`` for an empty one and for one the release filled with ``"NA"``.
    """
    return text.strip() in {"", _NA}


def _split_prediction(text: str, kind: TextKind, row: int) -> tuple[str | None, str | None]:
    """Separate a prediction cell into its long-term and short-term halves.

    The release writes the two halves with a ``;`` between them. A half that states nothing comes
    back as ``None``. A cell with no separator does not say which half it holds, and nothing here
    guesses from its words, so both halves come back as ``None``.

    Args:
        text: The cell, verbatim.
        kind: Which text file the cell is from, for the error message.
        row: The release's row index, for the error message.

    Returns:
        The long-term half and the short-term half, each ``None`` when the cell states none.

    Raises:
        TimeFFormatError: If the cell has more than one separator.
    """
    parts = text.split(_PREDICTION_SEPARATOR)
    if len(parts) == 1:
        return None, None
    if len(parts) > 2:  # noqa: PLR2004 - a cell holds a long-term half, a separator, and a short-term half
        raise TimeFFormatError(
            f"{kind} row {row}: a prediction holds one {_PREDICTION_SEPARATOR!r} between its long-term and "
            f"short-term halves, found {len(parts) - 1}"
        )
    long_term, short_term = parts
    return (None if _states_nothing(long_term) else long_term, None if _states_nothing(short_term) else short_term)


def _text_annotations(rows: list[TextRow], kind: TextKind, origin_us: int) -> Iterator[Annotation]:
    """Turn the rows of one text file into interval annotations.

    A row covers the days from its start date through its end date, so its span runs from the
    midnight that starts the first day to the midnight that ends the last. The fact cell is one
    annotation and each half of the prediction cell is another. A cell or half that states
    nothing, because it is empty or the release filled it with ``"NA"``, gets no annotation, and
    so does a prediction cell with no ``;``, since it does not say which half it holds.

    Args:
        rows: The rows of the file.
        kind: Which text file they come from, which names the annotation keys.
        origin_us: The record's origin, which the spans are measured from.

    Yields:
        One annotation per cell or half that states something, in row order: the fact, then the
        long-term prediction, then the short-term one.

    Raises:
        TimeFFormatError: If a prediction cell holds more than one ``;``.
    """  # noqa: DOC502 - raised by _split_prediction
    for row in rows:
        span = TimeInterval.micros(_midnight_us(row.start) - origin_us, _midnight_us(row.end + ONE_DAY) - origin_us)
        halves = _split_prediction(row.prediction, kind, row.row)
        cells = ((_FACT_FIELD, row.fact), *zip(_PREDICTION_FIELDS, halves, strict=True))
        for (field, description), text in cells:
            if text is None or _states_nothing(text):
                continue
            yield Annotation(
                key=f"{kind}_{field}",
                value=text,
                span=span,
                description=f"From {_TEXT_SOURCES[kind]}. {description}",
                occurrence_metadata={"source_row": row.row},
            )


def _iter_tasks(entries: tuple[tuple[Record, Signal, DomainSource], ...]) -> Iterator[ForecastingTask]:
    """Give the forecasting tasks of every converted domain, in record order.

    Args:
        entries: Each record with its target signal and its domain declaration.

    Yields:
        The tasks of each domain, domain by domain.
    """
    for record, target, source in entries:
        yield from tasks.iter_forecasting_tasks(record, target, lookback=source.lookback, horizons=source.horizons)


class TimeMmdConnector(BaseConnector[TimeMmdDomainFiles]):
    """Connector for the Energy and Environment domains of Time-MMD (GitHub ``AdityaLab/Time-MMD``)."""

    async def download_async(self, cache_dir: Path) -> list[TimeMmdDomainFiles]:  # noqa: PLR6301 (override: the files are fixed by DOMAINS)
        """Fetch the three CSV files of each domain from GitHub at the pinned commit.

        Args:
            cache_dir: The directory the files land in, mirroring the repository layout.

        Returns:
            One handle per domain, in :data:`~timenet_connectors.datasets.adityalab.time_mmd.sources.DOMAINS` order.
        """
        artifacts: list[Artifact] = []
        refs: list[TimeMmdDomainFiles] = []
        for source in DOMAINS:
            paths = {
                relative: cache_dir / relative
                for relative in (source.numerical_file, source.text_file("report"), source.text_file("search"))
            }
            artifacts.extend(Artifact(f"{_RAW_BASE}/{relative}", dest) for relative, dest in paths.items())
            numerical, report, search = paths.values()
            refs.append(TimeMmdDomainFiles(domain=source.name, numerical=numerical, report=report, search=search))
        await download_files(artifacts)
        return refs

    def convert(self, raw_refs: list[TimeMmdDomainFiles]) -> TimeFDataset:
        """Build one record per domain, with its text as annotations, and stream the forecasting tasks.

        The text annotations attach to each record before the task stream is set, so that a task can
        name the ones that fall in its context window as its ``input_annotations``.

        Args:
            raw_refs: The handles from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeFValidationError: If there is no handle, a handle names a domain this connector
                does not declare, or two handles name the same domain.
        """
        declared = {source.name: source for source in DOMAINS}
        if not raw_refs:
            raise TimeFValidationError("Time-MMD needs at least one domain to convert")
        names = [ref.domain for ref in raw_refs]
        unknown = sorted(set(names) - declared.keys())
        if unknown or len(set(names)) != len(names):
            raise TimeFValidationError(
                f"Time-MMD converts each of {sorted(declared)} at most once, got {names}"
                + (f"; unknown: {unknown}" if unknown else "")
            )

        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            ("source_repository", SOURCE_REPOSITORY),
            ("source_revision", SOURCE_REVISION),
            ("target_signal", TARGET_COLUMN),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        entries: list[tuple[Record, Signal, DomainSource]] = []
        for ref in raw_refs:
            entries.append(self._convert_domain(dataset, ref, declared[ref.domain]))
        dataset.set_task_stream([ForecastingTask], partial(_iter_tasks, tuple(entries)))
        return dataset

    @staticmethod
    def _convert_domain(
        dataset: TimeFDataset, files: TimeMmdDomainFiles, source: DomainSource
    ) -> tuple[Record, Signal, DomainSource]:
        """Read the three files of one domain and register its record.

        The origin of the record is the earliest date any of the three files states, and its
        ``time_span`` reaches to the latest. The signals sit on the numerical rows' cadence,
        offset from the origin by however much text precedes them.

        Args:
            dataset: The dataset to add the record to.
            files: The domain's downloaded files.
            source: The domain declaration.

        Returns:
            The record, its target signal, and the declaration, for the task stream.
        """
        table = sources.read_numerical_table(files.numerical, source)
        texts = {kind: sources.read_text_rows(files.text_path(kind)) for kind in _TEXT_KINDS}
        rows = [row for kind in _TEXT_KINDS for row in texts[kind]]
        origin = min([table.first_date, *(row.start for row in rows)])
        end = max([table.last_date + source.period, *(row.end + ONE_DAY for row in rows)])
        origin_us = _midnight_us(origin)

        axis = RegularAxis(
            period_us=Fraction(source.period // timedelta(microseconds=1)),
            offset_us=_midnight_us(table.first_date) - origin_us,
        )
        signals = tuple(_signal(column, table.values[column.header], axis) for column in source.columns)
        record = Record(
            sources=(Source(name=source.source_name, signals=signals),),
            start_time=TimeOrigin(origin_us),
            time_span=TimeInterval.micros(0, _midnight_us(end) - origin_us),
            metadata={
                "domain": source.name,
                "numerical_file": source.numerical_file,
                "report_file": source.text_file("report"),
                "search_file": source.text_file("search"),
            },
        )
        dataset.add_record(record=record)
        record.add_annotations(
            Annotation(key=constant.key, value=table.constants[constant.key], description=constant.description)
            for constant in source.constant_columns
        )
        for kind in _TEXT_KINDS:
            record.add_annotations(_text_annotations(texts[kind], kind, origin_us), warn_when_outside=False)
        return record, signals[0], source


CONNECTOR = TimeMmdConnector
