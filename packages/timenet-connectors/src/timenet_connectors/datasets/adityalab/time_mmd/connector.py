"""Convert Time-MMD's Economy and Energy CSVs without resampling or rewriting text.

Each domain is one Record. Its numerical columns are lazy float64 Signals under
one Source; report/search fact and preds fields are interval annotations. Their
coverage is preserved, but publication and availability remain unknown.
"""

import calendar
import csv
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import partial
import hashlib
import math
from pathlib import Path
from typing import cast

import pyarrow as pa
from typing_extensions import override

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, AnswerTask, ForecastingTask, Split, TimeInterval, TimeOrigin, TimeSeriesSpec
from timenet_connectors.download import Artifact, download_files


SOURCE_REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"
SOURCE_URL = f"https://raw.githubusercontent.com/AdityaLab/Time-MMD/{SOURCE_REVISION}"
SOURCE_FILES = {
    "numerical/Economy/Economy.csv": "b6a5d18fef4d0ef20425e7704a4831a93cce69e3ead766f47880e90e1ae693a5",
    "textual/Economy/Economy_report.csv": "df620d0d72b0307920c641ad8d070acf0b6333ffbdd95deeba9f05c47cd05bee",
    "textual/Economy/Economy_search.csv": "f7bd7a4220e02c2124d04ecf9be6c9d4049a6233474c1ab61c21f7c8996c0465",
    "numerical/Energy/Energy.csv": "94313cefb3459f04b58ec814da5d817a694e767cc52de0ce09d4de4efdad02ee",
    "textual/Energy/Energy_report.csv": "36fa229dca003f40d4fb8c816a8ddfde2e020fca1f14572b3ce81a368e948fe8",
    "textual/Energy/Energy_search.csv": "f9023e6fcb635731a74f248dcf5ceb8aeca48870f3fc559f2173b7aae790b595",
}
_DAY_US = 86_400_000_000
_DATE_COLUMNS = {"Month", "date", "start_date", "end_date"}
_UNITS = {"Economy": "million USD", "Energy": "USD/US gallon"}
_PROVIDERS = {"Economy": "US Census Bureau goods trade", "Energy": "US EIA retail gasoline prices"}
# Fixed example-task protocol, also recorded in task metadata for consumers.
# Adapters select the native scoped history; they do not redefine the task.
_HISTORY = 64
_COMPARISON = 8
# Chosen QA no-change bands in source units, not published Time-MMD labels
# or measurement uncertainty: 1 million USD and 0.001 USD per US gallon.
_TOLERANCE = {"Economy": Decimal("1"), "Energy": Decimal("0.001")}


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read original CSV strings without interpreting their contents.

    Returns:
        Headers and original row dictionaries.

    Raises:
        TimeFFormatError: If headers are missing/repeated or rows are ragged.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        if not headers or len(headers) != len(set(headers)):
            raise TimeFFormatError(f"{path.name}: missing or repeated CSV headers")
        rows = []
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise TimeFFormatError(f"{path.name}: ragged CSV row {index}")
            rows.append(cast("dict[str, str]", row))
    return headers, rows


def _coverage(row: dict[str, str]) -> tuple[date, date]:
    """Parse the source's inclusive date coverage.

    Returns:
        The start and end dates.

    Raises:
        TimeFFormatError: If dates are missing, invalid or reversed.
    """
    try:
        start, end = date.fromisoformat(row["start_date"]), date.fromisoformat(row["end_date"])
    except (KeyError, ValueError) as error:
        raise TimeFFormatError("Time-MMD rows require ISO start_date and end_date") from error
    if end < start:
        raise TimeFFormatError(f"Reversed Time-MMD coverage: {start} through {end}")
    return start, end


def _number(value: str) -> Decimal:
    """Parse a source number without rounding its decimal representation.

    Returns:
        The finite Decimal value.

    Raises:
        TimeFFormatError: If the value is invalid or cannot fit a finite float64.
    """
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise TimeFFormatError(f"Invalid Time-MMD numerical value: {value!r}") from error
    if not result.is_finite() or not math.isfinite(float(result)):
        raise TimeFFormatError(f"Nonfinite Time-MMD numerical value: {value!r}")
    return result


def _values(rows: tuple[dict[str, str], ...], column: str) -> pa.Array:
    """Materialize a column when its lazy Signal loader is read.

    Returns:
        The column as a float64 Arrow array.
    """
    return pa.array((float(_number(row[column])) for row in rows), type=pa.float64())


def _offsets(values: tuple[int, ...]) -> pa.Array:
    """Materialize source date offsets when the irregular axis is read.

    Returns:
        The offsets as an int64 Arrow array.
    """
    return pa.array(values, type=pa.int64())


def _numerical_rows(
    path: Path, domain: str
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...], list[int], tuple[date, ...]]:
    """Validate the domain's numerical rows and calendar.

    Returns:
        Columns, ordered rows, original row indices and observation dates.

    Raises:
        TimeFFormatError: If values, dates or the calendar are inconsistent.
    """
    headers, raw = _read_csv(path)
    # Preserve every measured column; the task needs OT, not a fixed channel count.
    columns = tuple(column for column in headers if column not in _DATE_COLUMNS)
    if not raw or "OT" not in columns:
        raise TimeFFormatError(f"{domain}: expected numerical rows with an OT channel")
    indexed = sorted(enumerate(raw), key=lambda item: _coverage(item[1])[0])
    rows = tuple(row for _, row in indexed)
    dates = tuple(_coverage(row)[0] for row in rows)
    if len(set(dates)) != len(dates):
        raise TimeFFormatError(f"{domain}: duplicate numerical dates")
    for index, row in enumerate(rows):
        start, end = _coverage(row)
        if row.get("date") != start.isoformat():
            raise TimeFFormatError(f"{domain}: date differs from start_date")
        if domain == "Energy":
            valid_period = end == start + timedelta(days=6)
            valid_step = index == 0 or start - dates[index - 1] == timedelta(days=7)
        else:
            valid_period = start.day == 1 and end == start.replace(day=calendar.monthrange(start.year, start.month)[1])
            previous = dates[index - 1]
            valid_step = index == 0 or start.year * 12 + start.month == previous.year * 12 + previous.month + 1
        if not valid_period or not valid_step:
            raise TimeFFormatError(f"{domain}: inconsistent observation calendar at {start}")
        for column in columns:
            _number(row[column])
    return columns, rows, [index for index, _ in indexed], dates


def _record(paths: dict[str, Path], domain: str) -> tuple[Record, tuple[dict[str, str], ...]]:
    """Build a domain Record with lazy Signals and original text annotations.

    Returns:
        The Record and its chronological numerical rows.

    Raises:
        TimeFFormatError: If source schemas or coverage are inconsistent.
    """
    columns, rows, row_indices, dates = _numerical_rows(paths[f"{domain}.csv"], domain)
    text = []
    for kind in ("report", "search"):
        path = paths[f"{domain}_{kind}.csv"]
        headers, text_rows = _read_csv(path)
        if set(headers) != {"", "start_date", "end_date", "fact", "preds"}:
            raise TimeFFormatError(f"{path.name}: unsupported text columns")
        text.append((kind, path, text_rows))
    periods = [_coverage(row) for row in rows]
    periods.extend(_coverage(row) for _, _, text_rows in text for row in text_rows)
    # Earlier text must fit the Record too; numerical axes are relative to this origin.
    origin = min(start for start, _ in periods)
    offsets = tuple((day - origin).days * _DAY_US for day in dates)
    axis = (
        RegularAxis(period_us=Fraction(7 * _DAY_US), offset_us=offsets[0], axis_id=f"{domain}-calendar")
        if domain == "Energy"
        else IrregularAxis(first_us=offsets[0], last_us=offsets[-1], axis_id=f"{domain}-calendar")
    )
    # Bind original strings in loaders; TimeNet materializes arrays only when read/written.
    signals = tuple(
        Signal.from_loader(
            id=f"{domain}-channel-{index}",
            name=column,
            spec=TimeSeriesSpec(
                spec_type=f"time_mmd_{domain.lower()}",
                name=f"{domain} ({_UNITS[domain]})",
                unit_value=None,
                dtype="float64",
            ),
            time_axis=axis,
            loader=partial(_values, rows, column),
            n_values=len(rows),
            time_offsets_loader=partial(_offsets, offsets) if domain == "Economy" else None,
            metadata={"unit": _UNITS[domain], "source_column": column},
        )
        for index, column in enumerate(columns)
    )
    record = Record(
        record_id=domain.lower(),
        start_time=TimeOrigin(datetime.combine(origin, datetime.min.time(), UTC)),
        time_span=TimeInterval(
            start_us=0, end_us=(max(end for _, end in periods) + timedelta(days=1) - origin).days * _DAY_US
        ),
        sources=(Source(id=f"{domain}-numerical", name=_PROVIDERS[domain], signals=signals),),
        metadata={
            "domain": domain,
            "source_revision": SOURCE_REVISION,
            "numerical_sha256": hashlib.sha256(paths[f"{domain}.csv"].read_bytes()).hexdigest(),
            "numerical_source_rows": row_indices,
            "numerical_date_columns": [{key: row[key] for key in row if key in _DATE_COLUMNS} for row in rows],
            "date_convention": "Date-only source timestamps represented as UTC midnight; not release times",
            "numerical_vintage": "Archived snapshot; historical release and revision times unknown",
        },
    )
    for kind, path, text_rows in text:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, row in enumerate(text_rows):
            start, end = _coverage(row)
            for field in ("fact", "preds"):
                record.annotate(
                    Annotation(
                        id=f"{domain}-{kind}-{index}-{field}",
                        key=f"{kind}_{field}",
                        value=row[field],
                        span=TimeInterval(
                            start_us=(start - origin).days * _DAY_US,
                            # CSV coverage is inclusive; TimeF intervals are half-open.
                            end_us=(end + timedelta(days=1) - origin).days * _DAY_US,
                        ),
                        source=f"Time-MMD/{domain}/{kind}",
                        metadata={
                            "source_path": f"textual/{domain}/{path.name}",
                            "source_revision": SOURCE_REVISION,
                            "source_sha256": digest,
                            "row_index": index,
                            "source_row_id": row[""],
                            "coverage_start": row["start_date"],
                            "coverage_end": row["end_date"],
                            "published_at": None,
                            "available_at": None,
                            "availability_evidence": "Coverage dates only; publication/generation times not provided",
                            "processing": "Released LLM-processed text, not independently verified facts or targets",
                        },
                    ),
                    warn_when_outside=False,
                )
    return record, rows


def _add_tasks(dataset: TimeFDataset, record: Record, rows: tuple[dict[str, str], ...], domain: str) -> None:
    """Attach QA and next-value forecasting tasks over shared history scopes."""
    signal = next(signal for signal in record.signals if signal.name == "OT")
    for target_index in range(_HISTORY, len(rows)):
        # The next row is a forecast target, never part of the shared input history.
        history = rows[target_index - _HISTORY : target_index]
        # Decimal source arithmetic makes equality at either QA boundary explicit.
        first = sum((_number(row["OT"]) for row in history[:_COMPARISON]), Decimal(0)) / _COMPARISON
        last = sum((_number(row["OT"]) for row in history[-_COMPARISON:]), Decimal(0)) / _COMPARISON
        difference = last - first
        answer = (
            "higher" if difference > _TOLERANCE[domain] else "lower" if difference < -_TOLERANCE[domain] else "within"
        )
        scope = record.time_interval(
            datetime.combine(date.fromisoformat(history[0]["start_date"]), datetime.min.time(), UTC),
            datetime.combine(date.fromisoformat(rows[target_index]["start_date"]), datetime.min.time(), UTC),
            time_series_ids=(signal.id,),
        )
        # Coverage does not prove publication; no stored source text is admitted here.
        metadata = {
            "history_length": _HISTORY,
            "history_start_index": target_index - _HISTORY,
            "history_stop_index": target_index,
            "as_of": history[-1]["start_date"] + "T00:00:00Z",
            "cutoff_basis": "Simulated stored observation date, not verified historical release availability",
            "text_policy": "Released text has no verified availability; no source annotations supplied",
            "excluded_annotation_count": len(record.annotations),
            "numerical_vintage": "Archived snapshot; historical release/revision times unknown",
            "overlap": "Rolling one-observation origins; adjacent histories overlap",
        }
        # Split by target observation index; adjacent histories deliberately overlap.
        split = Split.TRAIN if target_index < len(rows) * 4 // 5 else Split.TEST
        dataset.add_task(
            task=AnswerTask(
                id=f"{domain}-qa-{target_index:06d}",
                inputs=(record,),
                scope=scope,
                prompt=(
                    f"Using the 64 supplied OT observations in {_UNITS[domain]}, compare the last-eight mean "
                    f"with the first-eight mean. Answer higher if the difference exceeds {_TOLERANCE[domain]}, "
                    f"lower if below -{_TOLERANCE[domain]}, or within otherwise, including equality."
                ),
                targets=(answer,),
                split=split,
                metadata={**metadata, "comparison_length": _COMPARISON, "tolerance": str(_TOLERANCE[domain])},
            )
        )
        dataset.add_task(
            task=ForecastingTask(
                id=f"{domain}-forecast-{target_index:06d}",
                inputs=(record,),
                scope=scope,
                prompt=f"Predict the next observed OT value in {_UNITS[domain]} from the 64 supplied observations.",
                targets=(float(_number(rows[target_index]["OT"])),),
                split=split,
                metadata={**metadata, "horizon": 1, "target_date": rows[target_index]["start_date"]},
            )
        )


class TimeMMDConnector(BaseConnector[Path]):
    """Fetch a pinned Time-MMD snapshot and retain both domains without resampling."""

    @override
    async def download_async(self, cache_dir: Path) -> list[Path]:
        """Fetch six CSVs, verifying downloaded and previously cached source bytes.

        Returns:
            Local CSV paths for conversion.

        Raises:
            TimeFFormatError: If an existing cached file does not match the pinned digest.
        """
        artifacts = []
        for relative, digest in SOURCE_FILES.items():
            path = cache_dir / SOURCE_REVISION / relative
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise TimeFFormatError(f"Cached Time-MMD checksum mismatch: {path}")
            artifacts.append(Artifact(f"{SOURCE_URL}/{relative}", path, sha256=digest))
        return await download_files(artifacts)

    @override
    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Build two shared domain Records with lazy numerical values and original text.

        Returns:
            The converted Economy/Energy dataset.

        Raises:
            TimeFFormatError: If required files or source schemas are inconsistent.
        """
        paths = {path.name: path for path in raw_refs}
        if len(paths) != len(raw_refs) or set(paths) != {Path(relative).name for relative in SOURCE_FILES}:
            raise TimeFFormatError("Expected exactly the six Economy/Energy Time-MMD CSVs")
        dataset = TimeFDataset(metadata=self.metadata())
        for domain in ("Economy", "Energy"):
            record, rows = _record(paths, domain)
            dataset.add_record(record=record)
            _add_tasks(dataset, record, rows, domain)
        return dataset


CONNECTOR = TimeMMDConnector
