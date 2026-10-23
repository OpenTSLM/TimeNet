"""Convert TRACE's preprocessed weather windows into lazy Records, event-type, and caption tasks.

The authors' Google Drive archive holds the ``dataset/`` folder their code loads. Its ``pretrain`` and
``forecasting`` JSON files contain all 74,337 windows of the paper, keyed by original window id and
partitioned by the authors' splits. Each file is one JSON object, so ``convert`` indexes every window
by byte range, and the loaders reread one window at a time.

Classification follows the authors' Stage 1: ``pretrain`` event windows with at least 100 values,
labelled with one of nine NOAA event types. Non-event and forecasting windows become Records without
tasks. Each task's rationale is the event report the authors pair with the window in Stage 2, read
from the ``retrieval`` parquet files.

Captioning follows TimeBraid's use of TRACE: every captioned window of the ``retrieval`` files gets
a caption task, with the authors' sample description as its target, under the authors' retrieval
split.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, get_args

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, Strict, StrictInt, StrictStr, StringConstraints

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, ClassificationTask, Split, Task, TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive
from timenet_connectors.sources.json_stream import iter_object, read_entry
from timenet_connectors.time_axes import axis_for_offsets


# The README's Google Drive link, as a direct download that skips the HTML confirmation page. The
# archive is 510,759,681 bytes (487 MiB) and unpacks to 1.75 GB.
SOURCE_URL = (
    "https://drive.usercontent.google.com/download?id=1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY&export=download&confirm=t"
)
# Drive has no immutable revision, so pin the archive's bytes instead.
SOURCE_SHA256 = "a5d24d75a0b9f0f846a185caa46c4f77e0ef98272c12808df76fac25a181be58"


class Inventory(NamedTuple):
    """One source file's pinned counts."""

    rows: int
    events: int
    """Windows that become classification tasks (``pretrain``), or rows with an event report
    (``retrieval``)."""


# Pinned inventory of the archive's split files, keyed by (task, split).
#
# The JSON files hold the windows and match Table 8 of the paper. Every one becomes a Record, so the
# dataset holds all 74,337 windows the paper describes. Only pretrain windows can become
# classification tasks, and these match classification/*_data/labels.npy. Forecasting rows carry no
# event_type: the authors use them only for forecasting fine-tuning. Their Records stay, as the
# 50,211 non-event pretrain Records do, because they are part of the authors' data, and a later
# forecasting task can use them unchanged.
#
# The Stage 2 parquet files hold the 44,186 captioned windows with at least 100 values. Each row's
# description becomes a caption task, and its event report, if any, a classification task's
# rationale. Their series column is not read.
SOURCE_FILES = {
    ("pretrain", "train"): Inventory(45339, 10110),
    ("pretrain", "val"): Inventory(12953, 2884),
    ("pretrain", "test"): Inventory(6484, 1452),
    ("forecasting", "train"): Inventory(6690, 0),
    ("forecasting", "val"): Inventory(1914, 0),
    ("forecasting", "test"): Inventory(957, 0),
    ("retrieval", "train"): Inventory(42180, 13787),
    ("retrieval", "test"): Inventory(2006, 659),
}
# TimeBraid's captioning prompt (arXiv:2609.29792, Figure 17). TimeBraid trains on TRACE captions but
# does not publish their prompt; the figure shows this one for its captioning examples.
CAPTION_PROMPT = (
    "You are an expert in time series analysis. Please generate a detailed caption for this "
    "time-series, describing it as accurately as possible."
)
# The archive's classification and pretraining arrays drop windows shorter than this.
MIN_CLASSIFICATION_VALUES = 100
# EVENT_MAP in the authors' src/common.py, in label order. Its 'None': -100 marks non-event windows,
# which the authors' loss and metrics ignore.
EVENT_TYPES = (
    "Lightning",
    "Debris Flow",
    "Flash Flood",
    "Heavy Rain",
    "Tornado",
    "Funnel Cloud",
    "Hail",
    "Flood",
    "Thunderstorm Wind",
)
_NoEvent = Literal[-100]
NO_EVENT: int = get_args(_NoEvent)[0]
_EVENT_SCHEMA = "trace-noaa-event-types"
_SPLITS = {"train": Split.TRAIN, "val": Split.VALIDATION, "test": Split.TEST}
# Window ids come in anchor triplets 3k, 3k+1, 3k+2, in this order.
_Mode = Literal["7day_hourly", "28day_4hourly", "180day_daily"]
_MODES: tuple[str, ...] = get_args(_Mode)
_CADENCE_US = {"7day_hourly": 3_600_000_000, "28day_4hourly": 14_400_000_000, "180day_daily": 86_400_000_000}
# GHCN-h sky cover codes 00-10, after NCEI's hourly coverage code table: 00-08 count oktas, and 09 and
# 10 are not amounts. The paper's prompt (Appendix B.4) labels the same codes with mixed fractions.
SKY_COVER = (
    "0 oktas",
    "1 okta",
    "2 oktas",
    "3 oktas",
    "4 oktas",
    "5 oktas",
    "6 oktas",
    "7 oktas",
    "8 oktas",
    "sky obscured",
    "partial obscuration",
)
_SKY_COVER_LABELS = pa.array(SKY_COVER, type=pa.string())
# Units follow the Hugging Face release's field table; the archive itself states none.
_CHANNELS = (
    # Values carry at most two decimals, so float32 storage keeps them.
    ("temperature", TimeSeriesSpec(spec_type="air_temperature", name="Air temperature", unit_value=ureg.degC)),
    ("precipitation", TimeSeriesSpec(spec_type="precipitation", name="Precipitation", unit_value=ureg.mm)),
    (
        "relative_humidity",
        TimeSeriesSpec(spec_type="relative_humidity", name="Relative humidity", unit_value=ureg.percent),
    ),
    ("visibility", TimeSeriesSpec(spec_type="visibility", name="Visibility", unit_value=ureg.km)),
    ("wind_u", TimeSeriesSpec(spec_type="eastward_wind", name="Eastward wind", unit_value=ureg.m / ureg.s)),
    ("wind_v", TimeSeriesSpec(spec_type="northward_wind", name="Northward wind", unit_value=ureg.m / ureg.s)),
    (
        "sky_code",
        TimeSeriesSpec(
            spec_type="sky_cover", name="Sky cover", unit_value=ureg.dimensionless, dtype="enum", categories=SKY_COVER
        ),
    ),
)
_CHANNEL_NAMES = tuple(name for name, _ in _CHANNELS)

# Strict types: a JSON true or "1" must not pass as a number, and a blank string not as text.
_Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
_Number = Annotated[FiniteFloat, Strict()]
_Numbers = Annotated[list[_Number], Field(min_length=1)]
_SkyCodes = Annotated[list[Annotated[StrictInt, Field(ge=0, lt=len(SKY_COVER))]], Field(min_length=1)]
# Naive ISO seconds, as every published timestamp is. NumPy would also take "now", "NaT", or a
# zone suffix, which it silently shifts to UTC; the pattern keeps "read as UTC" an invariant.
_Timestamp = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")]
_EventIndex = Annotated[StrictInt, Field(ge=0, lt=len(EVENT_TYPES))]


class _WindowRow(BaseModel):
    """One window as the JSON files publish it, without the cross-field checks."""

    # The title names the model in validation errors, which reach users with the file and key.
    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE window")

    station_id: _Text
    latitude: _Number
    longitude: _Number
    # Pretrain rows publish "COUNTY,STATE"; most forecasting rows publish ["County", "State"].
    location: _Text | Annotated[list[_Text], Field(min_length=1)]
    mode: _Mode
    DATE: Annotated[list[_Timestamp], Field(min_length=1)]
    temperature: _Numbers
    precipitation: _Numbers
    relative_humidity: _Numbers
    visibility: _Numbers
    wind_u: _Numbers
    wind_v: _Numbers
    sky_code: _SkyCodes


class _PretrainRow(_WindowRow):
    """A pretrain window, which adds its event label."""

    model_config = ConfigDict(title="TRACE pretrain window")

    event_type: _NoEvent | _EventIndex


class _Description(BaseModel):
    """One window's generated description, as the retrieval files publish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE description")

    DATE: _Text
    labels: _Text
    location: _Text
    temperature: _Text
    precipitation: _Text
    relative_humidity: _Text
    visibility: _Text
    wind_u: _Text
    wind_v: _Text
    sky_code: _Text

    @property
    def caption(self) -> str:
        """Return the description as the authors' ``generate_dsp`` composes it for their text encoder."""
        # Spacing as published: generate_dsp joins the channel sentences with " \n ".
        channels = " \n ".join(getattr(self, name) for name in _CHANNEL_NAMES)
        return (
            f"Weather time series location: {self.location} Time range: {self.DATE} "
            f"The weather is {self.labels}. {channels}"
        )


class _EventReport(BaseModel):
    """One event window's report, as the retrieval files publish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE event report")

    event_id: StrictInt
    event_type: _EventIndex
    narrative: _Text

    @property
    def text(self) -> str:
        """Return the report as the authors' ``generate_er`` writes it for their text encoder."""
        return f"The weather event is {EVENT_TYPES[self.event_type]}. {self.narrative}"


@dataclass(frozen=True)
class TraceFileRef:
    """One extracted split file of the archive."""

    path: Path
    task: str
    """The authors' task folder: ``pretrain``, ``forecasting``, or ``retrieval``."""
    split: str
    """The authors' split name: ``train``, ``val``, or ``test``."""
    expected: Inventory | None = None
    """Pinned counts to check the file against; ``None`` skips the checks, as local fixtures do."""

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the file's rows."""
        return _SPLITS[self.split]

    @property
    def archive_path(self) -> str:
        """Return the file's path inside the archive."""
        return _archive_path(self.task, self.split)


def _archive_path(task: str, split: str) -> str:
    """Return a split file's path inside the archive: JSON windows, or Stage 2 parquet.

    Returns:
        The path below the archive root.
    """
    return f"dataset/{task}/{split}.{'parquet' if task == 'retrieval' else 'json'}"


@dataclass(frozen=True)
class _WindowRef:
    file: TraceFileRef
    key: str
    offset: int
    length: int

    @property
    def id(self) -> str:
        return f"trace-{self.key}"

    @property
    def context(self) -> str:
        return f"{self.file.path} (key {self.key})"


@dataclass(frozen=True)
class _Window:
    row: _WindowRow
    time_offsets_us: np.ndarray
    start_us: int
    event_type: int | None
    """The window's label if the authors classify it, else ``None``."""


def _unix_us(dates: list[str]) -> np.ndarray:
    """Return naive ISO timestamps as Unix microseconds, reading them as UTC.

    NumPy datetimes are naive. Only a UTC reading keeps every published step on its cadence.

    Returns:
        One int64 microsecond timestamp per date.
    """
    return np.array(dates, dtype="datetime64[us]").astype(np.int64)


def _time_offsets(dates: list[str], mode: str) -> tuple[int, np.ndarray]:
    """Read naive ISO timestamps as UTC and check they sit on the mode's cadence.

    Returns:
        The first timestamp in Unix microseconds, and every offset from it.

    Raises:
        TimeFValidationError: If a timestamp is not a calendar date, out of order, or off the cadence.
    """
    try:
        moments = _unix_us(dates)
    except ValueError as exc:
        # The row model fixed the format, so only an impossible date such as 02-30 reaches here.
        raise TimeFValidationError(f"DATE: {exc}") from exc
    steps = np.diff(moments)
    if (steps <= 0).any() or (steps % _CADENCE_US[mode]).any():
        raise TimeFValidationError(f"DATE must increase on the {mode} cadence; gaps are allowed")
    return int(moments[0]), moments - moments[0]


def _parse_window(row: Any, ref: _WindowRef) -> _Window:
    """Validate one window and decode its timing, naming the source location on error.

    Only pretrain windows can be classified: forecasting windows have no ``event_type``, so they
    get no label, like non-event windows and windows shorter than the authors' classification
    arrays keep. Those remain Records without an event-type task.

    Returns:
        The validated row with its time offsets, start, and classified event label.

    Raises:
        TimeFValidationError: If the row is malformed, with file and key context.
    """
    try:
        # Pretrain rows must carry an event label, and forecasting rows must not.
        window = (_PretrainRow if ref.file.task == "pretrain" else _WindowRow).model_validate(row)
        if window.mode != _MODES[int(ref.key) % len(_MODES)]:
            raise TimeFValidationError(f"mode {window.mode!r} does not match the id's anchor position")
        start_us, offsets = _time_offsets(window.DATE, window.mode)
        for name in _CHANNEL_NAMES:
            values = getattr(window, name)
            if len(values) != len(offsets):
                raise TimeFValidationError(f"{name} has {len(values)} values for {len(offsets)} timestamps")
        event_type = window.event_type if isinstance(window, _PretrainRow) else None
        if event_type == NO_EVENT or len(offsets) < MIN_CLASSIFICATION_VALUES:
            event_type = None
        return _Window(window, offsets, start_us, event_type)
    # A pydantic ValidationError is a ValueError, so it gains the same context.
    except (ValueError, OverflowError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


class _WindowLoader:
    """Load values lazily, reusing one decoded window across its channels and time offsets.

    ``convert`` validated each window's byte range when it indexed the extracted file, and a
    build reads the same file, so a reread is not validated again. This assumes the extracted
    files do not change between ``convert`` and writing.
    """

    def __init__(self) -> None:
        self.cached: tuple[_WindowRef, dict[str, Any]] | None = None

    def _row(self, ref: _WindowRef) -> dict[str, Any]:
        cached = self.cached
        if cached is not None and cached[0] == ref:
            return cached[1]
        row = read_entry(ref.file.path, ref.offset, ref.length)
        self.cached = ref, row
        return row

    def values(self, ref: _WindowRef, name: str) -> pa.Array:
        values = self._row(ref)[name]
        if name == "sky_code":
            # The published codes are already indices into the codebook.
            return pa.DictionaryArray.from_arrays(pa.array(values, type=pa.int32()), _SKY_COVER_LABELS)
        return pa.array(values, type=pa.float32())

    def time_offsets(self, ref: _WindowRef) -> pa.Array:
        moments = _unix_us(self._row(ref)["DATE"])
        return pa.array(moments - moments[0], type=pa.int64())


def _make_record(ref: _WindowRef, window: _Window, loader: _WindowLoader) -> Record:
    """Build one seven-channel station Record whose values stay in the source file.

    Returns:
        A Record with lazy signals on one shared time axis, its location and dates as annotations,
        and its provenance as metadata.
    """
    row = window.row
    # One axis per window: its seven channels share every timestamp. A gap-free window is regular at
    # its mode's cadence; a window with missing timestamps keeps them as explicit offsets.
    axis = axis_for_offsets(window.time_offsets_us, period_us=Fraction(_CADENCE_US[row.mode]))
    offsets_loader = partial(loader.time_offsets, ref) if isinstance(axis, IrregularAxis) else None
    signals = tuple(
        Signal.from_loader(
            id=f"{ref.id}-{name.replace('_', '-')}",
            name=name,
            spec=spec,
            time_axis=axis,
            time_offsets_loader=offsets_loader,
            n_values=len(window.time_offsets_us),
            loader=partial(loader.values, ref, name),
        )
        for name, spec in _CHANNELS
    )
    record = Record(
        record_id=ref.id,
        start_time=TimeOrigin(window.start_us),
        sources=(
            Source(
                id=f"{ref.id}-station",
                name=row.station_id,
                signals=signals,
                # TimeF does not keep signal order on read, so record the source's channel order.
                metadata={"channel_order": list(_CHANNEL_NAMES)},
            ),
        ),
        # Provenance and station facts that no task reads.
        metadata={
            "source_id": int(ref.key),
            "anchor": int(ref.key) // len(_MODES),
            "source_file": ref.file.archive_path,
            "station_id": row.station_id,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "mode": row.mode,
        },
    )
    # The context the authors' caption generator received with the series (paper Appendix B.4):
    # the window's place and dates. Caption tasks take these as input annotations.
    # An annotation key has one value type, so the forecasting files' ["County", "State"] lists are
    # joined into a string like the pretrain files' "COUNTY,STATE".
    location = row.location if isinstance(row.location, str) else ", ".join(row.location)
    for key, value, description in (
        ("location", location, "Station location: the published string, or the published list joined"),
        ("window_start", row.DATE[0], "First timestamp, as published; read as UTC"),
        ("window_end", row.DATE[-1], "Last timestamp, as published; read as UTC"),
    ):
        record.annotate(
            Annotation(key=key, value=value, description=description, id=f"{ref.id}-{key.replace('_', '-')}")
        )
    return record


@dataclass(frozen=True)
class _IndexedWindow:
    """What the task stream needs of one window once its Record is built."""

    ref: _WindowRef
    record: Record
    event_type: int | None
    """The window's label if the authors classify it, else ``None``."""


# Rows per parquet read, so neither the checks nor the task stream hold a whole file's captions.
_RETRIEVAL_BATCH_ROWS = 1024


def _retrieval_rows(file: TraceFileRef) -> Iterator[tuple[str, str, _EventReport | None]]:
    """Read and validate a Stage 2 parquet file one batch at a time.

    Yields:
        Each row's window key, its caption, and its event report (``None`` for a non-event window).

    Raises:
        TimeFValidationError: If a window id, a description, or a report is invalid.
    """
    # Skip the series column: the Records already carry the values from the JSON files.
    columns = ["id", "description", "events"]
    with pq.ParquetFile(file.path) as parquet:
        for batch in parquet.iter_batches(batch_size=_RETRIEVAL_BATCH_ROWS, columns=columns):
            for window_id, description, event in zip(
                *(batch.column(name).to_pylist() for name in columns), strict=True
            ):
                if type(window_id) is not int:
                    raise TimeFValidationError(f"{file.path}: expected a distinct integer window id, got {window_id!r}")
                try:
                    caption = _Description.model_validate(description).caption
                    # A non-event window has a null report.
                    report = None if event is None else _EventReport.model_validate(event)
                # A pydantic ValidationError is a ValueError.
                except ValueError as exc:
                    raise TimeFValidationError(f"{file.path} (id {window_id}): {exc}") from exc
                yield str(window_id), caption, report


def _check_report(window: _IndexedWindow, report: _EventReport | None) -> None:
    """Check that a window has an event report exactly when it is classified, and the right one.

    Raises:
        TimeFValidationError: If the window's event report is missing, mismatched, or unexpected.
    """
    if window.event_type is not None:
        if report is None or report.event_type != window.event_type:
            raise TimeFValidationError(f"{window.ref.context}: no event report with event_type {window.event_type}")
    elif report is not None:
        raise TimeFValidationError(f"{window.ref.context}: event report for a window without an event-type task")


def _check_retrieval(files: list[TraceFileRef], windows: dict[str, _IndexedWindow]) -> None:
    """Check every retrieval row against the indexed windows, keeping no captions or reports.

    Every row must belong to a window, every classified window must have its event report, and every
    report a classified window. The task stream rereads the rows and relies on these checks.

    Raises:
        TimeFValidationError: If a row, a report, or a file's pinned inventory is invalid.
    """
    seen: set[str] = set()
    orphans: list[str] = []
    for file in files:
        rows = events = 0
        for key, _, report in _retrieval_rows(file):
            if key in seen:
                raise TimeFValidationError(f"{file.path}: expected a distinct integer window id, got {key}")
            seen.add(key)
            rows += 1
            events += report is not None
            if key in windows:
                _check_report(windows[key], report)
            else:
                orphans.append(key)
        _check_inventory(file, rows, events)
    if orphans:
        raise TimeFValidationError(f"{len(orphans)} retrieval rows have no window, such as {min(orphans)}")
    for window in windows.values():
        if window.event_type is not None and window.ref.key not in seen:
            _check_report(window, None)


def _check_inventory(file: TraceFileRef, rows: int, events: int) -> None:
    """Compare one file's counts with its pinned inventory, when the reference carries one.

    A reference without one, such as a test fixture, skips the check.

    Raises:
        TimeFValidationError: If the counts differ from the pinned archive's.
    """
    if file.expected is not None and (rows, events) != file.expected:
        raise TimeFValidationError(
            f"{file.path}: expected {file.expected.rows} rows and {file.expected.events} events, "
            f"found {rows} and {events}"
        )


def _event_task(window: _IndexedWindow, event_type: int, report: _EventReport | None) -> ClassificationTask:
    """Build a classified window's event-type task.

    Returns:
        The task, with the event report as its rationale when there is one.
    """
    return ClassificationTask(
        id=f"{window.ref.id}-event-type",
        inputs=(window.record,),
        targets=(EVENT_TYPES[event_type],),
        split=window.ref.file.timenet_split,
        target_schema=_EVENT_SCHEMA,
        # The report names the target and describes the event after the window, so it is
        # supervision only, never an input.
        rationale=None if report is None else report.text,
        metadata={"event_type_index": event_type} | ({} if report is None else {"event_id": report.event_id}),
    )


def _caption_task(window: _IndexedWindow, split: Split, caption: str) -> AnswerTask:
    """Build a window's caption task.

    Returns:
        The task, under the retrieval file's split.
    """
    # The description was generated from the window's own series, location, and dates. It does not
    # mention the event, which happens after the window.
    return AnswerTask(
        id=f"{window.ref.id}-caption",
        inputs=(window.record,),
        targets=(caption,),
        prompt=CAPTION_PROMPT,
        # The location and dates the caption states, which the series alone does not carry.
        input_annotations=window.record.annotations,
        split=split,
    )


def _iter_tasks(
    files: tuple[TraceFileRef, ...], windows: dict[str, _IndexedWindow]
) -> Iterator[ClassificationTask | AnswerTask]:
    """Rebuild every task from the window index and the retrieval files, which ``convert`` checked.

    Each call rereads the retrieval files, so the writer can iterate more than once.

    Yields:
        Each row's event-type task, if its window is classified, then its caption task. Without
        retrieval files, each classified window's event-type task, without a rationale.
    """
    if not files:
        for window in windows.values():
            if window.event_type is not None:
                yield _event_task(window, window.event_type, None)
        return
    for file in files:
        for key, caption, report in _retrieval_rows(file):
            window = windows[key]
            if report is not None:
                # convert checked that the report's event type is the window's.
                yield _event_task(window, report.event_type, report)
            yield _caption_task(window, file.timenet_split, caption)


def _add_windows(
    dataset: TimeFDataset, file: TraceFileRef, loader: _WindowLoader, windows: dict[str, _IndexedWindow]
) -> None:
    """Add one window file's Records, and index each window for the task stream.

    Raises:
        TimeFValidationError: If a window id, a window, or the file's pinned inventory is invalid.
    """
    rows = classified = 0
    for key, row, offset, length in iter_object(file.path):
        # Retrieval rows key windows by str(int id), so only a canonical decimal key can match its row.
        if not key.isascii() or not key.isdecimal() or key != str(int(key)):
            raise TimeFValidationError(f"{file.path}: expected a canonical decimal window id, got {key!r}")
        ref = _WindowRef(file, key, offset, length)
        window = _parse_window(row, ref)
        record = dataset.add_record(record=_make_record(ref, window, loader))
        windows[key] = _IndexedWindow(ref, record, window.event_type)
        rows += 1
        classified += windows[key].event_type is not None
    _check_inventory(file, rows, classified)


class TraceWeatherEventsConnector(BaseConnector[TraceFileRef]):
    """Build TRACE weather windows with the authors' splits, nine-class event labels, and captions."""

    async def download_async(self, cache_dir: Path) -> list[TraceFileRef]:  # noqa: PLR6301 — BaseConnector interface
        """Download and extract the pinned Drive archive once, then locate its split files.

        Returns:
            One reference per JSON window file and retrieval parquet file, with its pinned row count.

        Raises:
            TimeNetDownloadError: If the archive lacks a split file.
        """
        root = await ensure_archive(SOURCE_URL, cache_dir, filename="trace_dataset.zip", sha256=SOURCE_SHA256)
        # The sha256 pins the layout: every split file sits under the archive's top-level dataset/.
        refs = [
            TraceFileRef(root / _archive_path(task, split), task, split, expected)
            for (task, split), expected in SOURCE_FILES.items()
        ]
        missing = [str(ref.path) for ref in refs if not ref.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"TRACE archive is missing {missing}")
        return refs

    def convert(self, raw_refs: list[TraceFileRef]) -> TimeFDataset:
        """Index every window into a lazy Record, check the retrieval rows, and stream the tasks.

        Any subset of the window files works, with both ``retrieval`` files or neither. Given them,
        every classification task carries its window's event report as its rationale, every report
        must belong to a classification task, and every row becomes a caption task on an existing
        Record, so the window files must cover every retrieval row. Without them, as in fixture tests
        of the window files alone, there are no rationales and no caption tasks.

        Captions and reports are not held: ``convert`` checks them one parquet batch at a time, and
        the task stream rereads them when the dataset is written.

        Returns:
            A dataset with all windows, the authors' split membership, and both kinds of task.

        Raises:
            TimeFValidationError: If references, rows, reports, captions, or the pinned inventory are
                invalid.
        """
        files = [(ref.task, ref.split) for ref in raw_refs]
        retrieval_files = [ref for ref in raw_refs if ref.task == "retrieval"]
        if len(set(files)) != len(files) or not set(files) <= SOURCE_FILES.keys() or len(retrieval_files) == len(files):
            raise TimeFValidationError(
                f"TRACE requires distinct split files among {sorted(SOURCE_FILES)}, including a window file"
            )
        retrieval_parts = {part for part in SOURCE_FILES if part[0] == "retrieval"}
        if retrieval_files and len(retrieval_files) != len(retrieval_parts):
            # One file alone would leave the other file's classified windows without their reports.
            raise TimeFValidationError(
                f"TRACE requires both retrieval files or neither, got {sorted(ref.split for ref in retrieval_files)}"
            )
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key="no_event_label", value=NO_EVENT, id="trace-no-event-label"))
        dataset.annotate(Annotation(key="event_type_vocabulary", value=list(EVENT_TYPES), id=_EVENT_SCHEMA))
        loader = _WindowLoader()
        windows: dict[str, _IndexedWindow] = {}
        for file in raw_refs:
            if file.task != "retrieval":
                _add_windows(dataset, file, loader, windows)
        if retrieval_files:
            _check_retrieval(retrieval_files, windows)
        task_types: list[type[Task]] = []
        if any(window.event_type is not None for window in windows.values()):
            task_types.append(ClassificationTask)
        if retrieval_files:
            task_types.append(AnswerTask)
        if task_types:
            dataset.set_task_stream(task_types, partial(_iter_tasks, tuple(retrieval_files), windows))
        return dataset


CONNECTOR = TraceWeatherEventsConnector
