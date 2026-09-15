"""The Original side of the ARFBench row: the release's own CSV and Parquet, read directly.

The release ships no loader, so this lane is ours and the paper declares it as ours.
``datasets.load_dataset`` is not an option: it resolves to the series Parquet only, drops
``arfbench-qa.csv`` and collapses the six sampling intervals, so it cannot build the canonical item
at all. What is left is what a practitioner writes: :func:`pandas.read_csv` for the QA table and
Parquet for the series.

The canonical item is one QA pair with the one or two metrics it cites, which is the same item the
TimeF side delivers. A question picks the finest interval every series it cites publishes, the rule
the connector uses, and this lane reads that rule off the file names in ``arfbench-ts-data``. Rows
whose value is null are dropped, again as the connector does, because TimeF stores only finite
values and a lane that kept them would deliver different values.

Two variants of the same read, recorded as ``loader_variant``:

``as_shipped``
    :func:`pandas.read_parquet` with no arguments, which is the line someone writes first. It
    restores ``__index_level_0__``, a pandas index column holding 51.42 percent of the column-chunk
    bytes over the 205 files in scope, and it also reads ``num_groups`` and ``query_name``. None of
    the three reaches the item.
``lazy_capable``
    ``pq.ParquetFile(path).read_row_groups(..., columns=("epoch", "group", "value"))`` over a memory
    map, and the QA table read with ``usecols``. Every file but one holds a single row group, so
    row-group pruning buys nothing here and the whole difference is column projection.

Both variants run the same decode after the read, so the gap between them is the bytes each one
pulled off disk rather than two different pipelines.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import importlib
import itertools
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, ORIGINAL, PANDAS, TORCH


DATASET = "datadog/arfbench"

QA_CSV = "arfbench-qa.csv"
"""The QA table, one row per question. Its ``question`` field spans two physical lines."""

TS_DIR = "arfbench-ts-data"
"""The series directory. A file is ``{incident}_{index}_{interval_seconds}.parquet``."""

JOIN_COLUMN = "query_group"
"""The only correct join key to the series files. ``query_name`` names a different metric on 646
of the released files, so it is never used here."""

ITEM_COLUMNS = ("epoch", "group", "value")
"""The three columns the item needs. The files carry six."""

ANCHOR_US = 1_741_305_600_000_000
"""2025-03-07T00:00:00Z, the release's own zero and the record zero the connector writes."""

US_PER_S = 1_000_000

EMPTY_SIGNAL = "value"
"""Stands in for the empty tag label the release writes for a metric with no grouping."""

DECODE_CACHE = 2
"""How many decoded files the handle keeps, matching the connector's own cache.

A question needs at most two files at once, and consecutive questions often cite the same metric. A
larger cache would hold all 205 files after one pass and turn the full read into something the TimeF
side, which re-reads a shared series per record, does not get.
"""


class Signal(NamedTuple):
    """One tag group of one metric, after the null rows are dropped.

    Attributes:
        name: The tag label, or ``value`` when the release wrote an empty one.
        time_us: Microseconds from :data:`ANCHOR_US`, ascending.
        values: The finite values, in the same order.
        axis: The axis the connector gives the same signal, decided by the same rule.
    """

    name: str
    time_us: np.ndarray
    values: np.ndarray
    axis: Any


class Question(NamedTuple):
    """One QA row, resolved to the files that answer it.

    Attributes:
        item_id: The canonical item id, equal to the TimeF record id.
        series_ids: The metrics the row cites, in the order it lists them.
        interval_s: The finest interval every cited metric publishes.
    """

    item_id: str
    series_ids: tuple[str, ...]
    interval_s: int

    def files(self) -> tuple[str, ...]:
        """Return the series file names this question reads.

        Returns:
            One name per cited metric.
        """
        return tuple(f"{series_id}_{self.interval_s}.parquet" for series_id in self.series_ids)


def published_intervals(ts_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read the sampling intervals each metric publishes off the file names.

    The connector takes this from the Hub's tree listing of all 748 files, this lane from the 205
    that are on disk. The two agree on every question: the interval a question resolves to is always
    downloaded for all of its metrics, and a finer shared interval on disk would be a finer shared
    interval in the repo as well.

    Args:
        ts_dir: The series directory.

    Returns:
        Per metric id, its intervals in seconds, ascending.

    Raises:
        LaneError: If the directory holds no series file.
    """
    found: dict[str, set[int]] = {}
    for path in ts_dir.glob("*.parquet"):
        incident, _, rest = path.stem.partition("_")
        index, _, interval = rest.partition("_")
        if incident and index and interval.isdigit():
            found.setdefault(f"{incident}_{index}", set()).add(int(interval))
    if not found:
        raise LaneError(f"{ts_dir} holds no ARFBench series file, so no question resolves to one")
    return {series_id: tuple(sorted(intervals)) for series_id, intervals in found.items()}


def read_questions(root: Path, loader_variant: str = LAZY_CAPABLE) -> tuple[Question, ...]:
    """Read the QA table and resolve every row to its metrics and its interval.

    Args:
        root: The release directory holding the QA table and the series directory.
        loader_variant: ``as_shipped`` reads the whole table, ``lazy_capable`` reads the join column
            alone. The ``question`` field spans two lines, so the parser scans the file either way.

    Returns:
        One entry per row, in the order the table lists them.

    Raises:
        LaneError: If the table is missing, or a row cites metrics with no shared interval.
    """
    import pandas as pd  # noqa: PLC0415 - a lane dependency, imported per lane rather than per module

    qa_csv = root / QA_CSV
    published = published_intervals(root / TS_DIR)
    try:
        columns = None if loader_variant == AS_SHIPPED else [JOIN_COLUMN]
        table = pd.read_csv(qa_csv, usecols=columns)
    except (OSError, ValueError) as error:
        raise LaneError(f"could not read the ARFBench QA table {qa_csv}: {error}") from error

    questions = []
    for ordinal, group in enumerate(table[JOIN_COLUMN]):
        series_ids = tuple(part.strip() for part in str(group).split(","))
        shared = set(published.get(series_ids[0], ()))
        for series_id in series_ids[1:]:
            shared &= set(published.get(series_id, ()))
        if not shared:
            raise LaneError(
                f"the metrics {list(series_ids)} publish no sampling interval in common under {root}, "
                f"so question {ordinal} has no series to carry"
            )
        questions.append(Question(f"arfbench-{ordinal:03d}", series_ids, min(shared)))
    return tuple(questions)


def item_ids(root: Path) -> tuple[str, ...]:
    """Return the canonical enumeration: one id per QA row, in table order.

    The ids are the TimeF record ids, so the item registry can gate the two sides against each other.

    Args:
        root: The release directory.

    Returns:
        The item ids.
    """
    return tuple(question.item_id for question in read_questions(root))


def tier_a_bytes(root: Path) -> int:
    """Return the bytes this lane opens: the QA table plus the series files it resolves to.

    Args:
        root: The release directory.

    Returns:
        The apparent size of those files.
    """
    wanted = {name for question in read_questions(root) for name in question.files()}
    ts_dir = root / TS_DIR
    return (root / QA_CSV).stat().st_size + sum((ts_dir / name).stat().st_size for name in sorted(wanted))


@dataclass
class ARFBenchHandle:
    """An open release directory, delivering QA rows as one consumer's items.

    Attributes:
        root: The release directory.
        questions: The canonical enumeration, one entry per ordinal.
        consumer: ``pandas`` or ``torch``.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        decoded: The most recently decoded files, at most :data:`DECODE_CACHE` of them.
    """

    root: Path
    questions: tuple[Question, ...]
    consumer: str
    loader_variant: str
    decoded: dict[str, tuple[Signal, ...]] = field(default_factory=dict)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.questions)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of questions and convert them to the consumer's type.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal, in the order asked for.
        """
        for ordinal in ordinals:
            question = self._question(ordinal)
            series = self._series_of(question)
            yield DeliveredItem(ordinal=ordinal, item_id=question.item_id, buffers=self._buffers(question, series))

    def group_of(self, ordinal: int) -> object:
        """Return the source files this item came from.

        Returns:
            The file names, which is the Original side's storage group.
        """
        return self._question(ordinal).files()

    def close(self) -> None:
        """Drop the decoded files the handle was holding."""
        self.decoded.clear()

    def _question(self, ordinal: int) -> Question:
        """Return the question at one ordinal.

        Returns:
            The question.

        Raises:
            LaneError: If the ordinal is outside the enumeration.
        """
        if not 0 <= ordinal < len(self.questions):
            raise LaneError(f"a request asked for an ordinal outside the {len(self.questions)} items")
        return self.questions[ordinal]

    def _series_of(self, question: Question) -> tuple[tuple[str, str, Signal], ...]:
        """Return one question's signals, with the ids the TimeF side gives them.

        Returns:
            Per signal, its time series id, its signal name and its values.
        """
        series = []
        for series_id in question.series_ids:
            path = self.root / TS_DIR / f"{series_id}_{question.interval_s}.parquet"
            for index, signal in enumerate(self._signals(path, question.interval_s)):
                series.append((f"arf-{series_id}-{question.interval_s}-{index}", f"{series_id}/{signal.name}", signal))
        return tuple(series)

    def _signals(self, path: Path, interval_s: int) -> tuple[Signal, ...]:
        """Decode one series file, reusing the last :data:`DECODE_CACHE` decodes.

        Returns:
            The file's signals, ordered by tag label.
        """
        key = str(path)
        held = self.decoded.get(key)
        if held is not None:
            return held
        signals = decode_file(path, self.loader_variant, interval_s)
        self.decoded[key] = signals
        while len(self.decoded) > DECODE_CACHE:
            del self.decoded[next(iter(self.decoded))]
        return signals

    def _buffers(self, question: Question, series: Sequence[tuple[str, str, Signal]]) -> tuple[Buffer, ...]:
        """Build the consumer's own type and return its value planes.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            frame = array_frame(
                question.item_id,
                [(signal_name, signal.axis, signal.values) for _, signal_name, signal in series],
            )
            return array_buffers(frame)
        if self.consumer == TORCH:
            import torch  # noqa: PLC0415 - as above

            return tuple(as_buffer(torch.from_numpy(signal.values).numpy()) for _, _, signal in series)
        raise LaneError(f"the ARFBench lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class ARFBenchLane:
    """One Original lane over the ARFBench release: a directory, a consumer and a read variant.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The release directory holding the QA table and the series directory.
        tier_a_bytes: Tier A bytes of this release scope, which sizes the blocks.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        dataset: Dataset id.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    tier_a_bytes: int
    loader_variant: str = LAZY_CAPABLE
    dataset: str = DATASET
    name: str = ""
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the variant is unknown, or Tier A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the ARFBench lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in {AS_SHIPPED, LAZY_CAPABLE}:
            raise LaneError(f"the ARFBench lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {self.loader_variant!r}")
        if self.tier_a_bytes < 1:
            raise LaneError(
                f"the ARFBench lane needs positive Tier A bytes to size its blocks, got {self.tier_a_bytes}"
            )
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{ORIGINAL}/{self.consumer}/{self.loader_variant}")

    def preload(self) -> None:
        """Import everything this lane reads with, without touching the release directory.

        The QA table always needs pandas, the variant decides the series reader, and the consumer
        decides the delivered type. All three are imported on first use in the normal path, which is
        right for a timed read but wrong for the memory baseline. This pulls them forward so the
        baseline can be taken after them.
        """
        modules = ["pandas", "timenet.dataset.axis"]
        if self.loader_variant == LAZY_CAPABLE:
            modules.append("pyarrow.parquet")
        modules.append("torch" if self.consumer == TORCH else "timenet.pandas")
        for module in modules:
            importlib.import_module(module)

    def open(self) -> ARFBenchHandle:
        """Open the release: read the QA table and resolve every row to its files.

        No series file is touched here. That is the honest split for the first-item cell: opening
        costs the QA table, and the first delivery costs the one file it needs.

        Returns:
            The open handle.

        Raises:
            LaneError: If the release directory is not there.
        """
        if not (self.root / QA_CSV).is_file():
            raise LaneError(f"lane {self.name} could not open {self.root}: no {QA_CSV} in it")
        return ARFBenchHandle(
            root=self.root,
            questions=read_questions(self.root, self.loader_variant),
            consumer=self.consumer,
            loader_variant=self.loader_variant,
        )


def axis_for(offsets: np.ndarray, interval_s: int) -> Any:
    """Return the axis the connector gives a signal, decided by the connector's own rule.

    A stream that starts on the interval and steps by it is a cadence, and the axis is that cadence
    started at the right index. Anything else keeps its offsets, which the axis states the ends of.
    Read off ``timenet_connectors.datasets.datadog.arfbench.connector``, so both sides of the row
    describe a signal's clock the same way.

    Args:
        offsets: The signal's microsecond offsets, ascending.
        interval_s: The file's sampling interval in seconds.

    Returns:
        A :class:`~timenet.dataset.axis.RegularAxis` or an
        :class:`~timenet.dataset.axis.IrregularAxis`.
    """
    # A lane dependency, imported per call rather than at module import.
    from timenet.dataset.axis import IrregularAxis, RegularAxis  # noqa: PLC0415

    step_us = interval_s * US_PER_S
    first = int(offsets[0])
    if first % step_us == 0 and bool(np.all(np.diff(offsets) == step_us)):
        return RegularAxis.from_rate_hz(Fraction(1, interval_s)).at_index(first // step_us)
    return IrregularAxis(first_us=first, last_us=int(offsets[-1]))


def decode_file(path: Path, loader_variant: str, interval_s: int) -> tuple[Signal, ...]:
    """Read one series file and cut it into signals.

    The two variants differ in the read alone. Both then drop the null rows, sort by tag label and
    then by time, and split on the label boundaries. Row order in the release is arbitrary and the
    index the files carry does not restore it, so the sort is not optional.

    Args:
        path: The series file.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        interval_s: The file's sampling interval in seconds, which decides each signal's axis.

    Returns:
        The signals, ordered by tag label.

    Raises:
        LaneError: If the variant is unknown, the file cannot be read, no row survives the null
            drop, or a surviving row carries a null tag label.
    """
    if loader_variant == AS_SHIPPED:
        read = _read_as_shipped
    elif loader_variant == LAZY_CAPABLE:
        read = _read_lazy_capable
    else:
        raise LaneError(f"the ARFBench lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {loader_variant!r}")
    try:
        labels, codes, time_us, values = read(path)
    except (OSError, ValueError) as error:
        raise LaneError(f"could not read the ARFBench series file {path}: {error}") from error

    keep = np.isfinite(values)
    codes, time_us, values = codes[keep], time_us[keep], values[keep]
    if codes.size == 0:
        raise LaneError(f"the series file {path} holds no finite value, so it carries no signal")
    if bool((codes < 0).any()):
        raise LaneError(
            f"the series file {path} holds a finite value under a null tag label; the release writes "
            f"an empty label for a metric with no grouping, so a null is a shape this lane has no "
            f"signal name for"
        )
    order = np.lexsort((time_us, codes))
    codes, time_us, values = codes[order], time_us[order], values[order]
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1, [len(codes)]))
    signals = [
        Signal(
            labels[codes[start]] or EMPTY_SIGNAL,
            time_us[start:stop],
            values[start:stop],
            axis_for(time_us[start:stop], interval_s),
        )
        for start, stop in itertools.pairwise(bounds)
    ]
    return tuple(sorted(signals, key=lambda signal: signal.name))


def _read_as_shipped(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read one series file the way someone writes it first: ``pandas.read_parquet(path)``.

    Every column comes back, and pandas restores ``__index_level_0__`` as the frame's index because
    the file's own metadata names it as one. That index is 51.42 percent of the column-chunk bytes
    across the files in scope and no part of it reaches the item.

    Args:
        path: The series file.

    Returns:
        The distinct tag labels, one label code per row, the microsecond offsets and the values.
    """
    # A lane dependency, imported per lane rather than at module import.
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_parquet(path)
    codes, labels = pd.factorize(frame["group"])
    return (
        [str(label) for label in labels],
        np.asarray(codes, dtype=np.int64),
        _offsets_us(frame["epoch"].to_numpy()),
        frame["value"].to_numpy(dtype=np.float64, na_value=np.nan),
    )


def _read_lazy_capable(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read one series file's three needed columns off a memory map, row group by row group.

    204 of the 205 files in scope hold a single row group, so this is the same rows as the eager
    read and the saving is the three columns it never opens.

    Args:
        path: The series file.

    Returns:
        The distinct tag labels, one label code per row, the microsecond offsets and the values.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 - as above

    handle = pq.ParquetFile(path, memory_map=True)
    table = handle.read_row_groups(range(handle.metadata.num_row_groups), columns=list(ITEM_COLUMNS))
    labels = table.column("group").combine_chunks().dictionary_encode()
    return (
        [str(label) for label in labels.dictionary.to_pylist()],
        labels.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64),
        _offsets_us(table.column("epoch").to_numpy(zero_copy_only=False)),
        table.column("value").to_numpy(zero_copy_only=False).astype(np.float64, copy=False),
    )


def _offsets_us(epochs: Any) -> np.ndarray:
    """Return timestamps as microseconds from :data:`ANCHOR_US`.

    Args:
        epochs: The ``epoch`` column, as datetime64.

    Returns:
        Int64 microseconds, one per row.
    """
    return np.asarray(epochs).astype("datetime64[us]").astype(np.int64) - ANCHOR_US
