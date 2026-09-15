"""The Original side of the HEARTS row: the release's own pickled test cases.

The release ships one Python pickle per frozen test case, in a flat ``<corpus>/<task>/<index>.pkl``
tree, and its reference harness reads a case with ``pickle.load``. So the canonical item and the
source file are the same unit here: 1,005 cases, 1,005 files, one item per file. That is the same
item the TimeF side delivers, where one case is one record.

Reading goes through the connector's restricted unpickler instead of a bare ``pickle.load``.
Rebuilding a payload runs whatever the stream names, so this release needs a reader that refuses a
global it does not know, and the allowlist holds the fifteen pairs the corpus asks for. The values
are the same either way: the guard only decides which globals may be imported.

Two variants of the same read, recorded as ``loader_variant``:

``as_shipped``
    ``pickle.load`` on the whole file, which is what the release's own harness does.
``lazy_capable``
    The same call, for the reason :data:`WHY_NO_LAZY_READ` records.

A pickle has no header, no column table and no offsets. Nothing about a payload, not its signals,
not its rates, not its lengths, is known until the whole opcode stream has been run, so there is no
way to read one signal or one range of a case without rebuilding all of it. Both variants call the
same function, and the gap between the two columns on this row is measurement noise. That is a
property of the format the release chose, and a lane that invented a lazy path would report on a
format the release does not ship.

A payload mixes modalities and rates: 100 Hz respiration, 1 Hz heart rate, and audio at 16 kHz to
48 kHz. The axis is decided per series from that series' own data, never per corpus, which is the
rule the connector follows and the rule the TimeF artifact stores. The 1,005 cases carry 1,555
series between them, so a case holds one, two or four of them.

The tree ships no index, so opening it walks the twenty-one task directories and reads the
enumeration off the file names. That walk is inside the first-item cell, because a consumer cannot
name the first case before it has one. It costs a fraction of the first case's own read, and it is
the price of an item that is addressed by a path rather than by a row.

Three things come from the connector rather than from a table of this module's own: the unpickler,
the task table that says which directories are in scope and in which order they are walked, and the
conversion of a frame's time column to microseconds. The enumeration is what the two sides are gated
on, and a clock read two ways is two different items, so both sides read those off one declaration.
Everything after that, the walk into a payload and the shaping of a consumer's type, is this lane's
own. :meth:`HeartsLane.preload` imports the three, so none of it lands inside a timed read.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import pickle  # noqa: S403 - imported for its error type; every read goes through the connector
from typing import Any, NamedTuple

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, ORIGINAL, PANDAS, TORCH
from timenet.errors import TimeFFormatError


DATASET = "yang-ai-lab/hearts"

WHY_NO_LAZY_READ = (
    "a pickle has no header and no random access, so the cheapest defensible read of one case is "
    "the whole case; lazy_capable is the same read as as_shipped"
)
"""Why the two variants coincide. Recorded here so the campaign can print the reason beside them."""

PICKLES_MODULE = "timenet_connectors.datasets.yang_ai_lab.hearts.pickles"
"""The connector's restricted unpickler and its two-payload cache."""

TASKS_MODULE = "timenet_connectors.datasets.yang_ai_lab.hearts.tasks"
"""The connector's task table: the directories in scope, in the order both sides walk them."""

SERIES_MODULE = "timenet_connectors.datasets.yang_ai_lab.hearts.series"
"""The connector's time-column conversion, so both sides read a frame's clock the same way."""

CASE_GLOB = "*.pkl"

ANSWER_KEY = "GT"
"""The key holding the answer. It carries no series and the walk never descends into it."""

TIME_COLUMNS = ("Timestamp", "timestamp", "Time (min)")
"""Columns that place a frame's rows in time. They become the time column, never a signal."""

INDEX_COLUMNS = ("timestamp_min",)
"""Columns that restate the time column as whole minutes. The time column already says this."""

COLUMN_DTYPES: dict[tuple[str, str], str] = {
    ("cgmacros", "Libre GL"): "float64",
    ("cgmacros", "CGM (mg/dL)"): "float64",
    ("harespod", "rsp"): "float64",
    ("harespod", "spo"): "float64",
    ("harespod", "hr"): "float64",
}
"""Value dtype per (corpus, frame column), matching the spec the connector gives that column."""

AUDIO_DTYPES: dict[str, str] = {"coswara": "float32", "coughvid": "float32", "vctk": "float32"}
"""Value dtype per corpus that stores bare audio buffers. Every buffer in the release is float32."""

VCTK_RATE_HZ = 16_000
"""VCTK payloads carry no rate. The release's ``exp/vctk/base.py`` defaults to this one."""

US_PER_S = 1_000_000


class Signal(NamedTuple):
    """One series of one test case, decoded and placed in time.

    Attributes:
        signal: The signal name, which is the key path that reached the values.
        time_series_id: The id the TimeF side gives the same series.
        axis: The axis the connector gives the same series, read off the connector's own rule.
        values: The values, in the spec's dtype.
    """

    signal: str
    time_series_id: str
    axis: Any
    values: np.ndarray


@dataclass(frozen=True)
class ReleaseCase:
    """One frozen test case, addressed the way both sides address it.

    Attributes:
        item_id: The canonical item id, equal to the TimeF record id.
        source: The corpus directory, such as ``harespod``.
        path: The ``.pkl`` file holding the case.
    """

    item_id: str
    source: str
    path: Path


def release_cases(root: Path) -> tuple[ReleaseCase, ...]:
    """Walk the release tree in canonical order: corpus, then task, then case index.

    The order is the connector's task table, so both sides of the row enumerate the same cases in
    the same order. A directory the tree does not hold is skipped rather than refused, which is what
    lets a small stand-in tree exercise the lane.

    Args:
        root: The directory holding the ``<corpus>/<task>/<index>.pkl`` tree.

    Returns:
        One entry per test case found, in enumeration order.

    Raises:
        LaneError: If a file is not named after its case index, or the tree holds no case in scope.
    """
    definitions = importlib.import_module(TASKS_MODULE).TASK_DEFINITIONS
    cases: list[ReleaseCase] = []
    for definition in definitions:
        directory = root / definition.source / definition.task
        if not directory.is_dir():
            continue
        try:
            paths = sorted(directory.glob(CASE_GLOB), key=lambda path: int(path.stem))
        except ValueError as error:
            raise LaneError(f"HEARTS {definition.directory} holds a file whose name is not a case index") from error
        for path in paths:
            item_id = f"hearts-{definition.source}-{definition.task}-{int(path.stem):02d}"
            cases.append(ReleaseCase(item_id=item_id, source=definition.source, path=path))
    if not cases:
        raise LaneError(f"the HEARTS tree at {root} holds no test case in scope, so it enumerates no item")
    return tuple(cases)


def item_ids(root: Path) -> tuple[str, ...]:
    """Return the canonical enumeration: one id per test case, in walk order.

    The ids are the TimeF record ids, so the item registry can gate the two sides against each
    other.

    Args:
        root: The release tree.

    Returns:
        The item ids.
    """
    return tuple(case.item_id for case in release_cases(root))


def tier_a_bytes(root: Path) -> int:
    """Return the bytes this lane opens: every test case in scope.

    Args:
        root: The release tree.

    Returns:
        The apparent size of those files.
    """
    return sum(case.path.stat().st_size for case in release_cases(root))


def read_payload(path: Path, loader_variant: str) -> dict[str, Any]:
    """Read one test case with the release's own loader.

    Both variants land here, for the reason :data:`WHY_NO_LAZY_READ` records.

    Args:
        path: The test-case file.
        loader_variant: ``as_shipped`` or ``lazy_capable``.

    Returns:
        The payload.

    Raises:
        LaneError: If the variant is unknown, or the file cannot be read.
    """
    if loader_variant not in {AS_SHIPPED, LAZY_CAPABLE}:
        raise LaneError(f"the HEARTS lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {loader_variant!r}")
    load_payload = importlib.import_module(PICKLES_MODULE).load_payload
    try:
        return load_payload(str(path))
    except (OSError, pickle.UnpicklingError, TimeFFormatError) as error:
        raise LaneError(f"could not read the HEARTS test case {path}: {error}") from error


def discard_payloads() -> None:
    """Drop the payloads the connector's reader is holding.

    That reader caches the last two payloads. Each item reads its own file, so the cache never hits
    inside a pass, and clearing it on close keeps a payload from crossing from one pass into the
    next.
    """
    importlib.import_module(PICKLES_MODULE).load_payload.cache_clear()


def signals_of(source: str, payload: dict[str, Any], item_id: str) -> tuple[Signal, ...]:
    """Cut one payload into the series it carries, in signal order.

    A frame becomes one series per value column, placed by its own time column. A bare array is an
    audio buffer and takes its rate from the payload. The answer is skipped: it holds no series, and
    on the TimeF side it lives on the task rather than on the record.

    Args:
        source: The corpus the payload came from.
        payload: The payload.
        item_id: The canonical item id, which prefixes every time series id.

    Returns:
        The case's series, sorted by signal so the order does not depend on dict order.

    Raises:
        LaneError: If the payload holds a shape this lane does not read.
    """
    signals: list[Signal] = []
    for keys, node, parent in _walk(payload, item_id):
        if isinstance(node, np.ndarray):
            signals.append(_audio_signal(source, parent, keys, node, item_id))
        else:
            signals.extend(_frame_signals(source, keys, node, item_id))
    if not signals:
        raise LaneError(f"the HEARTS test case {item_id} holds no series, so it carries no item")
    return tuple(sorted(signals, key=lambda entry: entry.signal))


@dataclass
class HeartsHandle:
    """An open release tree, delivering test cases as one consumer's items.

    The item and the file are the same unit, so the handle holds no decoded cache: every item reads
    its own file exactly once per pass.

    Attributes:
        root: The release tree.
        cases: The canonical enumeration, one entry per ordinal.
        consumer: ``pandas`` or ``torch``.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
    """

    root: Path
    cases: tuple[ReleaseCase, ...]
    consumer: str
    loader_variant: str

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.cases)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of cases and convert them to the consumer's type.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal, in the order asked for.
        """
        for ordinal in ordinals:
            case = self._case(ordinal)
            payload = read_payload(case.path, self.loader_variant)
            signals = signals_of(case.source, payload, case.item_id)
            yield DeliveredItem(ordinal=ordinal, item_id=case.item_id, buffers=self._buffers(case, signals))

    def group_of(self, ordinal: int) -> object:
        """Return the source file this item came from.

        Returns:
            The file's path under the release tree, which is the Original side's storage group.
        """
        return str(self._case(ordinal).path.relative_to(self.root))

    def close(self) -> None:  # noqa: PLR6301 - the LaneHandle protocol closes the handle, not the module
        """Drop the payload the connector's reader is holding."""
        discard_payloads()

    def _case(self, ordinal: int) -> ReleaseCase:
        """Return the test case at one ordinal.

        Returns:
            The case.

        Raises:
            LaneError: If the ordinal is outside the enumeration.
        """
        if not 0 <= ordinal < len(self.cases):
            raise LaneError(f"a request asked for an ordinal outside the {len(self.cases)} items")
        return self.cases[ordinal]

    def _buffers(self, case: ReleaseCase, signals: Sequence[Signal]) -> tuple[Buffer, ...]:
        """Build the consumer's own type and return its value planes.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the consumer is not one this lane implements.
        """
        if self.consumer == PANDAS:
            frame = array_frame(case.item_id, [(signal.signal, signal.axis, signal.values) for signal in signals])
            return array_buffers(frame)
        if self.consumer == TORCH:
            import torch  # noqa: PLC0415 - as above

            # copy() gives torch a writable array, which is the copy the TimeF side pays as well.
            return tuple(as_buffer(torch.from_numpy(signal.values.copy()).numpy()) for signal in signals)
        raise LaneError(f"the HEARTS lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class HeartsLane:
    """One Original lane over the HEARTS release: a tree, a consumer and a read variant.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The directory holding the ``<corpus>/<task>/<index>.pkl`` tree.
        tier_a_bytes: Tier A bytes of this release scope, which sizes the blocks.
        loader_variant: ``as_shipped`` or ``lazy_capable``. Both read the same way.
        dataset: Dataset id.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    tier_a_bytes: int
    loader_variant: str = AS_SHIPPED
    dataset: str = DATASET
    name: str = ""
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the variant is unknown, or Tier A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the HEARTS lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in {AS_SHIPPED, LAZY_CAPABLE}:
            raise LaneError(f"the HEARTS lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {self.loader_variant!r}")
        if self.tier_a_bytes < 1:
            raise LaneError(f"the HEARTS lane needs positive Tier A bytes to size its blocks, got {self.tier_a_bytes}")
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{ORIGINAL}/{self.consumer}/{self.loader_variant}")

    def preload(self) -> None:
        """Import everything this lane reads with, without touching the release tree.

        Rebuilding a payload needs pandas whichever consumer is asked for, because the released
        cases hold pandas frames. The connector's reader and its task table come in here too, and
        torch is a few hundred megabytes on its own. All of them are imported on first use in the
        normal path, which is right for a timed read but wrong for the memory baseline.
        """
        modules = ["pandas", "timenet.dataset.axis", PICKLES_MODULE, TASKS_MODULE, SERIES_MODULE]
        modules.append("torch" if self.consumer == TORCH else "timenet.pandas")
        for module in modules:
            importlib.import_module(module)

    def open(self) -> HeartsHandle:
        """Open the release: walk the tree and enumerate its cases.

        No payload is read here. That is the honest split for the first-item cell: opening costs the
        directory walk, and the first delivery costs the one case it needs.

        Returns:
            The open handle.

        Raises:
            LaneError: If the release tree is not there.
        """
        if not self.root.is_dir():
            raise LaneError(f"lane {self.name} could not open {self.root}: it is not a directory")
        return HeartsHandle(
            root=self.root,
            cases=release_cases(self.root),
            consumer=self.consumer,
            loader_variant=self.loader_variant,
        )


def _walk(payload: dict[str, Any], item_id: str) -> Iterator[tuple[tuple[str, ...], Any, dict[str, Any]]]:
    """Yield every frame and array in a payload, depth first, skipping the answer.

    Args:
        payload: The payload to walk.
        item_id: The canonical item id, for the error messages.

    Yields:
        The key path, the frame or array found there, and the dict holding it.
    """
    # A lane dependency, imported per lane rather than at module import.
    import pandas as pd  # noqa: PLC0415

    yield from _descend(payload, (), payload, item_id, pd.DataFrame)


def _descend(
    node: Any,
    keys: tuple[str, ...],
    parent: dict[str, Any],
    item_id: str,
    frame_type: type,
) -> Iterator[tuple[tuple[str, ...], Any, dict[str, Any]]]:
    """Walk one node of a payload.

    Args:
        node: The node to walk.
        keys: The key path that reached it.
        parent: The dict holding it.
        item_id: The canonical item id, for the error messages.
        frame_type: The frame class, passed in so the walk needs no import of its own.

    Yields:
        The key path, the frame or array found there, and the dict holding it.

    Raises:
        LaneError: If an array is not the 1-D float buffer this lane reads.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key != ANSWER_KEY:
                yield from _descend(value, (*keys, str(key)), node, item_id, frame_type)
    elif isinstance(node, np.ndarray):
        if node.ndim != 1 or node.dtype.kind != "f":
            raise LaneError(
                f"HEARTS array {'.'.join(keys)} of {item_id} has dtype {node.dtype} and ndim "
                f"{node.ndim}; this lane reads only the 1-D float buffers the release holds"
            )
        yield keys, node, parent
    elif isinstance(node, frame_type):
        yield keys, node, parent


def _audio_signal(
    source: str,
    parent: dict[str, Any],
    keys: tuple[str, ...],
    values: np.ndarray,
    item_id: str,
) -> Signal:
    """Return one audio buffer as a signal, paced by the rate its payload states.

    Args:
        source: The corpus the payload came from.
        parent: The dict holding the buffer, which is where its rate sits.
        keys: The key path of the buffer.
        values: The buffer.
        item_id: The canonical item id.

    Returns:
        The signal.

    Raises:
        LaneError: If the corpus has no audio dtype, or the stated rate is not a positive integer.
    """
    from timenet.dataset.axis import RegularAxis  # noqa: PLC0415 - a lane dependency

    dtype = AUDIO_DTYPES.get(source)
    if dtype is None:
        raise LaneError(f"HEARTS {source} has a bare array at {'.'.join(keys)} in {item_id} but no audio dtype")
    rate_hz = VCTK_RATE_HZ if source == "vctk" else parent.get("sr")
    if not isinstance(rate_hz, int) or isinstance(rate_hz, bool) or rate_hz <= 0:
        raise LaneError(f"HEARTS {source} audio in {item_id} has sampling rate {rate_hz!r}, which is not positive")
    signal = ".".join(keys)
    return Signal(
        signal=signal,
        time_series_id=f"{item_id}-{signal}",
        axis=RegularAxis.from_rate_hz(rate_hz),
        values=np.ascontiguousarray(values, dtype=np.dtype(dtype)),
    )


def _frame_signals(source: str, keys: tuple[str, ...], frame: Any, item_id: str) -> Iterator[Signal]:
    """Yield one signal per value column of one frame.

    Args:
        source: The corpus the payload came from.
        keys: The key path of the frame.
        frame: The frame itself.
        item_id: The canonical item id.

    Yields:
        One signal per value column.

    Raises:
        LaneError: If the frame has no time column, or a value column has no dtype here.
    """
    time_column = next((name for name in TIME_COLUMNS if name in frame.columns), None)
    if time_column is None:
        raise LaneError(
            f"HEARTS frame {'.'.join(keys)} of {item_id} has no time column; its columns are {list(frame.columns)}"
        )
    axis = _frame_axis(frame[time_column])
    for column in frame.columns:
        if column in TIME_COLUMNS or column in INDEX_COLUMNS:
            continue
        dtype = COLUMN_DTYPES.get((source, str(column)))
        if dtype is None:
            raise LaneError(f"HEARTS column {column!r} of {source} has no dtype in this lane; add the pair")
        signal = ".".join((*keys, str(column)))
        yield Signal(
            signal=signal,
            time_series_id=f"{item_id}-{signal}",
            axis=axis,
            values=np.ascontiguousarray(frame[column].to_numpy(), dtype=np.dtype(dtype)),
        )


def _frame_axis(column: Any) -> Any:
    """Return the axis a frame's rows sit on, read off the connector's own two rules.

    A frame whose steps are all equal is a cadence and gets a regular axis. Everything else keeps
    the offsets its column gives, and the axis states their ends. Both rules come from the
    connector, so the two sides of the row describe a clock the same way.

    Args:
        column: The frame's time column.

    Returns:
        A :class:`~timenet.dataset.axis.RegularAxis` or an
        :class:`~timenet.dataset.axis.IrregularAxis`.

    Raises:
        LaneError: If the column has a dtype this lane does not read as time.
    """
    series = importlib.import_module(SERIES_MODULE)
    try:
        return series.axis_for(series.time_offsets_us(column))
    except (TimeFFormatError, ValueError) as error:
        raise LaneError(f"a HEARTS time column could not be read as time: {error}") from error
