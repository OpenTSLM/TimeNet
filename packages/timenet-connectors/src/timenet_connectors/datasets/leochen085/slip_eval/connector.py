"""The SLIP evaluation benchmarks connector.

The release ships eleven folders, each holding fixed-length windows cut from somebody else's
recordings, split into train and test. One row is one window and one class. The folder fixes the
shape: how many signals, at what rate, over what vocabulary.

Three of the folders — ``PPG_CVA``, ``PPG_DM`` and ``PPG_HTN`` — hold the same windows under
three different diagnoses. They become one record each carrying up to three tasks, rather than the
same values stored three times.

A label is registered once per folder and every task points at it, because the vocabularies are
small and one of them, ``ptbxl``, is whole paragraphs reused across every row of that folder.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, ClassificationTask
from timenet_connectors.datasets.leochen085.slip_eval import folders, specs, tasks
from timenet_connectors.datasets.leochen085.slip_eval.keys import SlipEvalKey


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


_LOG = logging.getLogger(__name__)

_ID_PREFIX = "slip-eval"

_BATCH_ROWS = 512  # rows decoded per batch while walking a split


@dataclass(frozen=True)
class SlipEvalSource:
    """What ``download`` hands ``convert``: the folder root, and no rows."""

    root: Path  # the directory holding the eleven folders


@dataclass
class _Building:
    """What the walk accumulates and the dataset needs once the walk is done."""

    shared: dict[str, Annotation]  # every annotation a task references, keyed by id
    of_window: dict[tuple[str, str, int], str]  # which record each window became
    vocabularies: dict[str, dict[str, int | None]]  # the classes per folder, and the index the release gave each

    @classmethod
    def empty(cls) -> _Building:
        """Give a state holding nothing yet.

        Returns:
            An empty state.
        """
        return cls(shared={}, of_window={}, vocabularies={})


@dataclass(frozen=True)
class SlipEvalWindow:
    """One row of one split, without its values."""

    folder: str  # the directory it came from
    split: str  # train or test
    index: int  # the row's position within the split, counting across that split's files
    path: Path  # the parquet file
    row_in_file: int  # the row's index within that file
    signals: int  # how many signals the window holds
    length: int  # how many values each holds; checked equal across the window when it is read
    label: str  # the readable class, from text_label
    index_of_label: int | None  # the release's own class index, where the folder ships an integer
    participant: str | None  # the subject, where the folder ships one
    prompt: str  # the folder's own template, verbatim


def _iter_windows(root: Path, folder: str) -> Iterator[SlipEvalWindow]:
    """Walk one folder's splits, giving each window's facts without its values.

    Args:
        root: The directory holding the eleven folders.
        folder: The folder's directory name.

    Yields:
        One :class:`SlipEvalWindow` per row.

    Raises:
        TimeFFormatError: If a file lacks a column every folder is expected to ship.
    """
    for split in ("train", "test"):
        index = 0
        for path in sorted((root / folder).glob(f"{split}-*.parquet")):
            reader = pq.ParquetFile(path)
            present = set(reader.schema_arrow.names)
            for required in ("X", "text_label", "prompt"):
                if required not in present:
                    raise TimeFFormatError(f"{path} has no {required!r} column; it holds {sorted(present)}")
            wanted = [name for name in ("label", "text_label", "participant_id", "prompt") if name in present]
            in_file = 0
            for batch in reader.iter_batches(batch_size=_BATCH_ROWS, columns=[*wanted, "X"]):
                scalars = batch.select(wanted).to_pylist()
                column = batch.column("X")
                counts = column.value_lengths().to_pylist()
                lengths = column.flatten().value_lengths().to_pylist()
                cursor = 0
                for row, count in zip(scalars, counts, strict=True):
                    window = lengths[cursor : cursor + count]
                    if len(set(window)) > 1:
                        raise TimeFFormatError(
                            f"{path} row {in_file}: the window's signals have lengths {sorted(set(window))}, "
                            f"but every signal of one window covers the same span"
                        )
                    participant = row.get("participant_id")
                    yield SlipEvalWindow(
                        folder=folder,
                        split=split,
                        index=index,
                        path=path,
                        row_in_file=in_file,
                        signals=count,
                        length=window[0],
                        label=str(row["text_label"]),
                        index_of_label=row["label"] if isinstance(row.get("label"), int) else None,
                        participant=None if participant is None else f"{folder}-{participant}",
                        prompt=str(row["prompt"]),
                    )
                    cursor += count
                    in_file += 1
                    index += 1


@functools.lru_cache(maxsize=1)
def _x_column(path: Path) -> pa.ChunkedArray:
    """Give one file's ``X`` column, reading the file only when it is not the one already held.

    Every series reads its values through :func:`_read_signal`. The writer asks for values grouped
    by spec type and then by signal name, and within one such group in the order this connector
    built the records, so a cache of one turns a read per signal into a read per file. The cache
    holds exactly one, because the release gives no bound on how many would fit. One entry is not
    one row: it is a whole file's column, and the largest of those is big enough that holding a
    second would matter. It stays reachable until another file displaces it.

    The function stands alone because its argument is the cache key. Folded into its caller the key
    would gain the row and the signal, every signal would take an entry, and a cache of one would
    hold nothing useful.

    Args:
        path: The parquet file.

    Returns:
        The file's ``X`` column.
    """
    return pq.ParquetFile(path).read(columns=["X"]).column("X")


def _read_signal(path: Path, row: int, signal: int) -> pa.Array:
    """Read one signal of one window back out of its file.

    Args:
        path: The parquet file.
        row: The row's index within that file.
        signal: Which signal of the window.

    Returns:
        The values as an Arrow array.
    """
    return _x_column(path)[row][signal].values


def _loader_for(path: Path, row: int, signal: int) -> Callable[[], pa.Array]:
    """Give a callable that reads one signal's values when it is asked.

    Args:
        path: The parquet file.
        row: The row's index within that file.
        signal: Which signal of the window.

    Returns:
        A no-argument callable giving the values as an Arrow array.
    """
    return lambda: _read_signal(path, row, signal)


def _series_for(window: SlipEvalWindow) -> tuple[TimeSeries, ...]:
    """Build one :class:`TimeSeries` per signal of one window.

    The axis is the rate the card states for that folder, used as stated.

    Args:
        window: The window's facts.

    Returns:
        The series, in the order the ``X`` column stores them.

    Raises:
        TimeFFormatError: If the window holds a different number of signals than the card names for
            its folder, which means the release has changed shape.
    """
    folder = folders.BY_NAME[window.folder]
    axis = RegularAxis.from_rate_hz(folder.rate_hz)
    names = folder.signals
    if window.signals != len(names):
        raise TimeFFormatError(
            f"{window.path} row {window.row_in_file}: {window.folder} ships {window.signals} signals, "
            f"but the card names {len(names)}: {list(names)}"
        )
    return tuple(
        TimeSeries(
            spec=specs.BY_FOLDER[window.folder],
            signal=names[index],
            time_axis=axis,
            loader=_loader_for(window.path, window.row_in_file, index),
            n_values=window.length,
        )
        for index in range(window.signals)
    )


class SlipEvalConnector(BaseConnector[SlipEvalSource]):
    """Connector for the SLIP evaluation benchmarks (Hub repo ``LeoChen085/SlipDataset``)."""

    HF_REPO = "LeoChen085/SlipDataset"  # the Hub repo the release lives in

    def download(self, cache_dir: Path) -> list[SlipEvalSource]:
        """Fetch the eleven evaluation folders and give back the directory holding them.

        The pretraining corpus of the same repository is not fetched; it is a different dataset.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One handle naming the directory the folders sit in.

        Raises:
            TimeNetDownloadError: If the fetch returned none of the folders.
        """
        patterns = [f"{folder.name}/*.parquet" for folder in folders.FOLDERS]
        root = Path(
            snapshot_download(self.HF_REPO, repo_type="dataset", cache_dir=str(cache_dir), allow_patterns=patterns)
        )
        if not any((root / folder.name).is_dir() for folder in folders.FOLDERS):
            raise TimeNetDownloadError(f"{self.HF_REPO!r} returned none of the evaluation folders under {root}")
        return [SlipEvalSource(root=root)]

    def convert(self, raw_refs: list[SlipEvalSource]) -> TimeFDataset:
        """Build one record per window, and stream one classification task per window.

        The tasks are streamed rather than added. ``add_task`` rebuilds the set of every registered
        task id on each call, so adding this release's tasks one at a time is quadratic. Streaming
        is also what lets the labels be registered once and referenced.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.
        """
        root = raw_refs[0].root
        dataset = TimeFDataset(metadata=self.metadata())
        state = _Building.empty()

        for folder in folders.FOLDERS:
            if folder.name in folders.PPG_FOLDERS:
                continue
            for window in _iter_windows(root, folder.name):
                record_id = f"{_ID_PREFIX}-{folder.name}-{window.split}-{window.index:06d}"
                dataset.add_record(
                    time_series=_series_for(window),
                    record_id=record_id,
                    subject_ids=() if window.participant is None else (window.participant,),
                )
                _register(window, state)
                state.of_window[folder.name, window.split, window.index] = record_id

        # The three PPG folders hold the same windows under three diagnoses. A window is matched
        # across them by its values, so the merge does not depend on the files agreeing about order.
        seen: dict[bytes, str] = {}
        for name in folders.PPG_FOLDERS:
            for window in _iter_windows(root, name):
                digest = _fingerprint_of(window)
                record_id = seen.get(digest)
                if record_id is None:
                    record_id = f"{_ID_PREFIX}-{name}-{window.split}-{window.index:06d}"
                    dataset.add_record(
                        time_series=_series_for(window),
                        record_id=record_id,
                        subject_ids=() if window.participant is None else (window.participant,),
                    )
                    seen[digest] = record_id
                _register(window, state)
                state.of_window[name, window.split, window.index] = record_id

        _report(root)
        dataset.register_annotations([*state.shared.values(), *tasks.vocabularies(state.vocabularies)])
        dataset.set_task_stream([ClassificationTask], lambda: _iter_tasks(root, state.of_window))
        return dataset


def _report(root: Path) -> None:
    """Warn once for each way the release departs from what its card states.

    One warning per kind, naming the folders it applies to, rather than one per record: a property
    of a whole folder is one fact about the release.

    Kept out of ``convert`` because inlining it puts that method over the branch limit ``ruff``
    enforces.

    Args:
        root: The directory holding the eleven folders.
    """
    extra_classes, leaking, duplicated = [], [], []
    for folder in folders.FOLDERS:
        windows = list(_iter_windows(root, folder.name))
        if not windows:
            continue
        labels = {w.label for w in windows}
        if len(labels) != folder.classes:
            extra_classes.append(f"{folder.name} has {len(labels)}, the card says {folder.classes}")
        by_split = {
            split: {w.participant for w in windows if w.split == split and w.participant is not None}
            for split in ("train", "test")
        }
        shared_subjects = by_split["train"] & by_split["test"]
        if shared_subjects:
            leaking.append(f"{folder.name} shares {len(shared_subjects)}")
        distinct = {
            _read_signal(windows[0].path, windows[0].row_in_file, i).to_numpy().tobytes()
            for i in range(windows[0].signals)
        }
        if len(distinct) < windows[0].signals:
            duplicated.append(f"{folder.name} ships {windows[0].signals} signals, {len(distinct)} distinct")
    for found, what in (
        (extra_classes, "folders hold a different number of classes than the card states"),
        (leaking, "folders share subjects between their train and test splits"),
        (duplicated, "folders ship signals that are copies of one another"),
    ):
        if found:
            _LOG.warning("%d %s: %s", len(found), what, "; ".join(found))


def _iter_tasks(root: Path, of_window: dict[tuple[str, str, int], str]) -> Iterator[ClassificationTask]:
    """Yield one :class:`ClassificationTask` per window of every folder.

    Args:
        root: The directory holding the eleven folders.
        of_window: Which record each window belongs to, keyed by folder, split and index. The three
            PPG folders share records, so this is not one record per window.

    Yields:
        The task each window answers, its label held by reference.
    """
    for folder in folders.FOLDERS:
        for window in _iter_windows(root, folder.name):
            yield tasks.task_for(window, of_window[window.folder, window.split, window.index])


def _register(window: SlipEvalWindow, state: _Building) -> None:
    """Register the annotations one window's task will reference, on first sight of each.

    Args:
        window: The window's facts.
        state: What the walk accumulates. Extended in place.
    """
    for key, value, annotation_id in (
        (SlipEvalKey.LABEL, window.label, tasks.label_id(window.folder, window.label)),
        (SlipEvalKey.SOURCE_BENCHMARK, window.folder, tasks.benchmark_id(window.folder)),
        (SlipEvalKey.SPLIT, window.split, tasks.split_id(window.split)),
    ):
        state.shared.setdefault(annotation_id, Annotation(key=key, value=value, id=annotation_id))
    state.vocabularies.setdefault(window.folder, {})[window.label] = window.index_of_label


def fingerprint(signals: Iterable[np.ndarray]) -> bytes:
    """Give a value that is equal for two windows holding the same numbers.

    Args:
        signals: The window's signals, in the order the file stores them.

    Returns:
        A digest over the values, which is what decides whether two folders hold the same window.
    """
    digest = hashlib.blake2b(digest_size=16)
    for values in signals:
        digest.update(values.tobytes())
    return digest.digest()


def _fingerprint_of(window: SlipEvalWindow) -> bytes:
    """Read one window's values and fingerprint them.

    Args:
        window: The window to fingerprint.

    Returns:
        The digest :func:`fingerprint` gives for its values.
    """
    return fingerprint(
        _read_signal(window.path, window.row_in_file, index).to_numpy() for index in range(window.signals)
    )


CONNECTOR = SlipEvalConnector
