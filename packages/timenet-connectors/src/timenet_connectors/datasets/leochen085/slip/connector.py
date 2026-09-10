"""The SLIP pretraining corpus connector.

The release ships 17 parquet shards under ``data/``. One row is one record: a series or a set of
series, four captions of it, and the name of the corpus it was drawn from. ``meta.csv`` describes
those corpora, and the ``dataset`` column joins to it, which is how a row recovers its sampling rate.

The corpus is 4.3 GB and 618,508 rows, so nothing here holds values. ``download`` gives back one
handle naming the shards, ``convert`` walks them, every series reads its own values when asked, and
the tasks are streamed.

It does read them twice. ``convert`` decodes every value once to count, per series, how many of
them are finite: none, which 96 of the release's 1,659,875 signals answer and no length or header
states, or some but not all, which none of them answers today. The loaders decode the values again
when the writer asks. That second pass is what the ``all_nan_signals`` annotation and the two
warnings cost.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import functools
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask
from timenet_connectors.datasets.leochen085.slip import tables
from timenet_connectors.datasets.leochen085.slip.keys import SlipKey
from timenet_connectors.datasets.leochen085.slip.specs import SERIES


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


_LOG = logging.getLogger(__name__)

_ID_PREFIX = "slip"

_SHARDS = "data/*.parquet"
_META = "meta.csv"
_CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")
_ROW_COLUMNS = ["category", "dataset", *_CAPTION_COLUMNS]
_BATCH_ROWS = 2048  # rows decoded per batch while walking a shard
# Every shard of the release is named this way, and a record id is built from the number in it.
_SHARD_NAME = re.compile(r"\Atrain-(?P<number>\d{5})-of-\d{5}\Z")
# What the release leaves behind where a caption should be: "Paraphrase 1:", "The second paraphrase:".
_STUB = re.compile(r"^\s*(the\s+\w+\s+)?paraphrase\s*\d*\s*:?\s*$", re.IGNORECASE)
# A caption the generator finished ends a sentence. One that stopped at a length limit does not.
_ENDS_A_SENTENCE = (".", "!", "?", '"')


class _Tally:
    """Counts one kind of thing the release ships, so the build warns once about it and not once each.

    A property shared by thousands of records is one fact about the release, so the report gives a
    count and one example rather than a line per record.
    """

    def __init__(self) -> None:
        self.seen = 0
        self.first = ""

    def count(self, example: str) -> None:
        """Record one occurrence, keeping the first as the example.

        Args:
            example: What to show if this kind is reported.
        """
        self.seen += 1
        if not self.first:
            self.first = example

    def report(self, what: str) -> None:
        """Log this kind once, with its count and its example. Silent when nothing was counted.

        Args:
            what: The sentence naming the kind, e.g. "captions stop mid-sentence".
        """
        if self.seen:
            _LOG.warning("%d %s; the first is %s", self.seen, what, self.first)


@dataclass(frozen=True)
class SlipSource:
    """What ``download`` hands ``convert``: paths, and no rows."""

    shards: tuple[Path, ...]  # the 17 data/train-*.parquet files, in name order
    meta_csv: Path  # the table describing the corpora the rows were drawn from


@dataclass(frozen=True)
class SlipRow:
    """One row of one shard, without its values."""

    index: int  # the row's absolute index in the shard
    row_group: int  # which row group of the shard holds it
    offset: int  # its index within that row group
    scalars: dict[str, Any]  # category, dataset and the four captions
    lengths: list[int]  # the length of each of the row's series
    empty: tuple[int, ...]  # which of those series hold no finite number at all
    partial: tuple[tuple[int, int], ...]  # which hold some, and how many of their values are not finite


@dataclass(frozen=True)
class _SignalRef:
    """Where one signal's values sit, so a loader can find them again without holding them.

    The row group and the offset within it are resolved when the reference is built, so reading a
    signal is one row-group read and no lookup. Resolving them per read would be a scan per signal,
    and a full build calls the loaders 1.66 million times.
    """

    shard: Path  # the parquet file
    row_group: int  # which row group of that file holds the row
    offset: int  # the row's index within that row group
    signal: int  # which inner list of the row's time_series column


@functools.lru_cache(maxsize=1)
def _row_group(shard: Path, group: int) -> pa.ChunkedArray:
    """Give one row group's ``time_series`` column, decoding it only when it is not the one held.

    Every series reads its values through a loader, and the writer asks for them in the order this
    connector built the records. Without a cache each of the release's 1,659,875 signals would
    decode a whole row group of roughly 7,300 rows to take one of them. With one, the whole release
    decodes 85 row groups, five per shard.

    The function stands alone because its arguments are the cache key. Folded into its caller the
    key would gain the row and the signal, every signal would take an entry, and a cache of one
    would hold nothing useful.

    Args:
        shard: The parquet file.
        group: Which row group of it.

    Returns:
        That row group's ``time_series`` column.
    """
    return pq.ParquetFile(shard).read_row_groups([group], columns=["time_series"]).column("time_series")


def _shard_number(shard: Path) -> str:
    """Give the number a shard's name states, which a record id is built from.

    Args:
        shard: The parquet file.

    Returns:
        The five digits between ``train-`` and ``-of-``.

    Raises:
        TimeFFormatError: If the name states no number. The id is built from that number, so a
            release that renames its shards would renumber every record.
    """
    match = _SHARD_NAME.match(shard.stem)
    if match is None:
        raise TimeFFormatError(
            f"{shard.name} is not named train-NNNNN-of-NNNNN, so it states no number to build an id from"
        )
    return match["number"]


def _record_id(shard: Path, row: int) -> str:
    """Give a record's id: which shard it came from, and where in it.

    The release ships no id of its own, so the id is positional. Two builds of one release agree; a
    new release does not.

    Args:
        shard: The parquet file the row came from.
        row: The row's absolute index in that file.

    Returns:
        An id of the form ``slip-shard-00000-row-000000``.

    Raises:
        TimeFFormatError: If the shard's name states no number.
    """  # noqa: DOC502 (raised by _shard_number, which convert calls for every shard up front)
    return f"{_ID_PREFIX}-shard-{_shard_number(shard)}-row-{row:06d}"


class SlipConnector(BaseConnector[SlipSource]):
    """Connector for the SLIP pretraining corpus (Hub repo ``LeoChen085/SlipDataset``, ``data/``)."""

    HF_REPO = "LeoChen085/SlipDataset"  # the Hub repo the release lives in

    def download(self, cache_dir: Path) -> list[SlipSource]:
        """Fetch the corpus shards and ``meta.csv``, and give back their paths.

        The evaluation folders of the same repository are not fetched; they are a different dataset.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One handle naming every shard and the table beside them.

        Raises:
            ImportError: If ``huggingface_hub``, declared in this connector's ``requirements.txt``,
                is not installed.
            TimeNetDownloadError: If the fetch returned no shards.
        """
        # discovery.available() imports every connector module to read its CONNECTOR, and
        # huggingface_hub is declared in this connector's requirements.txt rather than by the
        # package. A module-level import would break dataset listing for every connector in an
        # environment without it.
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415 (see the comment above)
        except ImportError as exc:
            raise ImportError(
                f"reading {self.HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc

        root = Path(
            snapshot_download(
                self.HF_REPO, repo_type="dataset", cache_dir=str(cache_dir), allow_patterns=[_SHARDS, _META]
            )
        )
        shards = tuple(sorted(root.glob(_SHARDS)))
        if not shards:
            raise TimeNetDownloadError(f"{self.HF_REPO!r} returned no shards matching {_SHARDS!r} under {root}")
        return [SlipSource(shards=shards, meta_csv=root / _META)]

    def convert(self, raw_refs: list[SlipSource]) -> TimeFDataset:
        """Build one record per row of every shard, and stream one task per caption.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.

        Raises:
            TimeNetDownloadError: If the handle names no shards.
            TimeFFormatError: If ``meta.csv`` states a rate this connector cannot read, or names a
                corpus twice, or names none of the corpus a row was drawn from; or if a shard's name
                states no number.
            TimeFValidationError: If a row states a series with no values, which no record can hold.
        """  # noqa: DOC502 (both come from the helpers below and from TimeSeries, not from here)
        source = raw_refs[0]
        if not source.shards:
            raise TimeNetDownloadError("the SLIP handle names no shards")
        # Every shard's name is read before any row is, so a mis-named shard stops the build in
        # milliseconds rather than after every shard before it has been converted.
        for shard in source.shards:
            _shard_number(shard)
        with source.meta_csv.open(encoding="utf-8", newline="") as handle:
            corpora = tables.corpora(list(csv.DictReader(handle)))
        dataset = TimeFDataset(metadata=self.metadata())
        empty, partial, stubs, truncated = _Tally(), _Tally(), _Tally(), _Tally()
        for shard in source.shards:
            for row in _iter_rows(shard):
                name = row.scalars["dataset"]
                corpus = _corpus(corpora, name, shard, row.index)
                record_id = _record_id(shard, row.index)
                axis = OrdinalAxis() if corpus.period_us is None else RegularAxis(period_us=corpus.period_us)
                record = dataset.add_record(
                    time_series=tuple(
                        TimeSeries(
                            spec=SERIES,
                            signal=f"s{index}",
                            time_axis=axis,
                            loader=_loader_for(
                                _SignalRef(shard=shard, row_group=row.row_group, offset=row.offset, signal=index)
                            ),
                            n_values=length,
                            time_series_id=f"{record_id}-s{index}",
                        )
                        for index, length in enumerate(row.lengths)
                    ),
                    record_id=record_id,
                )
                annotations = [
                    Annotation(key=SlipKey.SOURCE_DATASET, value=name),
                    Annotation(key=SlipKey.DOMAIN, value=row.scalars["category"]),
                    Annotation(key=SlipKey.SOURCE_URL, value=corpus.source_url),
                ]
                if row.empty:
                    names = [f"s{signal}" for signal in row.empty]
                    annotations.append(Annotation(key=SlipKey.ALL_NAN_SIGNALS, value=names))
                    empty.count(f"{record_id} ({name}): {', '.join(names)}")
                if row.partial:
                    counts = [f"s{signal} ({many})" for signal, many in row.partial]
                    partial.count(f"{record_id} ({name}): {', '.join(counts)}")
                record.add_annotations(annotations)
                _count_captions(row, record_id, stubs, truncated)
        empty.report("records hold a series with no finite numbers at all")
        partial.report("records hold a series with some values that are not finite numbers")
        stubs.report("captions hold a paraphraser stub rather than text")
        truncated.report("captions stop mid-sentence, where the generator hit a length limit")
        dataset.set_task_stream([AnswerTask], lambda: _iter_tasks(source.shards))
        return dataset


def _corpus(corpora: dict[str, tables.SourceCorpus], name: str, shard: Path, row: int) -> tables.SourceCorpus:
    """Give what ``meta.csv`` says about the corpus a row names.

    Args:
        corpora: What the table states, keyed by corpus name.
        name: The value the row's ``dataset`` column holds.
        shard: The parquet file the row came from.
        row: The row's index in that shard.

    Returns:
        That corpus.

    Raises:
        TimeFFormatError: If the table names no such corpus, which means the release grew one and
            its rate and its source are unknown rather than absent.
    """
    corpus = corpora.get(name)
    if corpus is None:
        raise TimeFFormatError(
            f"{shard.name} row {row}: dataset holds {name!r}, which meta.csv does not name. "
            f"It states {len(corpora)} corpora, so the release has grown one"
        )
    return corpus


def _count_captions(row: SlipRow, record_id: str, stubs: _Tally, truncated: _Tally) -> None:
    """Count the captions of one row that are not captions.

    Counted here rather than in the task stream, because the writer reads a stream more than once
    and the same caption would be counted on each pass.

    Args:
        row: The row, whose scalars hold all four captions.
        record_id: The record the row became, for the example a warning prints.
        stubs: Counts the captions holding a paraphraser stub. Extended in place.
        truncated: Counts the captions that stop mid-sentence. Extended in place.
    """
    for column in _CAPTION_COLUMNS:
        caption = (row.scalars[column] or "").strip()
        if _STUB.match(caption):
            stubs.count(f"{record_id} {column} holds {caption!r}")
        elif not caption.endswith(_ENDS_A_SENTENCE):
            truncated.count(f"{record_id} {column} ends {caption[-40:]!r}")


def _loader_for(ref: _SignalRef) -> Callable[[], pa.Array]:
    """Give a callable that reads one signal's values when it is asked.

    Args:
        ref: Where the values sit.

    Returns:
        A no-argument callable giving the values as an Arrow array.
    """
    return lambda: _row_group(ref.shard, ref.row_group)[ref.offset][ref.signal].values


def _iter_rows(shard: Path) -> Iterator[SlipRow]:
    """Walk one shard, giving each row's scalars, the lengths of its series, and where it sits.

    The walk goes row group by row group, so each row knows which group holds it and where in that
    group it is. A loader built from that reads one row group and looks nothing up. The
    ``time_series`` column is read for its list lengths and for how many of each series' values are
    numbers; no values are kept.

    Args:
        shard: The parquet file to walk.

    Yields:
        One :class:`SlipRow` per row of the shard.
    """
    reader = pq.ParquetFile(shard)
    index = 0
    for group in range(reader.metadata.num_row_groups):
        group_start = index
        for batch in reader.iter_batches(
            batch_size=_BATCH_ROWS, columns=[*_ROW_COLUMNS, "time_series"], row_groups=[group]
        ):
            scalars = batch.select(_ROW_COLUMNS).to_pylist()
            column = batch.column("time_series")
            outer = column.value_lengths().to_pylist()
            inner = column.flatten().value_lengths().to_pylist()
            # A series of all NaN has a length like any other, so emptiness only shows in the values.
            finite = np.isfinite(column.flatten().flatten().to_numpy(zero_copy_only=False))
            cursor, at = 0, 0
            for row, count in zip(scalars, outer, strict=True):
                lengths = inner[cursor : cursor + count]
                empty, partial = [], []
                for signal, length in enumerate(lengths):
                    numbers = int(finite[at : at + length].sum())
                    if numbers == 0:
                        empty.append(signal)
                    elif numbers < length:
                        partial.append((signal, length - numbers))
                    at += length
                cursor += count
                yield SlipRow(
                    index=index,
                    row_group=group,
                    offset=index - group_start,
                    scalars=row,
                    lengths=lengths,
                    empty=tuple(empty),
                    partial=tuple(partial),
                )
                index += 1


def _iter_tasks(shards: tuple[Path, ...]) -> Iterator[AnswerTask]:
    """Yield one unprompted :class:`AnswerTask` per caption, for every row of every shard.

    This yields and counts nothing. The writer reads a stream more than once, so anything counted
    here would be counted on each pass; ``convert`` counts the captions instead.

    Args:
        shards: The parquet files, in the order :meth:`convert` walked them.

    Yields:
        Four tasks per record, each carrying one caption as its target.
    """
    for shard in shards:
        index = 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=_BATCH_ROWS, columns=list(_CAPTION_COLUMNS)):
            for row in batch.to_pylist():
                record_id = _record_id(shard, index)
                for column in _CAPTION_COLUMNS:
                    # A streamed task never reaches Record.task_ids, so nothing resolves its id.
                    # The census found no null caption; `or ""` keeps a task's answer a string if a
                    # later release ships one, rather than a task with neither prompt nor target.
                    yield AnswerTask(target=row[column] or "", record_ids=(record_id,))
                index += 1


CONNECTOR = SlipConnector
