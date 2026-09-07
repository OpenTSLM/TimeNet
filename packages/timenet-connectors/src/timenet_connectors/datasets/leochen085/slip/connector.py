"""The SLIP pretraining corpus connector.

The release ships 17 parquet shards under ``data/``. One row is one record: a series or a set of
series, four captions of it, and the name of the corpus it was drawn from. ``meta.csv`` describes
those corpora, and the ``dataset`` column joins to it, which is how a row recovers its sampling rate.

The corpus is 4.3 GB and 618,508 rows, so nothing here holds values. ``download`` gives back one
handle naming the shards, ``convert`` walks them, every series reads its own values when asked, and
the tasks are streamed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import functools
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeNetDownloadError
from timenet.types import Annotation, AnswerTask
from timenet_connectors.datasets.leochen085.slip import tables
from timenet_connectors.datasets.leochen085.slip.keys import SlipKey
from timenet_connectors.datasets.leochen085.slip.specs import SERIES


if TYPE_CHECKING:
    from collections.abc import Iterator


_LOG = logging.getLogger(__name__)

_ID_PREFIX = "slip"

HF_REPO = "LeoChen085/SlipDataset"  # the Hub repo the release lives in
_SHARDS = "data/*.parquet"
_META = "meta.csv"
_CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")
_ROW_COLUMNS = ["category", "dataset", *_CAPTION_COLUMNS]
_BATCH_ROWS = 2048  # rows decoded per batch while walking a shard
# What the release leaves behind where a caption should be: "Paraphrase 1:", "The second paraphrase:".
_STUB = re.compile(r"^\s*(the\s+\w+\s+)?paraphrase\s*\d*\s*:?\s*$", re.IGNORECASE)
# A caption the generator finished ends a sentence. One that stopped at a length limit does not.
_ENDS_A_SENTENCE = (".", "!", "?", '"')


class _Tally:
    """Counts one kind of thing the release ships, so the build warns once about it and not once each.

    ``fidelity.md`` asks for one warning per kind with a count and an example. A property shared by
    thousands of records is one fact about the release.
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
class _Row:
    """One row of one shard, without its values."""

    index: int  # the row's absolute index in the shard
    row_group: int  # which row group of the shard holds it
    offset: int  # its index within that row group
    scalars: dict[str, Any]  # category, dataset and the four captions
    lengths: list[int]  # the length of each of the row's series
    empty: tuple[int, ...]  # which of those series hold no numbers at all


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

    Args:
        shard: The parquet file.
        group: Which row group of it.

    Returns:
        That row group's ``time_series`` column.
    """
    return pq.ParquetFile(shard).read_row_groups([group], columns=["time_series"]).column("time_series")


def _read_signal(ref: _SignalRef) -> pa.Array:
    """Read one signal's values back out of its shard.

    Args:
        ref: Where the values sit.

    Returns:
        The values as an Arrow array.
    """
    return _row_group(ref.shard, ref.row_group)[ref.offset][ref.signal].values


def _record_id(shard: Path, row: int) -> str:
    """Give a record's id: which shard it came from, and where in it.

    The release ships no id of its own, so the id is positional. Two builds of one release agree; a
    new release does not.

    Args:
        shard: The parquet file the row came from.
        row: The row's absolute index in that file.

    Returns:
        An id of the form ``slip-shard-00000-row-000000``.
    """
    return f"{_ID_PREFIX}-shard-{shard.stem.split('-')[1]}-row-{row:06d}"


class SlipConnector(BaseConnector[SlipSource]):
    """Connector for the SLIP pretraining corpus (Hub repo ``LeoChen085/SlipDataset``, ``data/``)."""

    @override
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
                f"reading {HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc

        root = Path(
            snapshot_download(HF_REPO, repo_type="dataset", cache_dir=str(cache_dir), allow_patterns=[_SHARDS, _META])
        )
        shards = tuple(sorted(root.glob(_SHARDS)))
        if not shards:
            raise TimeNetDownloadError(f"{HF_REPO!r} returned no shards matching {_SHARDS!r} under {root}")
        return [SlipSource(shards=shards, meta_csv=root / _META)]

    def convert(self, raw_refs: list[SlipSource]) -> TimeFDataset:
        """Build one record per row of every shard, and stream one task per caption.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.

        Raises:
            TimeNetDownloadError: If the handle names no shards.
        """
        source = raw_refs[0]
        if not source.shards:
            raise TimeNetDownloadError("the SLIP handle names no shards")
        corpora = tables.corpora(_read_meta(source.meta_csv))
        dataset = TimeFDataset(metadata=self.metadata())
        empty = _Tally()
        for shard in source.shards:
            for row in _walk(shard):
                _add_record(dataset, corpora, shard, row, empty)
        empty.report("records hold a series with no numbers at all")
        dataset.set_task_stream([AnswerTask], lambda: _iter_tasks(source.shards))
        return dataset


def _add_record(
    dataset: TimeFDataset,
    corpora: dict[str, tables.SourceCorpus],
    shard: Path,
    row: _Row,
    tally: _Tally,
) -> None:
    """Add one row of one shard as a record, with its series and its annotations.

    Args:
        dataset: The dataset being built.
        corpora: What ``meta.csv`` says about each corpus, keyed by name.
        shard: The parquet file the row came from.
        row: The row's index, its scalar columns and the lengths of its series.
        tally: Counts the records holding a series with no numbers, so the build warns once.
    """
    name = row.scalars["dataset"]
    corpus = corpora.get(name)
    period_us = corpus.period_us if corpus else None
    axis = RegularAxis(period_us=period_us) if period_us is not None else OrdinalAxis()
    record_id = _record_id(shard, row.index)
    series = tuple(
        TimeSeries(
            spec=SERIES,
            signal=f"s{index}",
            time_axis=axis,
            loader=_loader_for(_SignalRef(shard=shard, row_group=row.row_group, offset=row.offset, signal=index)),
            n_values=length,
            time_series_id=f"{record_id}-s{index}",
        )
        for index, length in enumerate(row.lengths)
    )
    record = dataset.add_record(time_series=series, record_id=record_id)
    # Nothing resolves these by id, so they take the default UUIDv7.
    annotations = [
        Annotation(key=SlipKey.SOURCE_DATASET, value=name),
        Annotation(key=SlipKey.DOMAIN, value=row.scalars["category"]),
    ]
    if corpus and corpus.source_url:
        annotations.append(Annotation(key=SlipKey.SOURCE_URL, value=corpus.source_url))
    if row.empty:
        empty = [f"s{signal}" for signal in row.empty]
        annotations.append(Annotation(key=SlipKey.ALL_NAN_SIGNALS, value=empty))
        tally.count(f"{record_id} ({name}): {', '.join(empty)}")
    record.add_annotations(annotations)


def _loader_for(ref: _SignalRef) -> Any:
    """Give a callable that reads one signal's values when it is asked.

    Args:
        ref: Where the values sit.

    Returns:
        A no-argument callable giving the values as an Arrow array.
    """
    return lambda: _read_signal(ref)


def _read_meta(path: Path) -> list[dict[str, str]]:
    """Read ``meta.csv`` into rows.

    Args:
        path: The table's path.

    Returns:
        One dict per row, header names as the file writes them.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _walk(shard: Path) -> Iterator[_Row]:
    """Walk one shard, giving each row's scalars, the lengths of its series, and where it sits.

    The walk goes row group by row group, so each row knows which group holds it and where in that
    group it is. A loader built from that reads one row group and looks nothing up. The
    ``time_series`` column is read for its list lengths and for whether a series holds any number at
    all; no values are kept.

    Args:
        shard: The parquet file to walk.

    Yields:
        One :class:`_Row` per row of the shard.
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
                empty = []
                for signal, length in enumerate(lengths):
                    if not finite[at : at + length].any():
                        empty.append(signal)
                    at += length
                cursor += count
                yield _Row(
                    index=index,
                    row_group=group,
                    offset=index - group_start,
                    scalars=row,
                    lengths=lengths,
                    empty=tuple(empty),
                )
                index += 1


def _iter_tasks(shards: tuple[Path, ...]) -> Iterator[AnswerTask]:
    """Yield one unprompted :class:`AnswerTask` per caption, for every row of every shard.

    Args:
        shards: The parquet files, in the order :meth:`convert` walked them.

    Yields:
        Four tasks per record, each carrying one caption as its target.
    """
    stubs, truncated = _Tally(), _Tally()
    for shard in shards:
        index = 0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=_BATCH_ROWS, columns=list(_CAPTION_COLUMNS)):
            for row in batch.to_pylist():
                record_id = _record_id(shard, index)
                for column in _CAPTION_COLUMNS:
                    caption = (row[column] or "").strip()
                    if _STUB.match(caption):
                        stubs.count(f"{record_id} {column} holds {caption!r}")
                    elif not caption.endswith(_ENDS_A_SENTENCE):
                        truncated.count(f"{record_id} {column} ends {caption[-40:]!r}")
                    # A streamed task never reaches Record.task_ids, so nothing resolves its id.
                    yield AnswerTask(target=row[column], record_ids=(record_id,))
                index += 1
    stubs.report("captions hold a paraphraser stub rather than text")
    truncated.report("captions stop mid-sentence, where the generator hit a length limit")


CONNECTOR = SlipConnector
