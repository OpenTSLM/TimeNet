"""Stream SLIP training rows into sensor records and caption tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, InputModality
from timenet_connectors.datasets.leochen085.slip_common import REPO, REVISION, SERIES, download_slip, nested_row_group
from timenet_connectors.datasets.leochen085.slip_train import tables
from timenet_connectors.datasets.leochen085.slip_train.keys import SlipKey


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

_ID_PREFIX = "slip"

_SHARDS = "data/*.parquet"
_META = "meta.csv"
_CAPTION_COLUMNS = ("caption0", "caption1", "caption2", "caption3")
_ROW_COLUMNS = ["category", "dataset", *_CAPTION_COLUMNS]
_BATCH_ROWS = 2048  # rows decoded per batch while walking a shard
# Every shard of the release is named this way, and a record id is built from the number in it.
_SHARD_NAME = re.compile(r"^train-(?P<number>\d{5})-of-\d{5}$")


@dataclass(frozen=True)
class SlipSource:
    """What ``download`` hands ``convert``: paths, and no rows."""

    shards: tuple[Path, ...]  # the data/train-*.parquet files, in name order
    meta_csv: Path  # the table describing the corpora the rows were drawn from


@dataclass(frozen=True)
class SlipRow:
    """One row of one shard, without its values."""

    index: int  # the row's absolute index in the shard
    row_group: int  # which row group of the shard holds it
    offset: int  # its index within that row group
    scalars: dict[str, Any]  # category, dataset and the four captions
    lengths: list[int]  # the length of each of the row's series


@dataclass(frozen=True)
class _SignalRef:
    """Where one signal's values sit, so a loader can find them again without holding them.

    The row group and the offset within it are resolved when the reference is built, so reading a
    signal is one row-group read and no lookup. Resolving them per read would be a scan per signal,
    and a build calls the loaders once per signal of the release.
    """

    shard: Path  # the parquet file
    row_group: int  # which row group of that file holds the row
    offset: int  # the row's index within that row group
    signal: int  # which inner list of the row's time_series column


def _read_signal(ref: _SignalRef) -> pa.Array:
    """Read one signal's values back out of its shard.

    Args:
        ref: Where the values sit.

    Returns:
        The values as an Arrow array.
    """
    return nested_row_group(ref.shard, ref.row_group, "time_series")[ref.offset][ref.signal].values


def _shard_number(shard: Path) -> str:
    """Give the number a shard's name states.

    Args:
        shard: The parquet file.

    Returns:
        The five-digit number, as the name writes it.

    Raises:
        TimeFFormatError: If the shard's name states no number. A record id is built from that
            number, so a release that renames its shards would renumber every record.
    """
    match = _SHARD_NAME.match(shard.stem)
    if match is None:
        raise TimeFFormatError(
            f"{shard.name} is not named train-NNNNN-of-NNNNN, so it states no number to build an id from"
        )
    return str(match["number"])


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
    """  # noqa: DOC502 (raised by _shard_number, not directly here)
    return f"{_ID_PREFIX}-shard-{_shard_number(shard)}-row-{row:06d}"


def _caption_id(record_id: str, index: int) -> str:
    """Give the id of one caption annotation of a record.

    The tasks answer by reference to the caption occurrences the record carries, so a reader who
    follows one arrives at an id that names its record and its column rather than a generated one.

    Args:
        record_id: The record the caption belongs to.
        index: Which caption column, 0 to 3.

    Returns:
        An id of the form ``slip-shard-00000-row-000000-caption0``.
    """
    return f"{record_id}-{_CAPTION_COLUMNS[index]}"


class SlipTrainConnector(BaseConnector[SlipSource]):
    """Connector for the SLIP pretraining corpus (Hub repo ``LeoChen085/SlipDataset``, ``data/``)."""

    def download(self, cache_dir: Path) -> list[SlipSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch the corpus shards and ``meta.csv``, and give back their paths.

        The evaluation folders of the same repository are not fetched; they are a different dataset.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            One handle naming every shard and the table beside them.

        Raises:
            TimeNetDownloadError: If the fetch returned no shards.
        """
        root = download_slip(cache_dir, _SHARDS, _META)
        shards = tuple(sorted(root.glob(_SHARDS)))
        if not shards:
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} returned no shards matching {_SHARDS!r} under {root}")
        return [SlipSource(shards=shards, meta_csv=root / _META)]

    def convert(self, raw_refs: list[SlipSource]) -> TimeFDataset:
        """Build one record per row of every shard, and stream one task per caption.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.

        Raises:
            TimeFFormatError: If ``meta.csv`` states a rate this connector cannot read, or names a
                corpus twice, or names none of the corpus a row was drawn from; if a shard's name
                states no number; or if a row states no caption.
        """  # noqa: DOC502 (raised by the helpers below, not directly here)
        source = raw_refs[0]
        # Checked over every shard before any row is read, so a renamed shard stops the build in
        # milliseconds rather than after the shards before it have been converted.
        for shard in source.shards:
            _shard_number(shard)
        corpora = tables.corpora(_read_meta(source.meta_csv))
        dataset = TimeFDataset(metadata=self.metadata())
        for shard in source.shards:
            for row in _iter_rows(shard):
                name = row.scalars["dataset"]
                corpus = _corpus(corpora, name, shard, row.index)
                record_id = _record_id(shard, row.index)
                axis = OrdinalAxis() if corpus.period_us is None else RegularAxis(period_us=corpus.period_us)
                signals = tuple(
                    Signal.from_loader(
                        spec=SERIES,
                        name=f"s{index}",
                        time_axis=axis,
                        loader=_loader_for(
                            _SignalRef(shard=shard, row_group=row.row_group, offset=row.offset, signal=index)
                        ),
                        n_values=length,
                        source_id=record_id,
                        id=f"{record_id}-s{index}",
                    )
                    for index, length in enumerate(row.lengths)
                )
                # The release states which corpus a row was cut from and nothing finer, so that
                # corpus is the one source every signal of the row has.
                record = Record(
                    record_id=record_id,
                    sources=(Source(id=f"{record_id}-source", name=name, signals=signals),),
                )
                dataset.add_record(record=record)
                captions = _captions(row, shard)
                annotations = [
                    Annotation(key=SlipKey.SOURCE_DATASET, value=name),
                    Annotation(key=SlipKey.DOMAIN, value=row.scalars["category"]),
                    Annotation(key=SlipKey.SOURCE_URL, value=corpus.source_url),
                    *(
                        Annotation(key=column, value=captions[index], id=_caption_id(record_id, index))
                        for index, column in enumerate(_CAPTION_COLUMNS)
                    ),
                ]
                record.add_annotations(annotations)
        dataset.set_task_stream([AnswerTask], lambda: _iter_tasks(dataset.records))
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


def _captions(row: SlipRow, shard: Path) -> list[str]:
    """Give one row's four captions, in column order.

    Args:
        row: The row to read them from.
        shard: The parquet file the row came from.

    Returns:
        The four captions, as the release wrote them.

    Raises:
        TimeFFormatError: If a caption column states nothing. Every row of this release states four
            captions, so a null means the release changed.
    """
    captions = []
    for column in _CAPTION_COLUMNS:
        caption = row.scalars[column]
        if caption is None:
            raise TimeFFormatError(
                f"{shard.name} row {row.index}: {column} states nothing, and every row of this "
                "release states four captions"
            )
        captions.append(caption)
    return captions


def _loader_for(ref: _SignalRef) -> Callable[[], pa.Array]:
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


def _iter_rows(shard: Path) -> Iterator[SlipRow]:
    """Walk one shard, giving each row's scalars, the lengths of its series, and where it sits.

    The walk goes row group by row group, so each row knows which group holds it and where in that
    group it is. A loader built from that reads one row group and looks nothing up. The
    ``time_series`` column is read for its list lengths. The values are loaded by the writer.

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
            signals = column.flatten()
            counts = column.value_lengths().to_pylist()
            sizes = signals.value_lengths().to_pylist()
            cursor = 0
            for row, count in zip(scalars, counts, strict=True):
                lengths = sizes[cursor : cursor + count]
                cursor += count
                yield SlipRow(
                    index=index,
                    row_group=group,
                    offset=index - group_start,
                    scalars=row,
                    lengths=lengths,
                )
                index += 1


def _iter_tasks(records: Sequence[Record]) -> Iterator[AnswerTask]:
    """Yield one unprompted :class:`AnswerTask` per caption, for every record.

    The captions are annotations of the record, so a task answers by reference to the occurrence the
    record carries and the stream reads no shard again. It counts nothing: ``convert`` reads every
    caption once and reports there.

    Args:
        records: The records :meth:`SlipTrainConnector.convert` built, in the order it built them.

    Yields:
        Four tasks per record, each answering with one of that record's caption annotations, in
        column order.
    """
    for record in records:
        captions = {
            annotation.key: annotation for annotation in record.annotations if annotation.key in _CAPTION_COLUMNS
        }
        for column in _CAPTION_COLUMNS:
            yield AnswerTask(
                id=f"{record.id}-{column}-task",
                inputs=(record,),
                target_annotations=(captions[column],),
                input_modalities=frozenset({InputModality.TIME_SERIES}),
            )


CONNECTOR = SlipTrainConnector
