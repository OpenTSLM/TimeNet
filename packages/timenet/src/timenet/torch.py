"""A read-only PyTorch view of the records a control-plane reader holds.

An item is one record. It carries the record's ``series`` as tensors that keep the stored dtype,
its ``series_masks``, its ``tasks``, and its ``annotations``. :class:`TimeFTorchDataset` delivers
items by position and :class:`TimeFIterableStream` streams them, and both build the item with
:func:`record_item`, so the two cannot drift apart.

Both views read a whole batch of records' values in one call to
:meth:`~timenet.control_plane.reader.TimeFReader.values_for`, then build the batch's items from
that one mapping. Reading per signal instead re-opens the shard and re-decodes the row group for
every signal, which is why :func:`record_item` takes values it did not read.

``series_masks`` holds one entry per series and every entry is ``None``. The mask states which
timesteps are present, and this values plane records no such thing: a signal arrives as a plain
``numpy`` array, which cannot express an absent value, and a shard column is a list of the
modality's dtype with no validity to read back. The ``specs`` table does keep a ``nullable`` flag,
which :class:`~timenet.control_plane.reader.SignalView` carries, but that is a claim about the
modality rather than a record of what is stored. An all-``True`` mask would therefore assert
per-timestep validity that nothing on disk backs, so the slot stays empty until the values plane
stores a mask to fill it with.

This module needs ``torch``, which is an optional extra. It imports it where it is used rather than
at module import, so a caller without torch can still import this module, and only the calls that
build tensors raise.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np

from timenet.control_plane.reader import RecordView, SignalView, TimeFReader
from timenet.errors import TimeFValidationError


try:
    from torch.utils.data import Dataset, IterableDataset

    _DATASET_BASE: Any = Dataset
    _ITERABLE_BASE: Any = IterableDataset
except ModuleNotFoundError:  # pragma: no cover - only reachable without the extra
    _DATASET_BASE = object
    _ITERABLE_BASE = object


ITEM_FIELDS = ("record_id", "series", "tasks", "annotations", "series_masks")
"""The fields of one item, in the order :func:`record_item` writes them."""

DEFAULT_BATCH_SIZE = 512
"""Records hydrated per round of queries, and values read per pass over the values plane."""


class TimeFTorchDataset(_DATASET_BASE):
    """Shows a version's records as a map-style ``torch.utils.data.Dataset``.

    Position ``i`` is the record the control plane assigned surrogate id ``i``. Those ids run
    0, 1, 2, ... with no gap and no repeat, which the writer checks before it commits a version, so
    a position resolves to a record without an index of its own.

    ``__getitem__`` reads one record, which costs a round of queries and a pass over the values
    plane for a single item. A ``DataLoader`` with a batch sampler calls ``__getitems__`` instead,
    which hydrates the whole batch in one round and reads its values in one pass. Use batched
    access, or :class:`TimeFIterableStream` for a full walk.

    Series lengths and dtypes differ between records, so a ``DataLoader`` that batches records needs
    a ``collate_fn`` of its own, or ``batch_size=1``.
    """

    def __init__(self, reader: TimeFReader, *, transform: Callable[[dict[str, Any]], Any] | None = None) -> None:
        """Wrap an open reader.

        Args:
            reader: The reader to read through. It stays open for as long as the dataset is used.
            transform: An optional callable applied to each item before it is returned.
        """
        self._reader = reader
        self._transform = transform
        row = reader.connection.execute("SELECT count(*) FROM records").fetchone()
        self._length = 0 if row is None else row[0]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:  # noqa: PLW3201 - torch's own name for it
        """Read several positions with one round of queries and one pass over the values plane.

        ``torch.utils.data`` calls this rather than ``__getitem__`` when a sampler hands over a
        whole batch, which is what makes a batched read cost one request instead of one per item.

        Args:
            indices: The positions to read.

        Returns:
            One item per position, in the order asked for.

        Raises:
            IndexError: If a position is outside the records this version holds.
        """
        wanted = [int(index) for index in indices]
        outside = sorted(index for index in wanted if not 0 <= index < self._length)
        if outside:
            raise IndexError(f"position(s) {outside} are outside the {self._length} records of this version")
        # A position is a surrogate id, not the id a builder gave, so this takes the by-id entry
        # point rather than `records`, which would have to look the builder's ids up first.
        return _items(self._reader, self._reader.records_by_id(wanted), self._transform)


class TimeFIterableStream(_ITERABLE_BASE):
    """Streams a version's records as items, holding one batch rather than the corpus.

    :class:`TimeFTorchDataset` addresses records by position, so a full walk through it pays a
    sampler's order rather than the stored one. This one asks the reader for records in stored
    order, a batch at a time, so a corpus larger than memory can be trained on and the first item
    arrives without reading the rest.

    Inside a ``DataLoader`` worker the stream reads the slice the reader partitions for that worker,
    so ``num_workers`` workers together see every record exactly once. The reader drops its
    connection when it is pickled to a worker and opens a new one there.
    """

    def __init__(
        self,
        reader: TimeFReader,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Wrap an open reader.

        Args:
            reader: The reader to stream from. It stays open for as long as the stream is iterated.
            batch_size: How many records to hydrate per round of queries. The same number of
                records' values are read in one pass over the values plane.
            transform: An optional callable applied to each item before it is returned.
        """
        self._reader = reader
        self._batch_size = batch_size
        self._transform = transform

    def __iter__(self) -> Iterator[Any]:
        """Walk this worker's slice of the corpus, a batch at a time.

        Yields:
            One item per record, in stored order.
        """
        worker_index, num_workers = _worker_split()
        batch: list[RecordView] = []
        for record in self._reader.iter_records(
            batch_size=self._batch_size, worker_index=worker_index, num_workers=num_workers
        ):
            batch.append(record)
            if len(batch) == self._batch_size:
                yield from _items(self._reader, batch, self._transform)
                batch = []
        if batch:
            yield from _items(self._reader, batch, self._transform)


def record_item(record: RecordView, values: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """Convert one record and its already-read values into the item both views deliver.

    Args:
        record: The record, as the reader rebuilt it.
        values: Values for at least this record's signals, keyed by signal id, as
            :meth:`~timenet.control_plane.reader.TimeFReader.values_for` returns them. They are
            passed in rather than read here so one pass over the values plane can serve a batch of
            records.

    Returns:
        The item, with the fields of :data:`ITEM_FIELDS`. ``series`` holds one tensor per signal, in
        the order the record's sources are walked, each keeping the dtype it was stored under.
        ``series_masks`` holds one ``None`` per signal, for the reason the module docstring gives.
        ``tasks`` is empty: a task points at the records it uses, so a walk that starts from a
        record reaches its tasks only by querying for them, which is a query per item that a
        streaming read does not pay.
    """
    tensors = tuple(_series_tensor(signal, values) for signal in record.signals())
    return {
        "record_id": record.external_id if record.external_id is not None else str(record.record_id),
        "series": tensors,
        "tasks": (),
        "annotations": record.annotations,
        "series_masks": (None,) * len(tensors),
    }


def _items(
    reader: TimeFReader, records: Sequence[RecordView], transform: Callable[[dict[str, Any]], Any] | None
) -> list[Any]:
    """Build one item per record, reading every one of their signals in one pass.

    Args:
        reader: The reader holding the values plane.
        records: The records to convert.
        transform: An optional callable applied to each item.

    Returns:
        One item per record, in the order given.
    """
    values = reader.values_for([signal.signal_id for record in records for signal in record.signals()])
    items = [record_item(record, values) for record in records]
    return items if transform is None else [transform(item) for item in items]


def _series_tensor(signal: SignalView, values: Mapping[int, np.ndarray]) -> Any:
    """Convert one signal's values into a tensor.

    Args:
        signal: The signal to convert.
        values: Values for at least this signal, keyed by signal id.

    Returns:
        The values as a tensor of shape ``(n_values,)``, keeping the stored dtype.

    Raises:
        TimeFValidationError: If this signal's values were not read, or it holds text, which has no
            tensor form.
    """
    array = values.get(signal.signal_id)
    if array is None:
        raise TimeFValidationError(
            f"signal {signal.signal_id} ({signal.name!r}) has no values here; ask values_for() for "
            "every signal of the record before building its item"
        )
    if array.dtype.kind in "OUS":
        raise TimeFValidationError(
            f"signal {signal.signal_id} ({signal.name!r}) holds {signal.dtype} values, which have no "
            "tensor form; read them as an array with TimeFReader.values_for() instead"
        )
    # A Parquet row group hands back a read-only view of its own buffer. Copying it makes the tensor
    # writable, which torch.from_numpy otherwise warns about, and releases the row group.
    return _torch().from_numpy(array if array.flags.writeable else array.copy())


def _worker_split() -> tuple[int, int]:
    """Return which slice of the corpus this process reads.

    Returns:
        The worker index and the worker count. Outside a ``DataLoader`` worker that is ``(0, 1)``,
        which is the whole corpus.
    """
    info = _torch().utils.data.get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


@cache
def _torch() -> Any:
    """Import torch on first use, so this module stays importable without the extra.

    Returns:
        The ``torch`` module.

    Raises:
        ModuleNotFoundError: If torch is not installed.
    """
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the timenet torch view needs torch: pip install 'timenet[torch]', or install a build "
            "from https://download.pytorch.org/whl/cpu for a small CPU-only one"
        ) from exc
    return torch


__all__: Sequence[str] = (
    "DEFAULT_BATCH_SIZE",
    "ITEM_FIELDS",
    "TimeFIterableStream",
    "TimeFTorchDataset",
    "record_item",
)
