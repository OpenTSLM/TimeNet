"""A read-only PyTorch view of a :class:`~timenet.dataset.TimeFDataset`.

This module needs the ``torch`` extra (``pip install 'timenet[torch]'``). The method
:meth:`timenet.client.TimeNet.load_torch` loads this module only when it is used. Users who do not
use PyTorch do not need to install it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from torch.utils.data import Dataset

from timenet.dataset import Record, Signal, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import EntityId, Task


if TYPE_CHECKING:
    from timenet.reader import TimeFReader


# NumPy counterpart of torch.long for enum category indices.
_ENUM_INDEX_DTYPE = np.int64


class _ItemSource(Protocol):
    """Where a :class:`TimeFTorchDataset` gets records and their tasks."""

    def __len__(self) -> int: ...

    def fetch(self, indices: Sequence[int]) -> list[tuple[Record, tuple[Task, ...]]]:
        """Return the records at ``indices`` with their tasks, in order."""
        ...


class TimeFTorchDataset(Dataset):
    """Shows a dataset's records as a map-style ``torch.utils.data.Dataset``.

    ``__getitem__`` returns a dict. Its ``series`` field holds tensors with the original dtype and
    shape ``(n_steps, *value_shape)``. Its ``series_masks`` field holds one boolean tensor per series.
    Each mask is all true. Missing values raise ``TimeFValidationError``, including when the Signal
    spec permits nulls. Consumers must handle missing data before conversion to PyTorch.

    The dict also contains ``record_id``, resolved ``tasks``, and ``annotations``.
    Use ``transform`` to reshape items for a model. A ``collate_fn`` combines records into a batch.
    Variable shapes and custom task or annotation objects need a suitable transform or ``collate_fn``.
    ``batch_size=1`` still combines records into a batch and needs the same handling.
    Uniform tensors with empty tasks and annotations support default batching.

    Build it from a materialized :class:`~timenet.dataset.TimeFDataset`, or with
    :meth:`from_reader` from a :class:`~timenet.reader.TimeFReader` so records and tasks are built
    per batch instead of all at once. A ``DataLoader`` calls :meth:`__getitems__` with a whole batch
    of indices, and the reader-backed view answers it with one round of queries for the batch.
    """

    def __init__(
        self,
        dataset: TimeFDataset | None = None,
        *,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        items: _ItemSource | None = None,
    ) -> None:
        """Wrap a dataset.

        Args:
            dataset: The dataset to view. Its per-series values load only when accessed.
            transform: An optional callable applied to each item dict before it is returned.
            items: A prepared item source, used by :meth:`from_reader` instead of ``dataset``.

        Raises:
            TimeFValidationError: If neither or both of ``dataset`` and ``items`` are given.
        """
        if (dataset is None) == (items is None):
            raise TimeFValidationError("TimeFTorchDataset needs a dataset or an item source, not both")
        self._items: _ItemSource = items if items is not None else _DatasetItems(cast("TimeFDataset", dataset))
        self._transform = transform

    @classmethod
    def from_reader(
        cls,
        reader: TimeFReader,
        *,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> TimeFTorchDataset:
        """View an open reader without materializing its records or tasks.

        Each ``__getitems__`` call builds the batch's records in one query round and their tasks
        in another, so memory stays proportional to the batch. The reader pickles without its
        connection, so ``DataLoader`` workers each reopen the control database.

        Args:
            reader: The reader to draw records and tasks from.
            transform: An optional callable applied to each item dict before it is returned.

        Returns:
            The lazy view.
        """
        return cls(items=_ReaderItems(reader), transform=transform)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:  # noqa: PLW3201 - DataLoader batch hook
        """Return the items at ``indices``, building the whole batch together.

        Args:
            indices: Record positions, in the order the items are returned.

        Returns:
            One item dict, or transformed item, per index.
        """
        items = [self._item(record, tasks) for record, tasks in self._items.fetch(indices)]
        return items if self._transform is None else [self._transform(item) for item in items]

    @staticmethod
    def _item(record: Record, tasks: tuple[Task, ...]) -> dict[str, Any]:
        pairs = tuple(_series_tensor_and_mask(signal) for signal in record.signals)
        return {
            "record_id": record.id,
            "series": tuple(values for values, _ in pairs),
            "series_masks": tuple(mask for _, mask in pairs),
            "tasks": tasks,
            "annotations": record.annotations,
        }


class _DatasetItems:
    """Items from a materialized dataset, each record with the tasks that take it as input."""

    def __init__(self, dataset: TimeFDataset) -> None:
        self._records = dataset.records
        self._tasks_by_record: dict[int, list[Task]] = {}
        for task in dataset.tasks:
            for record in task.inputs:
                self._tasks_by_record.setdefault(id(record), []).append(task)

    def __len__(self) -> int:
        return len(self._records)

    def fetch(self, indices: Sequence[int]) -> list[tuple[Record, tuple[Task, ...]]]:
        """Return the records at ``indices`` with their tasks, in order."""
        records = [self._records[index] for index in indices]
        return [(record, tuple(self._tasks_by_record.get(id(record), ()))) for record in records]


class _ReaderItems:
    """Items built from a reader per batch: one query round for the records, one for their tasks."""

    def __init__(self, reader: TimeFReader) -> None:
        self._reader = reader
        self._record_ids: tuple[EntityId, ...] | None = None

    def _ids(self) -> tuple[EntityId, ...]:
        if self._record_ids is None:
            self._record_ids = self._reader.record_ids()
        return self._record_ids

    def __len__(self) -> int:
        return len(self._ids())

    def fetch(self, indices: Sequence[int]) -> list[tuple[Record, tuple[Task, ...]]]:
        """Return the records at ``indices`` with their tasks, in order."""
        ids = self._ids()
        records = tuple(self._reader.iter_records([ids[index] for index in indices]))
        tasks_by_record: dict[EntityId, list[Task]] = {record.id: [] for record in records}
        for task in self._reader.read_tasks(records):
            for record in task.inputs:
                attached = tasks_by_record.get(record.id)
                if attached is not None:
                    attached.append(task)
        return [(record, tuple(tasks_by_record[record.id])) for record in records]


def _series_tensor_and_mask(ts: Signal) -> tuple[Shaped[Tensor, " time *value"], Tensor]:
    """Convert one series to a tensor plus its validity mask.

    Args:
        ts: The series to load.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)`` and an all-true boolean tensor shaped
        ``(n_steps,)``.

    Raises:
        TimeFValidationError: If the series holds free-form string values, unknown enum labels,
            null values, or partially missing tensor timesteps.
    """
    if ts.spec.dtype == "str":
        raise TimeFValidationError(
            f"string series {ts.id!r} has no tensor representation; read it via Signal.to_arrow() instead"
        )
    if ts.spec.dtype == "enum":
        # index_in finds each label's position in the categories using C code.
        # This avoids a Python lookup for every value.
        arrow = ts.to_arrow()
        if arrow.null_count:
            raise TimeFValidationError(
                f"series {ts.id!r} contains null values. Handle missing data before conversion to PyTorch"
            )
        categories = pa.array(ts.spec.categories, type=arrow.type.value_type)
        codes = pc.index_in(arrow, value_set=categories)  # ty: ignore[unresolved-attribute]
        if pc.any(codes.is_null()).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFValidationError(f"enum series for {ts.spec.spec_type!r} has values outside its categories")
        dense = codes.to_numpy(zero_copy_only=False).astype(_ENUM_INDEX_DTYPE, copy=False)
    else:
        dense = ts.to_numpy()
    # Give PyTorch writable, contiguous buffers without changing the signal's data.
    return torch.from_numpy(dense.copy()), torch.ones(len(dense), dtype=torch.bool, device="cpu")
