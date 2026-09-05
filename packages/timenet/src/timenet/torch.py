"""A read-only PyTorch view of a :class:`~timenet.dataset.TimeFDataset`.

This module needs the ``torch`` extra (``pip install 'timenet[torch]'``). The method
:meth:`timenet.client.TimeNet.load_torch` loads this module only when it is used. Users who do not
use PyTorch do not need to install it.
"""

from collections.abc import Callable
from typing import Any

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from torch.utils.data import Dataset

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Task


class TimeFTorchDataset(Dataset):
    """Shows a dataset's records as a map-style ``torch.utils.data.Dataset``.

    ``__getitem__`` returns a dict. The dict has the record's ``series`` as tensors that keep the
    original dtype, with shape ``(n_steps, *value_shape)``. It also has ``series_masks``: one
    boolean tensor per nullable series, ``True`` where the timestep is present, and ``None`` for a
    series that cannot hold nulls. An absent timestep still occupies its slot in ``series`` with a
    zero-equivalent value, so a model must read the mask rather than treat that value as observed.
    The dict also has the record's ``record_id``, its resolved ``tasks``, and its ``annotations``.
    Use the ``transform`` argument to reshape items for a model. Series lengths and trailing shapes
    can vary between records. Because of this, a ``DataLoader`` that batches records needs a custom
    ``collate_fn``, or you must set ``batch_size=1``.
    """

    def __init__(self, dataset: TimeFDataset, *, transform: Callable[[dict[str, Any]], Any] | None = None) -> None:
        """Wrap a dataset.

        Args:
            dataset: The dataset to view. Its per-series values load only when accessed.
            transform: An optional callable applied to each item dict before it is returned.
        """
        self._records = dataset.records
        self._tasks_by_id = {task.id: task for task in dataset.tasks}
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Any:
        record = self._records[index]
        pairs = tuple(_series_tensor_and_mask(ts) for ts in record.time_series)
        item: dict[str, Any] = {
            "record_id": record.record_id,
            "series": tuple(values for values, _ in pairs),
            "series_masks": tuple(mask for _, mask in pairs),
            "tasks": tuple(self._resolve_task(task_id, record.record_id) for task_id in record.task_ids),
            "annotations": record.annotations,
        }
        return self._transform(item) if self._transform is not None else item

    def _resolve_task(self, task_id: str, record_id: str) -> Task:
        """Return the task that a record references. Raise an error if the id does not exist.

        Args:
            task_id: A task id from the record's ``task_ids``.
            record_id: The id of the record that references the task. Used in the error message.

        Returns:
            The resolved :class:`~timenet.types.Task`.

        Raises:
            TimeFValidationError: If no task with ``task_id`` exists in the dataset.
        """
        task = self._tasks_by_id.get(task_id)
        if task is None:
            raise TimeFValidationError(f"record {record_id!r} references unknown task id {task_id!r}")
        return task


def _series_tensor(ts: TimeSeries) -> Shaped[Tensor, " time *value"]:
    """Convert one series to a tensor with its time and per-step dimensions.

    Args:
        ts: The series to load. A free-form string series has no tensor representation and raises,
            as :func:`_series_tensor_and_mask` documents.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)``. An absent timestep is filled with a
        zero-equivalent value, so use :func:`_series_tensor_and_mask` to tell it apart from an
        observation.
    """
    return _series_tensor_and_mask(ts)[0]


def _series_tensor_and_mask(ts: TimeSeries) -> tuple[Shaped[Tensor, " time *value"], Tensor | None]:
    """Convert one series to a tensor plus its validity mask.

    Args:
        ts: The series to load.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)`` and, for a nullable series, a boolean
        tensor shaped ``(n_steps,)`` that is ``True`` where the timestep is present. A series that
        cannot hold nulls returns ``None`` for the mask.

    Raises:
        TimeFValidationError: If the series holds free-form string values, which have no tensor
            representation.
    """
    if ts.spec.dtype == "str":
        raise TimeFValidationError(
            f"string series {ts.time_series_id!r} has no tensor representation; "
            "read it via TimeSeries.to_arrow() instead"
        )
    if ts.spec.dtype == "enum":
        # index_in maps each label to its codebook position in C. This replaces a to_pylist() plus
        # a Python lookup for every value in the series.
        arrow = ts.to_arrow()
        categories = pa.array(ts.spec.categories, type=arrow.type.value_type)
        codes = pc.index_in(arrow, value_set=categories)  # ty: ignore[unresolved-attribute]
        valid = codes.is_valid().to_numpy(zero_copy_only=False)
        values: Tensor = torch.from_numpy(codes.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64))
    else:
        # to_numpy_and_mask reads the series once and returns its mask, so use that mask rather
        # than reading the series a second time. copy() gives a writable C-contiguous array:
        # Arrow's zero-copy view is read-only, and torch.from_numpy warns about that.
        dense, valid = ts.to_numpy_and_mask()
        values = torch.from_numpy(dense.copy())
    mask = torch.from_numpy(valid.copy()) if ts.spec.nullable else None
    return values, mask
