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

    ``__getitem__`` returns a dict. Its ``series`` field holds tensors with the original dtype and
    shape ``(n_steps, *value_shape)``. Its ``series_masks`` field holds one boolean tensor per series.
    Each mask is ``True`` where the timestep is present. A series that cannot hold nulls has an
    all-true mask. Missing positions hold zero or false, so a model must use the mask to identify them.

    The dict also contains ``record_id``, resolved ``tasks``, and ``annotations``.
    Use ``transform`` to reshape items for a model. A ``collate_fn`` combines records into a batch.
    Variable shapes and custom task or annotation objects need a suitable transform or ``collate_fn``.
    ``batch_size=1`` still combines records into a batch and needs the same handling.
    Uniform tensors with empty tasks and annotations support default batching.
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
        ts: The series to load. Free-form strings have no tensor representation.
            See :func:`_series_tensor_and_mask` for the errors this method raises.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)``. Missing positions hold zero or false.
        Use :func:`_series_tensor_and_mask` to distinguish them from observations.
    """
    return _series_tensor_and_mask(ts)[0]


def _series_tensor_and_mask(ts: TimeSeries) -> tuple[Shaped[Tensor, " time *value"], Tensor]:
    """Convert one series to a tensor plus its validity mask.

    Args:
        ts: The series to load.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)`` and a boolean tensor shaped
        ``(n_steps,)`` that is ``True`` where the timestep is present. A series that cannot hold
        nulls returns an all-true mask.

    Raises:
        TimeFValidationError: If the series holds free-form string values, unknown enum labels,
            or null enum values with a non-nullable spec.
    """
    if ts.spec.dtype == "str":
        raise TimeFValidationError(
            f"string series {ts.time_series_id!r} has no tensor representation; "
            "read it via TimeSeries.to_arrow() instead"
        )
    if ts.spec.dtype == "enum":
        # index_in finds each label's position in the categories using C code.
        # This avoids a Python lookup for every value.
        arrow = ts.to_arrow()
        if arrow.null_count and not ts.spec.nullable:
            raise TimeFValidationError(f"series for {ts.spec.spec_type!r} has null values but nullable=False")
        categories = pa.array(ts.spec.categories, type=arrow.type.value_type)
        codes = pc.index_in(arrow, value_set=categories)  # ty: ignore[unresolved-attribute]
        if pc.any(pc.and_(arrow.is_valid(), codes.is_null())).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFValidationError(f"enum series for {ts.spec.spec_type!r} has values outside its categories")
        valid = arrow.is_valid().to_numpy(zero_copy_only=False)
        values: Tensor = torch.from_numpy(codes.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64))
    else:
        # Reuse the mask from the same read. copy() gives an array with writable, contiguous memory.
        # Arrow's view is read-only, which causes a warning from torch.from_numpy.
        dense, valid = ts.to_numpy_and_mask()
        values = torch.from_numpy(dense.copy())
    mask = torch.from_numpy(valid.copy())
    return values, mask
