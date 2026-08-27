"""A read-only PyTorch view of a :class:`~timenet.dataset.TimeFDataset`.

This module needs the ``torch`` extra (``pip install 'timenet[torch]'``). The method
:meth:`timenet.client.TimeNet.load_torch` loads this module only when it is used. Users who do not
use PyTorch do not need to install it.
"""

from collections.abc import Callable
from typing import Any

from jaxtyping import Shaped
import torch
from torch import Tensor
from torch.utils.data import Dataset

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.errors import TimeFValidationError
from timenet.types import Task


class TimeFTorchDataset(Dataset):
    """Shows a dataset's samples as a map-style ``torch.utils.data.Dataset``.

    ``__getitem__`` returns a dict. The dict has the sample's ``series`` as tensors that keep the
    original dtype, with shape ``(n_steps, *value_shape)``. The dict also has the sample's
    ``sample_id``, its resolved ``tasks``, and its ``annotations``. Use the ``transform`` argument
    to reshape items for a model. Series lengths and trailing shapes can vary between samples.
    Because of this, a ``DataLoader`` that batches samples needs a custom ``collate_fn``, or you
    must set ``batch_size=1``.
    """

    def __init__(self, dataset: TimeFDataset, *, transform: Callable[[dict[str, Any]], Any] | None = None) -> None:
        """Wrap a dataset.

        Args:
            dataset: The dataset to view. Its per-series values load only when accessed.
            transform: An optional callable applied to each item dict before it is returned.
        """
        self._samples = dataset.samples
        self._tasks_by_id = {task.id: task for task in dataset.tasks}
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        sample = self._samples[index]
        item: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "series": tuple(_series_tensor(ts) for ts in sample.time_series),
            "tasks": tuple(self._resolve_task(task_id, sample.sample_id) for task_id in sample.task_ids),
            "annotations": sample.annotations,
        }
        return self._transform(item) if self._transform is not None else item

    def _resolve_task(self, task_id: str, sample_id: str) -> Task:
        """Return the task that a sample references. Raise an error if the id does not exist.

        Args:
            task_id: A task id from the sample's ``task_ids``.
            sample_id: The id of the sample that references the task. Used in the error message.

        Returns:
            The resolved :class:`~timenet.types.Task`.

        Raises:
            TimeFValidationError: If no task with ``task_id`` exists in the dataset.
        """
        task = self._tasks_by_id.get(task_id)
        if task is None:
            raise TimeFValidationError(f"sample {sample_id!r} references unknown task id {task_id!r}")
        return task


def _series_tensor(ts: TimeSeries) -> Shaped[Tensor, " time *value"]:
    """Convert one series to a tensor with its time and per-step dimensions.

    Args:
        ts: The series to load.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)``.

    Raises:
        TimeFValidationError: If the series holds free-form string values, which have no tensor
            representation.
    """
    # copy(): Arrow's zero-copy numpy view is read-only. torch.from_numpy warns when an array is
    # read-only.
    if ts.spec.dtype == "str":
        raise TimeFValidationError(
            f"string series {ts.time_series_id!r} has no tensor representation; "
            "read it via TimeSeries.to_arrow() instead"
        )
    if ts.spec.dtype == "enum":
        return torch.from_numpy(ts.to_arrow().indices.to_numpy().copy()).to(torch.int64)
    return torch.from_numpy(ts.to_numpy().copy())
