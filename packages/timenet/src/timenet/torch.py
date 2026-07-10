"""A read-only PyTorch view over a :class:`~timenet.dataset.TimeFDataset`.

Requires the ``torch`` extra (``pip install 'timenet[torch]'``). :meth:`timenet.client.TimeNet.load_torch`
imports this module lazily, so base users who never touch PyTorch don't need it installed.
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
    """Exposes a dataset's samples as a map-style ``torch.utils.data.Dataset``.

    ``__getitem__`` returns a dict with the sample's ``series`` as float32 tensors (one per channel),
    its ``sample_id``, its resolved ``tasks``, and its ``annotations``. Pass ``transform`` to reshape
    items into whatever a model expects. Series lengths may vary between samples, so a ``DataLoader``
    that batches them needs a custom ``collate_fn`` (or ``batch_size=1``).
    """

    def __init__(self, dataset: TimeFDataset, *, transform: Callable[[dict[str, Any]], Any] | None = None) -> None:
        """Wrap a dataset.

        Args:
            dataset: The dataset to view (its per-series values load lazily on access).
            transform: Optional callable applied to each item dict before it is returned.
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
        """Return the task a sample references, or raise if the id is dangling.

        Args:
            task_id: A task id from the sample's ``task_ids``.
            sample_id: The referencing sample's id, for the error message.

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
    """Materialize one series as a tensor with its temporal and per-step dimensions.

    Args:
        ts: The series to load.

    Returns:
        The values with shape ``(n_steps, *spec.value_shape)``.
    """
    # copy(): Arrow's zero-copy numpy view is read-only, which torch.from_numpy warns about.
    return torch.from_numpy(ts.to_numpy().copy())
