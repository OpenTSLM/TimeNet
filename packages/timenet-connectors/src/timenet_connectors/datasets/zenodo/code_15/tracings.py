"""Read CODE-15%'s HDF5 parts: scan each trace's zero runs once, and reread one trace on demand.

Each part holds an ``exam_id`` dataset and a ``tracings`` dataset in the same order, which isn't
``exams.csv``'s. Its last row is a filler that ``exams.csv`` doesn't list. h5py, declared in the connector's
requirements, is imported only when a part is opened.
"""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple
import weakref

import numpy as np

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.zenodo.code_15.preprocessing import TRACE_SAMPLES
from timenet_connectors.datasets.zenodo.code_15.release import (
    EXAM_ID_DATASET,
    FILLER_EXAM_ID,
    TRACINGS_DATASET,
    Part,
)
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Lead


# Rows read at a time while scanning: 512 traces are 96 MiB of float32.
_SCAN_ROWS = 512


def _h5py() -> ModuleType:
    """Import h5py, which reads the parts.

    Returns:
        The imported module.

    Raises:
        ImportError: If h5py, declared in the connector's requirements, is missing.
    """
    try:
        import h5py  # noqa: PLC0415 - optional connector dependency
    except ImportError as exc:
        raise ImportError(
            "reading CODE-15% needs h5py, declared in the connector's requirements.txt. Run the build "
            "without --no-isolation, or install it yourself"
        ) from exc
    return h5py


class ExamLocation(NamedTuple):
    """Where an exam's trace is: its part's file and its row there."""

    path: Path
    row: int


class TraceScan(NamedTuple):
    """One row of a part, scanned: its exam, where it is, and its runs of all-lead zeros.

    An all-zero trace has :data:`~timenet_connectors.datasets.zenodo.code_15.preprocessing.TRACE_SAMPLES` zeros
    at each end.
    """

    exam_id: int
    location: ExamLocation
    leading_zeros: int
    trailing_zeros: int

    @property
    def all_zero(self) -> bool:
        """Return whether the trace is zero in every lead and sample."""
        return self.leading_zeros == TRACE_SAMPLES


def zero_runs(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count each trace's samples, from the first and from the last, that are zero in every lead.

    Args:
        traces: ``(n, 4096, 12)`` traces.

    Returns:
        The leading and the trailing counts, one per trace; 4096 for an all-zero trace.
    """
    nonzero = np.any(traces != 0, axis=2)
    any_nonzero = nonzero.any(axis=1)
    leading = np.where(any_nonzero, nonzero.argmax(axis=1), TRACE_SAMPLES)
    trailing = np.where(any_nonzero, nonzero[:, ::-1].argmax(axis=1), TRACE_SAMPLES)
    return leading, trailing


def _datasets(handle: Any, path: Path) -> tuple[Any, Any]:
    """Return a part's ``exam_id`` and ``tracings`` datasets, checked against the release's shapes and types.

    Returns:
        The two datasets.

    Raises:
        TimeFFormatError: If either is missing, or their shapes or types aren't the release's.
    """
    if set(handle) != {EXAM_ID_DATASET, TRACINGS_DATASET}:
        raise TimeFFormatError(
            f"{path}: expected the datasets {EXAM_ID_DATASET} and {TRACINGS_DATASET}, got {set(handle)}"
        )
    exam_ids, tracings = handle[EXAM_ID_DATASET], handle[TRACINGS_DATASET]
    if exam_ids.dtype != np.int64 or tracings.dtype != np.float32:
        raise TimeFFormatError(
            f"{path}: expected int64 ids and float32 tracings, got {exam_ids.dtype}, {tracings.dtype}"
        )
    if exam_ids.ndim != 1 or tracings.shape != (exam_ids.shape[0], TRACE_SAMPLES, len(Lead)):
        raise TimeFFormatError(
            f"{path}: expected {EXAM_ID_DATASET} (N,) and {TRACINGS_DATASET} (N, {TRACE_SAMPLES}, {len(Lead)}), "
            f"got {exam_ids.shape} and {tracings.shape}"
        )
    return exam_ids, tracings


def scan_part(path: Path) -> Iterator[TraceScan]:
    """Read a part once, block by block, and scan each exam's trace.

    Args:
        path: The part's HDF5 file.

    Yields:
        One scan per row but the filler, in the file's order.

    Raises:
        TimeFFormatError: If the file's datasets aren't the release's, its last row isn't an all-zero filler,
            another row has the filler's id, or a trace holds a value that isn't finite.
    """
    with _h5py().File(path, "r") as handle:
        exam_ids_dataset, tracings = _datasets(handle, path)
        exam_ids = exam_ids_dataset[:]
        n_exams = exam_ids.shape[0] - 1
        if n_exams < 0 or exam_ids[-1] != FILLER_EXAM_ID or np.any(tracings[n_exams] != 0):
            raise TimeFFormatError(f"{path}: the last row isn't the all-zero filler with exam id {FILLER_EXAM_ID}")
        if np.any(exam_ids[:n_exams] == FILLER_EXAM_ID):
            raise TimeFFormatError(f"{path}: a row other than the last has the filler's exam id {FILLER_EXAM_ID}")
        for start in range(0, n_exams, _SCAN_ROWS):
            block = tracings[start : min(start + _SCAN_ROWS, n_exams)]
            finite = np.isfinite(block).all(axis=(1, 2))
            if not finite.all():
                row = start + int(np.argmin(finite))
                raise TimeFFormatError(f"{path}: exam {exam_ids[row]}'s trace holds a value that isn't finite")
            leading, trailing = zero_runs(block)
            for offset in range(block.shape[0]):
                row = start + offset
                yield TraceScan(
                    exam_id=int(exam_ids[row]),
                    location=ExamLocation(path=path, row=row),
                    leading_zeros=int(leading[offset]),
                    trailing_zeros=int(trailing[offset]),
                )


def part_path(root: Path, part: Part) -> Path:
    """Return a part's HDF5 file under the extracted release.

    Returns:
        The path.

    Raises:
        TimeFFormatError: If the file is missing.
    """
    path = root / part.hdf5_name
    if not path.is_file():
        raise TimeFFormatError(f"{root}: {part.hdf5_name} is missing")
    return path


def _close(files: dict[Path, Any]) -> None:
    """Close every open part and forget it."""
    for handle in files.values():
        handle.close()
    files.clear()


class TraceReader:
    """Reread one exam's published trace, keeping only the part last read open.

    A reader serves the Records of one load, which come by part, then by row, so it opens each part once.
    Reading another part closes the open one, and the reader closes the last when it is closed or collected:
    a load holds at most one HDF5 file, and its chunk cache, at a time.
    """

    def __init__(self) -> None:
        # At most one part, by path. The finalizer holds the dict, not the reader, so the reader can be collected.
        self._files: dict[Path, Any] = {}
        weakref.finalize(self, _close, self._files)

    def read(self, location: ExamLocation) -> np.ndarray:
        """Return one published trace, closing the part read before if it is another.

        Returns:
            The ``(4096, 12)`` float32 trace, as the file holds it.
        """
        handle = self._files.get(location.path)
        if handle is None:
            self.close()
            handle = self._files[location.path] = _h5py().File(location.path, "r")
        return handle[TRACINGS_DATASET][location.row]

    def close(self) -> None:
        """Close the open part, if any. A later read reopens the part it needs."""
        _close(self._files)
