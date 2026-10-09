"""Bounded jobs with cooperative cancellation and expiring results."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
from threading import Event, Lock, local
import time
from typing import Literal
import uuid

from timenet.errors import TimeFValidationError


_context = local()
JobState = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]
_TERMINAL = {"succeeded", "failed", "cancelled"}


def checkpoint(
    completed: int | None = None,
    total: int | None = None,
    *,
    phase: str = "reading",
    unit: str = "steps",
) -> None:
    """Report progress and stop a cancelled operation at a bounded read boundary.

    Raises:
        CancelledError: When the current job has been asked to stop.
    """
    job = getattr(_context, "job", None)
    if job is not None:
        if completed is not None:
            job.progress = {
                "completed": str(completed),
                "total": None if total is None else str(total),
                "phase": phase,
                "unit": unit,
            }
        if job.cancelled.is_set():
            raise CancelledError


@dataclass
class Job:
    """One retained operation; cancellation is terminal only after work stops."""

    job_id: str
    operation: str = "inspection"
    state: JobState = "queued"
    result: dict[str, object] | None = None
    error: str | None = None
    progress: dict[str, str | None] = field(default_factory=dict)
    cancelled: Event = field(default_factory=Event, repr=False)
    future: Future | None = field(default=None, repr=False)
    finished: float | None = None
    result_bytes: int = 0


class JobManager:
    """Bound admission and retained results without evicting active operations."""

    def __init__(
        self,
        max_jobs: int = 32,
        *,
        executor: ThreadPoolExecutor | None = None,
        max_bytes: int = 32 * 2**20,
        ttl: float = 300,
        max_active: int = 8,
    ) -> None:
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._ttl = ttl
        self._max_active = max_active
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="timenet-viewer")
        self._closed = False

    def _expire(self) -> None:
        now = time.monotonic()
        for key, job in list(self._jobs.items()):
            if job.finished is not None and now - job.finished >= self._ttl:
                del self._jobs[key]

    def submit(self, work: Callable[[], dict[str, object]], *, operation: str = "inspection") -> Job:
        """Admit an operation if space is available.

        Returns:
            The queued job.

        Raises:
            TimeFValidationError: If the queue is full or the manager has closed.
        """
        job = Job(job_id=uuid.uuid4().hex, operation=operation)
        with self._lock:
            self._expire()
            if self._closed:
                raise TimeFValidationError("job manager is closed")
            if sum(item.state not in _TERMINAL for item in self._jobs.values()) >= self._max_active:
                raise TimeFValidationError("job queue is full")
            if len(self._jobs) >= self._max_jobs:
                completed = next((key for key, item in self._jobs.items() if item.state in _TERMINAL), None)
                if completed is None:
                    raise TimeFValidationError("job queue is full")
                del self._jobs[completed]
            self._jobs[job.job_id] = job
            job.future = self._executor.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[], dict[str, object]]) -> None:
        _context.job = job
        try:
            checkpoint()
            job.state = "running"
            result = work()
            if result.get("coverage") != "partial":
                checkpoint()
            size = len(json.dumps(result, allow_nan=False).encode())
            if size > min(self._max_bytes, 8 * 2**20):
                raise TimeFValidationError("job result exceeds byte limit")
            with self._lock:
                for key, item in list(self._jobs.items()):
                    if sum(entry.result_bytes for entry in self._jobs.values()) + size <= self._max_bytes:
                        break
                    if item.state in _TERMINAL:
                        del self._jobs[key]
                job.result, job.result_bytes, job.state = result, size, "succeeded"
                if job.cancelled.is_set():
                    job.state = "cancelled"
        except CancelledError:
            job.state = "cancelled"
        except Exception:
            logging.getLogger(__name__).exception(
                "Viewer job failed: job_id=%s operation=%s",
                job.job_id,
                job.operation,
                extra={"job_id": job.job_id, "operation": job.operation},
            )
            job.state, job.error = "failed", "operation failed"
        finally:
            job.finished = time.monotonic()
            _context.job = None

    def get(self, job_id: str) -> Job | None:
        """Return a retained job, expiring completed results first."""
        with self._lock:
            self._expire()
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Request cancellation.

        Returns:
            The current job status, or None if it expired.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.state not in _TERMINAL:
                job.cancelled.set()
                if job.future is not None and job.future.cancel():
                    job.state, job.finished = "cancelled", time.monotonic()
                else:
                    job.state = "cancelling"
            return job

    def close(self) -> None:
        """Cancel work and wait for read loops to release their resources."""
        with self._lock:
            self._closed = True
            ids = list(self._jobs)
        for job_id in ids:
            self.cancel(job_id)
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
