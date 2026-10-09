"""Cancellation, admission, retention and shutdown behavior."""

from threading import Event

import pytest

from timenet.errors import TimeFValidationError
from timenet.viewer.jobs import JobManager, checkpoint


def test_cancel_waits_for_cooperative_read_boundary_and_keeps_active_jobs():
    started, release = Event(), Event()
    manager = JobManager(max_jobs=2)

    def work():
        started.set()
        assert release.wait(5)
        checkpoint()
        pytest.fail("cancelled scan continued")

    try:
        running = manager.submit(work)
        assert started.wait(5)
        queued = manager.submit(dict)
        with pytest.raises(TimeFValidationError, match="full"):
            manager.submit(dict)
        assert manager.get(running.job_id) is running
        manager.cancel(running.job_id)
        assert running.state == "cancelling"
        manager.cancel(queued.job_id)
        assert queued.state == "cancelled"
        release.set()
        assert running.future is not None
        running.future.result(timeout=5)
        assert running.state == "cancelled"
        assert running.result is None
    finally:
        release.set()
        manager.close()


def test_result_bytes_expiry_and_shutdown():
    manager = JobManager(max_bytes=128, ttl=0)
    job = manager.submit(lambda: {"value": "x" * 200})
    assert job.future is not None
    job.future.result(timeout=5)
    assert job.state == "failed"
    assert manager.get(job.job_id) is None
    manager.close()
    with pytest.raises(TimeFValidationError, match="closed"):
        manager.submit(dict)
