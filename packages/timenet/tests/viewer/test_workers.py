"""Reader thread ownership and lifetime isolation."""

import asyncio
from threading import Event, get_ident

import pytest

from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.reader.inspection import ReaderInspection
from timenet.viewer.workers import ReaderWorker


def test_worker_owns_creation_calls_and_close(reader, monkeypatch):
    calls = []
    original_init, original_close = TimeFReader.__init__, TimeFReader.close

    def init(self, *args, **kwargs):
        calls.append(("create", get_ident(), self))
        original_init(self, *args, **kwargs)

    def close(self):
        calls.append(("close", get_ident(), self))
        original_close(self)

    monkeypatch.setattr(TimeFReader, "__init__", init)
    monkeypatch.setattr(TimeFReader, "close", close)
    worker = ReaderWorker(reader, "test-owner")
    try:
        with pytest.raises(TimeFValidationError, match="owning worker"):
            worker.run(ReaderInspection.overview)

        def query(inspection):
            calls.append(("query", get_ident(), inspection._reader))
            return inspection.records_page(limit=1)

        async def requests():
            return await asyncio.gather(worker.call(query), worker.call(query))

        assert all(page.items for page in asyncio.run(requests()))
    finally:
        worker.close()
    assert [name for name, _, _ in calls] == ["create", "query", "query", "close"]
    assert len({thread for _, thread, _ in calls}) == 1
    assert calls[0][1] != get_ident()
    clone = calls[0][2]
    assert clone is not reader
    assert all(owner is clone for _, _, owner in calls)
    assert reader.record_ids() == ("record-0", "record-1", "record-2")


def test_worker_call_keeps_the_event_loop_responsive(reader):
    worker = ReaderWorker(reader, "test-responsive")
    release = Event()

    async def request():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def query(inspection):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "the event loop could not release the reader operation"
            return inspection.records_page(limit=1)

        pending = asyncio.create_task(worker.call(query))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            release.set()
            assert (await asyncio.wait_for(pending, timeout=5)).items[0].record_id == "record-0"
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)

    try:
        asyncio.run(request())
    finally:
        release.set()
        worker.close()


def test_worker_close_waits_for_queued_requests(reader, monkeypatch):
    worker = ReaderWorker(reader, "test-drain")
    release, queued, closing = Event(), Event(), Event()
    calls = []
    original_submit = worker._executor.submit
    original_close = TimeFReader.close
    submissions = 0

    def submit(*args, **kwargs):
        nonlocal submissions
        future = original_submit(*args, **kwargs)
        submissions += 1
        if submissions == 2:
            queued.set()
        elif submissions == 3:
            closing.set()
        return future

    def close(self):
        calls.append("close")
        original_close(self)

    monkeypatch.setattr(worker._executor, "submit", submit)
    monkeypatch.setattr(TimeFReader, "close", close)

    def first(inspection):
        assert release.wait(5), "the test did not release the queued requests"
        calls.append("first")
        return inspection.records_page(limit=1)

    def second(inspection):
        calls.append("second")
        return inspection.records_page(limit=1)

    def close_worker():
        worker.close()
        assert calls == ["first", "second", "close"]

    async def requests():
        pending = [asyncio.create_task(worker.call(first)), asyncio.create_task(worker.call(second))]
        close_task = None
        try:
            assert await asyncio.to_thread(queued.wait, 5)
            close_task = asyncio.create_task(asyncio.to_thread(close_worker))
            assert await asyncio.to_thread(closing.wait, 5)
            assert not close_task.done()
            assert calls == []
            release.set()
            pages = await asyncio.gather(*pending)
            assert [page.items[0].record_id for page in pages] == ["record-0", "record-0"]
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
            if close_task is not None:
                await close_task
            else:
                await asyncio.to_thread(worker.close)

    asyncio.run(requests())
    assert calls == ["first", "second", "close"]
