from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import weakref

import pyarrow as pa
import pytest

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.sources import channels
from timenet_connectors.sources.channels import SharedChannelLoader


def test_channels_share_a_lazy_read_and_preserve_values_types_and_lengths():
    calls = []
    values = (pa.array([1, None, 3], type=pa.int16()), pa.array([0.25, -0.5], type=pa.float64()))

    def read_channels(key):
        calls.append(key)
        return values

    shared_channels = SharedChannelLoader(read_channels)
    first, second = shared_channels.channel_loader("record", 0), shared_channels.channel_loader("record", 1)
    assert calls == []
    assert second().equals(values[1])
    assert first().equals(values[0])
    assert second().type == pa.float64()
    assert first().type == pa.int16()
    assert calls == ["record"]


def test_eviction_and_clear_release_arrays_without_invalidating_loaders():
    calls, references = [], []

    def read_channels(key):
        calls.append(key)
        array = pa.array([key])
        references.append(weakref.ref(array))
        return (array,)

    shared_channels = SharedChannelLoader(read_channels)
    first, second = shared_channels.channel_loader(1, 0), shared_channels.channel_loader(2, 0)
    assert first().to_pylist() == [1]
    assert second().to_pylist() == [2]
    assert references[0]() is None
    assert first().to_pylist() == [1]
    shared_channels.clear()
    assert references[-1]() is None
    assert first().to_pylist() == [1]
    assert calls == [1, 2, 1, 1]


def test_two_record_cache_keeps_the_most_recently_used_record():
    calls = []

    def read_channels(key):
        calls.append(key)
        return (pa.array([key]),)

    shared_channels = SharedChannelLoader(read_channels, max_recordings=2)
    for key in (1, 2, 1, 3, 1, 2):
        assert shared_channels.channel_loader(key, 0)().to_pylist() == [key]
    assert calls == [1, 2, 3, 2]


def test_concurrent_channels_decode_once(monkeypatch):
    read_started, finish_read, second_request_waiting = Event(), Event(), Event()
    calls = []

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                second_request_waiting.set()
                self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(channels, "Lock", ObservedLock)

    def read_channels(key):
        calls.append(key)
        read_started.set()
        assert finish_read.wait(timeout=5)
        return (pa.array([1]), pa.array([2]))

    shared_channels = SharedChannelLoader(read_channels)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(shared_channels.channel_loader("record", 0))
        try:
            assert read_started.wait(timeout=5)
            second = pool.submit(shared_channels.channel_loader("record", 1))
            # The second request reaches the held lock while the first read is still running.
            assert second_request_waiting.wait(timeout=5)
            assert not second.done()
        finally:
            finish_read.set()
        assert first.result().to_pylist() == [1]
        assert second.result().to_pylist() == [2]
    assert calls == ["record"]


def test_failed_reads_are_retried_and_keep_the_original_error():
    calls = []

    def read_channels(key):
        calls.append(key)
        if len(calls) == 1:
            raise TimeNetConnectorError("record.dat: invalid samples")
        return (pa.array([1]),)

    load = SharedChannelLoader(read_channels).channel_loader("record", 0)
    with pytest.raises(TimeNetConnectorError, match=r"record\.dat: invalid samples"):
        load()
    assert load().to_pylist() == [1]
    assert len(calls) == 2


@pytest.mark.parametrize("result", [None, (), ([1, 2],)])
def test_rejects_invalid_reader_results(result):
    load = SharedChannelLoader(lambda key: result).channel_loader("bad-record", 0)
    with pytest.raises(TimeFValidationError, match=r"bad-record.*Arrow channel arrays"):
        load()


def test_rejects_a_missing_channel():
    load = SharedChannelLoader(lambda key: (pa.array([1]),)).channel_loader("record", 1)
    with pytest.raises(TimeFValidationError, match="record: channel 1 is out of range"):
        load()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_rejects_invalid_cache_sizes(size):
    with pytest.raises(TimeFValidationError, match="positive integer"):
        SharedChannelLoader(lambda key: (), max_recordings=size)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_rejects_invalid_channel_indices(index):
    with pytest.raises(TimeFValidationError, match="nonnegative integer"):
        SharedChannelLoader(lambda key: ()).channel_loader("record", index)


def test_eviction_releases_the_previous_array_before_reading_a_replacement():
    previous = None

    def read_channels(key):
        nonlocal previous
        if previous is not None:
            assert previous() is None
        array = pa.array([key])
        previous = weakref.ref(array)
        return (array,)

    shared_channels = SharedChannelLoader(read_channels)
    assert shared_channels.channel_loader(1, 0)().to_pylist() == [1]
    assert shared_channels.channel_loader(2, 0)().to_pylist() == [2]


def test_failed_replacement_leaves_old_record_evicted_and_can_be_retried():
    calls = []
    failure = TimeNetConnectorError("record 2: invalid samples")

    def read_channels(key):
        calls.append(key)
        if key == 2 and calls.count(2) == 1:
            raise failure
        return (pa.array([key]),)

    shared_channels = SharedChannelLoader(read_channels)
    shared_channels.channel_loader(1, 0)()
    with pytest.raises(TimeNetConnectorError) as error:
        shared_channels.channel_loader(2, 0)()
    assert error.value is failure
    shared_channels.channel_loader(1, 0)()
    assert shared_channels.channel_loader(2, 0)().to_pylist() == [2]
    assert calls == [1, 2, 1, 2]
