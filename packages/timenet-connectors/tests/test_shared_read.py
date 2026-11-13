from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import time
import weakref

import pyarrow as pa
import pytest

from timenet.dataset import Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import TimeSeriesSpec
from timenet_connectors.sources.shared_read import SharedReadCache


def test_lazy_signals_share_across_sources_and_evict():
    cache = SharedReadCache()
    calls = []
    units = {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]}

    def read(key):
        calls.append(key)
        return units[key]

    def channel(key, index):
        return pa.array([row[index] for row in cache.get(key, partial(read, key))], type=pa.float32())

    spec = TimeSeriesSpec(spec_type="other", name="test", unit_value=None, dtype="float32")

    def signal(name, key, index):
        return Signal.from_loader(
            name=name, spec=spec, time_axis=RegularAxis.from_rate_hz(1), n_values=2, loader=partial(channel, key, index)
        )

    left = Source(name="left", signals=(signal("first", "a", 0),))
    right = Source(name="right", signals=(signal("second", "a", 1), signal("other", "b", 0)))
    assert calls == []
    assert left.signals[0].to_arrow().to_pylist() == [1, 3]
    array = right.signals[0].to_arrow()
    assert array.type == pa.float32() and array.to_pylist() == [2, 4]
    assert calls == ["a"]
    assert right.signals[1].to_arrow().to_pylist() == [5, 7]
    assert left.signals[0].to_arrow().to_pylist() == [1, 3]
    assert calls == ["a", "b", "a"]
    other = SharedReadCache()
    assert other.get("a", partial(read, "a")) == units["a"]
    assert calls == ["a", "b", "a", "a"]


def test_failure_evicts_and_retry_releases_old_value():
    class Value:
        pass

    cache = SharedReadCache()
    value = cache.get("old", Value)
    ref = weakref.ref(value)
    del value
    error = RuntimeError("decode failed")

    def fail():
        assert ref() is None
        raise error

    with pytest.raises(RuntimeError) as exc:
        cache.get("new", fail)
    assert exc.value is error
    replacement = Value()
    assert cache.get("new", lambda: replacement) is replacement
    assert cache.get("new", fail) is replacement


def test_concurrent_reads_are_serialized_and_return_correct_units():
    cache = SharedReadCache()
    active = 0
    guard = threading.Lock()

    def read(key):
        nonlocal active
        with guard:
            active += 1
            assert active == 1
        time.sleep(0.005)
        with guard:
            active -= 1
        return key

    keys = [i % 3 for i in range(60)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda key: cache.get(key, partial(read, key)), keys))
    assert results == keys
