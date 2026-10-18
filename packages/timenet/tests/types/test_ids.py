import os
import sys

import numpy as np
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import check_id, ids as ids_module, new_id


def test_new_id_fits_a_float64_exactly():
    value = new_id()
    assert type(value) is int
    assert 2**52 <= value < 2**53
    assert int(float(value)) == value


def test_new_id_is_unique():
    values = [new_id() for _ in range(10_000)]
    assert len(set(values)) == len(values)


def test_new_id_draws_a_new_tag_when_the_counter_wraps(monkeypatch):
    first = new_id()
    monkeypatch.setattr(ids_module._state, "counter", iter([ids_module._COUNTER_LIMIT, 0]))
    monkeypatch.setattr(ids_module._state, "redraw", lambda: setattr(ids_module._state, "tag", 1 << 29))
    after_wrap = new_id()
    assert after_wrap == (1 << 29) << ids_module._COUNTER_BITS
    assert after_wrap != first


def test_the_fork_hook_replaces_a_held_lock():
    ids_module._state.lock.acquire()
    ids_module._state.reset()
    assert not ids_module._state.lock.locked()
    new_id()


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_draws_a_new_tag():
    parent_tag = ids_module._state.tag
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.write(write_fd, str(new_id() >> ids_module._COUNTER_BITS).encode())
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    child_tag = int(os.read(read_fd, 64).decode())
    os.close(read_fd)
    os.waitpid(pid, 0)
    assert child_tag != parent_tag


@pytest.mark.parametrize("value", [1, 42, 2**63 - 1, np.int64(7), np.uint32(7)])
def test_check_id_accepts_a_positive_int64(value):
    result = check_id(value, "record_id")
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [True, False, 0, -1, "17", 1.0, None, 2**63])
def test_check_id_rejects_anything_else(value):
    with pytest.raises(TimeFValidationError, match="record_id must be a positive int64 id"):
        check_id(value, "record_id")


def test_is_provisional_splits_at_two_to_the_52():
    assert not ids_module.is_provisional(1)
    assert not ids_module.is_provisional(ids_module.FINAL_ID_LIMIT - 1)
    assert ids_module.is_provisional(ids_module.FINAL_ID_LIMIT)
    assert ids_module.is_provisional(new_id())
