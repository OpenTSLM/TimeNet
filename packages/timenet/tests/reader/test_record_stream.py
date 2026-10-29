"""Tests for the batched Record stream, ``TimeFReader.iter_records``."""

import shutil

from _task_release import _TEXT, _handles, _metadata, _write
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFNotFoundError, TimeFValidationError
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.stored_array import StoredArray
from timenet.reader import RECORD_BATCH_SIZE, Loading, LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.types import read_value


@pytest.fixture
def batches(monkeypatch) -> list[tuple[int, ...]]:
    """Record the IDs of every batch of Records that the stream reads from the catalog."""
    batches: list[tuple[int, ...]] = []
    original = DuckDBControlReader.read_records

    def recording(self, record_ids=None, **kwargs):
        if record_ids is not None:
            record_ids = tuple(record_ids)
            batches.append(record_ids)
        return original(self, record_ids, **kwargs)

    monkeypatch.setattr(DuckDBControlReader, "read_records", recording)
    return batches


@pytest.mark.parametrize(
    ("batch_size", "sizes"),
    [(1, [1, 1, 1, 1]), (2, [2, 2]), (3, [3, 1]), (4, [4]), (None, [4])],
)
def test_the_stream_reads_records_in_batches_of_the_given_size(release, batches, batch_size, sizes):
    with TimeFReader(open_local_version(release.path)) as reader:
        order = reader.record_ids()
        stream = reader.iter_records() if batch_size is None else reader.iter_records(batch_size=batch_size)
        streamed = [record.id for record in stream]
    assert RECORD_BATCH_SIZE == 1_000
    assert streamed == list(order)
    assert [len(batch) for batch in batches] == sizes
    assert [record_id for batch in batches for record_id in batch] == list(order)


def test_requested_ids_keep_their_order_across_batches(release, batches):
    r1, r2, r4 = (release.records[name] for name in ("r1", "r2", "r4"))
    with TimeFReader(open_local_version(release.path)) as reader:
        streamed = [record.id for record in reader.iter_records(iter((r4, r1, r4, r2)), batch_size=3)]
    assert streamed == [r4, r1, r4, r2]
    assert batches == [(r4, r1, r4), (r2,)]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "2"])
def test_a_batch_size_that_is_not_a_positive_int_is_rejected(release, batch_size):
    with (
        TimeFReader(open_local_version(release.path)) as reader,
        pytest.raises(TimeFValidationError, match="batch_size must be a positive int"),
    ):
        next(reader.iter_records(batch_size=batch_size))


def test_an_unknown_id_in_a_later_batch_raises_before_the_first_batch(release, batches):
    with (
        TimeFReader(open_local_version(release.path)) as reader,
        pytest.raises(TimeFNotFoundError, match=r"no such record\(s\) in control\.duckdb: \[99\]"),
    ):
        next(reader.iter_records((release.records["r1"], 99), batch_size=1))
    assert batches == []


@pytest.mark.parametrize(
    "plan",
    [
        LoadPlan.ALL_LAZY,
        LoadPlan(signal_values=Loading.EAGER),
        LoadPlan(time_offsets=Loading.EAGER),
        LoadPlan(text=Loading.EAGER),
        LoadPlan.ALL_EAGER,
    ],
    ids=["lazy", "signal_values", "time_offsets", "text", "eager"],
)
def test_each_batch_loads_the_kinds_of_values_its_plan_names(release, plan):
    with TimeFReader(open_local_version(release.path)) as reader:
        for record in reader.iter_records(load=plan, batch_size=2):
            # Each record holds every kind of stored value: arrays, time offsets, and packed text.
            for kind, handles in _handles(record).items():
                assert handles
                assert all(handle.is_loaded is (getattr(plan, kind) is Loading.EAGER) for handle in handles)


def test_the_values_of_each_eager_batch_stay_readable_without_storage(release):
    with TimeFReader(open_local_version(release.path)) as reader:
        records = list(reader.iter_records(load=LoadPlan.ALL_EAGER, batch_size=1))
    shutil.rmtree(release.path)
    assert {record.id for record in records} == set(release.records.values())
    for record in records:
        name = record.metadata["name"]
        regular, irregular = record.signals
        assert [regular.to_numpy().tolist(), irregular.to_numpy().tolist()] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
        assert irregular.time_offsets_us().tolist() == [0, 2, 5]
        assert read_value(record.annotations[0].value) == f"{_TEXT}{name}"
        assert read_value(regular.annotations[0].value) == [_TEXT, name]


def test_an_empty_request_or_release_yields_no_batch(tmp_path, release, batches):
    with TimeFReader(open_local_version(release.path)) as reader:
        assert list(reader.iter_records((), load=LoadPlan.ALL_EAGER)) == []
    empty = _write(tmp_path / "empty", TimeFDataset(metadata=_metadata()))
    with TimeFReader(open_local_version(empty)) as reader:
        assert list(reader.iter_records(load=LoadPlan.ALL_EAGER)) == []
    assert batches == []


def test_a_batch_loads_only_its_own_records(release, batches, monkeypatch):
    filled: list[StoredArray] = []
    original = StoredArray._fill

    def recording(self, *args):
        filled.append(self)
        return original(self, *args)

    monkeypatch.setattr(StoredArray, "_fill", recording)
    with TimeFReader(open_local_version(release.path)) as reader:
        stream = reader.iter_records(load=LoadPlan.ALL_EAGER, batch_size=2)
        first = [next(stream), next(stream)]
        # The first batch is loaded. The second one is neither read from the catalog nor loaded.
        assert len(batches) == 1
        assert {id(handle) for handle in filled} == {id(handle) for handle in _handles(*first)["signal_values"]}
        second = list(stream)
    assert len(batches) == 2
    assert len(filled) == 8
    assert {id(handle) for handle in filled[4:]} == {id(handle) for handle in _handles(*second)["signal_values"]}
