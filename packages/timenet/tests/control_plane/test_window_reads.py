"""Read a step window of a signal and count the row groups it cost.

A window read is only worth having if it reads less than a whole-signal read, so these tests count
what the values plane decodes rather than only checking the values that come back. The count comes
from ``pyarrow.parquet.ParquetFile.read_row_group``, which is the call that turns a row group into
memory, so a window that plans well but decodes everything anyway cannot pass here.

The fixture writes one long signal with a small chunk budget: 40,960 float32 values, 1,024 per chunk
and four chunks per row group, so the signal spans 40 chunks in 10 row groups. NaN, both infinities
and a negative zero sit at known steps, and every comparison is made on ``tobytes()``, because
``==`` calls NaN unequal to itself and -0.0 equal to 0.0.
"""

from fractions import Fraction

import numpy as np
import pyarrow.parquet as pq
import pytest

from timenet.control_plane import DeclarativeDataset, Record, Signal, Source, TimeFReader, TimeFWriter
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.testing import make_metadata
from timenet.types import TimeSeriesSpec, ureg


CHUNK_VALUES = 1_024
"""Values per chunk: the 4 KiB chunk budget over four bytes a float32."""

CHUNKS_PER_GROUP = 4
"""Chunks per row group: the 16 KiB row group budget over the 4 KiB chunk budget."""

N_VALUES = 40 * CHUNK_VALUES
"""The long signal's length: 40 chunks, so 10 row groups."""

SPECIALS = {7: np.float32("nan"), 3_000: np.float32("inf"), 20_481: np.float32("-inf"), 20_482: np.float32("-0.0")}
"""Payloads that a comparison on values rather than bytes would wave through, at known steps."""

EEG_SPEC = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.Unit("uV"), dtype="float32")
EEG_AXIS = RegularAxis(period_us=Fraction(1_000_000, 100))


def _values() -> np.ndarray:
    """Return the long signal's values, with the special payloads planted in them."""
    values = np.random.default_rng(0).standard_normal(N_VALUES).astype(np.float32)
    for index, payload in SPECIALS.items():
        values[index] = payload
    return values


@pytest.fixture(scope="module")
def version(tmp_path_factory):
    """Write one record whose long signal spans 40 chunks in 10 row groups."""
    root = tmp_path_factory.mktemp("windows")
    long_signal = Signal(id="long", name="EEG Fpz-Cz", values=_values(), time_axis=EEG_AXIS, spec=EEG_SPEC)
    # Both short signals fit one chunk each and arrive after the long signal's last flush, so they
    # end up as two rows of one trailing row group. That is what a shared decode is measured on.
    short_signal = Signal(
        id="short", name="EEG Pz-Oz", values=np.arange(100, dtype=np.float32), time_axis=EEG_AXIS, spec=EEG_SPEC
    )
    other_signal = Signal(
        id="other", name="EOG", values=np.arange(100, 200, dtype=np.float32), time_axis=EEG_AXIS, spec=EEG_SPEC
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[
            Record(
                id="night-000",
                sources=[Source(id="psg", name="PSG", signals=[long_signal, short_signal, other_signal])],
            )
        ],
    )
    with TimeFWriter(root, dataset.metadata, chunk_max_bytes=4 * 1024, row_group_target_bytes=16 * 1024) as writer:
        writer.write(dataset)
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def reader(version):
    """Open the written version for reading."""
    with TimeFReader(version) as opened:
        yield opened


@pytest.fixture
def signals(reader):
    """Return the two signals' surrogate ids, keyed by the id they were built under."""
    return {signal.external_id: signal.signal_id for signal in reader.record("night-000").signals()}


@pytest.fixture
def row_group_reads(monkeypatch):
    """Count every row group the values plane decodes, whichever call path reaches it."""
    decoded: list[int] = []
    original = pq.ParquetFile.read_row_group

    def counted(self, index, *arguments, **options):
        decoded.append(index)
        return original(self, index, *arguments, **options)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", counted)
    return decoded


def test_the_signal_spans_the_chunks_and_row_groups_the_fixture_claims(reader, signals):
    """Every count below is against these two numbers, so a layout change cannot pass silently."""
    locators = reader.chunk_locators(signals["long"])
    assert len(locators) == N_VALUES // CHUNK_VALUES
    assert {locator.n_values for locator in locators} == {CHUNK_VALUES}
    groups = sorted({locator.chunk_major_idx for locator in locators})
    assert len(groups) == N_VALUES // CHUNK_VALUES // CHUNKS_PER_GROUP


def test_a_window_inside_one_chunk_reads_one_row_group(reader, signals, row_group_reads):
    """The point of the whole exercise: 3,000 steps of a 40,960 step signal cost one row group."""
    window = reader.values_window(signals["long"], 5_000, 8_000)
    assert len(row_group_reads) == 1
    assert window.tobytes() == _values()[5_000:8_000].tobytes()


def test_a_whole_signal_read_costs_every_row_group(reader, signals, row_group_reads):
    """What the window is measured against: ten row groups, against the one a window reads.

    Measured through ``values_for``, which groups by row group the way a window read does. The
    per-signal ``values`` decodes once per chunk instead, so it would report 40 for the same bytes.
    """
    reader.values_for([signals["long"]])
    assert len(row_group_reads) == N_VALUES // CHUNK_VALUES // CHUNKS_PER_GROUP


def test_a_window_spanning_a_chunk_boundary_stays_inside_its_row_group(reader, signals, row_group_reads):
    """Chunks 0 and 1 share a row group, so crossing between them is still one decode."""
    window = reader.values_window(signals["long"], CHUNK_VALUES - 10, CHUNK_VALUES + 10)
    assert len(row_group_reads) == 1
    assert window.tobytes() == _values()[CHUNK_VALUES - 10 : CHUNK_VALUES + 10].tobytes()


def test_a_window_spanning_a_row_group_boundary_reads_both(reader, signals, row_group_reads):
    """Chunk 3 ends a row group and chunk 4 starts the next, so this crosses two and reads two."""
    start = CHUNKS_PER_GROUP * CHUNK_VALUES - 10
    window = reader.values_window(signals["long"], start, start + 20)
    assert sorted(row_group_reads) == [0, 1]
    assert window.tobytes() == _values()[start : start + 20].tobytes()


def test_a_window_over_the_whole_signal_equals_a_whole_read(reader, signals, row_group_reads):
    """A window is not a different read of the same values, so the widest one must agree exactly."""
    window = reader.values_window(signals["long"], 0, N_VALUES)
    assert len(row_group_reads) == N_VALUES // CHUNK_VALUES // CHUNKS_PER_GROUP
    assert window.dtype == np.float32
    assert window.tobytes() == _values().tobytes()


def test_an_empty_window_reads_nothing_and_keeps_the_dtype(reader, signals, row_group_reads):
    """An empty window is a legitimate ask, and it must not decode a row group to answer it."""
    window = reader.values_window(signals["long"], 5_000, 5_000)
    assert row_group_reads == []
    assert (len(window), window.dtype) == (0, np.float32)


def test_a_window_past_the_end_is_clamped_to_the_last_step(reader, signals, row_group_reads):
    """A scoring can overrun the signals it scores, so a window delivers the steps that exist."""
    window = reader.values_window(signals["long"], N_VALUES - 100, N_VALUES + 10_000)
    assert len(row_group_reads) == 1
    assert window.tobytes() == _values()[N_VALUES - 100 :].tobytes()


def test_a_window_entirely_past_the_end_reads_nothing(reader, signals, row_group_reads):
    """Clamped to nothing rather than raised, so a caller walking a grid of epochs runs off cleanly."""
    window = reader.values_window(signals["long"], N_VALUES + 5, N_VALUES + 100)
    assert row_group_reads == []
    assert (len(window), window.dtype) == (0, np.float32)


def test_a_backwards_window_is_refused(reader, signals):
    """A reversed or negative window is a caller mistake, not an empty read."""
    with pytest.raises(TimeFValidationError):
        reader.values_window(signals["long"], 900, 100)
    with pytest.raises(TimeFValidationError):
        reader.values_window(signals["long"], -1, 100)


def test_every_window_matches_the_slice_of_the_whole_signal(reader, signals):
    """Compared as bytes: NaN is not equal to itself and -0.0 compares equal to 0.0."""
    whole = _values()
    for start, stop in ((0, 1), (7, 8), (2_990, 3_010), (20_470, 20_500), (39_000, N_VALUES), (0, N_VALUES)):
        window = reader.values_window(signals["long"], start, stop)
        assert window.dtype == whole.dtype
        assert window.tobytes() == whole[start:stop].tobytes(), f"window [{start}, {stop}) differs"


def test_two_windows_in_one_row_group_cost_one_decode(reader, signals, row_group_reads):
    """Both short signals sit in the trailing row group, so their windows share its one decode."""
    found = reader.values_windows([(signals["short"], 10, 20), (signals["other"], 30, 40)])
    assert len(row_group_reads) == 1
    assert found[0].tobytes() == np.arange(10, 20, dtype=np.float32).tobytes()
    assert found[1].tobytes() == np.arange(130, 140, dtype=np.float32).tobytes()


def test_windows_on_different_row_groups_read_one_each(reader, signals, row_group_reads):
    """A window per signal of a record, which is what an item cut from a record asks for."""
    whole = _values()
    found = reader.values_windows([(signals["long"], 100, 200), (signals["short"], 10, 20)])
    assert sorted(row_group_reads) == [0, N_VALUES // CHUNK_VALUES // CHUNKS_PER_GROUP]
    assert found[0].tobytes() == whole[100:200].tobytes()
    assert found[1].tobytes() == np.arange(10, 20, dtype=np.float32).tobytes()


def test_many_windows_of_one_signal_share_its_row_group(reader, signals, row_group_reads):
    """A batch of items cut from one record: four windows of one signal, one decode between them."""
    whole = _values()
    wanted = ((10, 20), (1_100, 1_200), (2_200, 2_300), (3_300, 3_400))
    found = reader.values_windows([(signals["long"], start, stop) for start, stop in wanted])
    assert row_group_reads == [0]
    for window, (start, stop) in zip(found, wanted, strict=True):
        assert window.tobytes() == whole[start:stop].tobytes()


def test_reading_windows_one_at_a_time_pays_per_call(reader, signals, row_group_reads):
    """Why the batched call exists: four windows of one row group cost four decodes apart."""
    whole = _values()
    for start, stop in ((10, 20), (1_100, 1_200), (2_200, 2_300), (3_300, 3_400)):
        assert reader.values_window(signals["long"], start, stop).tobytes() == whole[start:stop].tobytes()
    assert row_group_reads == [0, 0, 0, 0]


def test_an_unknown_signal_reads_back_empty(reader, signals):
    """The same answer values() gives for an id that names nothing."""
    unknown = max(signals.values()) + 1_000
    assert len(reader.values_window(unknown, 0, 10)) == 0
    assert [len(window) for window in reader.values_windows([(unknown, 0, 10)])] == [0]
