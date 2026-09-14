"""Read many signals' values in one call, which is the call every consumer goes through.

The torch views and the pandas frame builder both read a batch of records by asking
:meth:`TimeFReader.values_for` for every signal of the batch at once. Reading per signal instead
re-opens the shard and re-decodes the row group for each one, so these tests count the row groups
the values plane decodes as well as checking the values that come back. The count comes from
``pyarrow.parquet.ParquetFile.read_row_group``, which is the call that turns a row group into memory.

The fixture writes one long signal and four short ones with small byte budgets, so the long signal
spans 10 row groups and three of the short ones share the one trailing row group of their shard. The
fourth is another modality, so it lands in a shard of its own. One short signal is used by two
records, which is the case a shared series makes: a batch of records lists it once per record and it
must still be decoded once.
"""

from fractions import Fraction

import numpy as np
import pyarrow.parquet as pq
import pytest

from timenet.control_plane import DeclarativeDataset, Record, Signal, Source, TimeFReader, TimeFWriter
from timenet.dataset.axis import RegularAxis
from timenet.testing import make_metadata
from timenet.types import TimeSeriesSpec, ureg


CHUNK_VALUES = 1_024
"""Values per chunk: the 4 KiB chunk budget over four bytes a float32."""

CHUNKS_PER_GROUP = 4
"""Chunks per row group: the 16 KiB row group budget over the 4 KiB chunk budget."""

N_VALUES = 40 * CHUNK_VALUES
"""The long signal's length: 40 chunks, so 10 row groups."""

ROW_GROUPS = N_VALUES // CHUNK_VALUES // CHUNKS_PER_GROUP
"""How many row groups the long signal spans, which is what a whole-signal read costs."""

EEG_AXIS = RegularAxis(period_us=Fraction(1_000_000, 100))
EEG_SPEC = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.Unit("uV"), dtype="float32")
STEP_SPEC = TimeSeriesSpec(spec_type="steps", name="Steps", unit_value=ureg.Unit(""), dtype="int16")


def _long_values() -> np.ndarray:
    """Return the long signal's values, with payloads a value comparison would wave through."""
    values = np.random.default_rng(0).standard_normal(N_VALUES).astype(np.float32)
    for index, payload in {7: "nan", 3_000: "inf", 20_481: "-inf", 20_482: "-0.0"}.items():
        values[index] = np.float32(payload)
    return values


@pytest.fixture(scope="module")
def version(tmp_path_factory):
    """Write two records: one long signal, three short ones, and a series both records use."""
    root = tmp_path_factory.mktemp("values-for")
    long_signal = Signal(id="long", name="EEG Fpz-Cz", values=_long_values(), time_axis=EEG_AXIS, spec=EEG_SPEC)
    short_signal = Signal(
        id="short", name="EEG Pz-Oz", values=np.arange(100, dtype=np.float32), time_axis=EEG_AXIS, spec=EEG_SPEC
    )
    other_signal = Signal(
        id="other", name="EOG", values=np.arange(100, 200, dtype=np.float32), time_axis=EEG_AXIS, spec=EEG_SPEC
    )
    shared_signal = Signal(
        id="shared", name="EMG", values=np.arange(200, 300, dtype=np.float32), time_axis=EEG_AXIS, spec=EEG_SPEC
    )
    counts_signal = Signal(
        id="counts", name="steps", values=np.arange(100, dtype=np.int16), time_axis=EEG_AXIS, spec=STEP_SPEC
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[
            Record(
                id="night-000",
                sources=[Source(id="psg", name="PSG", signals=[long_signal, short_signal, shared_signal])],
            ),
            Record(
                id="night-001",
                sources=[Source(id="psg-2", name="PSG", signals=[other_signal, shared_signal, counts_signal])],
            ),
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
    """Return every signal's surrogate id, keyed by the id it was built under."""
    found = {}
    for external_id in ("night-000", "night-001"):
        for signal in reader.record(external_id).signals():
            found[signal.external_id] = signal.signal_id
    return found


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


def test_the_fixture_lays_the_signals_out_the_way_the_counts_below_assume(reader, signals):
    """The three EEG short signals must share one row group, or every count below means nothing."""
    assert len(reader.chunk_locators(signals["long"])) == N_VALUES // CHUNK_VALUES
    assert len({locator.chunk_major_idx for locator in reader.chunk_locators(signals["long"])}) == ROW_GROUPS
    groups = {
        name: {(locator.chunk_file, locator.chunk_major_idx) for locator in reader.chunk_locators(signals[name])}
        for name in ("short", "other", "shared")
    }
    assert all(len(group) == 1 for group in groups.values())
    assert len(set().union(*groups.values())) == 1


def test_every_signal_comes_back_keyed_by_its_surrogate_id(reader, signals):
    found = reader.values_for(list(signals.values()))
    assert set(found) == set(signals.values())
    assert found[signals["long"]].shape == (N_VALUES,)
    assert [len(found[signals[name]]) for name in ("short", "other", "shared", "counts")] == [100, 100, 100, 100]


def test_a_batch_matches_the_per_signal_read_byte_for_byte(reader, signals):
    """Compared as bytes: NaN is not equal to itself and -0.0 compares equal to 0.0."""
    found = reader.values_for(list(signals.values()))
    for name, signal_id in signals.items():
        alone = reader.values(signal_id)
        assert found[signal_id].dtype == alone.dtype, name
        assert found[signal_id].tobytes() == alone.tobytes(), name
    assert found[signals["long"]].tobytes() == _long_values().tobytes()
    assert found[signals["counts"]].dtype == np.int16


def test_signals_sharing_a_row_group_cost_one_decode(reader, signals, row_group_reads):
    """The whole point of the batched read: three signals in one row group, one decode between them."""
    found = reader.values_for([signals["short"], signals["other"], signals["shared"]])
    assert len(row_group_reads) == 1
    assert len(found) == 3
    assert found[signals["other"]].tolist() == list(range(100, 200))


def test_reading_them_one_at_a_time_pays_per_signal(reader, signals, row_group_reads):
    """Why the batched call exists: the same three signals cost a decode each read apart."""
    for name in ("short", "other", "shared"):
        reader.values(signals[name])
    assert len(row_group_reads) == 3


def test_a_shared_series_named_once_per_record_is_decoded_once(reader, signals, row_group_reads):
    """A batch of records lists a shared series once per record, and it must still cost one decode."""
    found = reader.values_for([signals["shared"], signals["shared"], signals["short"]])
    assert len(row_group_reads) == 1
    assert set(found) == {signals["shared"], signals["short"]}
    assert found[signals["shared"]].tolist() == list(range(200, 300))


def test_a_signal_spanning_several_row_groups_is_rebuilt_in_chunk_order(reader, signals, row_group_reads):
    found = reader.values_for([signals["long"]])
    assert len(row_group_reads) == ROW_GROUPS
    assert found[signals["long"]].tobytes() == _long_values().tobytes()


def test_an_id_naming_no_signal_is_skipped(reader, signals):
    """The same answer values() gives for an id that names nothing, without failing the batch."""
    unknown = max(signals.values()) + 1_000
    found = reader.values_for([signals["short"], unknown])
    assert set(found) == {signals["short"]}
    assert len(reader.values(unknown)) == 0


def test_an_empty_request_reads_nothing(reader, row_group_reads):
    """A record with no signals yields an empty ask, which must not touch the values plane."""
    assert reader.values_for([]) == {}
    assert row_group_reads == []
