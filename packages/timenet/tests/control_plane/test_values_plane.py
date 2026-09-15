"""What a values read hands back: the same bytes either way round, and an array the caller owns.

``values`` and ``values_for`` read the same signal through different code, so the two have to agree
on more than the numbers. A caller that scales or fills what it got back must not care which call
produced it, and a 400 byte signal must not keep the row group it was decoded from alive.

The fixture writes one signal long enough to span several chunks and row groups, and two short ones
that fit a single chunk each, which is the case the batched read used to return as a view.

Owning the result must not cost a second copy of it. The batched read is the path a training loader
takes, so a whole-signal read that allocates its result twice is measured here as well as read.
"""

from fractions import Fraction
import gc
import tracemalloc

import numpy as np
import pytest

from timenet.control_plane import DeclarativeDataset, Record, Signal, Source, TimeFReader, TimeFWriter
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.testing import make_metadata
from timenet.types import TimeSeriesSpec, ureg


CHUNK_VALUES = 1_024
"""Values per chunk: the 4 KiB chunk budget over four bytes a float32."""

N_VALUES = 10 * CHUNK_VALUES
"""The long signal's length, so it spans ten chunks over several row groups."""

EEG_SPEC = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.Unit("uV"), dtype="float32")
EEG_AXIS = RegularAxis(period_us=Fraction(1_000_000, 100))


def _values() -> np.ndarray:
    """Return the long signal's values."""
    return np.arange(N_VALUES, dtype=np.float32)


@pytest.fixture(scope="module")
def version(tmp_path_factory):
    """Write one record holding a multi-chunk signal and two single-chunk ones."""
    root = tmp_path_factory.mktemp("values")
    signals = [
        Signal(id="long", name="EEG Fpz-Cz", values=_values(), time_axis=EEG_AXIS, spec=EEG_SPEC),
        Signal(
            id="short", name="EEG Pz-Oz", values=np.arange(100, dtype=np.float32), spec=EEG_SPEC, time_axis=EEG_AXIS
        ),
        Signal(id="other", name="EOG", values=np.arange(100, 200, dtype=np.float32), spec=EEG_SPEC, time_axis=EEG_AXIS),
    ]
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=signals)])],
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
    """Return the three signals' surrogate ids, keyed by the id they were built under."""
    return {signal.external_id: signal.signal_id for signal in reader.record("night-000").signals()}


def test_a_batched_read_owns_its_arrays(reader, signals):
    """A 400 byte signal used to come back as a view pinning its whole row group."""
    found = reader.values_for(list(signals.values()))
    for external_id, signal_id in signals.items():
        values = found[signal_id]
        assert values.flags.writeable, external_id
        assert values.flags.owndata, external_id


def test_a_batched_read_is_writable_wherever_a_single_read_is(reader, signals):
    """The two calls read the same signal, so they cannot disagree on whether it can be written to."""
    single = reader.values(signals["short"])
    batched = reader.values_for([signals["short"]])[signals["short"]]
    single[0] = 0.0
    batched[0] = 0.0
    assert single.tobytes() == batched.tobytes()


def test_a_batched_read_returns_what_a_single_read_returns(reader, signals):
    """Both the multi-chunk signal, which is concatenated, and the two that fit one chunk."""
    found = reader.values_for(list(signals.values()))
    for signal_id in signals.values():
        single = reader.values(signal_id)
        assert found[signal_id].dtype == single.dtype
        assert found[signal_id].tobytes() == single.tobytes()


def test_a_multi_chunk_signal_is_stitched_back_in_chunk_order(reader, signals):
    """Ten chunks read out of several row groups, so a read that ordered by row group would shuffle."""
    assert len(reader.chunk_locators(signals["long"])) == N_VALUES // CHUNK_VALUES
    assert reader.values_for([signals["long"]])[signals["long"]].tobytes() == _values().tobytes()


def test_a_whole_signal_read_allocates_its_result_once(tmp_path):
    """A batched read that copies each chunk and then concatenates allocates the whole signal twice.

    Only NumPy's arrays are traced here, not Arrow's buffers, so the peak is the result plus the
    small fixed cost of the read. Two full passes show up as a peak of twice what came back.
    """
    values = np.arange(64 * 1024, dtype=np.float32)
    signal = Signal(id="long", name="EEG Fpz-Cz", values=values, time_axis=EEG_AXIS, spec=EEG_SPEC)
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[signal])])],
    )
    with TimeFWriter(tmp_path, dataset.metadata, chunk_max_bytes=16 * 1024, row_group_target_bytes=64 * 1024) as writer:
        writer.write(dataset)
    version = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(version) as opened:
        signal_id = opened.record("night-000").signals()[0].signal_id
        assert len(opened.chunk_locators(signal_id)) == 16
        # Read once untraced so the query plans and the imports they pull in are not counted.
        opened.values_for([signal_id])
        gc.collect()
        tracemalloc.start()
        found = opened.values_for([signal_id])
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert found[signal_id].tobytes() == values.tobytes()
    assert peak < 1.5 * values.nbytes, f"peak {peak} bytes for {values.nbytes} bytes returned"


def test_a_string_signal_round_trips_through_parquet(tmp_path):
    """The counterpart of the zarr backend's refusal: this backend has no trouble with 'str'."""
    stage_spec = TimeSeriesSpec(spec_type="sleep-stage", name="Sleep stage", unit_value=ureg.dimensionless, dtype="str")
    stages = Signal(
        id="stages",
        name="Sleep stage",
        values=np.array(["W", "N1", "N2", "N3", "REM"]),
        time_axis=RegularAxis(period_us=Fraction(30_000_000, 1)),
        spec=stage_spec,
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages])])],
    )
    with TimeFWriter(tmp_path, dataset.metadata, values_backend="parquet") as writer:
        writer.write(dataset)
    version = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(version) as opened:
        signal_id = opened.record("night-000").signals()[0].signal_id
        assert list(opened.values(signal_id)) == ["W", "N1", "N2", "N3", "REM"]


def test_a_string_signal_reads_the_same_way_through_the_batched_path(tmp_path):
    """'str' is the one dtype the batched read decodes to objects, and it spans chunks like any other.

    The stages are chunked small enough to cross several chunks, so this covers both the concatenate
    and the ownership the batched read promises, on the dtype furthest from a fixed-width float.
    """
    stage_spec = TimeSeriesSpec(spec_type="sleep-stage", name="Sleep stage", unit_value=ureg.dimensionless, dtype="str")
    labels = np.array(["W", "N1", "N2", "N3", "REM"] * 40)
    stages = Signal(
        id="stages",
        name="Sleep stage",
        values=labels,
        time_axis=RegularAxis(period_us=Fraction(30_000_000, 1)),
        spec=stage_spec,
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages])])],
    )
    with TimeFWriter(tmp_path, dataset.metadata, chunk_max_bytes=256, row_group_target_bytes=512) as writer:
        writer.write(dataset)
    version = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(version) as opened:
        signal_id = opened.record("night-000").signals()[0].signal_id
        assert len(opened.chunk_locators(signal_id)) > 1
        batched = opened.values_for([signal_id])[signal_id]
        assert list(batched) == list(labels)
        assert list(batched) == list(opened.values(signal_id))
        assert batched.flags.writeable
        assert batched.flags.owndata


def test_an_enum_signal_round_trips_as_its_labels(tmp_path):
    """The reader calls 'enum' a text dtype, so the values plane has to be able to store one."""
    stage_spec = TimeSeriesSpec(
        spec_type="sleep-stage",
        name="Sleep stage",
        unit_value=ureg.dimensionless,
        dtype="enum",
        categories=("W", "N1", "N2", "N3", "REM"),
    )
    stages = Signal(
        id="stages",
        name="Sleep stage",
        values=np.array(["W", "N1", "N2", "N3", "REM"]),
        time_axis=RegularAxis(period_us=Fraction(30_000_000, 1)),
        spec=stage_spec,
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(),
        records=[Record(id="night-000", sources=[Source(id="psg", name="PSG", signals=[stages])])],
    )
    with TimeFWriter(tmp_path, dataset.metadata, values_backend="parquet") as writer:
        writer.write(dataset)
    version = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(version) as opened:
        signal = opened.record("night-000").signals()[0]
        assert signal.dtype == "enum"
        # The labels themselves come back, not their positions in the codebook.
        assert list(opened.values(signal.signal_id)) == ["W", "N1", "N2", "N3", "REM"]
        assert opened.values_window(signal.signal_id, 2, 2).dtype == np.dtype(object)


def test_a_signal_with_fewer_time_offsets_than_values_is_refused():
    """The values plane addresses both arrays with the values offset, so a short pair desynchronises."""
    with pytest.raises(TimeFValidationError, match="one offset per value"):
        Signal(
            id="chest",
            name="Chest temperature",
            values=np.arange(100, dtype=np.float32),
            time_axis=IrregularAxis(first_us=0, last_us=99),
            time_offsets_us=np.arange(99, dtype=np.int64),
            spec=EEG_SPEC,
        )


def test_a_signal_with_more_time_offsets_than_values_is_refused():
    """Refused from the other side too: the check is on the pair, not on one array being short."""
    with pytest.raises(TimeFValidationError, match="one offset per value"):
        Signal(
            id="chest",
            name="Chest temperature",
            values=np.arange(10, dtype=np.float32),
            time_axis=IrregularAxis(first_us=0, last_us=99),
            time_offsets_us=np.arange(11, dtype=np.int64),
            spec=EEG_SPEC,
        )
