"""Cover the torch view: the item both walks build, the two walks, and the reads they refuse.

The item is built in one place, :func:`record_item`, and the map-style dataset and the iterable
stream both go through it, so most of what is checked here is checked once on the item and then once
per walk on the order and the slicing.

torch is an optional extra, so the module is skipped when it is missing, the way the zarr and moto
tests are.
"""

from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from torch.utils.data import Dataset, IterableDataset  # noqa: E402

from timenet.control_plane import (  # noqa: E402
    DeclarativeDataset,
    Record,
    Signal,
    Source,
    TimeFReader,
    TimeFWriter,
)
from timenet.dataset.axis import OrdinalAxis, RegularAxis  # noqa: E402
from timenet.errors import TimeFValidationError  # noqa: E402
from timenet.testing import make_dataset, make_metadata  # noqa: E402
from timenet.torch import ITEM_FIELDS, TimeFIterableStream, TimeFTorchDataset, record_item  # noqa: E402
from timenet.types import TimeSeriesSpec, ureg  # noqa: E402


N_RECORDS = 4
"""Records in the fixture corpus. Every count below is against this number."""

N_VALUES = 64
"""Values per ECG lead. The temperature signal carries 10, which is what make_record builds."""


@pytest.fixture(scope="module")
def version(tmp_path_factory):
    """Write the fixture corpus and return its committed version directory."""
    root = tmp_path_factory.mktemp("torch")
    dataset = make_dataset(n_records=N_RECORDS, n_values=N_VALUES)
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def reader(version):
    """Open the written version for reading."""
    with TimeFReader(version) as opened:
        yield opened


@pytest.fixture
def mixed_dtype_version(tmp_path_factory):
    """Write one record whose two signals are stored under different dtypes."""
    root = tmp_path_factory.mktemp("torch-dtypes")
    axis = RegularAxis(period_us=Fraction(1_000_000, 100))
    counts = Signal(
        id="counts",
        name="step count",
        values=np.arange(32, dtype=np.int16),
        time_axis=axis,
        spec=TimeSeriesSpec(spec_type="steps", name="Steps", unit_value=ureg.Unit(""), dtype="int16"),
    )
    voltage = Signal(
        id="voltage",
        name="lead I",
        values=np.arange(32, dtype=np.float32),
        time_axis=axis,
        spec=TimeSeriesSpec(spec_type="ecg-voltage", name="ECG", unit_value=ureg.Unit("mV"), dtype="float32"),
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(dataset_id="test/dtypes"),
        records=[Record(id="mixed", sources=[Source(id="wrist", name="Wrist", signals=[counts, voltage])])],
    )
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


@pytest.fixture
def text_version(tmp_path_factory):
    """Write one record whose only signal holds text, which no tensor can carry."""
    root = tmp_path_factory.mktemp("torch-text")
    note = Signal(
        id="note",
        name="clinical note",
        values=np.array(["fine", "lead fell off"]),
        time_axis=OrdinalAxis(),
        spec=TimeSeriesSpec(spec_type="note", name="Note", unit_value=ureg.Unit(""), dtype="str"),
    )
    dataset = DeclarativeDataset(
        metadata=make_metadata(dataset_id="test/text"),
        records=[Record(id="notes-000", sources=[Source(id="chart", name="Chart", signals=[note])])],
    )
    with TimeFWriter(root, dataset.metadata) as writer:
        writer.write(dataset)
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_the_dataset_is_a_torch_dataset_of_every_record(reader):
    dataset = TimeFTorchDataset(reader)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == N_RECORDS


def test_the_item_carries_every_field_in_order(reader):
    item = TimeFTorchDataset(reader)[0]
    assert tuple(item) == ITEM_FIELDS
    assert item["record_id"] == "record-000"
    assert [tensor.shape for tensor in item["series"]] == [(N_VALUES,), (N_VALUES,), (10,)]
    assert item["tasks"] == ()
    assert item["series_masks"] == (None, None, None)
    assert [annotation.name for annotation in item["annotations"]] == ["patient_sex", "patient_age"]


def test_record_item_builds_an_item_from_values_read_ahead(reader):
    """The item is also built directly, from values a caller read for a whole batch in one pass."""
    record = reader.record("record-001")
    values = reader.values_for([signal.signal_id for signal in record.signals()])
    item = record_item(record, values)
    assert tuple(item) == ITEM_FIELDS
    assert item["record_id"] == "record-001"
    assert [tensor.numpy().tobytes() for tensor in item["series"]] == [
        values[signal.signal_id].tobytes() for signal in record.signals()
    ]


def test_a_series_keeps_the_dtype_it_was_stored_under(mixed_dtype_version):
    """A tensor built through float() would widen the int16 signal, so both dtypes are checked."""
    with TimeFReader(mixed_dtype_version) as reader:
        item = TimeFTorchDataset(reader)[0]
        assert [tensor.dtype for tensor in item["series"]] == [torch.int16, torch.float32]
        assert item["series"][0].tolist() == list(range(32))


def test_a_series_holds_the_values_the_reader_reads(reader):
    record = reader.record("record-000")
    item = TimeFTorchDataset(reader)[record.record_id]
    for tensor, signal in zip(item["series"], record.signals(), strict=True):
        assert tensor.numpy().tobytes() == reader.values(signal.signal_id).tobytes()


def test_a_position_is_the_surrogate_id_the_control_plane_assigned(reader):
    dataset = TimeFTorchDataset(reader)
    for position in range(N_RECORDS):
        assert dataset[position]["record_id"] == reader.records_by_id([position])[0].external_id


def test_getitems_reads_a_batch_in_the_order_asked_for(reader):
    dataset = TimeFTorchDataset(reader)
    wanted = [3, 0, 2]
    batch = dataset.__getitems__(wanted)
    assert [item["record_id"] for item in batch] == [dataset[position]["record_id"] for position in wanted]
    for item, position in zip(batch, wanted, strict=True):
        assert [tensor.numpy().tobytes() for tensor in item["series"]] == [
            tensor.numpy().tobytes() for tensor in dataset[position]["series"]
        ]


def test_a_position_outside_the_version_raises_index_error(reader):
    dataset = TimeFTorchDataset(reader)
    with pytest.raises(IndexError, match="outside"):
        _ = dataset[N_RECORDS]
    with pytest.raises(IndexError, match="outside"):
        _ = dataset[-1]
    with pytest.raises(IndexError, match="outside"):
        dataset.__getitems__([0, N_RECORDS + 10])


def test_a_transform_is_applied_on_both_paths(reader):
    dataset = TimeFTorchDataset(reader, transform=lambda item: item["record_id"])
    assert dataset[1] == "record-001"
    assert dataset.__getitems__([1, 2]) == ["record-001", "record-002"]
    stream = TimeFIterableStream(reader, transform=lambda item: item["record_id"])
    assert sorted(stream) == ["record-000", "record-001", "record-002", "record-003"]


def test_the_stream_walks_the_corpus_in_stored_order(reader):
    stream = TimeFIterableStream(reader, batch_size=3)
    assert isinstance(stream, IterableDataset)
    walked: list[Any] = list(stream)
    assert [item["record_id"] for item in walked] == [record.external_id for record in reader.iter_records()]
    assert len(walked) == N_RECORDS
    assert all(len(item["series"]) == 3 for item in walked)


def test_workers_split_the_corpus_into_disjoint_slices(reader, monkeypatch):
    """Two DataLoader workers must together see every record exactly once.

    The split comes from ``torch.utils.data.get_worker_info``, so standing in for it is what a
    worker looks like from inside the stream, without paying for two spawned processes.
    """
    slices = []
    for worker_id in range(2):
        monkeypatch.setattr(
            torch.utils.data, "get_worker_info", lambda worker=worker_id: SimpleNamespace(id=worker, num_workers=2)
        )
        slices.append([item["record_id"] for item in TimeFIterableStream(reader, batch_size=2)])
    assert all(len(part) == N_RECORDS // 2 for part in slices), slices
    flat = [record_id for part in slices for record_id in part]
    assert sorted(flat) == reader.record_ids()


def test_a_text_signal_has_no_tensor_form(text_version):
    """Text is a legitimate signal dtype, so the view has to refuse it rather than build a tensor."""
    with TimeFReader(text_version) as reader, pytest.raises(TimeFValidationError, match="no tensor form") as raised:
        _ = TimeFTorchDataset(reader)[0]
    assert "clinical note" in str(raised.value)


def test_a_signal_whose_values_were_not_read_names_itself(reader):
    record = reader.record("record-000")
    missing = record.signals()[0]
    with pytest.raises(TimeFValidationError, match="no values here") as raised:
        record_item(record, {})
    assert str(missing.signal_id) in str(raised.value)
    assert missing.name in str(raised.value)
