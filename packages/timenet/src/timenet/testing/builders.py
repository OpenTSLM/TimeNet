"""Deterministic hierarchies the tests build on.

The shapes here exercise the parts of the model that are easy to get wrong: a source tree deeper
than one level, two signals sharing one axis and one spec, an annotation attached to more than one
object, and annotations at every level of the hierarchy.
"""

from fractions import Fraction

import numpy as np

from timenet.control_plane.model import Annotation, DeclarativeDataset, Record, Signal, Source, Task
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.types import Access, DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg


ECG_AXIS = RegularAxis(period_us=Fraction(1_000_000, 500))
TEMPERATURE_AXIS = RegularAxis(period_us=Fraction(1_000_000, 1))
ECG_SPEC = TimeSeriesSpec(spec_type="ecg-voltage", name="ECG voltage", unit_value=ureg.Unit("mV"), dtype="float32")
TEMPERATURE_SPEC = TimeSeriesSpec(
    spec_type="body-temperature", name="Body temperature", unit_value=ureg.Unit("degC"), dtype="float32"
)


def make_metadata(dataset_id: str = "test/bedside", version: str = "1.0.0") -> DatasetMetadata:
    """Return metadata for a test dataset.

    Args:
        dataset_id: The ``org/name`` id.
        version: The version string.

    Returns:
        The metadata block.
    """
    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version.parse(version),
        name="Bedside monitor",
        description="A test dataset.",
        license=License.CC_BY_4_0,
        domains=(Domain.CARDIOLOGY,),
        access=Access.OPEN,
    )


def make_record(record_id: str, *, seed: int = 0, n_values: int = 1_000, shared: Annotation | None = None) -> Record:
    """Build one record: a bedside monitor holding an ECG and a temperature sensor.

    The tree is three levels deep (monitor -> ECG -> signals), the two ECG leads share one axis and
    one spec, and annotations land on the record, a source, and a signal.

    Args:
        record_id: The record's id. Every id beneath it is derived from this, so two records never
            collide.
        seed: Seed for the generated values.
        n_values: How many values each ECG lead carries.
        shared: An annotation to attach to the record, so a caller can reuse one payload across
            several records.

    Returns:
        The record.
    """
    rng = np.random.default_rng(seed)
    lead_i = Signal(
        id=f"{record_id}-lead-i",
        name="I",
        values=rng.standard_normal(n_values).astype(np.float32),
        time_axis=ECG_AXIS,
        spec=ECG_SPEC,
    )
    lead_ii = Signal(
        id=f"{record_id}-lead-ii",
        name="II",
        values=rng.standard_normal(n_values).astype(np.float32),
        time_axis=ECG_AXIS,
        spec=ECG_SPEC,
    )
    lead_i.annotate(Annotation.point(name="lead_status", value="Lead fell off", at_us=6_000_000))

    temperature_values = (36.5 + rng.standard_normal(10) * 0.1).astype(np.float32)
    chest = Signal(
        id=f"{record_id}-chest",
        name="Chest temperature",
        values=temperature_values,
        time_axis=IrregularAxis(first_us=0, last_us=9_000_000),
        time_offsets_us=np.arange(10, dtype=np.int64) * 1_000_000,
        spec=TEMPERATURE_SPEC,
    )

    ecg = Source(id=f"{record_id}-ecg", name="ECG", signals=[lead_i, lead_ii])
    ecg.annotate(Annotation.static(name="device_model", value="Monitor 9000"))
    temperature = Source(id=f"{record_id}-temperature", name="Temperature sensor", signals=[chest])
    monitor = Source(id=f"{record_id}-monitor", name="Bedside monitor", sources=[ecg, temperature])

    record = Record(id=record_id, sources=[monitor], start_time_us=1_700_000_000_000_000)
    record.annotate(shared if shared is not None else Annotation.static(name="patient_sex", value="male"))
    record.annotate(Annotation.static(name="patient_age", value=65, unit="years"))
    return record


def make_dataset(n_records: int = 2, *, n_values: int = 1_000) -> DeclarativeDataset:
    """Build a dataset whose records share one annotation payload and one task each.

    Args:
        n_records: How many records to build.
        n_values: How many values each ECG lead carries.

    Returns:
        The populated dataset.
    """
    dataset = DeclarativeDataset(metadata=make_metadata())
    shared = Annotation.static(name="patient_sex", value="male")
    for index in range(n_records):
        record = make_record(f"record-{index:03d}", seed=index, n_values=n_values, shared=shared)
        dataset.add_record(record)
        dataset.add_task(
            Task(
                id=f"diagnosis-{index:03d}",
                prompt="Diagnose this patient.",
                inputs=[record],
                target=["The patient is stable."],
            ).annotate(Annotation.static(name="task_type", value="diagnosis"))
        )
    dataset.annotate(Annotation.static(name="collection", value="test fixtures"))
    return dataset
