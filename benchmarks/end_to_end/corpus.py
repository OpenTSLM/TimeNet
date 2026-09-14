"""The deterministic corpus the regression suite writes and reads back.

Every value is derived from its position, so two revisions building the same corpus build exactly the
same bytes. The shapes here are the ones most likely to break: a source tree deeper than one level,
a series shared by two records, annotations at every level, an irregular axis, a task with two
inputs, and a signal whose values are constant.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from timenet.control_plane import Annotation, DeclarativeDataset, Record, RecordRef, Signal, Source, Task
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis
from timenet.types import Access, DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg


ECG = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.Unit("mV"), dtype="float32")
TEMPERATURE = TimeSeriesSpec(spec_type="temperature", name="Temperature", unit_value=ureg.Unit("degC"), dtype="float32")
COUNTS = TimeSeriesSpec(spec_type="counts", name="Counts", unit_value=ureg.Unit(""), dtype="float64")

FAST = RegularAxis(period_us=Fraction(1_000_000, 500))
SLOW = RegularAxis(period_us=Fraction(1_000_000, 1))
ORDINAL = OrdinalAxis()

# The lead whose values are constant, the records that share a series, and the task built with two
# inputs. Each is a shape that has broken a writer before, so each is named rather than inlined.
CONSTANT_LEAD = 2
SHARING_RECORDS = 2
TWO_INPUT_TASK = 2


def metadata(scale: int) -> DatasetMetadata:
    """Return the corpus metadata, naming the scale so a mix-up is visible.

    Args:
        scale: The step multiplier the corpus was built with.

    Returns:
        The metadata block.
    """
    return DatasetMetadata(
        dataset_id="timenet/end-to-end-benchmark",
        dataset_version=Version.parse("1.0.0"),
        name="End-to-end benchmark corpus",
        description=f"Deterministic corpus at scale {scale}.",
        license=License.MIT,
        domains=(Domain.CARDIOLOGY,),
        access=Access.OPEN,
    )


def _values(seed: int, count: int, *, constant: bool = False) -> np.ndarray:
    """Return a deterministic values array.

    Args:
        seed: Chooses the series.
        count: How many values.
        constant: Emit one repeated value, which compresses differently from noise.

    Returns:
        The values.
    """
    if constant:
        return np.full(count, float(seed), dtype=np.float32)
    index = np.arange(count, dtype=np.float32)
    return (np.sin(index / (seed + 3)) * (seed + 1)).astype(np.float32)


def build_corpus(*, profile: str = "portable", scale: int = 1) -> DeclarativeDataset:
    """Build the benchmark corpus.

    Args:
        profile: ``"portable"`` for the shapes every backend accepts, or ``"rich"`` to add an
            ordinal axis and a float64 series.
        scale: Positive multiplier for each series' number of steps.

    Returns:
        A deterministic dataset.

    Raises:
        ValueError: If ``profile`` or ``scale`` is invalid.
    """
    if profile not in {"portable", "rich"}:
        raise ValueError(f"profile must be 'portable' or 'rich', got {profile!r}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    dataset = DeclarativeDataset(metadata=metadata(scale))
    # One annotation object on several records: its payload must be stored once.
    cohort = Annotation.static(name="cohort", value="benchmark")
    # One signal used by two records: its values must be stored once and linked twice.
    shared = Signal(id="shared-series", name="shared", values=_values(7, 64 * scale), time_axis=SLOW, spec=TEMPERATURE)
    shared.annotate(Annotation.static(name="calibration", value="factory"))

    for index in range(4):
        record_id = f"record-{index:03d}"
        leads = [
            Signal(
                id=f"{record_id}-lead-{lead}",
                name=f"lead-{lead}",
                values=_values(index * 3 + lead, 256 * scale, constant=lead == CONSTANT_LEAD),
                time_axis=FAST,
                spec=ECG,
            )
            for lead in range(3)
        ]
        leads[0].annotate(Annotation.point(name="lead_status", value="fell off", at_us=1_000_000))
        leads[1].annotate(Annotation.interval(name="artifact", value="motion", start_us=0, end_us=500_000))

        irregular = Signal(
            id=f"{record_id}-irregular",
            name="irregular",
            values=_values(index + 11, 16 * scale),
            time_axis=IrregularAxis(first_us=0, last_us=15_000 * scale),
            time_offsets_us=np.arange(16 * scale, dtype=np.int64) * 1_000,
            spec=TEMPERATURE,
        )

        ecg = Source(id=f"{record_id}-ecg", name="ECG", signals=leads)
        ecg.annotate(Annotation.static(name="device", value="monitor-9000"))
        aux = Source(id=f"{record_id}-aux", name="Auxiliary", signals=[irregular])
        if index < SHARING_RECORDS:
            aux.signals.append(shared)
        if profile == "rich":
            aux.signals.append(
                Signal(
                    id=f"{record_id}-ordinal",
                    name="ordinal",
                    values=_values(index + 23, 32 * scale).astype(np.float64),
                    time_axis=ORDINAL,
                    spec=COUNTS,
                )
            )
        monitor = Source(id=f"{record_id}-monitor", name="Bedside monitor", sources=[ecg, aux])

        record = Record(id=record_id, sources=[monitor], start_time_us=1_700_000_000_000_000 + index)
        record.annotate(cohort)
        record.annotate(Annotation.static(name="subject_age", value=40 + index, unit="years"))
        dataset.add_record(record)

    for index in range(3):
        inputs: list[Record | RecordRef | str] = [dataset.records[index]]
        if index == TWO_INPUT_TASK:
            # a task with two inputs, where the order is meaningful
            inputs = [dataset.records[0], dataset.records[1]]
        task = Task(
            id=f"task-{index:03d}",
            prompt=f"Describe recording {index}.",
            inputs=inputs,
            target=[f"Recording {index} looks normal."],
        )
        task.annotate(Annotation.static(name="task_kind", value="describe"))
        dataset.add_task(task)

    dataset.annotate(Annotation.static(name="corpus", value="end-to-end"))
    return dataset
