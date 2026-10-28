"""A release whose tasks reach records in every way, and helpers to write and read it."""

from pathlib import Path

from timenet.dataset import Record, Signal, TimeFDataset
from timenet.format.stored_array import StoredArray
from timenet.format.stored_values import StoredOffsets, StoredText, StoredTextList
from timenet.reader import TASK_BATCH_SIZE, LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.types import Annotation, DatasetMetadata, Domain, License, Task, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report. " * 300
_KINDS = ("signal_values", "time_offsets", "text")
_SPEC = TimeSeriesSpec(spec_type="ecg", name="lead", unit_value=ureg.millivolt)


def _write(tmp_path, dataset: TimeFDataset) -> Path:
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _metadata() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="test/plans",
        dataset_version=Version(1, 0, 0),
        name="Plans",
        description="Packed text, irregular axes, and tasks that reach records in every way.",
        license=License.CC_BY_4_0,
        domains=(Domain.GENERAL,),
    )


def _handles(*objects) -> dict[str, list]:
    """Return the stored value handles on records, signals, annotations, and tasks, by plan field."""
    found: dict[str, dict[int, object]] = {kind: {} for kind in _KINDS}

    def add(handle):
        for kind, types in zip(_KINDS, (StoredArray, StoredOffsets, StoredText | StoredTextList), strict=True):
            if isinstance(handle, types):
                found[kind][id(handle)] = handle

    def signal(item):
        add(item._values)
        add(item._time_offsets)
        for annotation in item.annotations:
            add(annotation.value)

    for item in objects:
        if isinstance(item, Record):
            for annotation in item.annotations:
                add(annotation.value)
            for source in item.walk_sources():
                for annotation in source.annotations:
                    add(annotation.value)
                for each in source.signals:
                    signal(each)
        elif isinstance(item, Signal):
            signal(item)
        elif isinstance(item, Annotation):
            add(item.value)
        else:
            for value in (item.prompt, item.rationale, *(item.targets or ())):
                add(value)
            for annotation in item.annotations:
                add(annotation.value)
    return {kind: list(handles.values()) for kind, handles in found.items()}


def _all(handles, *, loaded: bool) -> bool:
    return all(handle.is_loaded is loaded for values in handles.values() for handle in values)


def _streamed(release, batch_size: int = TASK_BATCH_SIZE) -> dict[str, Task]:
    """Stream every task of the release with every value loaded, ``batch_size`` tasks a batch."""
    with TimeFReader(open_local_version(release.path)) as reader:
        return release.named(reader.iter_tasks(load=LoadPlan.ALL_EAGER, batch_size=batch_size))
