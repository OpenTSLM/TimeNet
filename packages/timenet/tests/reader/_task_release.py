"""A release whose tasks reach records in every way, and helpers to write and read it."""

from pathlib import Path

from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import DatasetMetadata, Domain, License, Task, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter


_TEXT = "A detailed clinical report. " * 300
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


def _streamed(release, batch_size: int = 5_000) -> dict[str, Task]:
    """Stream every task of the release, ``batch_size`` tasks a batch."""
    with TimeFReader(open_local_version(release.path)) as reader:
        return release.named(reader.iter_tasks(batch_size=batch_size))
