"""Derive HEARTS CGMacros inputs from the pinned original CGMacros parent."""

from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, TimeFDataset
from timenet.types import ForecastingTask, TSEditingTask
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.records import case_records
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.release import TASKS, VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.tasks import convert_case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, iter_cases, moment_us
from timenet_connectors.datasets.yang_ai_lab.hearts_core.derived import (
    derived_signal,
    sample_offsets,
    timestamp_positions,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import Built
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import REPO, REVISION
from timenet_connectors.sources.huggingface_hub import hub_snapshot


class HeartsCgmacrosConnector(BaseConnector[Path]):
    """The CGMacros sub-benchmark, independently buildable and usable."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch the pinned case selection and benchmark annotations.

        Returns:
            The frozen release root.
        """
        return [hub_snapshot(REPO, REVISION, cache_dir, ("cgmacros/*/*.pkl",))]

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: PLR6301
        """Select and transform parent recordings, then attach HEARTS tasks.

        Returns:
            A child dataset with independent transformed values.

        """
        parent = context.parent("physionet/cgmacros")
        parents = {
            record.id: record
            for record in parent.iter_records(
                record_id for record_id in parent.record_ids() if not record_id.startswith("cgmacros-photo-")
            )
        }
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.register_annotations(VOCABULARIES.values())
        for case in iter_cases(raw_refs[0], TASKS):
            records = case_records(case)
            for built in records:
                _derive_inputs(case, built, parents)
                dataset.add_record(record=built.record)
            task = convert_case(dataset, case, records)
            if isinstance(task, (ForecastingTask, TSEditingTask)):
                _derive_target(case, records[0], task, parents)
            dataset.add_tasks(tasks=[task])
        return dataset


def _derive_target(case: Case, built: Built, task: ForecastingTask | TSEditingTask, parents: dict[str, Record]) -> None:
    """Derive held-out target values from the unmasked parent recording."""
    target = cast("tuple[Record, ...]", task.targets)[0]
    parent = parents[f"cgmacros-{case.payload['subject_id']}"]
    target.sources[0].signals = tuple(_signal(case, built, signal, parent, mask=False) for signal in target.signals)


def _derive_inputs(case: Case, built: Built, parents: dict[str, Record]) -> None:
    name = built.record.id.removeprefix(f"{case.id}-")
    subject = case.payload.get(f"{name}_subject", case.payload.get("subject_id"))
    parent = parents[f"cgmacros-{subject}"]
    built.record.metadata.update({"parent_dataset": "physionet/cgmacros@1.0.0", "parent_record": parent.id})
    for source in built.record.sources:
        if source.name == "image_mapping":
            # These are separately published benchmark assets, not copies of a parent signal.
            source.metadata["source"] = f"{REPO}@{REVISION}/{case.directory}/{case.index}.pkl"
            continue
        source.signals = tuple(_signal(case, built, signal, parent, mask=True) for signal in source.signals)


def _signal(case: Case, built: Built, expected: Signal, parent: Record, *, mask: bool) -> Signal:
    column = "Libre GL" if expected.name == "CGM (mg/dL)" else expected.name
    original = next(signal for signal in parent.signals if signal.name == column)
    origin = moment_us(parent.metadata["recording_start_local"])
    timestamps = sample_offsets(expected) + built.origin_us
    if case.task == "iauc_calculation":
        timestamps += moment_us(case.payload["meal_time"])
    positions = timestamp_positions(original, timestamps, origin_us=origin)
    masked = np.zeros(len(positions), dtype=bool)
    if mask and "mask_start" in case.payload and column == "Libre GL":
        masked = (timestamps >= moment_us(case.payload["mask_start"])) & (
            timestamps <= moment_us(case.payload["mask_end"])
        )

    def load() -> pa.Array:
        values = original.to_arrow().take(pa.array(positions)).to_numpy(zero_copy_only=False).copy()
        values[masked] = 0
        return pa.array(values, type=pa.float64())

    return derived_signal(expected, original, load, parent="physionet/cgmacros@1.0.0")


CONNECTOR = HeartsCgmacrosConnector
