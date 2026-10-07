"""Derive HEARTS COUGHVID waveforms from original recordings."""

from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import iter_cases
from timenet_connectors.datasets.yang_ai_lab.hearts_core.derived import derived_signal
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import waveform_record
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import REPO, REVISION, Waveform
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.release import TASKS, VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.tasks import convert_case
from timenet_connectors.sources.huggingface_hub import hub_snapshot


PCM16_SCALE = 32768


class HeartsCoughvidConnector(BaseConnector[Path]):
    """The six COUGHVID benchmark tasks over child-owned mono waveforms."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch pinned benchmark selections and annotations.

        Returns:
            The frozen release root.
        """
        return [hub_snapshot(REPO, REVISION, cache_dir, ("coughvid/*/*.pkl",))]

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: PLR6301
        """Select original recordings and reproduce the benchmark's audio conversion.

        Returns:
            The COUGHVID child, with the published tasks and labels.

        """
        root = raw_refs[0]
        requested = {f"coughvid-{case.payload['subject_id']}" for case in iter_cases(root, TASKS)}
        parents = {record.id: record for record in context.parent("epfl/coughvid").iter_records(sorted(requested))}
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.register_annotations(VOCABULARIES.values())
        for case in iter_cases(root, TASKS):
            original = parents[f"coughvid-{case.payload['subject_id']}"]
            waveform = cast("Waveform", case.definition.inputs[0])
            record = waveform_record(case, waveform, subject=case.payload.get("subject_id"))
            record.sources[0].signals = (_audio(record.signals[0], original),)
            record.metadata.update({"parent_dataset": "epfl/coughvid@1.0.0", "parent_record": original.id})
            dataset.add_record(record=record)
            dataset.add_tasks(tasks=[convert_case(case, (record,))])
        return dataset


def _audio(expected: Signal, record: Record) -> Signal:
    if not record.signals:
        raise TimeFFormatError(f"{expected.id}: original COUGHVID recording has no decodable audio")
    original = record.signals[0]
    if (
        not isinstance(expected.time_axis, RegularAxis)
        or not isinstance(original.time_axis, RegularAxis)
        or expected.time_axis.period_us != original.time_axis.period_us
    ):
        raise TimeFFormatError(f"{expected.id}: HEARTS audio rate differs from the original COUGHVID recording")

    def load() -> pa.Array:
        values = original.to_numpy()
        if str(record.metadata["filename"]).endswith((".webm", ".ogg")):
            values = np.clip(np.rint(values * PCM16_SCALE), -PCM16_SCALE, PCM16_SCALE - 1) / PCM16_SCALE
        if original.spec.value_shape:
            values = values.mean(axis=1)
        return pa.array(values, type=pa.float32())

    # HEARTS used a PCM16 decoder. Decoder versions can round differently by one PCM16 step.
    signal = derived_signal(expected, original, load, parent="epfl/coughvid@1.0.0", atol=1 / PCM16_SCALE)
    signal.metadata.update(transform="mono; PCM16 for WebM and Ogg", comparison_atol=1 / PCM16_SCALE)
    return signal


CONNECTOR = HeartsCoughvidConnector
