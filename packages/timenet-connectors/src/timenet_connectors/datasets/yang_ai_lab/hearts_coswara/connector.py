"""Derive Coswara benchmark inputs from original recordings and participant metadata."""

from pathlib import Path
from typing import cast

import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import Annotation
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, iter_cases
from timenet_connectors.datasets.yang_ai_lab.hearts_core.derived import derived_signal
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import waveform_record
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import REPO, REVISION, Waveform
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.release import (
    AUDIO_QUALITY_DESCRIPTION,
    TASKS,
    VOCABULARIES,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.tasks import convert_case
from timenet_connectors.sources.huggingface_hub import hub_snapshot


class HeartsCoswaraConnector(BaseConnector[Path]):
    """Keep Coswara originals separate from the HEARTS selection and labels."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch pinned benchmark selections and annotations.

        Returns:
            The frozen release root.
        """
        return [hub_snapshot(REPO, REVISION, cache_dir, ("coswara/*/*.pkl",))]

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: PLR6301
        """Create child-owned mono recordings and the five Coswara task types.

        Returns:
            The benchmark child, with provenance for waveform and symptoms-only cases.

        """
        root = raw_refs[0]
        requested = set()
        for case in iter_cases(root, TASKS):
            participant = f"coswara-{case.payload['subject_id']}"
            requested.add(participant)
            if "data" in case.payload:
                requested.add(f"{participant}-{case.payload['audio_type']}")
        parents = {record.id: record for record in context.parent("iiscleap/coswara").iter_records(sorted(requested))}
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.register_annotations(VOCABULARIES.values())
        for case in iter_cases(root, TASKS):
            participant = parents[f"coswara-{case.payload['subject_id']}"]
            _check_symptoms(case, participant)
            inputs = ()
            if case.definition.inputs:
                original = parents[f"{participant.id}-{case.payload['audio_type']}"]
                record = _audio_record(case, original)
                dataset.add_record(record=record)
                inputs = (record,)
            task = convert_case(case, inputs)
            task.metadata.update({"parent_dataset": "iiscleap/coswara@1.0.0", "parent_participant": participant.id})
            dataset.add_tasks(tasks=[task])
        return dataset


def _check_symptoms(case: Case, participant: Record) -> None:
    """Check benchmark symptoms against the original participant metadata.

    Raises:
        TimeFFormatError: If a symptom differs from the source metadata.
    """
    for key, expected in case.payload.get("symptoms", {}).items():
        present = str(participant.metadata[key]).lower() in {"true", "y", "1"}
        if present != bool(expected):
            raise TimeFFormatError(f"{case.id}: symptom {key} differs from the original participant metadata")


def _audio_record(case: Case, original: Record) -> Record:
    """Build a mono benchmark recording with its audio quality annotation.

    Returns:
        The child-owned audio record.

    Raises:
        TimeFFormatError: If the original recording has no audio.
    """
    if not original.signals:
        raise TimeFFormatError(f"{case.id}: original Coswara recording has no audio")
    waveform = cast("Waveform", case.definition.inputs[0])
    record = waveform_record(case, waveform, subject=case.payload.get("subject_id"))
    record.annotate(
        Annotation(
            key="audio_quality",
            value=int(case.at(("data", "quality"))),
            description=AUDIO_QUALITY_DESCRIPTION,
        )
    )
    record.sources[0].signals = (_audio(record.signals[0], original.signals[0]),)
    record.metadata.update({"parent_dataset": "iiscleap/coswara@1.0.0", "parent_record": original.id})
    return record


def _audio(expected: Signal, original: Signal) -> Signal:
    if (
        not isinstance(expected.time_axis, RegularAxis)
        or not isinstance(original.time_axis, RegularAxis)
        or expected.time_axis.period_us != original.time_axis.period_us
    ):
        raise TimeFFormatError(f"{expected.id}: HEARTS audio rate differs from the original Coswara recording")

    def load() -> pa.Array:
        values = original.to_numpy()
        return pa.array(values.mean(axis=1) if original.spec.value_shape else values)

    return derived_signal(expected, original, load, parent="iiscleap/coswara@1.0.0")


CONNECTOR = HeartsCoswaraConnector
