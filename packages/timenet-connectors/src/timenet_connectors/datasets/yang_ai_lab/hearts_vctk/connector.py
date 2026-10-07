"""Select the original VCTK microphone, then resample and optionally reverse it."""

import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
from pydantic import TypeAdapter

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, iter_cases
from timenet_connectors.datasets.yang_ai_lab.hearts_core.derived import derived_signal
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import waveform_record
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import REPO, REVISION, Waveform
from timenet_connectors.datasets.yang_ai_lab.hearts_core.tasks import classification_task
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.release import PROMPTS, TASKS, VOCABULARIES
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_DIRECTION = TypeAdapter(Literal[0, 1])
TARGET_RATE = 16000
# HEARTS omits the microphone. Every entry was verified against both original recordings at
# the pinned release; the lazy loader checks the selected waveform again during each build.
MICROPHONES = TypeAdapter(dict[int, Literal["mic1", "mic2"]]).validate_json(
    Path(__file__).with_name("microphones.json").read_text()
)


class HeartsVctkConnector(BaseConnector[Path]):
    """Child-owned 16 kHz speech with the benchmark's temporal-direction task."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch pinned benchmark selections and direction labels.

        Returns:
            The frozen release root.
        """
        return [hub_snapshot(REPO, REVISION, cache_dir, ("vctk/*/*.pkl",))]

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: PLR6301
        """Select the verified source microphone, resample, and reverse selected cases.

        Returns:
            The direction-detection benchmark child.

        """
        root = raw_refs[0]
        selected = {}
        for case in iter_cases(root, TASKS):
            utterance = str(case.payload["recording_id"]).removeprefix(str(case.payload["speaker_id"]) + "_")
            selected[case.id] = f"vctk-{utterance}_{MICROPHONES[case.index]}"
        parents = {
            record.id: record for record in context.parent("cstr/vctk").iter_records(sorted(set(selected.values())))
        }
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.register_annotations(VOCABULARIES.values())
        for case in iter_cases(root, TASKS):
            original = parents[selected[case.id]]
            direction = _DIRECTION.validate_python(case.payload["GT"])
            record = _derive_record(case, original, reverse=direction == 1)
            dataset.add_record(record=record)
            prompt = PROMPTS[case.directory].substitute(speaker=str(case.payload["speaker_id"]))
            task = classification_task(
                case, (record,), prompt, VOCABULARIES[case.directory], label=case.definition.options[direction]
            )
            dataset.add_tasks(tasks=[task])
        return dataset


def _derive_record(case: Case, original: Record, *, reverse: bool) -> Record:
    """Build a transformed microphone recording with source provenance.

    Returns:
        The child-owned speech record.

    Raises:
        TimeFFormatError: If the original recording has no audio.
    """
    if not original.signals:
        raise TimeFFormatError(f"{case.id}: original VCTK recording has no audio")
    waveform = cast("Waveform", case.definition.inputs[0])
    record = waveform_record(case, waveform, subject=case.payload.get("speaker_id"))
    record.sources[0].signals = (_audio(record.signals[0], original.signals[0], reverse=reverse),)
    record.metadata.update(
        {
            "parent_dataset": "cstr/vctk@1.0.0",
            "parent_record": original.id,
            "recording_id": str(case.payload["recording_id"]),
        }
    )
    return record


def _audio(expected: Signal, original: Signal, *, reverse: bool) -> Signal:
    if not isinstance(original.time_axis, RegularAxis) or original.spec.value_shape:
        raise TimeFFormatError(f"{original.id}: VCTK microphone must contain regularly sampled mono audio")
    source_rate = float(1_000_000 / original.time_axis.period_us)

    def load() -> pa.Array:
        import soxr  # noqa: PLC0415 - connector dependency

        values = original.to_numpy()
        length = math.ceil(len(values) * TARGET_RATE / source_rate)
        resampled = soxr.resample(values, source_rate, TARGET_RATE, quality="HQ")
        # Match librosa's ceil-length convention used by the released benchmark.
        resampled = np.pad(resampled[:length], (0, max(0, length - len(resampled))))
        return pa.array(resampled[::-1] if reverse else resampled, type=pa.float32())

    # Float32 resampler implementations can differ by a few units of machine precision.
    signal = derived_signal(
        expected, original, load, parent="cstr/vctk@1.0.0", atol=4 * float(np.finfo(np.float32).eps)
    )
    signal.metadata.update(resampler="soxr HQ", sample_rate=TARGET_RATE, reversed=reverse)
    return signal


CONNECTOR = HeartsVctkConnector
