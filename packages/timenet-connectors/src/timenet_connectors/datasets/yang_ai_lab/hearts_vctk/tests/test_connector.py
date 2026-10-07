import pickle

import numpy as np
import pytest
import soxr

from timenet.composition import BuildContext
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet.types import InputModality, TimeSeriesSpec
from timenet_connectors.datasets.cstr.vctk.connector import VctkConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.connector import HeartsVctkConnector, _derive_record
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.release import TASKS


@pytest.fixture
def parent_and_cases(tmp_path):
    values = np.sin(np.arange(4800) / 37).astype(np.float32)
    signal = Signal(
        id="original-audio",
        name="audio",
        data=values,
        time_axis=RegularAxis.from_rate_hz(48000),
        spec=TimeSeriesSpec(
            spec_type="audio", name="Audio", dtype="float32", unit_value=None, modality=InputModality.AUDIO
        ),
    )
    parent = TimeFDataset(metadata=VctkConnector().metadata())
    parent.add_record(
        record=Record(
            record_id="vctk-p225_001_mic1", sources=(Source(id="original-source", name="mic1", signals=(signal,)),)
        )
    )
    second = Signal(id="second-audio", name="audio", data=-values, time_axis=signal.time_axis, spec=signal.spec)
    parent.add_record(
        record=Record(
            record_id="vctk-p225_001_mic2", sources=(Source(id="second-source", name="mic2", signals=(second,)),)
        )
    )
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(parent)
    folder = tmp_path / "cases/vctk/waveform_temporal_direction_detection"
    folder.mkdir(parents=True)
    downsampled = soxr.resample(values, 48000, 16000, quality="HQ")
    for index in (0, 1):
        (folder / f"{index}.pkl").write_bytes(
            pickle.dumps(
                {
                    "speaker_id": "p225",
                    "recording_id": "p225_p225_001",
                    "GT": index,
                    "waveform": downsampled if index == 0 else -downsampled[::-1],
                }
            )
        )
    return registry, tmp_path / "cases", downsampled


def test_resampled_forward_and_reversed_copies_roundtrip(parent_and_cases):
    registry, cases, expected = parent_and_cases
    connector = HeartsVctkConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        child.set_dependencies(context.dependency_lock())
        assert [task.targets for task in child.tasks] == [("forward",), ("reversed",)]
        np.testing.assert_array_equal(child.tasks[0].inputs[0].signals[0].to_numpy(), expected)
        np.testing.assert_array_equal(child.tasks[1].inputs[0].signals[0].to_numpy(), -expected[::-1])
        assert child.tasks[1].inputs[0].metadata["parent_record"] == "vctk-p225_001_mic2"
        registry.store(child)
    with registry.open_reader("cstr/vctk", "1.0.0") as reader:
        assert next(reader.iter_records()).signals[0].n_values == 4800


def test_mismatched_waveform_fails(parent_and_cases):
    registry, cases, expected = parent_and_cases
    path = cases / "vctk/waveform_temporal_direction_detection/0.pkl"
    path.write_bytes(
        pickle.dumps({"speaker_id": "p225", "recording_id": "p225_p225_001", "GT": 0, "waveform": expected + 0.1})
    )
    connector = HeartsVctkConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        child = connector.compose([cases], context)
        with pytest.raises(TimeFFormatError, match="do not reproduce"):
            child.tasks[0].inputs[0].signals[0].to_arrow()


def test_missing_parent_audio_is_rejected(tmp_path):
    directory = "vctk/waveform_temporal_direction_detection"
    case = Case(directory, TASKS[directory], 0, tmp_path / "0.pkl", {})
    original = Record(record_id="vctk-empty", sources=(Source(id="empty-source", name="mic1"),))
    with pytest.raises(TimeFFormatError, match="no audio"):
        _derive_record(case, original, reverse=False)
