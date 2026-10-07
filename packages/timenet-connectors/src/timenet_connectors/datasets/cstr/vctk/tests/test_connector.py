import numpy as np
from pydantic import ValidationError
import pytest
import soundfile as sf

from timenet.registry import LocalRegistry
from timenet_connectors.datasets.cstr.vctk.connector import Speaker, VctkConnector


def test_keeps_both_microphones_and_transcript(tmp_path):
    (tmp_path / "speaker-info.txt").write_text(
        "ID AGE GENDER ACCENT REGION COMMENTS\np225 23 F English Southern England\n"
    )
    audio = tmp_path / "wav48_silence_trimmed/p225"
    audio.mkdir(parents=True)
    for mic in (1, 2):
        sf.write(audio / f"p225_001_mic{mic}.flac", np.array([0.25, -0.5, 0]), 48000)
    text = tmp_path / "txt/p225"
    text.mkdir(parents=True)
    (text / "p225_001.txt").write_text("A test recording.\n")
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(VctkConnector().convert([tmp_path]), values_backend="zarr")
    with registry.open_reader("cstr/vctk", "1.0.0") as reader:
        records = tuple(reader.iter_records())
        assert len(records) == 2
        assert {record.metadata["microphone"] for record in records} == {"mic1", "mic2"}
        assert records[0].metadata["transcript"] == "A test recording."
        assert records[0].signals[0].to_numpy().tolist() == [0.25, -0.5, 0]


def test_invalid_speaker_is_rejected():
    with pytest.raises(ValidationError):
        Speaker.model_validate({"id": "p225", "age": -1, "gender": "F", "accent": "English"})
