import struct
import wave

from pydantic import ValidationError
import pytest

from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.epfl.coughvid.connector import CoughvidConnector, RecordingMetadata


IDS = (
    "00014dcc-0f06-4c27-8c7b-737b18a2cf4c",
    "00039425-7f3a-42aa-ac13-834aaa2b6b92",
    "0007c6f1-5441-40e6-9aaf-a761d8f2da3b",
)


@pytest.fixture
def originals(tmp_path):
    (tmp_path / "metadata_compiled.csv").write_text(
        "uuid,status,status_SSL,diagnosis_1\n"
        + "\n".join(f"{identity},healthy,healthy,healthy_cough" for identity in IDS)
    )
    with wave.open(str(tmp_path / f"{IDS[0]}.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(struct.pack("<6h", 8192, -16384, 0, 16384, -8192, 0))
    (tmp_path / f"{IDS[1]}.webm").write_bytes(b"invalid webm")
    return tmp_path


def test_preserves_native_audio_labels_and_unavailable_records(originals, tmp_path):
    connector = CoughvidConnector()
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(connector.convert([originals]), values_backend="zarr")
    with registry.open_reader("epfl/coughvid", "1.0.0") as reader:
        records = {record.id: record for record in reader.iter_records()}
        recorded = records[f"coughvid-{IDS[0]}"]
        assert recorded.metadata["diagnosis_1"] == "healthy_cough"
        assert recorded.signals[0].to_numpy().tolist() == [[0.25, -0.5], [0, 0.5], [-0.25, 0]]
        assert records[f"coughvid-{IDS[1]}"].metadata["audio_status"] == "undecodable"
        assert not records[f"coughvid-{IDS[1]}"].signals
        assert records[f"coughvid-{IDS[2]}"].metadata["audio_status"] == "missing"


def test_duplicate_audio_is_rejected(originals):
    (originals / f"{IDS[0]}.ogg").write_bytes(b"duplicate")
    with pytest.raises(TimeFFormatError, match="multiple"):
        CoughvidConnector().convert([originals])


def test_invalid_id_is_rejected():
    with pytest.raises(ValidationError):
        RecordingMetadata.model_validate({"uuid": "../not-a-recording"})
