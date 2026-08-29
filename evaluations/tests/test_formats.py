"""Every format must round-trip the frame it was given."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluations.errors import EvaluationError
from evaluations.formats.base import directory_size
from evaluations.formats.pandas_ import PandasFormat
from evaluations.formats.timef import TimeFFormat
from evaluations.formats.torch_ import TorchFormat
from evaluations.source import LABEL, PATIENT, SIGNAL, load_frame, signal_stack


N_EPOCHS = 4
N_CHANNELS = 3
N_SAMPLES = 300


@pytest.fixture
def frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    signals = generator.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES)).astype(np.float32)
    return pd.DataFrame(
        {
            SIGNAL: list(signals),
            LABEL: ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage R"],
            PATIENT: ["SC400", "SC400", "SC401", "SC401"],
        }
    )


@pytest.fixture(params=[PandasFormat, TorchFormat, TimeFFormat], ids=["pandas", "torch", "timef"])
def fmt(request):
    return request.param()


def test_round_trip_preserves_values(fmt, frame, tmp_path: Path) -> None:
    artifact = fmt.write(frame, tmp_path / fmt.name)
    restored = fmt.read_all(artifact.path)

    assert restored.shape == (N_EPOCHS, N_CHANNELS, N_SAMPLES)
    np.testing.assert_allclose(restored, signal_stack(frame), rtol=0, atol=0)


def test_artifact_reports_its_size(fmt, frame, tmp_path: Path) -> None:
    artifact = fmt.write(frame, tmp_path / fmt.name)

    assert artifact.format == fmt.name
    assert artifact.path.exists()
    assert artifact.size_bytes > 0


def test_directory_size_sums_every_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 25)

    assert directory_size(tmp_path) == 35
    assert directory_size(tmp_path / "a.bin") == 10


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="does not exist"):
        load_frame(tmp_path / "absent")
