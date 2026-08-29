"""Each format writes what its own library gave it, and reads every value back in."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluations.errors import EvaluationError
from evaluations.formats import timef as timef_module
from evaluations.formats.base import directory_size
from evaluations.formats.pandas_ import PandasFormat
from evaluations.formats.timef import DATASET_ID, TimeFFormat
from evaluations.formats.torch_ import TorchFormat
from evaluations.pyhealth_loader import (
    LABEL,
    PATIENT,
    SIGNAL,
    SUBJECT_TABLES,
    _release_root,
    load_frame,
    signal_stack,
)
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import DatasetMetadata, License, TimeSeriesSpec, Version, ureg


N_EPOCHS = 4
N_CHANNELS = 3
N_SAMPLES = 300

# The two series of the stub dataset. They differ in length, as the channels of one recording do
# when they were sampled at different rates.
STUB_LENGTHS = (300, 120)
EPOCH_SECONDS = 30


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


@pytest.fixture(params=[PandasFormat, TorchFormat], ids=["pandas", "torch"])
def frame_format(request):
    """The two formats that store a frame. PyHealth reads it; these write what it read."""
    return request.param()


def test_a_frame_format_round_trips_every_value(frame_format, frame, tmp_path: Path) -> None:
    artifact = frame_format._write_frame(frame, tmp_path / frame_format.name)
    restored = frame_format.read_all(artifact.path)

    assert len(restored) == N_EPOCHS
    np.testing.assert_allclose(np.stack(restored), signal_stack(frame), rtol=0, atol=0)


def test_a_frame_format_reports_its_size(frame_format, frame, tmp_path: Path) -> None:
    artifact = frame_format._write_frame(frame, tmp_path / frame_format.name)

    assert artifact.format == frame_format.name
    assert artifact.path.exists()
    assert artifact.size_bytes > 0


# What the stub was asked for, so a test can prove the format named the dataset and passed the
# source through rather than resolving either of its own.
asked: list[str] = []
downloaded: list[Path] = []


class StubConnector:
    """Stands in for the dataset's connector, so these tests need no release and no EDF reader."""

    def download(self, cache_dir: Path) -> list[Path]:
        downloaded.append(cache_dir)
        return [cache_dir]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        assert raw_refs
        dataset = TimeFDataset(
            metadata=DatasetMetadata(
                dataset_id=DATASET_ID,
                dataset_version=Version(1, 0, 0),
                name="stub",
                description="A stub of the dataset's connector, for these tests.",
                license=License.ODBL_1_0,
            )
        )
        spec = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.microvolt)
        generator = np.random.default_rng(0)
        dataset.add_sample(
            time_series=tuple(
                TimeSeries.from_values(
                    generator.standard_normal(length).astype(np.float32),
                    spec=spec,
                    channel=f"ch{index}",
                    time_axis=RegularAxis.from_rate_hz(Fraction(length, EPOCH_SECONDS)),
                )
                for index, length in enumerate(STUB_LENGTHS)
            ),
            subject_ids=("SC400",),
            sample_id="stub-0",
        )
        return dataset


def _resolve(dataset_id: str) -> type[StubConnector]:
    asked.append(dataset_id)
    return StubConnector


@pytest.fixture
def stub_connector(monkeypatch) -> None:
    asked.clear()
    downloaded.clear()
    monkeypatch.setattr(timef_module, "resolve", _resolve)


def test_timef_converts_with_the_connector_of_the_dataset(stub_connector, tmp_path: Path) -> None:
    release = tmp_path / "release"

    TimeFFormat().write(release, tmp_path / "timef")

    assert asked == [DATASET_ID]
    assert downloaded == [release]


def test_timef_round_trips_every_series(stub_connector, tmp_path: Path) -> None:
    fmt = TimeFFormat()
    artifact = fmt.write(tmp_path / "release", tmp_path / "timef")
    restored = fmt.read_all(artifact.path)

    assert [len(series) for series in restored] == list(STUB_LENGTHS)


def test_timef_reports_its_size(stub_connector, tmp_path: Path) -> None:
    artifact = TimeFFormat().write(tmp_path / "release", tmp_path / "timef")

    assert artifact.format == "timef"
    assert artifact.path.is_dir()
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


def test_a_source_without_a_release_raises(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="holds no release"):
        load_frame(tmp_path)


def test_the_release_root_is_the_directory_holding_both_tables(tmp_path: Path) -> None:
    root = tmp_path / "sleep-edfx-1.0.0"
    root.mkdir()
    for table in SUBJECT_TABLES:
        (root / table).touch()

    assert _release_root(tmp_path) == root


def test_a_release_missing_one_study_table_raises(tmp_path: Path) -> None:
    root = tmp_path / "sleep-edfx-1.0.0"
    root.mkdir()
    (root / SUBJECT_TABLES[0]).touch()

    with pytest.raises(EvaluationError, match="holds no release"):
        _release_root(tmp_path)
