import dataclasses
from pathlib import Path
import sys

import pytest


# Make sibling test helpers (e.g. ``_fake_registry``) importable by name under
# pytest's importlib import mode, which does not add test dirs to ``sys.path``.
sys.path.insert(0, str(Path(__file__).parent))

from timenet.control_plane import DeclarativeDataset, TimeFWriter
from timenet.testing import make_dataset, make_metadata
from timenet.types import Domain, License, Version


def _hello_world() -> DeclarativeDataset:
    """``timenet/hello-world`` at 1.0.0: general domain, CC-BY-4.0, no tags."""
    dataset = make_dataset(n_records=2, n_values=64)
    dataset.metadata = dataclasses.replace(
        make_metadata(dataset_id="timenet/hello-world"),
        name="Hello World",
        description="A tiny demo dataset.",
        license=License.CC_BY_4_0,
        domains=(Domain.GENERAL,),
    )
    return dataset


def _ecg() -> DeclarativeDataset:
    """``demo/ecg`` at 2.0.0: cardiology, MIT, tagged ``clinical``."""
    dataset = make_dataset(n_records=2, n_values=64)
    dataset.metadata = dataclasses.replace(
        make_metadata(dataset_id="demo/ecg"),
        dataset_version=Version(2, 0, 0),
        name="ECG Dataset",
        description="A clinical ECG dataset.",
        license=License.MIT,
        domains=(Domain.CARDIOLOGY,),
        tags=("clinical",),
    )
    return dataset


@pytest.fixture
def registry_root(tmp_path) -> Path:
    """A local registry directory holding two datasets: timenet/hello-world and demo/ecg."""
    for dataset in (_hello_world(), _ecg()):
        with TimeFWriter(tmp_path, dataset.metadata) as writer:
            writer.write(dataset)
    return tmp_path
