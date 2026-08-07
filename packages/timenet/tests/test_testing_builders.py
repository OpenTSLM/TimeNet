import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.testing import (
    CountingLoader,
    assert_datasets_equal,
    make_dataset,
    sine_loader,
)
from timenet.types import Annotation


def test_counting_loader_counts_and_returns():
    loader = CountingLoader([1.0, 2.0, 3.0])
    assert loader.calls == 0
    arr = loader()
    assert loader.calls == 1
    assert arr.to_pylist() == [1.0, 2.0, 3.0]


def test_sine_loader_is_deterministic():
    a = sine_loader(n=16, freq_hz=1.0, sampling_rate_hz=16.0)()
    b = sine_loader(n=16, freq_hz=1.0, sampling_rate_hz=16.0)()
    assert a.equals(b)
    assert np.asarray(a).dtype == np.float32


def test_make_dataset_builds_a_valid_dataset():
    ds = make_dataset()
    assert isinstance(ds, TimeFDataset)
    assert len(ds.samples) >= 1


def test_assert_datasets_equal_reflexive():
    assert_datasets_equal(make_dataset(), make_dataset())


def test_assert_datasets_equal_detects_difference():
    ds = make_dataset()
    other = make_dataset()
    other.samples[0].add_annotation(Annotation(key="extra", value=1))
    with pytest.raises(AssertionError):
        assert_datasets_equal(ds, other)
