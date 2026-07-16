from dataclasses import replace
import pickle

import pytest

from timenet.types import DataSource, TimeSeriesSpec, ureg


def _ecg_spec(**overrides):
    spec = TimeSeriesSpec(
        spec_type="ecg_lead",
        name="ECG Lead",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
    )
    return replace(spec, **overrides) if overrides else spec


def test_data_source_construction():
    ds = DataSource(data_source_type="holter_x", name="Holter Monitor X", provider="Acme")
    assert ds.data_source_type == "holter_x"
    assert ds.provider == "Acme"


def test_data_source_provider_optional():
    assert DataSource(data_source_type="synthetic", name="Synthetic").provider is None


def test_spec_construction_modality_only():
    spec = _ecg_spec()
    assert spec.spec_type == "ecg_lead"
    assert spec.unit_value == ureg.millivolt
    assert spec.data_source is None
    assert not hasattr(spec, "channel")  # channel lives on TimeSeries, not the spec


def test_spec_with_data_source():
    ds = DataSource(data_source_type="holter_x", name="Holter Monitor X")
    assert _ecg_spec(data_source=ds).data_source is ds


def test_spec_frozen():
    with pytest.raises(AttributeError):
        _ecg_spec().spec_type = "x"


def test_spec_equality_and_hash():
    assert _ecg_spec() == _ecg_spec()
    assert len({_ecg_spec(), _ecg_spec()}) == 1


def test_spec_is_picklable():
    # The whole point of descriptors over dynamic synthesis: read-back objects must pickle
    # for multiprocessing DataLoaders.
    spec = _ecg_spec(data_source=DataSource(data_source_type="holter_x", name="Holter"))
    restored = pickle.loads(pickle.dumps(spec))
    assert restored == spec


def test_spec_rejects_non_frequency_sampling_rate():
    with pytest.raises(ValueError, match="sampling"):
        _ecg_spec(unit_sampling_rate=ureg.volt)


def test_spec_rejects_non_time_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        _ecg_spec(unit_timestamp=ureg.volt)


def test_spec_value_unit_unconstrained():
    # Any unit is a valid channel value unit (mV, g, bpm, dimensionless, ...).
    assert _ecg_spec(unit_value=ureg.dimensionless).unit_value == ureg.dimensionless
    assert _ecg_spec(unit_value=ureg.bpm).unit_value == ureg.bpm
