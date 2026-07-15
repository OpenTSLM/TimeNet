from dataclasses import dataclass, replace
import pickle

import pint
import pytest

from timenet.errors import TimeFValidationError
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


def test_pickled_units_rebind_to_the_shared_registry():
    # Units pickle as names and are rebuilt against `ureg`, so a spec does not depend on whatever
    # registry happens to be process-global. Without that, comparing the two raises
    # "Cannot operate with Unit and Unit of different registries".
    restored = pickle.loads(pickle.dumps(_ecg_spec()))
    assert restored.unit_value._REGISTRY is ureg
    assert restored.unit_value == ureg.millivolt


def test_pickle_round_trips_a_custom_unit():
    # `bpm` exists only in `ureg`; resolving it against pint's default registry raises
    # UndefinedUnitError, so this is the case that fails hardest if units pickle registry-bound.
    spec = _ecg_spec(unit_value=ureg.bpm)
    assert pickle.loads(pickle.dumps(spec)).unit_value == ureg.bpm


def test_spec_rejects_non_frequency_sampling_rate():
    with pytest.raises(TimeFValidationError, match="sampling"):
        _ecg_spec(unit_sampling_rate=ureg.volt)


def test_spec_rejects_non_time_timestamp():
    with pytest.raises(TimeFValidationError, match="timestamp"):
        _ecg_spec(unit_timestamp=ureg.volt)


def test_spec_value_unit_unconstrained():
    # Any unit is a valid channel value unit (mV, g, bpm, dimensionless, ...).
    assert _ecg_spec(unit_value=ureg.dimensionless).unit_value == ureg.dimensionless
    assert _ecg_spec(unit_value=ureg.bpm).unit_value == ureg.bpm


@dataclass(frozen=True)
class _WithGain(TimeSeriesSpec):
    """A spec subclass adding its own unit field (module scope so it pickles)."""

    unit_gain: pint.Unit = ureg.bpm


def test_pickle_round_trips_a_subclass_added_unit_field():
    # Connectors are encouraged to subclass with field defaults. A subclass that adds its own unit
    # field must pickle by name too, or a custom unit in it fails to resolve on unpickle.
    spec = _WithGain(
        spec_type="hr",
        name="HR",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.millivolt,
    )
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.unit_gain == ureg.bpm
    assert restored.unit_gain._REGISTRY is ureg


def test_spec_nd_value_contract():
    spec = _ecg_spec(dtype="uint8", value_shape=(32, 32, 3), dimension_names=("height", "width", "color"))
    assert spec.dtype == "uint8"
    assert spec.value_shape == (32, 32, 3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"spec_type": ""}, "spec_type"),
        ({"dtype": "complex64"}, "dtype"),
        ({"value_shape": (32, 0, 3)}, "positive integers"),
        ({"value_shape": (32, 32, 3), "dimension_names": ("height",)}, "match value_shape"),
        (
            {"value_shape": (32, 32, 3), "dimension_names": ("space", "space", "color")},
            "unique",
        ),
    ],
)
def test_spec_rejects_invalid_nd_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        _ecg_spec(**overrides)
