import pint
import pytest

from timenet.types import ureg


def test_standard_units_available():
    assert ureg.hertz.dimensionality == ureg.Unit("1/second").dimensionality
    assert ureg.millivolt.dimensionality == ureg.volt.dimensionality


def test_conversion():
    assert (5.0 * ureg.millivolt).to(ureg.volt).magnitude == pytest.approx(0.005)


def test_bpm_custom_unit():
    # bpm is beats per minute; 60 bpm == 1 beat/second.
    assert (60.0 * ureg.bpm).to(ureg.beat / ureg.second).magnitude == pytest.approx(1.0)


def test_incompatible_conversion_raises():
    with pytest.raises(pint.DimensionalityError):
        (5.0 * ureg.millivolt).to(ureg.hertz)


def test_single_shared_registry():
    # Units built from the shared registry compare/convert cleanly; this is load-bearing
    # for the dimensionality checks on TimeSeriesSpec.
    assert ureg.hertz._REGISTRY is ureg
