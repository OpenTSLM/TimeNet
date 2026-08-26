import pint
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ureg
from timenet.types.units import normalize_unit


def test_standard_units_available():
    assert ureg.hertz.dimensionality == ureg.Unit("1/second").dimensionality
    assert ureg.millivolt.dimensionality == ureg.volt.dimensionality


def test_conversion():
    assert (5.0 * ureg.millivolt).to(ureg.volt).magnitude == pytest.approx(0.005)


def test_bpm_custom_unit():
    # bpm is beats per minute; 60 bpm == 1 beat/second.
    assert (60.0 * ureg.bpm).to(ureg.beat / ureg.second).magnitude == pytest.approx(1.0)


def test_brpm_custom_unit():
    # brpm is breaths per minute; 60 brpm == 1 breath/second.
    assert (60.0 * ureg.brpm).to(ureg.breath / ureg.second).magnitude == pytest.approx(1.0)


def test_incompatible_conversion_raises():
    with pytest.raises(pint.DimensionalityError):
        (5.0 * ureg.millivolt).to(ureg.hertz)


def test_single_shared_registry():
    # Units built from the shared registry compare/convert cleanly; this is load-bearing
    # for the dimensionality checks on TimeSeriesSpec.
    assert ureg.hertz._REGISTRY is ureg


def test_import_does_not_touch_the_application_registry():
    # A library must not reassign pint's process-wide registry behind the host's back; TimeNet's own
    # types pickle units by name instead, so nothing here needs the global.
    assert pint.get_application_registry().get() is not ureg


@pytest.mark.parametrize("bad_unit", ["/min", "10*3/uL", "not_a_unit"])
def test_normalize_unit_rejects_malformed_and_unknown_strings(bad_unit):
    with pytest.raises(TimeFValidationError, match="unknown unit"):
        normalize_unit(bad_unit)


def test_normalize_unit_passes_a_valid_string_through():
    assert normalize_unit("millivolt") == "millivolt"


def test_normalize_unit_converts_pint_unit_to_string():
    assert normalize_unit(ureg.millivolt) == "millivolt"


def test_normalize_unit_passes_none_through():
    assert normalize_unit(None) is None


def test_use_as_application_registry_is_opt_in_and_works():
    # The escape hatch for bare pint.Unit / Quantity pickling, which no per-type __getstate__ covers.
    import pickle  # noqa: PLC0415

    from timenet.types import use_as_application_registry  # noqa: PLC0415

    previous = pint.get_application_registry().get()
    try:
        use_as_application_registry()
        assert pickle.loads(pickle.dumps(ureg.bpm)) == ureg.bpm
        quantity = pickle.loads(pickle.dumps(60.0 * ureg.bpm))
        assert quantity.magnitude == pytest.approx(60.0)
        assert quantity.units == ureg.bpm
    finally:
        pint.set_application_registry(previous)
