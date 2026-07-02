from hypothesis import given, strategies as st
import pytest

from timenet.types import Version


def test_str_roundtrip():
    assert str(Version(1, 2, 3)) == "1.2.3"


def test_parse_roundtrip():
    assert Version.parse("1.2.3") == Version(1, 2, 3)


def test_ordering():
    assert Version(1, 2, 0) > Version(1, 1, 9)
    assert Version(1, 0, 0) < Version(2, 0, 0)


def test_frozen():
    v = Version(1, 0, 0)
    with pytest.raises(AttributeError):
        v.major = 2  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("bad", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_rejects_negative(bad):
    with pytest.raises(ValueError):
        Version(*bad)


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "1.x.0", "", "1.2.-1"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        Version.parse(bad)


@given(st.integers(0, 999), st.integers(0, 999), st.integers(0, 999))
def test_parse_str_roundtrip_property(major, minor, patch):
    v = Version(major, minor, patch)
    assert Version.parse(str(v)) == v
