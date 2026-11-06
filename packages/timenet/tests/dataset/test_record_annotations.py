from collections import UserList
from fractions import Fraction

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimeSeriesSpec, read_value, ureg


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _record() -> Record:
    signal = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[1.0, 2.0])
    return Record(sources=(Source(name="device", signals=(signal,)),))


def test_add_annotations_takes_a_mapping_of_keys_to_values():
    record = _record()
    attached = record.add_annotations({"sensor_location": "wrist", "age": 54, "tags": ["a", "b"], "pair": ("c", "d")})
    assert [(one.key, read_value(one.value)) for one in record.annotations] == [
        ("sensor_location", "wrist"),
        ("age", 54),
        ("tags", ["a", "b"]),
        ("pair", ["c", "d"]),
    ]
    assert record.annotations == attached


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ({"a": 1, "b": {"nested": 1}}, "annotation 'b': unsupported annotation value type"),
        ({"a": 1, 3: "x"}, "key must be text"),
        ({"tags": UserList(["a", "b"])}, "unsupported annotation value type: UserList"),
        ("ab", "not str"),
        (b"ab", "not bytes"),
        (None, "not NoneType"),
        (Annotation(key="a", value=1), "attach a single one with annotate"),
        ({"age": Annotation(key="age", value=1)}, r"pass mapping\.values\(\) or a list"),
        ([Annotation(key="a", value=1), "b"], "must be an Annotation; got str"),
    ],
)
def test_add_annotations_attaches_nothing_from_a_bad_batch(batch, message):
    record = _record()
    with pytest.raises(TimeFValidationError, match=message):
        record.add_annotations(batch)
    assert record.annotations == ()


def test_a_rejected_mapping_value_names_its_key_once():
    with pytest.raises(TimeFValidationError) as caught:
        _record().add_annotations({"k": None})  # ty: ignore[invalid-argument-type]
    assert str(caught.value).count("'k'") == 1


def test_errors_from_the_callers_iterator_pass_through():
    def broken():
        yield Annotation(key="a", value=1)
        raise KeyError("source row")

    with pytest.raises(KeyError, match="source row"):
        _record().add_annotations(broken())
