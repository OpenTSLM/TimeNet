import pytest

from timenet.errors import TimeFValidationError
from timenet.types import Label
from timenet.types._fields import as_tuple


def test_as_tuple_keeps_the_order_of_the_items():
    assert as_tuple(["b", "a"]) == ("b", "a")
    assert as_tuple(item for item in (1, 2)) == (1, 2)


@pytest.mark.parametrize("value", ["ab", {"a": 1}, 3])
def test_as_tuple_rejects_a_string_a_mapping_and_a_value_that_is_not_iterable(value):
    with pytest.raises(TimeFValidationError, match="iterable of items"):
        as_tuple(value)


def test_as_tuple_rejects_a_single_item_of_the_item_type_also_when_it_can_be_iterated():
    label = Label(name="yes")
    assert len(tuple(label)) > 0
    with pytest.raises(TimeFValidationError, match="got Label"):
        as_tuple(label, item_type=Label)
    assert as_tuple([label], item_type=Label) == (label,)
