"""Validation and immutable copies of metadata documented as JSON-compatible."""

from typing import Annotated, NoReturn, cast

from pydantic import (
    BeforeValidator,
    ConfigDict,
    JsonValue,
    StrictStr,
    TypeAdapter,
)
from typing_extensions import TypeAliasType

from timenet.errors import TimeFValidationError


def _reject_float_subclasses(value: object) -> object:
    """Reject float subclasses before Pydantic converts them to built-in floats.

    Returns:
        The original value for Pydantic to validate and copy.

    Raises:
        TimeFValidationError: If any nested value is a float subclass.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, float) and type(item) is not float:
            raise TimeFValidationError("must be a built-in float")
        if isinstance(item, (list, dict)) and id(item) not in seen:
            # Leave cycles intact so Pydantic reports them as validation errors.
            seen.add(id(item))
            pending.extend(item.values() if isinstance(item, dict) else item)
    return value


JsonMapping = TypeAliasType("JsonMapping", dict[StrictStr, JsonValue])
"""A strict JSON object with string keys."""

_JSON_CONFIG = ConfigDict(strict=True, allow_inf_nan=False)
_VALUE_ADAPTER = TypeAdapter(Annotated[JsonValue, BeforeValidator(_reject_float_subclasses)], config=_JSON_CONFIG)
_MAPPING_ADAPTER = TypeAdapter(Annotated[JsonMapping, BeforeValidator(_reject_float_subclasses)], config=_JSON_CONFIG)


def validate_json_value(value: object) -> JsonValue:
    """Validate and copy one strict JSON value.

    Args:
        value: Candidate JSON value.

    Returns:
        A detached hierarchy containing only JSON-native values.
    """
    return _VALUE_ADAPTER.validate_python(value)


def validate_json_mapping(value: object) -> JsonMapping:
    """Validate and copy a string-keyed JSON object.

    Args:
        value: Candidate metadata mapping.

    Returns:
        A detached JSON-native dictionary.
    """
    if type(value) is dict and not value:
        return {}
    return cast("JsonMapping", _MAPPING_ADAPTER.validate_python(value))


def _refuse_change(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse a change to immutable JSON metadata.

    Raises:
        TypeError: Always.
    """
    raise TypeError("JSON metadata cannot change; create a new mapping instead")


# Keep JSON-native subclasses so json.dumps accepts objects and arrays and equality with the
# corresponding dict and list is preserved. MappingProxyType and tuples do not provide both.
class _FrozenJsonObject(dict):
    """A JSON object that cannot change. It compares equal to a ``dict``, and ``json.dumps`` writes it as one."""

    __slots__ = ()

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse_change

    def __reduce__(self) -> tuple[type, tuple[dict]]:
        # Unpickling a dict subclass would otherwise set each item, which this class refuses.
        return (_FrozenJsonObject, (dict(self),))


class _FrozenJsonArray(list):
    """A JSON array that cannot change. It compares equal to a ``list``, and ``json.dumps`` writes it as one."""

    __slots__ = ()

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse_change
    append = clear = extend = insert = pop = remove = reverse = sort = _refuse_change

    def __reduce__(self) -> tuple[type, tuple[list]]:
        # Unpickling a list subclass would otherwise append each item, which this class refuses.
        return (_FrozenJsonArray, (list(self),))


def _frozen(value: object) -> object:
    """Return a copy of a validated JSON value in which no object or array can change."""
    if isinstance(value, dict):
        return _FrozenJsonObject({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenJsonArray(_frozen(item) for item in value)
    return value


def freeze_json_mapping(value: object) -> JsonMapping:
    """Validate a metadata mapping and return a copy of it that cannot change.

    Args:
        value: Candidate JSON metadata, validated with the same rules as :func:`validate_json_mapping`.

    Returns:
        A detached mapping with immutable nested objects and arrays, still serializable as JSON.
    """
    return cast("JsonMapping", _frozen(validate_json_mapping(value)))
