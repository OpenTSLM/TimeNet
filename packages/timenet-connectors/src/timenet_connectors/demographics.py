"""Shared annotation keys and values for the people in a dataset.

Users filter and join datasets by these annotations, so every dataset spells them the same way.
Each connector converts its source's own codes, such as ``is_male`` or ``F=1``, into these values
before it builds the annotations with the functions below.

The annotations are:

- ``subject_ids``: a list of the record's subject identifiers, as strings. The TimeF format
  documents this key.
- ``sex``: one :class:`Sex` value. :data:`SEXES` is its label set, for a task that asks for it.
- ``age``: a whole number of years, the unit in :data:`AGE_UNIT`. The release stores one value
  type for each key, so a dataset cannot mix whole and fractional ages.
"""

from collections.abc import Iterable, Mapping, Set
from enum import Enum, StrEnum, unique

from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, Label, LabelSet


@unique
class DemographicKey(StrEnum):
    """The annotation keys for facts about the people in a record."""

    SUBJECT_IDS = "subject_ids"
    SEX = "sex"
    AGE = "age"


@unique
class Sex(Enum):
    """A person's sex, after the connector decodes its source's code.

    It is a plain ``Enum``, not a ``str``, so that an undecoded source code never passes for it.
    Write ``sex.value`` to store or compare the text.
    """

    MALE = "male"
    FEMALE = "female"


SEXES = LabelSet(
    name=DemographicKey.SEX.value,
    labels=(
        Label(name=Sex.MALE.value, description="The person is male."),
        Label(name=Sex.FEMALE.value, description="The person is female."),
    ),
)
"""The label set for a classification task whose target is a :class:`Sex` value."""

_LARGEST_STORED_INT = 2**63 - 1
"""The largest int that a release stores in an int64 column."""

AGE_UNIT = "year"
"""The unit of every ``age`` annotation and every age target."""


def subject_ids_annotation(subject_ids: Iterable[str]) -> Annotation:
    """Build the ``subject_ids`` annotation of a record.

    Args:
        subject_ids: The source's identifiers of the people in the record, in source order.

    Returns:
        The annotation, with the identifiers as a list.

    Raises:
        TimeNetConnectorError: If ``subject_ids`` is a single string, a set (whose order can change
            between builds), or a mapping, or if it holds no identifier. Each identifier must be a
            unique, nonblank string with no whitespace at either end.
    """
    if isinstance(subject_ids, str):
        raise TimeNetConnectorError(f"subject_ids {subject_ids!r} is one string; pass a list of identifiers")
    if isinstance(subject_ids, Set):
        raise TimeNetConnectorError("subject_ids is a set, whose order can change between builds; pass a list")
    if isinstance(subject_ids, Mapping):
        raise TimeNetConnectorError("subject_ids is a mapping, which gives only its keys; pass a list")
    ids = list(subject_ids)
    if not ids:
        raise TimeNetConnectorError("a subject_ids annotation needs at least one identifier")
    for one in ids:
        if not isinstance(one, str) or not one.strip():
            raise TimeNetConnectorError(f"subject identifier {one!r} is not a nonblank string")
        if one != one.strip():
            # A filter on "a" does not match " a". Fix the source or the connector, never strip here.
            raise TimeNetConnectorError(f"subject identifier {one!r} has whitespace around it")
    if len(set(ids)) != len(ids):
        raise TimeNetConnectorError(f"subject identifiers {ids!r} repeat a value")
    return Annotation(key=DemographicKey.SUBJECT_IDS.value, value=ids)


def sex_annotation(sex: Sex, *, description: str | None = None) -> Annotation:
    """Build the ``sex`` annotation of a record.

    Args:
        sex: The decoded sex.
        description: How the source states the sex, if a reader needs to know.

    Returns:
        The annotation, with the value of ``sex``.

    Raises:
        TimeNetConnectorError: If ``sex`` is not a :class:`Sex` value, such as an undecoded source
            code like ``"F"``.
    """
    if not isinstance(sex, Sex):
        raise TimeNetConnectorError(
            f"expected a Sex value, got {sex!r}; convert the source code to Sex.MALE or Sex.FEMALE first"
        )
    return Annotation(key=DemographicKey.SEX.value, value=sex.value, description=description)


def age_annotation(age: int, *, description: str | None = None) -> Annotation:
    """Build the ``age`` annotation of a record, in whole years of :data:`AGE_UNIT`.

    Args:
        age: The age in whole years. For a source cell such as ``"56.0"``, check that the
            parsed number is finite and whole before converting it to an ``int``. If the source
            gives fractional years, use another key, because every ``age`` in a release must have
            the same value type.
        description: How the source states the age, if a reader needs to know.

    Returns:
        The annotation.

    Raises:
        TimeNetConnectorError: If ``age`` is not an integer of zero or more, or is too large for
            the 64-bit integer that a release stores it in.
    """
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise TimeNetConnectorError(f"age {age!r} must be an integer of zero or more years")
    if age > _LARGEST_STORED_INT:
        raise TimeNetConnectorError(f"age {age} is too large for the 64-bit integer that a release stores")
    return Annotation(key=DemographicKey.AGE.value, value=age, unit=AGE_UNIT, description=description)
