from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import AliasChoices, Field
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataRow, LocatedRow, NonBlankText, PublishedEnum, UniqueValues


class Row(DataRow):
    identifier: NonBlankText = Field(alias="id")
    age: Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class EntryLocation:
    index: int

    @property
    def context(self) -> str:
        return f"entry {self.index}"


def test_decoded_values_are_strict_while_cells_parse_strings():
    row = Row.from_strings({"id": "001", "age": "42"}, context="csv:2")
    assert row == Row.from_decoded({"id": "001", "age": 42}, context="json entry 1")
    with pytest.raises(TimeNetConnectorError, match="json entry 1: age:") as error:
        Row.from_decoded({"id": "001", "age": "42"}, context="json entry 1")
    assert error.value.__cause__ is not None


def test_nested_validation_errors_retain_field_path():
    class NestedRow(DataRow):
        children: list[Row]

    with pytest.raises(TimeNetConnectorError, match=r"entry 5: children\.0\.age:"):
        NestedRow.from_decoded({"children": [{"id": "001", "age": -1}]}, context="entry 5")


def test_column_contract_uses_validation_alias_and_optional_defaults():
    class AliasedRow(DataRow):
        identifier: str = Field(validation_alias="source_id")
        note: str = "unpublished"

    AliasedRow.check_columns(["source_id"], context="table")
    assert AliasedRow.from_strings({"source_id": "001"}, context="table:2").note == "unpublished"
    with pytest.raises(TimeNetConnectorError, match=r"missing columns.*source_id"):
        AliasedRow.check_columns(["identifier"], context="table")


def test_ambiguous_column_schemas_fail_explicitly():
    class AlternativeRow(DataRow):
        identifier: str = Field(validation_alias=AliasChoices("id", "identifier"))

    class DuplicateAliasRow(DataRow):
        first: str = Field(alias="id")
        second: str = Field(alias="id")

    with pytest.raises(TimeNetConnectorError, match="single string alias"):
        AlternativeRow.check_columns(["id"], context="table")
    with pytest.raises(TimeNetConnectorError, match="multiple fields"):
        DuplicateAliasRow.check_columns(["id"], context="table")


class Mode(Enum):
    DAILY = "daily"


class EnumRow(DataRow):
    mode: PublishedEnum[Mode]


def test_published_enum_parses_its_string_value():
    assert EnumRow.from_decoded({"mode": "daily"}, context="entry").mode is Mode.DAILY


@pytest.mark.parametrize("value", [1, True, b"daily", None, "unknown"])
def test_published_enum_rejects_other_types_and_unknown_values(value):
    with pytest.raises(TimeNetConnectorError, match="entry: mode:"):
        EnumRow.from_decoded({"mode": value}, context="entry")


def test_uniqueness_rejects_unhashable_fields():
    class ListRow(DataRow):
        values: list[int]

    unique = UniqueValues[ListRow, EntryLocation](ListRow, "values")
    with pytest.raises(TimeNetConnectorError, match="hashable"):
        unique.add(LocatedRow(ListRow(values=[1]), EntryLocation(0)))
