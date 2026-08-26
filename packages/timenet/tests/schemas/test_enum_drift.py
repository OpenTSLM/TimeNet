"""Guard: the enum blocks in the packaged JSON Schemas must match the Python enums they mirror.

Adding a license/domain/annotation/task in Python without updating the schema fails here.
"""

from timenet.schemas import DATASET_CARD_SCHEMA, MANIFEST_SCHEMA
from timenet.types import Access, AnnotationType, Domain, License, TaskType
from timenet.writer.value_encoding import ValueEncoding


def _schema_enum(schema, name):
    return sorted(schema["$defs"][name]["enum"])


def _values(enum):
    return sorted(member.value for member in enum)


def test_card_enums_match_python():
    assert _schema_enum(DATASET_CARD_SCHEMA, "license") == _values(License)
    assert _schema_enum(DATASET_CARD_SCHEMA, "domain") == _values(Domain)
    assert _schema_enum(DATASET_CARD_SCHEMA, "access") == _values(Access)


def test_manifest_shared_enums_match_python():
    assert _schema_enum(MANIFEST_SCHEMA, "license") == _values(License)
    assert _schema_enum(MANIFEST_SCHEMA, "domain") == _values(Domain)
    assert _schema_enum(MANIFEST_SCHEMA, "access") == _values(Access)


def test_manifest_annotation_and_task_enums_match_python():
    assert _schema_enum(MANIFEST_SCHEMA, "annotationType") == _values(AnnotationType)
    assert _schema_enum(MANIFEST_SCHEMA, "taskType") == _values(TaskType)


def test_manifest_value_type_enum_matches_the_closed_set():
    # Mirrors the tags returned by timenet.types.value_type_of (no Python enum backs it).
    assert _schema_enum(MANIFEST_SCHEMA, "valueType") == ["bool", "float", "int", "list", "map", "str"]


def test_manifest_value_encoding_enum_matches_python():
    assert _schema_enum(MANIFEST_SCHEMA, "valueEncoding") == _values(ValueEncoding)


def test_card_and_manifest_share_identical_shared_defs():
    for name in ("license", "domain", "datasetId", "semver"):
        assert DATASET_CARD_SCHEMA["$defs"][name] == MANIFEST_SCHEMA["$defs"][name]
