import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.metadata import MetadataExtension, RecordMetadata


class _OtherMetadata(MetadataExtension):
    NAME = "other"
    note: str


class _SubjectMetadata(MetadataExtension):
    NAME = "subject_id"


def test_metadata_extensions_keep_their_own_names():
    metadata = RecordMetadata(subject_id="S1").extend(_OtherMetadata(note="x"))
    assert metadata.to_json() == {"subject_id": "S1", "other": {"note": "x"}}
    assert RecordMetadata.from_json(metadata.to_json(), [_OtherMetadata]) == metadata
    # No extension may overwrite a core field or another extension.
    with pytest.raises(TimeFValidationError, match="already has a 'other' extension"):
        metadata.extend(_OtherMetadata(note="y"))
    with pytest.raises(TimeFValidationError, match="'subject_id' is a core metadata field"):
        metadata.extend(_SubjectMetadata())
    # A stored key that is neither a core field nor a known extension is not dropped silently.
    with pytest.raises(TimeFValidationError, match=r"neither core fields nor known extensions: \['other'\]"):
        RecordMetadata.from_json(metadata.to_json(), [])
