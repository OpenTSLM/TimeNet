from enum import StrEnum

import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.metadata import MetadataExtension, RecordMetadata


class _Sampling(StrEnum):
    RANDOM = "random"


class _OtherMetadata(MetadataExtension):
    NAME = "other"
    note: str


class _SubjectMetadata(MetadataExtension):
    NAME = "subject_id"


class _DeviceMetadata(MetadataExtension):
    NAME = "device"
    name: str


class _TypedMetadata(MetadataExtension):
    NAME = "typed"
    grams: float
    sampling: _Sampling


def test_metadata_extensions_keep_their_own_names():
    core = RecordMetadata(subject_id="S1")
    metadata = core.extend(_OtherMetadata(note="x"))
    assert core.to_json() == {"subject_id": "S1"}
    assert metadata.to_json() == {"subject_id": "S1", "other": {"note": "x"}}
    assert RecordMetadata.from_json(metadata.to_json(), [_OtherMetadata]) == metadata
    # No extension may overwrite a field or another extension.
    with pytest.raises(TimeFValidationError, match="already has a 'other' extension"):
        metadata.extend(_OtherMetadata(note="y"))
    with pytest.raises(TimeFValidationError, match="'subject_id' is a field of RecordMetadata"):
        metadata.extend(_SubjectMetadata())
    # A stored key that is neither a field nor a known extension is not dropped silently.
    with pytest.raises(TimeFValidationError, match=r"neither fields nor known extensions: \['other'\]"):
        RecordMetadata.from_json(metadata.to_json(), [])
    with pytest.raises(TimeFValidationError, match="two extension types share a name"):
        RecordMetadata.from_json({}, [_OtherMetadata, _OtherMetadata])
    with pytest.raises(TimeFValidationError, match="'other' is not a JSON object"):
        RecordMetadata.from_json({"other": "x"}, [_OtherMetadata])


def test_metadata_extensions_nest_and_keep_their_types():
    typed = _TypedMetadata(grams=45, sampling=_Sampling.RANDOM).extend(_DeviceMetadata(name="watch"))
    metadata = RecordMetadata(subject_id="S1").extend(typed)
    stored = metadata.to_json()
    assert stored == {
        "subject_id": "S1",
        "typed": {"grams": 45.0, "sampling": "random", "device": {"name": "watch"}},
    }
    assert repr(metadata) == (
        "RecordMetadata(subject_id='S1', typed=_TypedMetadata(grams=45.0, sampling=<_Sampling.RANDOM: 'random'>, "
        "device=_DeviceMetadata(name='watch')))"
    )
    restored = RecordMetadata.from_json(stored, [_TypedMetadata, _DeviceMetadata])
    assert restored == metadata
    assert restored.extension(_TypedMetadata).sampling is _Sampling.RANDOM
    assert restored.extension(_TypedMetadata).extension(_DeviceMetadata).name == "watch"
    with pytest.raises(TimeFValidationError, match="has no 'device' extension"):
        restored.extension(_DeviceMetadata)
