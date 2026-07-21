from timenet.errors import (
    DatasetNotFoundError,
    InvalidManifestError,
    RegistryError,
    TimeFFormatError,
    TimeFValidationError,
    TimeNetError,
)


def test_all_derive_from_base():
    for exc in (RegistryError, DatasetNotFoundError, TimeFValidationError, TimeFFormatError, InvalidManifestError):
        assert issubclass(exc, TimeNetError)


def test_validation_error_is_value_error():
    # So existing `except ValueError` handlers still catch dataset/writer validation failures.
    assert issubclass(TimeFValidationError, ValueError)


def test_invalid_manifest_is_format_and_value_error():
    assert issubclass(InvalidManifestError, TimeFFormatError)
    assert issubclass(InvalidManifestError, ValueError)


def test_catchable_as_base():
    try:
        raise DatasetNotFoundError("nope")
    except TimeNetError as exc:
        assert str(exc) == "nope"
