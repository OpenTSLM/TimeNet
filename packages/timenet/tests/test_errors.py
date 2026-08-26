from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetDownloadError,
    TimeNetError,
    TimeNetInvalidManifestError,
    TimeNetRegistryError,
)


def test_all_derive_from_base():
    for exc in (
        TimeNetRegistryError,
        TimeNetDatasetNotFoundError,
        TimeNetDownloadError,
        TimeFValidationError,
        TimeFFormatError,
        TimeNetInvalidManifestError,
    ):
        assert issubclass(exc, TimeNetError)


def test_validation_error_is_value_error():
    # So existing `except ValueError` handlers still catch dataset/writer validation failures.
    assert issubclass(TimeFValidationError, ValueError)


def test_invalid_manifest_is_format_and_value_error():
    assert issubclass(TimeNetInvalidManifestError, TimeFFormatError)
    assert issubclass(TimeNetInvalidManifestError, ValueError)


def test_catchable_as_base():
    try:
        raise TimeNetDatasetNotFoundError("nope")
    except TimeNetError as exc:
        assert str(exc) == "nope"
