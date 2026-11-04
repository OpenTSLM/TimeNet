import pickle

import pytest

from timenet.errors import (
    TimeFFormatError,
    TimeFNotFoundError,
    TimeFReaderError,
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetConnectorError,
    TimeNetDatasetNotFoundError,
    TimeNetDownloadError,
    TimeNetError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
    TimeNetVersionConflictError,
    TimeNetVersionUnavailableError,
)
from timenet.types import Access, Version
from timenet.types.access import AccessRequirement
from timenet.types.references import DatasetRef


def test_all_derive_from_base():
    for exc in (
        TimeNetRegistryError,
        TimeNetDatasetNotFoundError,
        TimeNetConnectorError,
        TimeNetDownloadError,
        TimeFValidationError,
        TimeFFormatError,
        TimeFReaderError,
    ):
        assert issubclass(exc, TimeNetError)


def test_a_download_error_is_a_connector_error():
    assert issubclass(TimeNetDownloadError, TimeNetConnectorError)


def test_validation_error_is_value_error():
    # So existing `except ValueError` handlers still catch dataset/writer validation failures.
    assert issubclass(TimeFValidationError, ValueError)


def test_not_found_error_is_a_validation_error_and_a_lookup_error():
    # Callers can tell a missing object from malformed input, and `except LookupError` catches it.
    assert issubclass(TimeFNotFoundError, TimeFValidationError)
    assert issubclass(TimeFNotFoundError, LookupError)


def test_catchable_as_base():
    try:
        raise TimeNetDatasetNotFoundError("nope")
    except TimeNetError as exc:
        assert str(exc) == "nope"


_A, _B, _C = DatasetRef.parse("org/a@1.0.0"), DatasetRef.parse("org/b@2.0.0"), DatasetRef.parse("org/c@3.0.0")
_PAGE_A = AccessRequirement(_A, Access.CREDENTIALED, "https://a.example")
_PAGE_B = AccessRequirement(_B, Access.RESTRICTED, "https://b.example")
_PAGE_C = AccessRequirement(_C, Access.RESTRICTED, "https://c.example")


@pytest.mark.parametrize(
    ("error", "facts", "message"),
    [
        (
            TimeNetMissingParentError(_A, _B, "here", payload_source=True),
            {"missing": _A, "needed_by": _B, "payload_source": True},
            "payload source org/a@1.0.0 of org/b@2.0.0 is not in here.",
        ),
        (
            TimeNetMissingParentError(_A, _B, "here", payload_source=False),
            {"payload_source": False},
            "parent release org/a@1.0.0 of org/b@2.0.0 is not in here.",
        ),
        (
            TimeNetFileError(_A, "org/a@1.0.0", "control.duckdb", "1,024 bytes", None),
            {"release": _A, "path": "control.duckdb", "expected": "1,024 bytes", "actual": None},
            "org/a@1.0.0: file 'control.duckdb' is missing, expected 1,024 bytes.",
        ),
        (
            TimeNetFileError(_A, "org/a@1.0.0", "control.duckdb", "1,024 bytes", "9 bytes"),
            {"expected": "1,024 bytes", "actual": "9 bytes"},
            "org/a@1.0.0: file 'control.duckdb' has 9 bytes, expected 1,024 bytes.",
        ),
        (
            TimeNetFileError(_A, "download of org/a@1.0.0", "values.parquet", "sha256:aa", "sha256:bb"),
            {"owner": "download of org/a@1.0.0", "expected": "sha256:aa", "actual": "sha256:bb"},
            "download of org/a@1.0.0: file 'values.parquet' has sha256:bb, expected sha256:aa.",
        ),
        (
            TimeNetAccessError(_A, Access.CREDENTIALED, (_PAGE_A,)),
            {"dataset": _A, "access": Access.CREDENTIALED, "requirements": (_PAGE_A,)},
            "org/a@1.0.0 is credentialed. Get access at https://a.example for org/a@1.0.0.",
        ),
        (
            TimeNetAccessError(_C, Access.RESTRICTED, (_PAGE_A, _PAGE_B)),
            {"dataset": _C, "access": Access.RESTRICTED, "requirements": (_PAGE_A, _PAGE_B)},
            "org/c@3.0.0 is restricted. "
            "Get access at https://a.example for org/a@1.0.0 and at https://b.example for org/b@2.0.0.",
        ),
        (
            TimeNetAccessError(_C, Access.RESTRICTED, (_PAGE_C, _PAGE_A, _PAGE_B)),
            {"requirements": (_PAGE_C, _PAGE_A, _PAGE_B)},
            "org/c@3.0.0 is restricted. Get access at https://c.example for org/c@3.0.0, "
            "at https://a.example for org/a@1.0.0 and at https://b.example for org/b@2.0.0.",
        ),
        (
            # A card that a test or a tool builds in code can lack the page.
            TimeNetAccessError(_A, Access.CREDENTIALED, ()),
            {"requirements": ()},
            "org/a@1.0.0 is credentialed.",
        ),
        (
            TimeNetVersionConflictError(_A, "registry here", "registry there"),
            {
                "release": _A,
                "registry": "registry here",
                "source": "registry there",
                "needed_by": None,
                "difference": "files",
            },
            "registry here already holds org/a@1.0.0 with other files than registry there. "
            "A published version cannot change, so a changed release needs a new version.",
        ),
        (
            TimeNetVersionConflictError(_A, "registry here", "registry there", _B, "card"),
            {"release": _A, "needed_by": _B, "difference": "card"},
            "registry here already holds org/a@1.0.0, a payload source of org/b@2.0.0, with another card than "
            "registry there. A published version cannot change, so a changed release needs a new version.",
        ),
        (
            TimeNetVersionUnavailableError(_B, Version(1, 0, 0)),
            {"release": _B, "declared": Version(1, 0, 0)},
            "connector for 'org/b' builds version 1.0.0, not the requested 2.0.0. "
            "Ask for version 1.0.0, or leave out the version.",
        ),
    ],
)
def test_an_error_with_facts_ends_with_the_hint_of_its_front_end(error, facts, message):
    assert str(error) == message
    error.hint = "Do this instead."
    # A worker process may pickle the error; its facts and its hint come back.
    restored = pickle.loads(pickle.dumps(error))
    assert {name: getattr(restored, name) for name in facts} == facts
    # The message ends with a period, and the hint follows as its own sentence.
    assert str(restored) == f"{message} Do this instead."


def test_a_version_that_the_connector_does_not_build_is_not_found_and_invalid():
    error = TimeNetVersionUnavailableError(_B, Version(1, 0, 0))
    # The client reports a missing version, and a build reports invalid input. Both catch it.
    assert isinstance(error, TimeNetDatasetNotFoundError)
    assert isinstance(error, TimeFValidationError)
